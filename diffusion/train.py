#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Train the conditional DDPM on the pooled cross-section of stocks (optional multi-GPU DDP
and bf16 AMP).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ivs.config import (
    AMP_DTYPE, BATCH_SIZE, COND_DIM, CKPT_EVERY_EPOCHS, DEFAULT_VERSION, DROPOUT,
    EMA_DECAY, EPOCHS, EXPERIMENT, GLOBAL_SEED, GRADIENT_CLIP, HIDDEN_DIM,
    LEARNING_RATE, LOG_EVERY_STEPS, NUM_BLOCKS, NUM_TIMESTEPS, BETA_SCHEDULE,
    M_GRID, ARB_CLAMP_NORM, SURFACE_DIM, SURFACE_M, SURFACE_T, TAU_GRID_YEARS,
    TARGET_DIM, TICKERS, TIME_EMB_DIM, USE_AMP, USE_EMA, VERSIONS, WEIGHT_DECAY,
    ckpt_dir, describe, log_dir, version_cfg, NORM_DIR, set_global_seed,
)
from diffusion.model.denoiser import DiffusionDenoiser, EMA
from diffusion.model.process import DiffusionSchedule, make_beta_schedule
from ivs.data.normalizer import Normalizer, fit_normalizer
from ivs.data.dataset import build_features, load_ticker, date_split, rates_per_example
from diffusion.losses import (
    diffusion_loss, grid_spacings, penalties_from_pred, total_loss,
)

_AMP_DTYPE = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16
_TERM_NAMES = ("L_m", "L_tau", "p_calendar", "p_spread", "p_butterfly")


# Variant switches
def variant_spec(version: str) -> tuple[bool, bool, str, tuple[float, ...]]:
    """(use_smoothness, use_arbitrage, smoothness_mode, five lambdas) from config."""
    v = version_cfg(version)
    use_smooth = bool(v.get("use_smoothness", False))
    use_arb = bool(v.get("use_arbitrage", False))
    lambdas = (
        float(v.get("lambda_m", 0.0)) if use_smooth else 0.0,
        float(v.get("lambda_tau", 0.0)) if use_smooth else 0.0,
        float(v.get("lambda_calendar", 0.0)) if use_arb else 0.0,
        float(v.get("lambda_spread", 0.0)) if use_arb else 0.0,
        float(v.get("lambda_butterfly", 0.0)) if use_arb else 0.0,
    )
    return use_smooth, use_arb, str(v.get("smoothness_mode", "logiv_fd")), lambdas


# DDP helpers
def setup_ddp(device_str: str):
    """Return (rank, local_rank, world_size, device, ddp_active).

    torchrun sets RANK/LOCAL_RANK/WORLD_SIZE -> NCCL process group on
    cuda:local_rank. No torchrun env -> single-process fallback on ``device_str``.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}"), True
    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, dev, False


def _load_dataset(ticker: str):
    arrays = load_ticker(ticker)
    cond, target, dates = build_features(arrays)
    r = rates_per_example(arrays)
    sl_tr, sl_te = date_split(dates)
    return cond, target, r, sl_tr, sl_te


def _plot_loss_curve(train_log: list, out_path: Path, use_penalty: bool) -> None:
    """Per-epoch loss curve (log-y) + CSV dump. Rank-0 only."""
    if not train_log:
        return
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in train_log]
    ld = [r["loss_mse"] for r in train_log]
    lt = [r["loss_total"] for r in train_log]

    with open(out_path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss_mse", "loss_total", *_TERM_NAMES])
        for r in train_log:
            w.writerow([r["epoch"], r["loss_mse"], r["loss_total"],
                        *(r.get(k, 0.0) for k in _TERM_NAMES)])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep, ld, lw=1.0, color="#2196F3", label="loss_mse (DDPM eps-MSE)")
    if use_penalty:
        ax.plot(ep, lt, lw=1.0, color="#C0392B", alpha=0.8, label="loss_total (+ penalties)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"universal DDPM training loss -- {EXPERIMENT}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[train] loss curve -> {out_path}", flush=True)


def train(version: str, device: torch.device, rank: int, world_size: int,
          ddp_active: bool, epochs: int | None = None) -> None:
    is_main = (rank == 0)
    use_smooth, use_arb, smooth_mode, lambdas = variant_spec(version)
    use_penalty = use_smooth or use_arb
    n_epochs: int = EPOCHS if epochs is None else int(epochs)

    if is_main:
        print(describe(), flush=True)
        print(f"[train] variant={version}  smoothness={use_smooth}  arbitrage={use_arb}",
              flush=True)
        if use_penalty:
            print("[train] lambdas  " + "  ".join(
                f"{n}={l:.4e}" for n, l in zip(_TERM_NAMES, lambdas)), flush=True)

    # 1. Load + split per ticker (every rank loads independently)
    all_cond_tr, all_tgt_tr, all_r_tr = [], [], []
    test_meta = {}
    if is_main:
        print(f"[train] loading data for {len(TICKERS)} tickers ...", flush=True)
    for tk in TICKERS:
        cond, target, r, sl_tr, sl_te = _load_dataset(tk)
        all_cond_tr.append(cond[sl_tr])
        all_tgt_tr.append(target[sl_tr])
        all_r_tr.append(r[sl_tr])
        test_meta[tk] = {"n_train": int(cond[sl_tr].shape[0]),
                         "n_test": int(cond[sl_te].shape[0])}

    cond_tr_pool = np.concatenate(all_cond_tr, axis=0)
    tgt_tr_pool = np.concatenate(all_tgt_tr, axis=0)
    r_tr_pool = np.concatenate(all_r_tr, axis=0).astype(np.float32)
    n_pool = cond_tr_pool.shape[0]
    if is_main:
        print(f"[train] pooled train = {n_pool} examples", flush=True)

    # 2. Fit normalizer (deterministic on all ranks; rank 0 persists)
    norm_path = NORM_DIR / "norm_params.npz"
    if norm_path.exists():
        norm = Normalizer.load(norm_path)
        if is_main:
            print(f"[train] loaded existing normalizer {norm_path}", flush=True)
    else:
        norm = fit_normalizer(all_cond_tr, all_tgt_tr)
        if is_main:
            norm.save(norm_path)
            with open(NORM_DIR / "data_meta.json", "w") as f:
                json.dump(test_meta, f, indent=2)
            print(f"[train] normalizer fitted + saved -> {norm_path}", flush=True)

    cond_tr_norm = norm.normalize_cond(cond_tr_pool)
    tgt_tr_norm = norm.normalize_target(tgt_tr_pool)

    # Device-resident tensors (per rank, on its own GPU)
    cond_n_gpu = torch.from_numpy(cond_tr_norm).to(device)
    tgt_n_gpu = torch.from_numpy(tgt_tr_norm).to(device)
    cond_raw_gpu = torch.from_numpy(cond_tr_pool).to(device)   # raw cond -> log sigma_{t-1}
    r_gpu = torch.from_numpy(r_tr_pool).to(device)             # per-example risk-free rate
    target_mean_t = torch.from_numpy(norm.target_mean).to(device)
    target_std_t = torch.from_numpy(norm.target_std).to(device)
    m_grid_t = torch.tensor(M_GRID, dtype=torch.float32, device=device)
    tau_grid_t = torch.tensor(TAU_GRID_YEARS, dtype=torch.float32, device=device)
    d_m_t, d_tau_t = grid_spacings(M_GRID, TAU_GRID_YEARS, device)

    # 3. Sharded on-GPU batching
    per_rank = n_pool // world_size                  # examples this rank owns / epoch
    n_batches = per_rank // BATCH_SIZE               # drop_last; identical across ranks
    if n_batches < 1:
        raise ValueError(
            f"per-rank examples ({per_rank}) < batch_size ({BATCH_SIZE}); "
            f"lower training.batch_size or training.world_size")
    if is_main:
        print(f"[train] world_size={world_size}  per-rank={per_rank}  "
              f"{n_batches} batches/epoch/rank  per-GPU batch={BATCH_SIZE}  "
              f"effective batch={BATCH_SIZE * world_size}  AMP={USE_AMP}({AMP_DTYPE})",
              flush=True)

    # 4. Model + schedule
    model = DiffusionDenoiser(
        x_dim=TARGET_DIM, cond_dim=COND_DIM,
        hidden_dim=HIDDEN_DIM, time_emb_dim=TIME_EMB_DIM,
        num_blocks=NUM_BLOCKS, dropout=DROPOUT,
    ).to(device)
    if ddp_active:
        model = DDP(model, device_ids=[device.index], output_device=device.index)
    core = model.module if ddp_active else model        # unwrapped for EMA / state_dict
    if is_main:
        print(f"[train] model: {sum(p.numel() for p in core.parameters()):,} parameters",
              flush=True)

    # EMA tracks the (DDP-synced) core params; maintained on rank 0 only.
    ema = EMA(core, decay=EMA_DECAY) if (USE_EMA and is_main) else None

    sched = DiffusionSchedule(make_beta_schedule(BETA_SCHEDULE, NUM_TIMESTEPS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    # 5. Training loop
    ckpt_path = ckpt_dir(version)
    log_path = log_dir(version)
    train_log = []
    global_step = 0
    t0_wall = time.time()
    if is_main:
        print(f"[train] starting {n_epochs} epochs ...", flush=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_mse = ep_total = 0.0
        ep_terms = [0.0] * len(_TERM_NAMES)
        n_done = 0

        # SHARED permutation (CPU, seeded by epoch only) -> identical on every rank
        g = torch.Generator().manual_seed(GLOBAL_SEED + epoch)
        perm = torch.randperm(n_pool, generator=g)
        shard = perm[rank * per_rank:(rank + 1) * per_rank].to(device)

        for bi in range(n_batches):
            idx = shard[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            cond_n = cond_n_gpu[idx]
            tgt_n = tgt_n_gpu[idx]
            cond_raw = cond_raw_gpu[idx]
            r_b = r_gpu[idx]

            B = tgt_n.shape[0]
            t_rand = torch.randint(1, NUM_TIMESTEPS + 1, (B,), device=device)
            eps = torch.randn_like(tgt_n)

            with torch.autocast(device_type=device.type, dtype=_AMP_DTYPE,
                                enabled=USE_AMP and device.type == "cuda"):
                x_t, _ = sched.q_sample(tgt_n, t_rand, noise=eps)
                eps_pred = model(x_t, t_rand, cond_n)
                loss_mse = diffusion_loss(eps_pred, eps)

            loss = loss_mse.float()
            term_vals = [0.0] * len(_TERM_NAMES)
            if use_penalty:
                # x0_hat in fp32, then the penalties with autocast disabled (one
                # surface reconstruction feeds every live term).
                x0_hat_norm = sched.predict_x0_from_eps(
                    x_t.float(), t_rand, eps_pred.float())
                with torch.autocast(device_type=device.type, enabled=False):
                    terms = penalties_from_pred(
                        x0_hat_norm, cond_raw, r_b,
                        surface_m=SURFACE_M, surface_t=SURFACE_T,
                        surface_dim=SURFACE_DIM,
                        target_mean=target_mean_t, target_std=target_std_t,
                        m_grid=m_grid_t, tau_grid=tau_grid_t,
                        d_m=d_m_t, d_tau=d_tau_t,
                        use_smoothness=use_smooth, use_arbitrage=use_arb,
                        smoothness_mode=smooth_mode, clamp_norm=ARB_CLAMP_NORM,
                    )
                loss = total_loss(loss_mse, terms, lambdas)
                term_vals = [float(t.item()) for t in terms]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            if ema is not None:
                ema.update(core)

            ep_mse += loss_mse.item()
            ep_total += loss.item()
            ep_terms = [a + b for a, b in zip(ep_terms, term_vals)]
            n_done += 1
            global_step += 1

            if is_main and global_step % LOG_EVERY_STEPS == 0:
                el = (time.time() - t0_wall) / 60
                extra = "  ".join(f"{n}={v:.3e}" for n, v in zip(_TERM_NAMES, term_vals)
                                  ) if use_penalty else ""
                print(f"[train] step={global_step:>7}  epoch={epoch:>5}  "
                      f"loss_mse={loss_mse.item():.6f}  loss_total={loss.item():.6f}  "
                      f"{extra}  elapsed={el:.1f}min", flush=True)

        nd = max(n_done, 1)
        train_log.append({
            "epoch": epoch, "global_step": global_step,
            "loss_mse": ep_mse / nd, "loss_total": ep_total / nd,
            **{n: v / nd for n, v in zip(_TERM_NAMES, ep_terms)},
        })

        if is_main and (epoch % CKPT_EVERY_EPOCHS == 0 or epoch == n_epochs):
            ckpt_file = ckpt_path / f"epoch_{epoch:05d}.pt"
            payload = {
                "epoch": epoch, "global_step": global_step,
                "model": core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "version": version,
            }
            if ema is not None:
                payload["ema"] = ema.state_dict()
            torch.save(payload, ckpt_file)
            print(f"[train] checkpoint -> {ckpt_file}", flush=True)

    # 6. Final checkpoint (rank 0)
    if is_main:
        avg_mse = train_log[-1]["loss_mse"]
        avg_total = train_log[-1]["loss_total"]
        final_path = ckpt_path / "final.pt"
        payload = {
            "epoch": n_epochs, "global_step": global_step,
            "model": core.state_dict(), "version": version,
            "experiment": EXPERIMENT,
            "lambdas": dict(zip(_TERM_NAMES, lambdas)),
            "loss_mse_final": avg_mse,
        }
        if ema is not None:
            backup = {n: p.data.clone() for n, p in core.named_parameters()}
            ema.apply(core)
            payload["model_ema"] = core.state_dict()
            ema.restore(core, backup)
            payload["ema"] = ema.state_dict()
        torch.save(payload, final_path)
        print(f"[train] final checkpoint -> {final_path}", flush=True)

        with open(log_path / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)
        _plot_loss_curve(train_log, log_path / "loss_curve.png", use_penalty)

        el = (time.time() - t0_wall) / 60
        print(f"\n[train] done: {n_epochs} epochs in {el:.1f} min  "
              f"final loss_mse={avg_mse:.6f}  loss_total={avg_total:.6f}", flush=True)

    if ddp_active:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    ap = argparse.ArgumentParser(description="Train the universal conditional DDPM")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=list(VERSIONS),
                    help="loss variant key inside the config (default: the only one)")
    ap.add_argument("--device", default="cuda:0",
                    help="single-process fallback device (ignored under torchrun)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training.epochs (smoke tests)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device, ddp_active = setup_ddp(args.device)
    seed = set_global_seed(GLOBAL_SEED + rank)
    print(f"[train] rank={rank}/{world_size} device={device} ddp={ddp_active} "
          f"seed={seed}", flush=True)
    train(args.version, device, rank, world_size, ddp_active, epochs=args.epochs)


if __name__ == "__main__":
    main()
