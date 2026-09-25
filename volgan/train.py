#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Based on VolGAN: https://github.com/milenavuletic/VolGAN

Train the pooled VolGAN benchmark with gradient-matched smoothness penalties.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ivs.config import (
    BATCH_SIZE, CKPT_EVERY_EPOCHS, COND_DIM, DEFAULT_VERSION, EPOCHS, HIDDEN_DIM,
    LR_D, LR_G, LOG_EVERY_STEPS, M_GRID, N_GRAD_MATCH, NOISE_DIM, NORM_DIR, SEED,
    SURFACE_DIM, TARGET_DIM, TAU_GRID_YEARS, TICKERS, VERSIONS, ckpt_dir, describe,
    log_dir, set_global_seed, version_cfg,
)
from volgan.model.networks import Discriminator, Generator
from volgan.losses import (
    build_smoothness_ops, discriminator_loss, generator_bce, smoothness_penalties,
)
from ivs.data.normalizer import Normalizer, fit_normalizer
from ivs.data.dataset import build_features, load_ticker, date_split


def _load_dataset(ticker: str):
    arrays = load_ticker(ticker)
    cond, target, dates = build_features(arrays)
    sl_tr, sl_te = date_split(dates)
    return cond, target, dates, sl_tr, sl_te


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.data.norm(2).item()) ** 2
    return math.sqrt(total)


def real_log_surface(fake_norm: torch.Tensor, prev_log_iv_raw: torch.Tensor,
                     target_mean: torch.Tensor, target_std: torch.Tensor
                     ) -> torch.Tensor:
    """Reconstruct the real log-IV surface g_t (B, SURFACE_DIM) from the
    generator's normalized output, for the smoothness penalty:

        inc_raw = fake_norm[:, 1:] * target_std[1:] + target_mean[1:]
        g_t     = inc_raw + prev_log_iv_raw

    The affine denormalization keeps the gradient flowing, so both alphas are
    calibrated on exactly the quantity the paper penalizes.
    """
    inc_raw = fake_norm[:, 1:] * target_std[1:] + target_mean[1:]
    return inc_raw + prev_log_iv_raw


def _parity_check_normalizer(norm: Normalizer) -> None:
    """Optional diagnostic comparing this normalizer against a reference one.

    The benchmark is only like-for-like if VolGAN standardizes its inputs exactly
    as the diffusion models do. Point $IVS_REFERENCE_NORMALIZER at a diffusion
    run's norm_params.npz to check that the parameters are bit-identical. Skipped
    when unset, and non-fatal either way.
    """
    ref_env = os.environ.get("IVS_REFERENCE_NORMALIZER")
    if not ref_env:
        print("[train] (normalizer parity check skipped: set "
              "$IVS_REFERENCE_NORMALIZER to enable)", flush=True)
        return
    ref = Path(ref_env).expanduser()
    if not ref.exists():
        print(f"[train] (reference normalizer not found: {ref})", flush=True)
        return
    z = np.load(ref, allow_pickle=False)
    d = max(
        float(np.abs(norm.cond_mean - z["cond_mean"]).max()),
        float(np.abs(norm.cond_std - z["cond_std"]).max()),
        float(np.abs(norm.target_mean - z["target_mean"]).max()),
        float(np.abs(norm.target_std - z["target_std"]).max()),
    )
    tag = "OK" if d < 1e-4 else "WARN(>1e-4)"
    print(f"[train] normalizer parity vs {ref}: max|delta|={d:.2e}  [{tag}]",
          flush=True)


def gradient_matching(gen, disc, gen_opt, disc_opt, criterion,
                      cond_n, tgt_n, prev_raw, target_mean, target_std, ops,
                      n_epochs, n_pool, device):
    bce_grad, m_grad, t_grad = [], [], []
    gen.train(); disc.train()
    n_batches = n_pool // BATCH_SIZE
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_pool, device=device)
        ep_g = ep_d = 0.0
        for bi in range(n_batches):
            idx = perm[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            cond = cond_n[idx]; real = tgt_n[idx]; prev = prev_raw[idx]
            B = cond.shape[0]
            # D step
            disc_opt.zero_grad(set_to_none=True)
            noise = torch.randn(B, NOISE_DIM, device=device)
            fake = gen(noise, cond).detach()
            d_real = disc(torch.cat((cond, real), dim=-1))
            d_fake = disc(torch.cat((cond, fake), dim=-1))
            d_loss = discriminator_loss(criterion, d_real, d_fake)
            d_loss.backward(); disc_opt.step()
            # G grads (BCE vs smoothness, measured separately)
            noise = torch.randn(B, NOISE_DIM, device=device)
            fake = gen(noise, cond)
            d_fake_g = disc(torch.cat((cond, fake), dim=-1))
            g_surf = real_log_surface(fake, prev, target_mean, target_std)
            m_pen, t_pen = smoothness_penalties(g_surf, ops)

            gen_opt.zero_grad(set_to_none=True)
            m_pen.backward(retain_graph=True); m_grad.append(grad_norm(gen))
            gen_opt.zero_grad(set_to_none=True)
            t_pen.backward(retain_graph=True); t_grad.append(grad_norm(gen))
            gen_opt.zero_grad(set_to_none=True)
            g_bce = generator_bce(criterion, d_fake_g)
            g_bce.backward(); bce_grad.append(grad_norm(gen)); gen_opt.step()

            ep_g += float(g_bce.item()); ep_d += float(d_loss.item())
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"[grad-match] {epoch:4d}/{n_epochs} G_BCE={ep_g/n_batches:+.4f} "
                  f"D={ep_d/n_batches:+.4f} |gBCE|={np.mean(bce_grad[-n_batches:]):.3e} "
                  f"|gM|={np.mean(m_grad[-n_batches:]):.3e} "
                  f"|gT|={np.mean(t_grad[-n_batches:]):.3e} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    eps = 1e-12
    a_m = float(np.mean(np.array(bce_grad) / np.maximum(m_grad, eps)))
    a_t = float(np.mean(np.array(bce_grad) / np.maximum(t_grad, eps)))
    print(f"[grad-match] DONE alpha_m={a_m:.4e} alpha_tau={a_t:.4e}", flush=True)
    return a_m, a_t


def train_main(gen, disc, gen_opt, disc_opt, criterion,
               cond_n, tgt_n, prev_raw, target_mean, target_std, ops,
               alpha_m, alpha_tau, n_epochs, n_pool, device, ckpt_path):
    history = []
    gen.train(); disc.train()
    n_batches = n_pool // BATCH_SIZE
    t0 = time.time(); global_step = 0
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n_pool, device=device)
        ep_g = ep_d = ep_m = ep_t = 0.0
        for bi in range(n_batches):
            idx = perm[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            cond = cond_n[idx]; real = tgt_n[idx]; prev = prev_raw[idx]
            B = cond.shape[0]
            # D
            disc_opt.zero_grad(set_to_none=True)
            noise = torch.randn(B, NOISE_DIM, device=device)
            fake = gen(noise, cond).detach()
            d_real = disc(torch.cat((cond, real), dim=-1))
            d_fake = disc(torch.cat((cond, fake), dim=-1))
            d_loss = discriminator_loss(criterion, d_real, d_fake)
            d_loss.backward(); disc_opt.step()
            # G
            gen_opt.zero_grad(set_to_none=True)
            noise = torch.randn(B, NOISE_DIM, device=device)
            fake = gen(noise, cond)
            d_fake_g = disc(torch.cat((cond, fake), dim=-1))
            g_bce = generator_bce(criterion, d_fake_g)
            g_surf = real_log_surface(fake, prev, target_mean, target_std)
            m_pen, t_pen = smoothness_penalties(g_surf, ops)
            g_loss = g_bce + alpha_m * m_pen + alpha_tau * t_pen
            g_loss.backward(); gen_opt.step()

            ep_g += float(g_loss.item()); ep_d += float(d_loss.item())
            ep_m += float(m_pen.item()); ep_t += float(t_pen.item())
            global_step += 1
            if global_step % LOG_EVERY_STEPS == 0:
                print(f"[train] step={global_step:>8} epoch={epoch:>5} "
                      f"G={g_loss.item():+.4f} D={d_loss.item():+.4f} "
                      f"m_pen={m_pen.item():.3e} t_pen={t_pen.item():.3e} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
        g_mean = ep_g / n_batches; d_mean = ep_d / n_batches
        history.append({"epoch": epoch, "g": g_mean, "d": d_mean,
                        "m_pen": ep_m / n_batches, "t_pen": ep_t / n_batches})
        if epoch % 25 == 0 or epoch == n_epochs:
            print(f"epoch {epoch:5d}/{n_epochs} G={g_mean:+.4f} D={d_mean:+.4f} "
                  f"m_pen={ep_m/n_batches:.3e} t_pen={ep_t/n_batches:.3e} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if epoch % CKPT_EVERY_EPOCHS == 0 or epoch == n_epochs:
            torch.save({"G": gen.state_dict(), "D": disc.state_dict(),
                        "epoch": epoch, "alpha_m": alpha_m, "alpha_tau": alpha_tau},
                       ckpt_path / f"epoch_{epoch:05d}.pt")
            print(f"[train] checkpoint -> {ckpt_path}/epoch_{epoch:05d}.pt", flush=True)
    return history


def plot_loss_curve(history, out_path, version):
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(10, 5))
    plt.plot(ep, [h["g"] for h in history], color="#1F4E9D", lw=1.0, label="Generator (G)")
    plt.plot(ep, [h["d"] for h in history], color="#C0392B", lw=1.0, label="Discriminator (D)")
    plt.axhline(math.log(2.0), color="gray", ls="--", lw=0.8, label="ln 2 (BCE equilibrium)")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title(f"Universal VolGAN [{version}] -- G/D training loss (pooled 50 IS tickers)")
    plt.legend(loc="upper right"); plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(out_path, dpi=140); plt.close()


def train(version: str, device_str: str, epochs: int | None, n_grad_match: int | None) -> None:
    vcfg = version_cfg(version)
    use_smooth = bool(vcfg.get("use_smoothness", True))
    n_epochs = EPOCHS if epochs is None else int(epochs)
    n_gm = N_GRAD_MATCH if n_grad_match is None else int(n_grad_match)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 1. Load + split per ticker, pool train
    all_cond_tr, all_tgt_tr = [], []
    test_meta = {}
    print(f"[train/{version}] Loading {len(TICKERS)} IS tickers ...", flush=True)
    for tk in TICKERS:
        cond, target, dates, sl_tr, sl_te = _load_dataset(tk)
        all_cond_tr.append(cond[sl_tr]); all_tgt_tr.append(target[sl_tr])
        test_meta[tk] = {"n_train": int(cond[sl_tr].shape[0]),
                         "n_test": int(cond[sl_te].shape[0])}
    cond_tr_pool = np.concatenate(all_cond_tr, axis=0).astype(np.float32)
    tgt_tr_pool = np.concatenate(all_tgt_tr, axis=0).astype(np.float32)
    n_pool = cond_tr_pool.shape[0]
    print(f"[train/{version}] pooled train = {n_pool} examples "
          f"(cond {cond_tr_pool.shape}, target {tgt_tr_pool.shape})", flush=True)

    # 2. Fit + save the ONE global normalizer (shared with the DDPM runs)
    norm_path = NORM_DIR / "norm_params.npz"
    if norm_path.exists():
        print(f"[train/{version}] Loading existing normalizer {norm_path}", flush=True)
        norm = Normalizer.load(norm_path)
    else:
        print(f"[train/{version}] Fitting GLOBAL z-score normalizer on pooled train ...",
              flush=True)
        norm = fit_normalizer(all_cond_tr, all_tgt_tr)
        norm.save(norm_path)
        print(f"[train/{version}] Normalizer saved -> {norm_path}", flush=True)
    _parity_check_normalizer(norm)
    with open(NORM_DIR / "data_meta.json", "w") as f:
        json.dump(test_meta, f, indent=2)

    cond_tr_norm = norm.normalize_cond(cond_tr_pool)
    tgt_tr_norm = norm.normalize_target(tgt_tr_pool)

    # 3. On-GPU resident pool (no CPU DataLoader)
    cond_n = torch.from_numpy(cond_tr_norm).to(device)
    tgt_n = torch.from_numpy(tgt_tr_norm).to(device)
    prev_raw = torch.from_numpy(cond_tr_pool[:, 3:].copy()).to(device)   # raw prev log-IV
    target_mean = torch.from_numpy(norm.target_mean).to(device)
    target_std = torch.from_numpy(norm.target_std).to(device)
    n_batches = n_pool // BATCH_SIZE
    print(f"[train/{version}] {n_batches} batches/epoch (batch={BATCH_SIZE}, "
          f"pool resident on {device})", flush=True)

    # 4. Model + optimizers + smoothness ops
    gen = Generator(noise_dim=NOISE_DIM, cond_dim=COND_DIM,
                    hidden_dim=HIDDEN_DIM, output_dim=TARGET_DIM).to(device)
    disc = Discriminator(in_dim=COND_DIM + TARGET_DIM, hidden_dim=HIDDEN_DIM).to(device)
    print(f"[train/{version}] G params={sum(p.numel() for p in gen.parameters()):,} "
          f"D params={sum(p.numel() for p in disc.parameters()):,}", flush=True)
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=LR_G)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=LR_D)
    criterion = nn.BCELoss().to(device)
    m_grid = torch.tensor(M_GRID, dtype=torch.float32)
    tau_grid = torch.tensor(TAU_GRID_YEARS, dtype=torch.float32)
    ops = build_smoothness_ops(m_grid, tau_grid, device)

    # 5. Gradient matching
    if use_smooth and n_gm > 0:
        print(f"[train/{version}] === GRADIENT MATCHING ({n_gm} epochs, REAL log-IV units) ===",
              flush=True)
        alpha_m, alpha_tau = gradient_matching(
            gen, disc, gen_opt, disc_opt, criterion, cond_n, tgt_n, prev_raw,
            target_mean, target_std, ops, n_gm, n_pool, device)
    else:
        alpha_m, alpha_tau = 0.0, 0.0
        print(f"[train/{version}] === smoothness OFF: alpha_m=alpha_tau=0 ===", flush=True)

    # 6. Main training
    ckpt_path = ckpt_dir(version)
    print(f"[train/{version}] === MAIN TRAINING ({n_epochs} epochs, "
          f"alpha_m={alpha_m:.3e}, alpha_tau={alpha_tau:.3e}) ===", flush=True)
    history = train_main(gen, disc, gen_opt, disc_opt, criterion, cond_n, tgt_n,
                         prev_raw, target_mean, target_std, ops, alpha_m, alpha_tau,
                         n_epochs, n_pool, device, ckpt_path)

    # 7. Save final + logs
    lp = log_dir(version)
    plot_loss_curve(history, lp / "loss_curve.png", version)
    final = ckpt_path / "final.pt"
    torch.save({"G": gen.state_dict(), "D": disc.state_dict(),
                "epoch": n_epochs, "alpha_m": alpha_m, "alpha_tau": alpha_tau}, final)
    meta = {"version": version, "use_smoothness": use_smooth, "epochs": n_epochs,
            "n_grad_match": n_gm if use_smooth else 0, "batch_size": BATCH_SIZE,
            "seed": SEED,
            "alpha_m": alpha_m, "alpha_tau": alpha_tau, "n_train": int(n_pool),
            "n_tickers": len(TICKERS), "noise_dim": NOISE_DIM, "hidden_dim": HIDDEN_DIM,
            "cond_dim": COND_DIM, "target_dim": TARGET_DIM, "history": history}
    with open(ckpt_path / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(lp / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train/{version}] DONE. saved {final}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Train the pooled (universal) VolGAN")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=list(VERSIONS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training.epochs (smoke tests)")
    ap.add_argument("--n-grad-match", type=int, default=None,
                    help="override training.n_grad_match")
    return ap.parse_args()


def main():
    args = parse_args()
    print(describe(), flush=True)
    seed = set_global_seed()
    print(f"[train/{args.version}] global seed locked = {seed}", flush=True)
    train(args.version, args.device, args.epochs, args.n_grad_match)


if __name__ == "__main__":
    main()
