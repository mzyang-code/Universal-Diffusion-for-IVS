#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Gradient-matching calibration of the penalty weights against the main-loss gradient.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from ivs.config import (
    ARB_CLAMP_NORM, ARB_TARGET_RATIO, CALIB_REF_EPOCHS, CONFIG_PATH,
    DEFAULT_VERSION, SMOOTH_TARGET_RATIO, BATCH_SIZE, BETA_SCHEDULE, COND_DIM,
    DROPOUT, GRADIENT_CLIP, HIDDEN_DIM, LEARNING_RATE, M_GRID, NUM_BLOCKS,
    NUM_TIMESTEPS, SURFACE_DIM, SURFACE_M, SURFACE_T, TARGET_DIM, TAU_GRID_YEARS,
    TIME_EMB_DIM, TICKERS, VERSIONS, WEIGHT_DECAY, NORM_DIR, describe,
    set_global_seed, version_cfg,
)
from diffusion.model.denoiser import DiffusionDenoiser
from diffusion.model.process import DiffusionSchedule, make_beta_schedule
from ivs.data.normalizer import Normalizer, fit_normalizer
from ivs.data.dataset import build_features, load_ticker, date_split, rates_per_example
from diffusion.losses import diffusion_loss, grid_spacings, penalties_from_pred

# Penalty term -> (config lambda key, target-ratio group). The order matches the
# penalties_from_pred return tuple.
_TERMS = (
    ("m",         "lambda_m",         "smoothness"),
    ("tau",       "lambda_tau",       "smoothness"),
    ("calendar",  "lambda_calendar",  "arbitrage"),
    ("spread",    "lambda_spread",    "arbitrage"),
    ("butterfly", "lambda_butterfly", "arbitrage"),
)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _load_train_pool():
    """Pooled TRAIN-split (cond, target, rates) lists across the whole universe."""
    all_cond_tr, all_tgt_tr, all_r_tr = [], [], []
    for tk in TICKERS:
        arrays = load_ticker(tk)
        cond, target, dates = build_features(arrays)
        r = rates_per_example(arrays)
        sl_tr, _ = date_split(dates)
        all_cond_tr.append(cond[sl_tr].astype(np.float32))
        all_tgt_tr.append(target[sl_tr].astype(np.float32))
        all_r_tr.append(r[sl_tr].astype(np.float32))
    return all_cond_tr, all_tgt_tr, all_r_tr


def _get_normalizer(all_cond_tr, all_tgt_tr) -> Normalizer:
    norm_path = NORM_DIR / "norm_params.npz"
    if norm_path.exists():
        print(f"[calibrate] loaded normalizer {norm_path}", flush=True)
        return Normalizer.load(norm_path)
    print("[calibrate] normalizer missing -> fitting on pooled train ...", flush=True)
    norm = fit_normalizer(all_cond_tr, all_tgt_tr)
    norm.save(norm_path)
    print(f"[calibrate] normalizer fitted + saved -> {norm_path}", flush=True)
    return norm


def calibrate(version: str = DEFAULT_VERSION,
              smooth_ratio: float = SMOOTH_TARGET_RATIO,
              arb_ratio: float = ARB_TARGET_RATIO,
              ref_epochs: int = CALIB_REF_EPOCHS,
              device_str: str = "cuda:0") -> dict:
    """Return the five gradient-matched lambda_* values."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    smooth_mode = str(version_cfg(version).get("smoothness_mode", "logiv_fd"))
    print(describe(), flush=True)
    print(f"[calibrate] version={version}  ref_epochs={ref_epochs}  "
          f"smooth_ratio={smooth_ratio}  arb_ratio={arb_ratio}  "
          f"smoothness_mode={smooth_mode}", flush=True)
    group_ratio = {"smoothness": smooth_ratio, "arbitrage": arb_ratio}

    all_cond_tr, all_tgt_tr, all_r_tr = _load_train_pool()
    norm = _get_normalizer(all_cond_tr, all_tgt_tr)

    cond_raw_pool = np.concatenate(all_cond_tr, axis=0)
    tgt_raw_pool = np.concatenate(all_tgt_tr, axis=0)
    r_pool = np.concatenate(all_r_tr, axis=0).astype(np.float32)
    cond_norm = norm.normalize_cond(cond_raw_pool)
    tgt_norm = norm.normalize_target(tgt_raw_pool)
    n_pool = cond_norm.shape[0]

    cond_n_gpu = torch.from_numpy(cond_norm).to(device)
    tgt_n_gpu = torch.from_numpy(tgt_norm).to(device)
    cond_raw_gpu = torch.from_numpy(cond_raw_pool).to(device)
    r_gpu = torch.from_numpy(r_pool).to(device)
    n_batches = n_pool // BATCH_SIZE
    print(f"[calibrate] pooled train={n_pool}  {n_batches} batches/epoch "
          f"(batch_size={BATCH_SIZE}, resident on {device})", flush=True)

    model = DiffusionDenoiser(
        x_dim=TARGET_DIM, cond_dim=COND_DIM,
        hidden_dim=HIDDEN_DIM, time_emb_dim=TIME_EMB_DIM,
        num_blocks=NUM_BLOCKS, dropout=DROPOUT,
    ).to(device)
    model.train()
    sched = DiffusionSchedule(make_beta_schedule(BETA_SCHEDULE, NUM_TIMESTEPS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    target_mean_t = torch.from_numpy(norm.target_mean).to(device)
    target_std_t = torch.from_numpy(norm.target_std).to(device)
    m_grid_t = torch.tensor(M_GRID, dtype=torch.float32, device=device)
    tau_grid_t = torch.tensor(TAU_GRID_YEARS, dtype=torch.float32, device=device)
    d_m_t, d_tau_t = grid_spacings(M_GRID, TAU_GRID_YEARS, device)

    ratios = {name: [] for name, _, _ in _TERMS}
    t0 = time.time()

    for epoch in range(1, ref_epochs + 1):
        perm = torch.randperm(n_pool, device=device)
        for bi in range(n_batches):
            idx = perm[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            cond_n = cond_n_gpu[idx]; tgt_n = tgt_n_gpu[idx]
            cond_raw = cond_raw_gpu[idx]; r_b = r_gpu[idx]
            B = tgt_n.shape[0]

            t_rand = torch.randint(1, NUM_TIMESTEPS + 1, (B,), device=device)
            eps = torch.randn_like(tgt_n)
            x_t, _ = sched.q_sample(tgt_n, t_rand, noise=eps)
            eps_pred = model(x_t, t_rand, cond_n)

            # Shared forward graph for the main loss + all five penalties.
            L_mse = diffusion_loss(eps_pred, eps)
            x0_hat_norm = sched.predict_x0_from_eps(x_t, t_rand, eps_pred)
            term_tensors = penalties_from_pred(
                x0_hat_norm, cond_raw, r_b,
                surface_m=SURFACE_M, surface_t=SURFACE_T, surface_dim=SURFACE_DIM,
                target_mean=target_mean_t, target_std=target_std_t,
                m_grid=m_grid_t, tau_grid=tau_grid_t, d_m=d_m_t, d_tau=d_tau_t,
                use_smoothness=True, use_arbitrage=True,
                smoothness_mode=smooth_mode, clamp_norm=ARB_CLAMP_NORM,
            )
            terms = {name: t for (name, _, _), t in zip(_TERMS, term_tensors)}

            # -- measure each penalty gradient (lambda = 1), DO NOT apply --
            gns = {}
            for name, term in terms.items():
                optimizer.zero_grad(set_to_none=True)
                if float(term.detach()) > 0.0:
                    term.backward(retain_graph=True)
                    gns[name] = _grad_norm(model)
                else:
                    gns[name] = 0.0   # no live penalty this batch -> no gradient

            # -- main-loss gradient: measured AND applied (frees the graph) --
            optimizer.zero_grad(set_to_none=True)
            L_mse.backward()
            gn_mse = _grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            for name in ratios:
                if gns[name] > 1e-12:
                    rr = gn_mse / gns[name]
                    if np.isfinite(rr):
                        ratios[name].append(rr)

        msg = "  ".join(
            f"{name}={group_ratio[grp] * np.mean(ratios[name]):.4e}(n={len(ratios[name])})"
            if ratios[name] else f"{name}=NA"
            for name, _, grp in _TERMS)
        print(f"[calibrate] epoch {epoch:>3}/{ref_epochs}  {msg}  "
              f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)

    out = {}
    print("\n[calibrate] DONE", flush=True)
    for name, key, grp in _TERMS:
        v = ratios[name]
        out[key] = float(group_ratio[grp] * np.mean(v)) if v else 0.0
        if v:
            print(f"  {key}: mean(G_mse/G)={np.mean(v):.4e}  n={len(v)}  "
                  f"({grp} ratio {group_ratio[grp]})  ->  lambda={out[key]:.6e}", flush=True)
        else:
            print(f"  {key}: no live penalty at the reference -> lambda=0", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gradient-matching calibration of the DDPM penalty weights")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=list(VERSIONS))
    ap.add_argument("--ref-epochs", type=int, default=CALIB_REF_EPOCHS)
    ap.add_argument("--smooth-ratio", type=float, default=SMOOTH_TARGET_RATIO,
                    help="target ratio for the smoothness pair (lambda_m, lambda_tau)")
    ap.add_argument("--arb-ratio", type=float, default=ARB_TARGET_RATIO,
                    help="target ratio for the arbitrage triple")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--update-config", action="store_true",
                    help="write the calibrated lambdas back into the config json")
    args = ap.parse_args()
    set_global_seed()

    vcfg = version_cfg(args.version)
    if not (vcfg.get("use_smoothness") or vcfg.get("use_arbitrage")):
        print(f"[calibrate] variant {args.version!r} uses no penalty "
              f"(DDPM_mse) -- nothing to calibrate.", flush=True)
        return

    lams = calibrate(version=args.version, smooth_ratio=args.smooth_ratio,
                     arb_ratio=args.arb_ratio, ref_epochs=args.ref_epochs,
                     device_str=args.device)

    if not args.update_config:
        print("\n[calibrate] DRY RUN. Add --update-config to write.", flush=True)
        return

    # Only the weights this variant actually uses are written back.
    keep = set()
    if vcfg.get("use_smoothness"):
        keep |= {"lambda_m", "lambda_tau"}
    if vcfg.get("use_arbitrage"):
        keep |= {"lambda_calendar", "lambda_spread", "lambda_butterfly"}
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    written = {}
    for key, val in lams.items():
        if key in keep:
            cfg["versions"][args.version][key] = round(val, 8)
            written[key] = round(val, 8)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"[calibrate] {CONFIG_PATH} updated: {written}", flush=True)


if __name__ == "__main__":
    main()
