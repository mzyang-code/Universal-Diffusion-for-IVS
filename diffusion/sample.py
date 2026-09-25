#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Generate conditional scenarios per test state from a trained DDPM and write samples.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from ivs.config import (
    BATCH_STATES, BETA_SCHEDULE, COND_DIM, DEFAULT_VERSION, DROPOUT, HIDDEN_DIM,
    NUM_BLOCKS, NUM_TIMESTEPS, SAMPLES_PER_STATE,
    SURFACE_M, SURFACE_T, TARGET_DIM, TENSOR_DIR, TICKERS,
    TIME_EMB_DIM, USE_EMA, VERSIONS, ckpt_dir, describe, sample_dir, NORM_DIR,
    set_global_seed,
)
from diffusion.model.denoiser import DiffusionDenoiser
from diffusion.model.process import DiffusionSchedule, make_beta_schedule
from ivs.data.normalizer import Normalizer
from ivs.data.dataset import build_features, load_ticker, date_split


def _load_dataset(ticker: str):
    arrays = load_ticker(ticker)
    cond, target, dates = build_features(arrays)
    sl_tr, sl_te = date_split(dates)
    return cond, target, dates, sl_tr, sl_te


def _load_model(version: str, device: torch.device) -> DiffusionDenoiser:
    model = DiffusionDenoiser(
        x_dim=TARGET_DIM, cond_dim=COND_DIM,
        hidden_dim=HIDDEN_DIM, time_emb_dim=TIME_EMB_DIM,
        num_blocks=NUM_BLOCKS, dropout=DROPOUT,
    ).to(device)
    final = ckpt_dir(version) / "final.pt"
    state = torch.load(final, map_location=device, weights_only=False)
    key = "model_ema" if (USE_EMA and "model_ema" in state) else "model"
    model.load_state_dict(state[key])
    model.eval()
    epoch = state.get("epoch", "?")
    loss = state.get("loss_mse_final", state.get("loss_diff_final", "?"))
    print(f"[sample/{version}] loaded {key} from {final}  (epoch={epoch}, loss={loss})",
          flush=True)
    return model


def sample_ticker(
    version: str,
    ticker: str,
    model: DiffusionDenoiser,
    norm: Normalizer,
    sched: DiffusionSchedule,
    device: torch.device,
    n_samples: int = SAMPLES_PER_STATE,
    max_states: int = 0,
) -> None:
    cond_all, target_all, dates, sl_tr, sl_te = _load_dataset(ticker)
    cond_te = cond_all[sl_te].astype(np.float32)      # (T, COND_DIM) raw
    tgt_te = target_all[sl_te].astype(np.float32)     # (T, TARGET_DIM) raw
    dates_te = dates[sl_te]

    n_states = cond_te.shape[0]
    if max_states > 0:
        n_states = min(n_states, max_states)
    cond_te = cond_te[:n_states]
    tgt_te = tgt_te[:n_states]
    dates_te = dates_te[:n_states]

    S = n_samples
    print(f"[sample/{version}/{ticker}] {n_states} states x {S} samples ...", flush=True)

    # Reconstruct real IV
    real_iv = np.empty((n_states, SURFACE_M, SURFACE_T), dtype=np.float32)
    for k in range(n_states):
        prev = cond_te[k, 3:].reshape(SURFACE_M, SURFACE_T)
        real_iv[k] = np.exp(tgt_te[k, 1:].reshape(SURFACE_M, SURFACE_T) + prev)

    # Normalize condition for model input
    cond_te_norm = norm.normalize_cond(cond_te)        # (T, COND_DIM)

    synth_target = np.empty((n_states, S, TARGET_DIM), dtype=np.float32)
    synth_log_ret = np.empty((n_states, S), dtype=np.float32)
    synth_iv = np.empty((n_states, S, SURFACE_M, SURFACE_T), dtype=np.float32)

    def model_fn(x, t, c):
        return model(x, t, c)

    with torch.no_grad():
        for s_start in range(0, n_states, BATCH_STATES):
            s_end = min(s_start + BATCH_STATES, n_states)
            nb = s_end - s_start
            c_norm = torch.from_numpy(cond_te_norm[s_start:s_end]).to(device)
            c_raw = torch.from_numpy(cond_te[s_start:s_end]).to(device)

            # Generate S samples per state by repeating the condition
            # Expand: (nb, COND_DIM) -> (nb*S, COND_DIM)
            c_norm_rep = c_norm.unsqueeze(1).expand(nb, S, COND_DIM).reshape(nb * S, COND_DIM)

            shape = (nb * S, TARGET_DIM)
            x0_norm = sched.ddpm_sample(model_fn, shape, c_norm_rep, device)

            # Denormalize to raw target space
            x0_raw = torch.from_numpy(
                norm.denormalize_target(x0_norm.cpu().numpy())
            ).to(device)                                         # (nb*S, TARGET_DIM)
            x0_raw = x0_raw.view(nb, S, TARGET_DIM)

            # Reconstruct IV surface
            log_iv_prev = c_raw[:, 3:].unsqueeze(1)              # (nb, 1, SURFACE_DIM)
            log_iv_inc = x0_raw[:, :, 1:]                        # (nb, S, SURFACE_DIM)
            iv = torch.exp(log_iv_prev + log_iv_inc)             # (nb, S, SURFACE_DIM)
            iv = iv.view(nb, S, SURFACE_M, SURFACE_T)

            synth_target[s_start:s_end] = x0_raw.cpu().numpy()
            synth_log_ret[s_start:s_end] = x0_raw[:, :, 0].cpu().numpy()
            synth_iv[s_start:s_end] = iv.cpu().numpy()

            if (s_start // BATCH_STATES) % 5 == 0:
                print(f"  ... states {s_start}/{n_states}", flush=True)

    # Load grids from tensor file
    zd = np.load(TENSOR_DIR / f"{ticker}_ivs.npz", allow_pickle=False)

    out_dir = sample_dir(version, ticker)
    out_path = out_dir / "samples.npz"
    np.savez_compressed(
        out_path,
        cond=cond_te.astype(np.float32),
        state_dates=np.asarray(dates_te, dtype="datetime64[D]"),
        real_target=tgt_te.astype(np.float32),
        real_iv=real_iv,
        synth_target=synth_target,
        synth_log_ret=synth_log_ret,
        synth_iv=synth_iv,
        m_grid=zd["m_grid"].astype(np.float32),
        tau_grid=zd["tau_grid"].astype(np.float32),
    )
    print(
        f"[sample/{version}/{ticker}] -> {out_path}  "
        f"({out_path.stat().st_size/1e6:.1f} MB, synth_iv={synth_iv.shape})",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Conditional diffusion sampling")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=list(VERSIONS))
    ap.add_argument("--ticker", nargs="+", default=None,
                    help="One or more tickers; default = the whole universe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_STATE,
                    help="Samples per test state")
    ap.add_argument("--max-states", type=int, default=0,
                    help="Cap test states (0 = use all)")
    args = ap.parse_args()

    print(describe(), flush=True)
    seed = set_global_seed()
    print(f"[sample/{args.version}] global seed locked = {seed}", flush=True)

    tickers = args.ticker if args.ticker else list(TICKERS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    norm = Normalizer.load(NORM_DIR / "norm_params.npz")
    print(f"[sample/{args.version}] Normalizer loaded.", flush=True)

    model = _load_model(args.version, device)

    betas = make_beta_schedule(BETA_SCHEDULE, NUM_TIMESTEPS)
    sched = DiffusionSchedule(betas).to(device)   # coeffs resident on GPU (fast sampling)

    for tk in tickers:
        sample_ticker(
            version=args.version, ticker=tk,
            model=model, norm=norm, sched=sched, device=device,
            n_samples=args.samples, max_states=args.max_states,
        )

    print(f"\n[sample/{args.version}] All tickers done.", flush=True)


if __name__ == "__main__":
    main()
