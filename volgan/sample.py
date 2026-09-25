#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Based on VolGAN: https://github.com/milenavuletic/VolGAN

Generate conditional scenarios per test state from the trained VolGAN and write samples.npz.
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
    BATCH_STATES, COND_DIM, DEFAULT_VERSION, HIDDEN_DIM, NOISE_DIM, NORM_DIR,
    SAMPLES_PER_STATE,
    SURFACE_M, SURFACE_T, TARGET_DIM, TENSOR_DIR, TICKERS, VERSIONS, ckpt_dir,
    sample_dir, set_global_seed,
)
from volgan.model.networks import Generator
from ivs.data.normalizer import Normalizer
from ivs.data.dataset import build_features, load_ticker, date_split


def _load_generator(version: str, device: torch.device) -> Generator:
    gen = Generator(noise_dim=NOISE_DIM, cond_dim=COND_DIM,
                    hidden_dim=HIDDEN_DIM, output_dim=TARGET_DIM).to(device)
    final = ckpt_dir(version) / "final.pt"
    state = torch.load(final, map_location=device, weights_only=False)
    gen.load_state_dict(state["G"])
    gen.eval()
    print(f"[sample/{version}] loaded G from {final} "
          f"(epoch={state.get('epoch','?')}, alpha_m={state.get('alpha_m','?')}, "
          f"alpha_tau={state.get('alpha_tau','?')})", flush=True)
    return gen


def _rates_for(ticker: str, state_dates: np.ndarray) -> np.ndarray:
    """Nearest-prior risk-free rate per state date, from the ticker tensor."""
    z = np.load(TENSOR_DIR / f"{ticker}_ivs.npz", allow_pickle=False)
    cd = z["dates"].astype("datetime64[D]")
    cr = z["rates"].astype(np.float64)
    idx = np.clip(np.searchsorted(cd, state_dates.astype("datetime64[D]")), 0, cd.shape[0] - 1)
    return cr[idx].astype(np.float32)


def sample_ticker(version: str, ticker: str, gen: Generator, norm: Normalizer,
                  device: torch.device, n_samples: int, batch_states: int) -> None:
    arrays = load_ticker(ticker)
    cond_all, target_all, dates = build_features(arrays)
    _, sl_te = date_split(dates)
    cond_te = cond_all[sl_te].astype(np.float32)        # (T, COND_DIM) raw
    tgt_te = target_all[sl_te].astype(np.float32)       # (T, TARGET_DIM) raw
    dates_te = dates[sl_te]
    n_states = cond_te.shape[0]
    S = n_samples

    # Real IV surface (reconstruct from raw prev log-IV + real increment)
    prev_raw = cond_te[:, 3:]                           # (T, SURFACE_DIM)
    real_iv = np.exp(tgt_te[:, 1:].astype(np.float64) + prev_raw.astype(np.float64))
    real_iv = real_iv.reshape(n_states, SURFACE_M, SURFACE_T).astype(np.float32)

    # Normalize condition for the generator
    cond_te_norm = norm.normalize_cond(cond_te)

    synth_target = np.empty((n_states, S, TARGET_DIM), dtype=np.float32)
    synth_log_ret = np.empty((n_states, S), dtype=np.float32)
    synth_iv = np.empty((n_states, S, SURFACE_M, SURFACE_T), dtype=np.float32)

    with torch.no_grad():
        for s0 in range(0, n_states, batch_states):
            s1 = min(s0 + batch_states, n_states)
            c = torch.from_numpy(cond_te_norm[s0:s1]).to(device)
            samp = gen.sample(c, S).cpu().numpy()        # (b, S, TARGET_DIM) NORMALIZED
            b = s1 - s0
            samp_raw = norm.denormalize_target(samp.reshape(b * S, TARGET_DIM)
                                               ).reshape(b, S, TARGET_DIM)
            prev_blk = prev_raw[s0:s1][:, None, :].astype(np.float64)  # (b,1,SURFACE_DIM)
            log_iv = samp_raw[:, :, 1:].astype(np.float64) + prev_blk  # (b,S,SURFACE_DIM)
            iv = np.exp(log_iv).reshape(b, S, SURFACE_M, SURFACE_T)
            synth_target[s0:s1] = samp_raw.astype(np.float32)
            synth_log_ret[s0:s1] = samp_raw[:, :, 0].astype(np.float32)
            synth_iv[s0:s1] = iv.astype(np.float32)

    zd = np.load(TENSOR_DIR / f"{ticker}_ivs.npz", allow_pickle=False)
    out_dir = sample_dir(version, ticker)
    out_path = out_dir / "samples.npz"
    np.savez_compressed(
        out_path,
        cond=cond_te, state_dates=np.asarray(dates_te, dtype="datetime64[D]"),
        real_target=tgt_te, real_iv=real_iv,
        synth_target=synth_target, synth_log_ret=synth_log_ret, synth_iv=synth_iv,
        m_grid=zd["m_grid"].astype(np.float32), tau_grid=zd["tau_grid"].astype(np.float32),
        rates=_rates_for(ticker, np.asarray(dates_te, dtype="datetime64[D]")),
        ticker=np.array([ticker] * n_states),
    )
    print(f"[sample/{version}/{ticker}] states={n_states} x S={S} "
          f"synth_iv={synth_iv.shape} -> {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pooled VolGAN conditional sampling")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=list(VERSIONS))
    ap.add_argument("--ticker", nargs="+", default=None,
                    help="One or more tickers; default = the whole universe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_STATE)
    ap.add_argument("--batch-states", type=int, default=BATCH_STATES)
    ap.add_argument("--shard", type=int, default=0,
                    help="This shard index in [0, num-shards) for multi-GPU splitting")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total shards; ticker k goes to shard (k %% num-shards)")
    args = ap.parse_args()

    seed = set_global_seed()
    print(f"[sample/{args.version}] global seed locked = {seed}", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    norm = Normalizer.load(NORM_DIR / "norm_params.npz")
    print(f"[sample/{args.version}] training normalizer loaded.", flush=True)
    gen = _load_generator(args.version, device)

    tickers = args.ticker if args.ticker else list(TICKERS)
    if args.num_shards > 1:
        tickers = [t for k, t in enumerate(tickers) if k % args.num_shards == args.shard]
        print(f"[sample/{args.version}] shard {args.shard}/{args.num_shards}: "
              f"{len(tickers)} tickers", flush=True)
    for tk in tickers:
        sample_ticker(args.version, tk, gen, norm, device, args.samples, args.batch_states)
    print(f"\n[sample/{args.version}] All {len(tickers)} tickers done.", flush=True)


if __name__ == "__main__":
    main()
