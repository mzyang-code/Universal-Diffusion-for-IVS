#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Per-ticker Nadaraya-Watson bandwidth search minimizing static arbitrage on the train window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Put the project root on the path so the smoother imports cleanly whether this
# is launched as a script or with -m.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ivs.config import (
    BANDWIDTH_SEARCH_FIRST_N_DAYS, BANDWIDTH_SEARCH_HIGH, BANDWIDTH_SEARCH_LOW,
    BANDWIDTH_SEARCH_STEP, BANDWIDTH_SEARCH_SEED, BANDWIDTHS_JSON, CLEAN_DIR,
    TENSOR_DIR, TICKERS, TRAIN_START, TRAIN_END,
)
from ivs.evaluate.arbitrage import arbitrage_phi
from ivs.data.prepare.tensor_prep import M_ARR, TAU_ARR_YEARS, _process_one_day


def _load_day_quotes(ticker: str, rng: np.random.Generator):
    """Return (iv, m, tau_years, vega, date) for one sampled early train day."""
    df = pd.read_parquet(
        CLEAN_DIR / f"{ticker}_clean.parquet",
        columns=["TradeDate", "CallPut", "MidIV", "Vega", "Moneyness", "TTM_Years"],
    )
    # OTM subset, as in the tensor build: calls K>=S, puts K<S
    is_call = df["CallPut"].eq("c")
    otm = (is_call & (df["Moneyness"] >= 1.0)) | (~is_call & (df["Moneyness"] < 1.0))
    df = df.loc[otm]

    # Restrict candidate days to the train window (no test-period look-ahead)
    td = pd.to_datetime(df["TradeDate"])
    in_train = (td >= pd.Timestamp(TRAIN_START)) & (td <= pd.Timestamp(TRAIN_END))
    df = df.loc[in_train]

    dates = np.sort(df["TradeDate"].unique())
    first_n = dates[: min(BANDWIDTH_SEARCH_FIRST_N_DAYS, len(dates))]
    day = first_n[rng.integers(0, len(first_n))]
    g = df.loc[df["TradeDate"] == day]

    from ivs.config import VEGA_MIN
    vega = g["Vega"].to_numpy(dtype=np.float64)
    vega = np.clip(vega, max(VEGA_MIN, 1e-4), None)   # same floor as the build
    return (
        g["MidIV"].to_numpy(dtype=np.float64),
        g["Moneyness"].to_numpy(dtype=np.float64),
        g["TTM_Years"].to_numpy(dtype=np.float64),
        vega,
        pd.Timestamp(day),
    )


def _penalty_for_surface(surface: np.ndarray) -> float:
    """Static-arbitrage penalty Phi of an (M, T) IV surface (relative call, r=0).
    Degenerate surfaces score +inf so the search never selects them."""
    if not np.isfinite(surface).all() or (surface <= 0).any():
        return float("inf")
    sig = torch.from_numpy(surface.astype(np.float64))
    m = torch.from_numpy(M_ARR)
    tau = torch.from_numpy(TAU_ARR_YEARS)
    r = torch.zeros((), dtype=torch.float64)
    phi = arbitrage_phi(sig, m, tau, r)["phi"]
    return float(phi.item())


def _ticker_seed(base_seed: int, ticker: str) -> int:
    """Stable per-ticker seed. The offset is a fixed hash of the symbol, not
    Python's salted hash(), so the sampled day and hence the chosen bandwidth do
    not depend on how many tickers run, in what order, or in which process."""
    h = int.from_bytes(hashlib.sha256(ticker.encode()).digest()[:4], "big")
    return (int(base_seed) + h) % (2**31 - 1)


def search_ticker(ticker: str, base_seed: int = BANDWIDTH_SEARCH_SEED) -> dict:
    t0 = time.time()
    seed = _ticker_seed(base_seed, ticker)
    rng = np.random.default_rng(seed)
    iv, m, tau, vega, day = _load_day_quotes(ticker, rng)
    grid = np.arange(
        BANDWIDTH_SEARCH_LOW,
        BANDWIDTH_SEARCH_HIGH + 1e-9,
        BANDWIDTH_SEARCH_STEP,
    )
    print(f"[{ticker}] sampled TRAIN day {day.date()} ({iv.size} OTM quotes); "
          f"searching {len(grid)}x{len(grid)} = {len(grid)**2} (h1,h2) pairs "
          f"onto the {M_ARR.size}x{TAU_ARR_YEARS.size} grid ...", flush=True)

    best = {"h_m": None, "h_tau": None, "penalty": float("inf")}
    n_valid = 0
    for h1 in grid:
        for h2 in grid:
            # Score on the same day pipeline the tensor build uses, so the
            # chosen bandwidth matches production.
            surface = _process_one_day(iv, m, tau, vega, float(h1), float(h2))
            pen = _penalty_for_surface(surface)
            if np.isfinite(pen):
                n_valid += 1
            if pen < best["penalty"]:
                best = {"h_m": float(h1), "h_tau": float(h2), "penalty": pen}

    if best["h_m"] is None:
        raise RuntimeError(
            f"[{ticker}] no (h1,h2) on day {day.date()} populated the full "
            f"{M_ARR.size}x{TAU_ARR_YEARS.size} grid -- check coverage / grid range.")

    dt = time.time() - t0
    print(f"[{ticker}] BEST h_m={best['h_m']:.4f}  h_tau={best['h_tau']:.4f}  "
          f"penalty={best['penalty']:.6g}  ({n_valid}/{len(grid)**2} pairs valid, "
          f"{dt:.1f}s)", flush=True)
    return {
        "ticker": ticker,
        "sampled_day": str(day.date()),
        "n_quotes": int(iv.size),
        "h_m": best["h_m"],
        "h_tau": best["h_tau"],
        "penalty": best["penalty"],
        "n_valid_pairs": int(n_valid),
        "grid_low": BANDWIDTH_SEARCH_LOW,
        "grid_high": BANDWIDTH_SEARCH_HIGH,
        "grid_step": BANDWIDTH_SEARCH_STEP,
        "train_window": [TRAIN_START, TRAIN_END],
        "base_seed": int(base_seed),
        "ticker_seed": int(seed),
    }


def _safe_search(ticker: str, base_seed: int) -> dict:
    try:
        return search_ticker(ticker, base_seed)
    except Exception as exc:  # noqa: BLE001 -- record, fall back to config defaults
        print(f"[{ticker}] bandwidth search FAILED: {exc!r}; "
              f"the build will use the config kernel_h_* defaults", flush=True)
        return {"ticker": ticker, "error": repr(exc)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Smoothing-bandwidth search (train-only, per-ticker)")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--tickers", nargs="+", default=list(TICKERS))
    ap.add_argument("--seed", type=int, default=BANDWIDTH_SEARCH_SEED)
    ap.add_argument("--ticker-jobs", type=int, default=1,
                    help="how many tickers to search CONCURRENTLY (each has an "
                         "independent, ticker-derived seed -> order-invariant).")
    args = ap.parse_args(argv)

    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(args.tickers)

    if args.ticker_jobs and args.ticker_jobs > 1:
        from joblib import Parallel, delayed
        tj = min(args.ticker_jobs, len(tickers))
        print(f"[bandwidth] searching {len(tickers)} tickers, {tj} concurrent ...",
              flush=True)
        results = Parallel(n_jobs=tj, backend="loky", verbose=5)(
            delayed(_safe_search)(t, args.seed) for t in tickers
        )
        out = {r["ticker"]: r for r in results}
    else:
        out = {t: _safe_search(t, args.seed) for t in tickers}

    with open(BANDWIDTHS_JSON, "w") as fp:
        json.dump(out, fp, indent=2)
    n_ok = sum(1 for r in out.values() if "h_m" in r)
    print(f"\n[bandwidth] {n_ok}/{len(tickers)} searched OK. Bandwidths -> {BANDWIDTHS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
