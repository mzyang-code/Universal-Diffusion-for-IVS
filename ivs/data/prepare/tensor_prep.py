#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Smooth each day's OTM quotes onto the fixed (m, tau) grid and write the per-ticker
tensor cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ivs.config import (
    BANDWIDTHS_JSON, CLEAN_DIR, KERNEL_H_M, KERNEL_H_TAU, LOG_DIR, M_GRID,
    N_JOBS, TAU_GRID_YEARS, TENSOR_DIR, TICKERS, VEGA_MIN,
)
from ivs.data.prepare.utils_rates import load_rf_lookup, load_spot_lookup


M_ARR = np.asarray(M_GRID, dtype=np.float64)
# tau grid expressed exactly in years
TAU_ARR_YEARS = np.asarray(TAU_GRID_YEARS, dtype=np.float64)


def _ticker_bandwidths(ticker: str) -> Tuple[float, float]:
    """Return (h_m, h_tau) for `ticker`: the searched optima, else config defaults.

    Both are a variance scale, used as the 2*h denominator in exp(-x^2/(2h)).
    h_m smooths the moneyness direction at each real expiry; h_tau pools nearby
    expiries into each slice before the tau interpolation.
    """
    h_m, h_tau = float(KERNEL_H_M), float(KERNEL_H_TAU)
    try:
        if BANDWIDTHS_JSON.exists():
            with open(BANDWIDTHS_JSON) as fp:
                bw = json.load(fp)
            if ticker in bw:
                h_m = float(bw[ticker]["h_m"])
                h_tau = float(bw[ticker]["h_tau"])
    except Exception as exc:  # noqa: BLE001 -- fall back loudly but keep going
        print(f"[{ticker}] WARN reading {BANDWIDTHS_JSON}: {exc}; using defaults",
              flush=True)
    return h_m, h_tau


def interpolate_and_extrapolate(x_data: np.ndarray, y_data: np.ndarray,
                                x: np.ndarray) -> np.ndarray:
    """Linear interpolation with linear extrapolation outside the support."""
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")
    if len(x_data) == 1:
        # single expiry: flat-extend (interp1d needs >=2 points)
        return np.full_like(np.asarray(x, dtype=np.float64), float(y_data[0]))
    sorted_indices = np.argsort(x_data)
    x_sorted = x_data[sorted_indices]
    y_sorted = y_data[sorted_indices]
    linear_interp = interp1d(x_sorted, y_sorted, kind="linear",
                             fill_value="extrapolate")
    return linear_interp(x)


# Day-level smoother on the real maturities
def _smooth_day_dynamic_tau(
    iv: np.ndarray, m: np.ndarray, tau: np.ndarray, vega: np.ndarray,
    h_m: float, h_tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth onto (M_GRID, the day's real expiries).

    Every distinct tau present that day gets a 1-D Vega-weighted NW smooth in
    moneyness onto M_ARR; the h_tau kernel pools nearby expiries so a thin expiry
    borrows strength from its neighbours. Returns (surface, tau_unique) with the
    surface shaped (|M|, n_unique_tau).
    """
    tau_unique = np.unique(tau)
    n_t = tau_unique.size
    surf = np.full((M_ARR.size, n_t), np.nan, dtype=np.float64)
    if iv.size == 0 or n_t == 0:
        return surf, tau_unique

    # precompute moneyness kernel distances: (|M|, N)
    dm = (M_ARR[:, None] - m[None, :]) ** 2 / (2.0 * h_m)         # (|M|, N)
    for j, tj in enumerate(tau_unique):
        # tau kernel weight of every quote w.r.t. this slice (pool neighbours)
        dt = (tj - tau) ** 2 / (2.0 * h_tau)                     # (N,)
        w = np.exp(-(dm + dt[None, :]))                          # (|M|, N)
        w = w * vega[None, :]                                    # vega-weighted
        num = w @ iv                                             # (|M|,)
        den = w.sum(axis=1)                                      # (|M|,)
        np.divide(num, den, out=surf[:, j], where=(den > 1e-12))
    return surf, tau_unique


def _unify_to_grid(surf_dyn: np.ndarray, tau_unique: np.ndarray) -> np.ndarray:
    """Interpolate each m-row from the day's real expiries onto TAU_ARR_YEARS,
    (|M|, n_unique_tau) -> (|M|, |TAU|).

    A NaN slice the moneyness kernel could not fill is dropped from that row's
    support first; an all-NaN row is filled from the nearest finite row.
    """
    out = np.full((M_ARR.size, TAU_ARR_YEARS.size), np.nan, dtype=np.float64)
    for i in range(M_ARR.size):
        row = surf_dyn[i]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            continue
        out[i] = interpolate_and_extrapolate(
            tau_unique[finite], row[finite], TAU_ARR_YEARS)
    # backfill any all-NaN m-rows from the nearest finite row (vertical hole)
    if np.isnan(out).any():
        good_rows = np.where(np.isfinite(out).all(axis=1))[0]
        if good_rows.size:
            for i in range(M_ARR.size):
                if not np.isfinite(out[i]).all():
                    nearest = good_rows[np.abs(good_rows - i).argmin()]
                    bad = ~np.isfinite(out[i])
                    out[i, bad] = out[nearest, bad]
    return out


def _process_one_day(
    iv: np.ndarray, m: np.ndarray, tau: np.ndarray, vega: np.ndarray,
    h_m: float, h_tau: float,
) -> np.ndarray:
    """One day end to end: build on the real maturities, unify onto the grid."""
    surf_dyn, tau_unique = _smooth_day_dynamic_tau(iv, m, tau, vega, h_m, h_tau)
    return _unify_to_grid(surf_dyn, tau_unique)


# Per-ticker driver
def build_tensor(ticker: str, n_jobs: int = 1) -> Dict:
    """Build the (T, |M|, |TAU|) IVS tensor for one ticker. n_jobs parallelizes the
    per-day smoothing; pass n_jobs=1 when tickers are built concurrently."""
    in_path = CLEAN_DIR / f"{ticker}_clean.parquet"
    out_path = TENSOR_DIR / f"{ticker}_ivs.npz"
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    print(f"[{ticker}] loading {in_path.name} ...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(in_path, columns=[
        "TradeDate", "CallPut", "Spot", "Strike", "MidIV", "Vega",
        "Moneyness", "TTM_Years",
    ])

    # OTM only: calls with K >= S, puts with K < S
    is_call = df["CallPut"].eq("c")
    otm = (is_call & (df["Moneyness"] >= 1.0)) | (~is_call & (df["Moneyness"] < 1.0))
    df = df.loc[otm].reset_index(drop=True)
    print(f"[{ticker}] OTM subset: {len(df):,} rows", flush=True)

    # Vega already passed the scrubber's gate; floor defensively.
    vega = df["Vega"].clip(lower=max(VEGA_MIN, 1e-4)).to_numpy(dtype=np.float64)
    df = df.assign(_vega=vega)

    grouped = df.groupby("TradeDate", sort=True)
    dates: List[pd.Timestamp] = []
    chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for d, g in grouped:
        dates.append(d)
        chunks.append((
            g["MidIV"].to_numpy(dtype=np.float64),
            g["Moneyness"].to_numpy(dtype=np.float64),
            g["TTM_Years"].to_numpy(dtype=np.float64),
            g["_vega"].to_numpy(dtype=np.float64),
        ))
    print(f"[{ticker}] grouped into {len(chunks):,} trade dates", flush=True)

    h_m, h_tau = _ticker_bandwidths(ticker)
    print(f"[{ticker}] smoothing bandwidths: h_m={h_m:.4f}  h_tau={h_tau:.4f} "
          f"(variance scale)", flush=True)

    if n_jobs and n_jobs > 1:
        surfaces = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_process_one_day)(*chk, h_m, h_tau) for chk in chunks
        )
    else:
        surfaces = [_process_one_day(*chk, h_m, h_tau) for chk in chunks]
    surfaces = np.stack(surfaces, axis=0)            # (T, |M|, |Tau|)
    print(f"[{ticker}] raw tensor shape={surfaces.shape}", flush=True)

    # Only drop a day that is entirely NaN (no usable OTM quote at all).
    valid = ~np.isnan(surfaces).all(axis=(1, 2))
    n_dropped_empty = int((~valid).sum())
    surfaces = surfaces[valid]
    dates_kept = [dates[i] for i in np.where(valid)[0]]

    # Any residual partial NaNs (should be ~none after unify) -> column-median fill.
    n_partial = int(np.isnan(surfaces).sum())
    if n_partial:
        col_med = np.nanmedian(surfaces, axis=0)     # (|M|,|Tau|)
        inds = np.where(np.isnan(surfaces))
        surfaces[inds] = col_med[inds[1], inds[2]]
        print(f"[{ticker}] filled {n_partial} residual NaN cells with column median",
              flush=True)

    dates_arr = np.array(dates_kept, dtype="datetime64[D]")
    n_valid = surfaces.shape[0]
    print(f"[{ticker}] valid days: {n_valid:,} "
          f"(dropped {n_dropped_empty:,} fully-empty days)",
          flush=True)

    # Spot + rate alignment
    spot_lookup = load_spot_lookup(ticker)
    rf_lookup = load_rf_lookup(ticker)
    spots = np.array(
        [spot_lookup[pd.Timestamp(d)] for d in dates_arr], dtype=np.float64,
    )
    rates = np.array(
        [rf_lookup[pd.Timestamp(d)] for d in dates_arr], dtype=np.float64,
    )
    log_ret = np.concatenate(([0.0], np.diff(np.log(spots))))

    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ivs=surfaces.astype(np.float32),
        dates=dates_arr,
        spots=spots.astype(np.float32),
        log_ret=log_ret.astype(np.float32),
        rates=rates.astype(np.float32),
        m_grid=M_ARR.astype(np.float32),
        tau_grid=TAU_ARR_YEARS.astype(np.float32),
    )
    dt = time.time() - t0
    print(f"[{ticker}] wrote {out_path.name} in {dt:.1f}s "
          f"(shape={surfaces.shape}, ~{out_path.stat().st_size/1e6:.1f} MB)",
          flush=True)

    return {
        "ticker": ticker,
        "shape": list(surfaces.shape),
        "valid_days": int(n_valid),
        "dropped_empty_days": n_dropped_empty,
        "residual_nan_cells_filled": n_partial,
        "first_date": str(dates_arr[0]),
        "last_date": str(dates_arr[-1]),
        "h_m": round(h_m, 6),
        "h_tau": round(h_tau, 6),
        "elapsed_sec": round(dt, 1),
        "output": str(out_path),
    }


def _safe_build(ticker: str, n_jobs: int) -> Dict:
    try:
        return build_tensor(ticker, n_jobs=n_jobs)
    except Exception as exc:  # noqa: BLE001
        print(f"[{ticker}] FAILED: {exc!r}", flush=True)
        return {"ticker": ticker, "error": repr(exc), "valid_days": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IVS tensor preparation")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--tickers", nargs="+", default=list(TICKERS))
    ap.add_argument("--n-jobs", type=int, default=N_JOBS,
                    help="inner per-day workers (used when --ticker-jobs=1)")
    ap.add_argument("--ticker-jobs", type=int, default=1,
                    help="how many tickers to build CONCURRENTLY (outer pool); "
                         ">1 forces inner n_jobs=1.")
    args = ap.parse_args(argv)

    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(args.tickers)

    if args.ticker_jobs and args.ticker_jobs > 1:
        tj = min(args.ticker_jobs, len(tickers))
        print(f"[tensor] building {len(tickers)} tensors, {tj} concurrent ...", flush=True)
        results = Parallel(n_jobs=tj, backend="loky", verbose=5)(
            delayed(_safe_build)(t, 1) for t in tickers
        )
        manifest = {r["ticker"]: r for r in results}
    else:
        manifest = {t: _safe_build(t, args.n_jobs) for t in tickers}

    manifest_path = TENSOR_DIR / "_manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2, default=str)
    n_ok = sum(1 for r in manifest.values() if r.get("valid_days", 0) > 0)
    print(f"\n[tensor] {n_ok}/{len(tickers)} tensors built. Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
