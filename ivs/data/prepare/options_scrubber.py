#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Apply quote-quality filters to the raw option quotes and write <TICKER>_clean.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ivs.config import (
    ABS_SPREAD_MAX_LOW_MID, CLEAN_DIR, DELTA_ABS_HIGH, DELTA_ABS_LOW, IV_HIGH,
    IV_LOW, LOG_DIR, MAX_IV_SPREAD, MIN_MID_PRICE, MIN_OPEN_INTEREST,
    MID_BREAKPOINT, N_JOBS, PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL,
    RAW_DIR, REL_SPREAD_MAX_HIGH_MID, TICKERS, TTM_DAYS_LOW, VEGA_MIN,
)
from ivs.data.prepare.utils_rates import load_rf_lookup, load_spot_lookup

OUTPUT_SCHEMA_COLS = [
    "TradeDate", "ExpiryDate", "Ticker", "CallPut", "Strike",
    "BidPrice", "AskPrice", "MidPrice", "BidIV", "AskIV", "MidIV",
    "OpenInterest", "Volume", "Delta", "Gamma", "Vega", "Theta", "Rho",
    "Spot", "RiskFreeRate", "TTM_Days", "TTM_Years",
    "Moneyness", "LogForwardMoneyness",
]


# Funnel bookkeeping
@dataclass
class FilterFunnel:
    input_rows: int = 0
    after_quote_validity: int = 0
    after_iv_validity: int = 0
    after_spread: int = 0
    after_liquidity: int = 0
    after_vega: int = 0
    after_delta: int = 0
    after_ttm: int = 0
    output_rows: int = 0

    def add(self, other: "FilterFunnel") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


# Per row-group worker
def _scrub_row_group(
    raw_path: str,
    rg_index: int,
    ticker: str,
    spot_lookup: Dict[pd.Timestamp, float],
    rf_lookup: Dict[pd.Timestamp, float],
) -> Tuple[pa.Table | None, FilterFunnel]:

    pf = pq.ParquetFile(raw_path)
    df = pf.read_row_group(rg_index).to_pandas()

    funnel = FilterFunnel(input_rows=len(df))
    if df.empty:
        return None, funnel

    df = df.rename(columns={
        "Call/Put": "CallPut",
        "BidImpliedVolatility": "BidIV",
        "AskImpliedVolatility": "AskIV",
    })
    df["CallPut"] = df["CallPut"].str.lower()
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
    df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"]).dt.normalize()

    # Quote validity
    df["MidPrice"] = 0.5 * (df["BidPrice"] + df["AskPrice"])
    mask = (
        (df["BidPrice"] > 0)
        & (df["AskPrice"] > 0)
        & (df["AskPrice"] >= df["BidPrice"])
        & (df["MidPrice"] >= MIN_MID_PRICE)
        & df["BidPrice"].notna()
        & df["AskPrice"].notna()
    )
    df = df.loc[mask]
    funnel.after_quote_validity = len(df)
    if df.empty:
        return None, funnel

    # IV validity
    df["MidIV"] = 0.5 * (df["BidIV"] + df["AskIV"])
    mask = (
        df["BidIV"].notna() & df["AskIV"].notna()
        & (df["BidIV"] > 0)
        & (df["AskIV"] > 0)
        & (df["BidIV"] < df["AskIV"])
        & ((df["AskIV"] - df["BidIV"]) <= MAX_IV_SPREAD)
        & (df["MidIV"] >= IV_LOW)
        & (df["MidIV"] <= IV_HIGH)
    )
    df = df.loc[mask]
    funnel.after_iv_validity = len(df)
    if df.empty:
        return None, funnel

    # Spread sanity (piece-wise on mid)
    spread = df["AskPrice"] - df["BidPrice"]
    rel_spread = spread / df["MidPrice"]
    high_mid = df["MidPrice"] >= MID_BREAKPOINT
    mask = (
        (high_mid & (rel_spread <= REL_SPREAD_MAX_HIGH_MID))
        | (~high_mid & (spread <= ABS_SPREAD_MAX_LOW_MID))
    )
    df = df.loc[mask]
    funnel.after_spread = len(df)
    if df.empty:
        return None, funnel

    # Liquidity
    df = df.loc[df["OpenInterest"] >= MIN_OPEN_INTEREST]
    funnel.after_liquidity = len(df)
    if df.empty:
        return None, funnel

    # Vega quality. An absolute Vega floor is price-level dependent and would
    # truncate low-priced / early-period data, so VEGA_MIN defaults to 0 and Vega
    # only has to be non-null (it is used as the kernel weight in the build).
    if VEGA_MIN > 0:
        df = df.loc[df["Vega"].notna() & (df["Vega"] >= VEGA_MIN)]
    else:
        df = df.loc[df["Vega"].notna()]
    funnel.after_vega = len(df)
    if df.empty:
        return None, funnel

    # Spot + rate join (needed for moneyness, ttm)
    df["Spot"] = df["TradeDate"].map(spot_lookup)
    df["RiskFreeRate"] = df["TradeDate"].map(rf_lookup)
    df = df.dropna(subset=["Spot", "RiskFreeRate"])
    if df.empty:
        return None, funnel

    df["TTM_Days"] = (df["ExpiryDate"] - df["TradeDate"]).dt.days.astype("int32")
    df["TTM_Years"] = df["TTM_Days"].astype("float64") / 365.0
    df["Moneyness"] = df["Strike"] / df["Spot"]
    forward = df["Spot"] * np.exp(df["RiskFreeRate"] * df["TTM_Years"])
    df["LogForwardMoneyness"] = np.log(df["Strike"] / forward)

    # Delta band
    abs_delta = df["Delta"].abs()
    df = df.loc[(abs_delta >= DELTA_ABS_LOW) & (abs_delta <= DELTA_ABS_HIGH)]
    funnel.after_delta = len(df)
    if df.empty:
        return None, funnel

    # TTM lower bound (no upper bound: long-dated contracts pass through)
    df = df.loc[df["TTM_Days"] >= TTM_DAYS_LOW]
    funnel.after_ttm = len(df)
    if df.empty:
        return None, funnel

    df["Ticker"] = ticker
    df = df[OUTPUT_SCHEMA_COLS].reset_index(drop=True)
    funnel.output_rows = len(df)
    return pa.Table.from_pandas(df, preserve_index=False), funnel


# Per-ticker driver
def process_ticker(ticker: str, n_jobs: int = 1, show_progress: bool = True) -> Dict:
    """Scrub one ticker. n_jobs parallelizes row groups within the ticker; when
    many tickers run concurrently pass n_jobs=1 and show_progress=False to avoid
    nested-pool oversubscription."""
    raw_path = RAW_DIR / f"{ticker}_options_2010_2024.parquet"
    out_path = CLEAN_DIR / f"{ticker}_clean.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    spot_lookup = load_spot_lookup(ticker)
    rf_lookup = load_rf_lookup(ticker)   # parity primary, per ticker

    pf = pq.ParquetFile(str(raw_path))
    n_rg = pf.num_row_groups
    if show_progress:
        print(f"[{ticker}] raw rows={pf.metadata.num_rows:,}  row_groups={n_rg}  "
              f"workers={n_jobs}", flush=True)

    if n_jobs and n_jobs > 1:
        jobs = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator",
                        verbose=0)(
            delayed(_scrub_row_group)(str(raw_path), i, ticker, spot_lookup, rf_lookup)
            for i in range(n_rg)
        )
    else:
        jobs = (_scrub_row_group(str(raw_path), i, ticker, spot_lookup, rf_lookup)
                for i in range(n_rg))

    writer: pq.ParquetWriter | None = None
    total = FilterFunnel()
    bar = tqdm(total=n_rg, desc=f"{ticker}", unit="rg", leave=True) if show_progress else None
    t0 = time.time()
    try:
        for table, funnel in jobs:
            total.add(funnel)
            if table is not None and table.num_rows > 0:
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(out_path), table.schema,
                        compression=PARQUET_COMPRESSION,
                        compression_level=PARQUET_COMPRESSION_LEVEL,
                    )
                writer.write_table(table)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        if writer is not None:
            writer.close()
    dt = time.time() - t0

    summary = {
        "ticker": ticker,
        "elapsed_sec": round(dt, 1),
        "raw_rows": total.input_rows,
        "clean_rows": total.output_rows,
        "survival_rate": (total.output_rows / total.input_rows) if total.input_rows else 0,
        "funnel": asdict(total),
        "output_file": str(out_path),
    }
    print(
        f"[{ticker}] done in {dt:6.1f}s | "
        f"{total.input_rows:,} -> {total.output_rows:,} rows "
        f"({summary['survival_rate']*100:.2f}% survival)",
        flush=True,
    )
    return summary


# Main
def _safe_process(ticker: str, n_jobs: int, show_progress: bool) -> Dict:
    """Record, rather than raise, a per-ticker failure so one bad ticker never
    aborts the whole batch."""
    try:
        return process_ticker(ticker, n_jobs=n_jobs, show_progress=show_progress)
    except Exception as exc:  # noqa: BLE001
        print(f"[{ticker}] FAILED: {exc!r}", flush=True)
        return {"ticker": ticker, "error": repr(exc), "clean_rows": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Options scrubber")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--tickers", nargs="+", default=list(TICKERS))
    ap.add_argument("--n-jobs", type=int, default=N_JOBS,
                    help="inner row-group workers per ticker (used when --ticker-jobs=1)")
    ap.add_argument("--ticker-jobs", type=int, default=1,
                    help="how many tickers to scrub CONCURRENTLY (outer pool). "
                         ">1 forces inner n_jobs=1 to avoid nested oversubscription.")
    args = ap.parse_args(argv)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(args.tickers)

    if args.ticker_jobs and args.ticker_jobs > 1:
        tj = min(args.ticker_jobs, len(tickers))
        print(f"[scrub] scrubbing {len(tickers)} tickers, {tj} concurrent "
              f"(inner single-threaded) ...", flush=True)
        results = Parallel(n_jobs=tj, backend="loky", verbose=5)(
            delayed(_safe_process)(t, 1, False) for t in tickers
        )
        overall = {r["ticker"]: r for r in results}
    else:
        overall = {t: _safe_process(t, args.n_jobs, True) for t in tickers}

    funnel_path = CLEAN_DIR / "_funnel.json"
    with open(funnel_path, "w") as fp:
        json.dump(overall, fp, indent=2, default=str)
    n_ok = sum(1 for r in overall.values() if r.get("clean_rows", 0) > 0)
    print(f"\n[scrub] {n_ok}/{len(tickers)} tickers produced clean data. "
          f"Funnel report -> {funnel_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
