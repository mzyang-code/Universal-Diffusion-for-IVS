#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Risk-free rate (put-call parity, T-bill fallback) and spot-price loaders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from ivs.config import (
    CLEAN_DIR, RAW_DIR, RATE_FALLBACK, RATE_PARITY_MIN_PAIRS, RATE_PARITY_R_HIGH,
    RATE_PARITY_R_LOW, RATE_SOURCE, MIN_MID_PRICE,
)

FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DGS3MO&cosd=2009-12-01&coed=2025-01-31"
)
RF_CACHE = CLEAN_DIR / "_dgs3mo.csv"


def _parity_cache_path(ticker: str) -> Path:
    """Per-ticker daily parity-rate cache (so we invert raw data only once)."""
    return CLEAN_DIR / f"_parity_rate_{ticker}.csv"


# DGS3MO (external): the fallback source
def fetch_rf_curve(force: bool = False) -> pd.Series:
    """Return a daily-indexed Series of continuously-compounded 3M T-bill rates."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    if RF_CACHE.exists() and not force:
        df = pd.read_csv(RF_CACHE, parse_dates=["date"])
    else:
        resp = requests.get(FRED_URL, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.lower() for c in df.columns]
        date_col = "observation_date" if "observation_date" in df.columns else "date"
        value_col = "dgs3mo"
        df = df.rename(columns={date_col: "date", value_col: "pct"})
        df["date"] = pd.to_datetime(df["date"])
        df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
        df.to_csv(RF_CACHE, index=False)

    df = df.dropna(subset=["pct"]).sort_values("date")
    simple = df["pct"].astype(float) / 100.0
    r_cc = np.log1p(simple * 0.25) / 0.25  # invert quarterly compounding => cc
    s = pd.Series(r_cc.values, index=df["date"].values, name="r_cc")
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full_idx).ffill().bfill()


def load_dgs3mo_lookup() -> Dict[pd.Timestamp, float]:
    s = fetch_rf_curve()
    return {ts.normalize(): float(v) for ts, v in s.items()}


# Put-call parity inversion (primary source), from the raw quotes
def _invert_parity_one_rg(df: pd.DataFrame,
                          spot_lookup: Dict[pd.Timestamp, float]) -> pd.DataFrame:
    """Invert r per (TradeDate, Strike, ExpiryDate) from matched call/put mids.
    Returns one row per matched pair: [TradeDate, r_pair, tau]."""
    df = df.rename(columns={"Call/Put": "CallPut"})
    df = df[["TradeDate", "ExpiryDate", "CallPut", "Strike", "BidPrice", "AskPrice"]].copy()
    df["CallPut"] = df["CallPut"].str.lower()
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
    df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"]).dt.normalize()
    df["MidPrice"] = 0.5 * (df["BidPrice"] + df["AskPrice"])
    # only quotable mids
    df = df[(df["BidPrice"] > 0) & (df["AskPrice"] >= df["BidPrice"])
            & (df["MidPrice"] >= MIN_MID_PRICE)]
    empty = pd.DataFrame(columns=["TradeDate", "r_pair", "tau"])
    if df.empty:
        return empty

    calls = df[df["CallPut"] == "c"][["TradeDate", "ExpiryDate", "Strike", "MidPrice"]]
    puts = df[df["CallPut"] == "p"][["TradeDate", "ExpiryDate", "Strike", "MidPrice"]]
    merged = pd.merge(
        calls, puts, on=["TradeDate", "ExpiryDate", "Strike"],
        suffixes=("_c", "_p"),
    )
    if merged.empty:
        return empty

    merged["Spot"] = merged["TradeDate"].map(spot_lookup)
    merged = merged.dropna(subset=["Spot"])
    merged["tau"] = (merged["ExpiryDate"] - merged["TradeDate"]).dt.days / 365.0
    merged = merged[merged["tau"] > 0]
    if merged.empty:
        return empty

    # r = ln( K / (S - C + P) ) / tau
    denom = merged["Spot"] - merged["MidPrice_c"] + merged["MidPrice_p"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = merged["Strike"] / denom
        merged["r_pair"] = np.log(ratio) / merged["tau"]
    merged = merged[np.isfinite(merged["r_pair"])]
    return merged[["TradeDate", "r_pair", "tau"]]


def compute_parity_rates(ticker: str, force: bool = False) -> pd.Series:
    """Per-TradeDate median put-call-parity rate for `ticker`, cached to CSV.

    Streams the raw option parquet row-group by row-group, never holding the
    whole file in memory. Days below parity_min_pairs or outside [r_low, r_high]
    come back NaN, and the caller falls back to DGS3MO.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cache = _parity_cache_path(ticker)
    if cache.exists() and not force:
        df = pd.read_csv(cache, parse_dates=["TradeDate"])
        return df.set_index("TradeDate")["r"]

    spot_lookup = load_spot_lookup(ticker)
    raw_path = RAW_DIR / f"{ticker}_options_2010_2024.parquet"
    pf = pq.ParquetFile(str(raw_path))

    parts = []
    cols = ["TradeDate", "ExpiryDate", "Call/Put", "Strike", "BidPrice", "AskPrice"]
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        part = _invert_parity_one_rg(df, spot_lookup)
        if not part.empty:
            parts.append(part)
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["TradeDate", "r_pair", "tau"])

    if long.empty:
        pd.DataFrame({"TradeDate": [], "r": [], "n_pairs": []}).to_csv(cache, index=False)
        s = pd.Series(dtype=float, name="r")
        s.index.name = "TradeDate"
        return s

    grp = long.groupby("TradeDate")["r_pair"]
    daily = grp.median()
    n_pairs = grp.size()
    # robustness: require enough pairs and a plausible rate
    ok = (n_pairs >= RATE_PARITY_MIN_PAIRS) \
        & (daily >= RATE_PARITY_R_LOW) & (daily <= RATE_PARITY_R_HIGH)
    daily = daily.where(ok)
    out = pd.DataFrame({"TradeDate": daily.index, "r": daily.values,
                        "n_pairs": n_pairs.reindex(daily.index).values})
    out.to_csv(cache, index=False)
    s = out.set_index("TradeDate")["r"]
    s.name = "r"
    return s


def load_rf_lookup(ticker: str | None = None) -> Dict[pd.Timestamp, float]:
    """Risk-free lookup: parity first, DGS3MO fallback.

    `ticker` is required when RATE_SOURCE == 'parity', since parity is per-ticker;
    with 'dgs3mo' (or no ticker) this returns the pure DGS3MO lookup.
    """
    dgs = load_dgs3mo_lookup()
    if RATE_SOURCE != "parity" or ticker is None:
        return dgs

    parity = compute_parity_rates(ticker)            # per-date median r (NaN where unusable)
    out: Dict[pd.Timestamp, float] = dict(dgs)
    n_parity = 0
    for ts, r in parity.items():
        if pd.notna(r):
            out[pd.Timestamp(ts).normalize()] = float(r)
            n_parity += 1
    print(f"[rates:{ticker}] parity rates used on {n_parity} days "
          f"(fallback={RATE_FALLBACK} on the rest)", flush=True)
    return out


def load_spot_lookup(ticker: str) -> Dict[pd.Timestamp, float]:
    """Load DlyCalDt -> close-price mapping from the per-ticker close parquet."""
    path = RAW_DIR / f"{ticker}_close_2010_2024.parquet"
    df = pd.read_parquet(path, columns=["DlyCalDt", "DlyPrc"])
    df = df.dropna(subset=["DlyPrc"])
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"]).dt.normalize()
    return dict(zip(df["DlyCalDt"], df["DlyPrc"].astype(float)))
