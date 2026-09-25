#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Draw the training universe from the eligible tickers and write it back into the config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ivs.config import (
    CLEAN_DIR, N_CANDIDATES, N_SELECT, RAW_DIR, SELECT_SEED,
    TENSOR_DIR, UNIVERSE_JSON, UNIVERSE_MIN_VALID_DAYS,
)

# The drawn universe is written back into the active experiment config.
from ivs.config import CONFIG_PATH  # noqa: E402


def discover_candidates() -> list[str]:
    """All tickers in raw_dir with BOTH an options and a close parquet."""
    opt = {p.name.split("_options_")[0]
           for p in RAW_DIR.glob("*_options_2010_2024.parquet")}
    cands = sorted(t for t in opt
                   if (RAW_DIR / f"{t}_close_2010_2024.parquet").exists())
    return cands


def n_trade_days(ticker: str) -> int:
    """Distinct trade days in the cleaned scatter (0 if missing/unreadable)."""
    p = CLEAN_DIR / f"{ticker}_clean.parquet"
    if not p.exists():
        return 0
    try:
        return int(pd.read_parquet(p, columns=["TradeDate"])["TradeDate"].nunique())
    except Exception:  # noqa: BLE001
        return 0


def scrub_all(cands: list[str], ticker_jobs: int) -> None:
    import ivs.data.prepare.options_scrubber as s1
    s1.main(["--tickers", *cands, "--ticker-jobs", str(ticker_jobs)])


def select(seed: int, n_select: int, min_days: int) -> dict:
    cands = discover_candidates()
    coverage = {t: n_trade_days(t) for t in cands}
    eligible = sorted(t for t in cands if coverage[t] >= min_days)

    rng = np.random.default_rng(seed)
    k = min(n_select, len(eligible))
    chosen = sorted(rng.choice(np.array(eligible), size=k, replace=False).tolist())

    return {
        "select_seed": int(seed),
        "min_valid_days": int(min_days),
        "n_candidates_found": len(cands),
        "n_eligible": len(eligible),
        "n_selected": len(chosen),
        "candidates": cands,
        "eligible": eligible,
        "selected": chosen,
        "coverage_days": {t: coverage[t] for t in cands},
    }


def write_outputs(result: dict) -> None:
    chosen = result["selected"]
    # tensor_dir/_universe.json (recorded universe)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNIVERSE_JSON, "w") as fp:
        json.dump(result, fp, indent=2)

    with open(CONFIG_PATH) as fp:
        cfg = json.load(fp)
    cfg["tickers"] = chosen
    with open(CONFIG_PATH, "w") as fp:
        json.dump(cfg, fp, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select the training universe")
    ap.add_argument("--config", default=None,
                    help="experiment config json (also settable via $IVS_CONFIG)")
    ap.add_argument("--scrub", action="store_true",
                    help="scrub ALL discovered candidates before selecting")
    ap.add_argument("--ticker-jobs", type=int, default=24,
                    help="concurrent tickers for the optional --scrub stage")
    ap.add_argument("--seed", type=int, default=SELECT_SEED)
    ap.add_argument("--n-select", type=int, default=N_SELECT)
    ap.add_argument("--min-days", type=int, default=UNIVERSE_MIN_VALID_DAYS)
    args = ap.parse_args(argv)

    cands = discover_candidates()
    print(f"[universe] discovered {len(cands)} candidates in {RAW_DIR} "
          f"(expected {N_CANDIDATES})", flush=True)

    if args.scrub:
        print(f"[universe] scrubbing all {len(cands)} candidates "
              f"({args.ticker_jobs} concurrent) ...", flush=True)
        scrub_all(cands, args.ticker_jobs)

    result = select(args.seed, args.n_select, args.min_days)
    write_outputs(result)

    print(f"\n[universe] candidates={result['n_candidates_found']}  "
          f"eligible(>= {args.min_days} days)={result['n_eligible']}  "
          f"selected={result['n_selected']}  (seed={args.seed})", flush=True)
    print(f"[universe] selected {result['n_selected']} -> {result['selected']}", flush=True)
    print(f"[universe] recorded -> {UNIVERSE_JSON}  +  config.json (.tickers)", flush=True)
    if result["n_eligible"] < args.n_select:
        print(f"[universe] WARNING: only {result['n_eligible']} eligible < "
              f"{args.n_select} requested -- selected all eligible.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
