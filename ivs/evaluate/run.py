#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Evaluate one experiment's samples (arbitrage, forecasting, PCA) and write the
per-ticker metrics to <eval_output_dir>/per_ticker_metrics.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from ivs.config import (
    CONFIG, DEFAULT_VERSION, EVAL_OUTPUT_DIR, EXPERIMENT, REWEIGHT_FIXED_BETA,
    REWEIGHT_KL_CONSTANT, TICKERS, describe, sample_dir,
)
from ivs.evaluate.arbitrage import arbitrage_metrics
from ivs.evaluate.forecasting import coverage_metrics
from ivs.evaluate.pca import pca_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="configs/<experiment>.json")
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tickers", nargs="+", default=None, help="default: all in config")
    args = ap.parse_args()
    print(describe(), flush=True)

    device = torch.device(args.device)
    group = "out-of-sample" if EXPERIMENT.endswith("_oos") else "in-sample"
    rows = []
    tickers = args.tickers or TICKERS
    for i, ticker in enumerate(tickers, 1):
        z = np.load(sample_dir(args.version, ticker) / "samples.npz", allow_pickle=False)
        row = {"model": CONFIG["paper_model"], "group": group, "ticker": ticker}
        row.update(arbitrage_metrics(z, device, REWEIGHT_KL_CONSTANT))
        row.update(coverage_metrics(z, device, REWEIGHT_FIXED_BETA))
        row.update(pca_metrics(z))
        rows.append(row)
        print(f"[{i}/{len(tickers)}] {ticker}: phi(beta>0)={row['phi_beta']:.3e} "
              f"coverage={row['coverage_beta0']:.3f} pc1={row['pc1_real']:.1f}/"
              f"{row['pc1_mean']:.1f}", flush=True)

    out = EVAL_OUTPUT_DIR / "per_ticker_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
