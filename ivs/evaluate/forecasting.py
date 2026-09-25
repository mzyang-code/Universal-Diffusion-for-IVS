#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Next-day return forecasting: coverage of the 95% predictive interval (Table 3).
"""
from __future__ import annotations

import numpy as np
import torch

from ivs.evaluate.arbitrage import arbitrage_phi


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """Per-day weighted quantile of (T, S) values -> (T,)."""
    order = np.argsort(values, axis=1)
    vs = np.take_along_axis(values, order, axis=1)
    cum = np.cumsum(np.take_along_axis(weights, order, axis=1), axis=1)
    out = np.empty(values.shape[0], dtype=np.float32)
    for t in range(values.shape[0]):
        out[t] = vs[t, min(int(np.searchsorted(cum[t], q)), vs.shape[1] - 1)]
    return out


def coverage_metrics(z, device: torch.device, beta: float = 50.0,
                     chunk: int = 32) -> dict:
    """Fraction of test days whose realized return lies inside the generated
    [q2.5%, q97.5%] band, unweighted (beta = 0) and reweighted with fixed beta."""
    real = z["real_target"][:, 0].astype(np.float32)
    synth = z["synth_target"][:, :, 0].astype(np.float32)
    synth_iv = z["synth_iv"]
    m = torch.as_tensor(z["m_grid"].astype(np.float64), device=device)
    tau = torch.as_tensor(z["tau_grid"].astype(np.float64), device=device)
    r0 = torch.zeros((), dtype=torch.float64, device=device)

    penalty = np.empty(synth.shape, dtype=np.float64)
    for a in range(0, synth.shape[0], chunk):
        sig = torch.as_tensor(synth_iv[a:a + chunk].astype(np.float64), device=device)
        penalty[a:a + chunk] = arbitrage_phi(sig, m, tau, r0)["phi"].cpu().numpy()
    logits = -beta * penalty
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

    def cov(lo, hi):
        return float(((real >= lo) & (real <= hi)).mean())

    lo0, hi0 = np.quantile(synth, [0.025, 0.975], axis=1).astype(np.float32)
    return {"coverage_beta0": cov(lo0, hi0),
            "coverage_beta": cov(_weighted_quantile(synth, w, 0.025),
                                 _weighted_quantile(synth, w, 0.975))}
