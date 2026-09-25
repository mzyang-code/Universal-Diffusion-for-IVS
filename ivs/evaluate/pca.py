#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

PCA of the daily log-IV increments: explained-variance ratios of PC1-3 (Table 4).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

N_PC = 3


def pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalue shares and eigenvectors (columns, descending) of the sample
    covariance of x (n, d)."""
    x = x.astype(np.float64, copy=False)
    xc = x - x.mean(axis=0)
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w, v = linalg.eigh(0.5 * (cov + cov.T), driver="evd")
    w, v = np.clip(w[::-1], 0.0, None), v[:, ::-1]
    return w / w.sum(), v


def pca_metrics(z) -> dict:
    """Observed PC1-3 variance shares of the daily log-IV increments, and the
    mean / std of the same shares over the generated scenario paths."""
    x_real = z["real_target"][:, 1:].astype(np.float32)
    x_fake = z["synth_target"][:, :, 1:].astype(np.float32)
    real = pca(x_real)[0][:N_PC]
    fake = np.stack([pca(x_fake[:, n])[0][:N_PC] for n in range(x_fake.shape[1])])
    out = {}
    for k in range(N_PC):
        out[f"pc{k + 1}_real"] = 100.0 * real[k]
        out[f"pc{k + 1}_mean"] = 100.0 * fake[:, k].mean()
        out[f"pc{k + 1}_std"] = 100.0 * fake[:, k].std(ddof=1)
    return out
