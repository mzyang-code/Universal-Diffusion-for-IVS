#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Per-dimension z-score normalizer for the condition and target vectors, fitted on the
pooled train split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


_EPS = 1e-8


class Normalizer:
    """Per-dimension z-score normalizer for cond and target."""

    def __init__(
        self,
        cond_mean: np.ndarray,
        cond_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = np.maximum(cond_std, _EPS).astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = np.maximum(target_std, _EPS).astype(np.float32)

    # forward transforms
    def normalize_cond(self, cond: np.ndarray) -> np.ndarray:
        return ((cond - self.cond_mean) / self.cond_std).astype(np.float32)

    def normalize_target(self, target: np.ndarray) -> np.ndarray:
        return ((target - self.target_mean) / self.target_std).astype(np.float32)

    # inverse (target only)
    def denormalize_target(self, target_norm: np.ndarray) -> np.ndarray:
        return (target_norm * self.target_std + self.target_mean).astype(np.float32)

    # persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            cond_mean=self.cond_mean,
            cond_std=self.cond_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
        )

    @staticmethod
    def load(path: Path) -> "Normalizer":
        z = np.load(path, allow_pickle=False)
        return Normalizer(
            cond_mean=z["cond_mean"],
            cond_std=z["cond_std"],
            target_mean=z["target_mean"],
            target_std=z["target_std"],
        )


def fit_normalizer(
    cond_list: list[np.ndarray],
    target_list: list[np.ndarray],
) -> Normalizer:
    """Fit one shared normalizer on the pooled train (cond, target) arrays."""
    cond_all = np.concatenate(cond_list, axis=0).astype(np.float64)
    tgt_all = np.concatenate(target_list, axis=0).astype(np.float64)
    return Normalizer(
        cond_mean=cond_all.mean(axis=0).astype(np.float32),
        cond_std=cond_all.std(axis=0).astype(np.float32),
        target_mean=tgt_all.mean(axis=0).astype(np.float32),
        target_std=tgt_all.std(axis=0).astype(np.float32),
    )
