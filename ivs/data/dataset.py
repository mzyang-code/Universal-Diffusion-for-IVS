#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Build (condition, target) pairs for joint generation of the next-day return and log-IV
surface increment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ivs.config import (
    COND_DIM, RET_ANNUALIZER_SQRT, RV_ANNUALIZER_SQRT, RV_WINDOW,
    SPLIT_LOG_RET_ABS_THRESHOLD, SURFACE_DIM, TARGET_DIM, TENSOR_DIR,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
)


@dataclass
class TickerArrays:
    ivs: np.ndarray            # (T, SURFACE_M, SURFACE_T) raw IV
    log_ret: np.ndarray        # (T,) daily log returns (unscaled)
    dates: np.ndarray          # (T,) datetime64[D]
    rates: np.ndarray          # (T,) per-day risk-free rate (aligned 1:1 with dates)


def load_ticker(ticker: str) -> TickerArrays:
    z = np.load(TENSOR_DIR / f"{ticker}_ivs.npz", allow_pickle=False)
    ivs = z["ivs"].astype(np.float32)
    log_ret = z["log_ret"].astype(np.float32)
    dates = z["dates"]
    rates = z["rates"].astype(np.float32)
    # Zero out split-day jumps (un-adjusted corporate actions)
    mask = np.abs(log_ret) > SPLIT_LOG_RET_ABS_THRESHOLD
    if mask.any():
        log_ret = log_ret.copy()
        log_ret[mask] = 0.0
    return TickerArrays(ivs=ivs, log_ret=log_ret, dates=dates, rates=rates)


def rates_per_example(arrays: TickerArrays) -> np.ndarray:
    """Risk-free rate r_t aligned to the target day of each example.

    The cached rates are aligned 1:1 with arrays.dates, so this is the same tail
    slice build_features takes. The arbitrage penalty and the evaluation both
    price with this r_t.
    """
    t0 = RV_WINDOW + 1
    return arrays.rates[t0:].astype(np.float32)


def realized_vol_per_example(log_ret: np.ndarray, n: int) -> np.ndarray:
    """Realized-vol feature rv_{t-1}[k], following the original VolGAN convention:

        rv[k] = sqrt(252/21) * sqrt( sum_{j=0..20} log_ret[k+j]^2 )

    The 21-day window spans [t-22, t-2] and so ends at t-2, not t-1. Returns rv
    of length n (= T - 22).
    """
    sq = log_ret.astype(np.float64) ** 2
    cs = np.concatenate(([0.0], np.cumsum(sq)))     # cs[i] = sum sq[:i]
    rv = np.empty(n, dtype=np.float64)
    for k in range(n):
        s = cs[k + RV_WINDOW] - cs[k]               # sum sq[k : k+RV_WINDOW]
        rv[k] = RV_ANNUALIZER_SQRT * np.sqrt(max(s, 0.0))
    return rv


def build_features(arrays: TickerArrays) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (N, COND_DIM) cond and (N, TARGET_DIM) target matrices.

    Example k has target day t = k + 22. Returns (cond, target, target_dates).
    """
    T = arrays.ivs.shape[0]
    log_iv = np.log(np.clip(arrays.ivs.astype(np.float64), 1e-6, None))
    log_iv_flat = log_iv.reshape(T, SURFACE_DIM).astype(np.float32)   # (T, SURFACE_DIM)

    lr = arrays.log_ret.astype(np.float32) * np.float32(RET_ANNUALIZER_SQRT)

    t0 = RV_WINDOW + 1                              # first valid `t` (= 22)
    if t0 >= T:
        raise ValueError(f"not enough days to build any (cond,target): T={T}")

    n = T - t0
    rv = realized_vol_per_example(arrays.log_ret, n).astype(np.float32)
    cond = np.empty((n, COND_DIM), dtype=np.float32)
    target = np.empty((n, TARGET_DIM), dtype=np.float32)

    for k in range(n):
        t = t0 + k
        cond[k, 0] = lr[t - 1]
        cond[k, 1] = lr[t - 2]
        cond[k, 2] = rv[k]
        cond[k, 3:] = log_iv_flat[t - 1]
        target[k, 0] = lr[t]
        target[k, 1:] = log_iv_flat[t] - log_iv_flat[t - 1]

    dates = arrays.dates[t0:]
    return cond, target, dates


def date_split(target_dates: np.ndarray) -> Tuple[slice, slice]:
    """Chronological split by target date, returned as two slices.

        train = target dates in [TRAIN_START, TRAIN_END]
        test  = target dates in [TEST_START,  TEST_END]

    target_dates is sorted and the two windows are contiguous on the trading
    calendar, so the split reduces to two contiguous slices.
    """
    d = np.asarray(target_dates, dtype="datetime64[D]")
    tr_end = np.datetime64(TRAIN_END)
    te_start, te_end = np.datetime64(TEST_START), np.datetime64(TEST_END)
    train_mask = (d >= np.datetime64(TRAIN_START)) & (d <= tr_end)
    test_idx = np.where((d >= te_start) & (d <= te_end))[0]
    n_train = int(train_mask.sum())
    if test_idx.size == 0:
        return slice(0, n_train), slice(n_train, n_train)
    te_lo, te_hi = int(test_idx[0]), int(test_idx[-1]) + 1
    # contiguity sanity: train block must end exactly where the test block starts
    assert te_lo == n_train, (
        f"non-contiguous date split (n_train={n_train}, first test idx={te_lo}); "
        "target dates are expected sorted & gapless across the split boundary")
    return slice(0, n_train), slice(te_lo, te_hi)
