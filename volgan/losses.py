#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Based on VolGAN: https://github.com/milenavuletic/VolGAN

VolGAN losses: BCE plus log-IV smoothness penalties.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SmoothnessOps:
    """Cached weight tensors for batched smoothness penalties.

    Layout is row-major (m, tau): a surface of shape (lk, lt) is flattened
    as `s.reshape(lk*lt)` -> position (i, j) maps to i*lt + j.

      w_m : (lk-1, lt)   inverse moneyness-spacing^2,
                         broadcast across all tau indices.
      w_t : (lk, lt-1)   inverse tau-spacing^2,
                         broadcast across all moneyness indices.
    """
    w_m: torch.Tensor   # (lk-1, lt)
    w_t: torch.Tensor   # (lk, lt-1)
    lk: int
    lt: int


def build_smoothness_ops(m_grid: torch.Tensor, tau_grid: torch.Tensor,
                         device: torch.device) -> SmoothnessOps:
    """Cache 1/(spacing)^2 weights for both axes on `device`."""
    lk = int(m_grid.shape[0])
    lt = int(tau_grid.shape[0])
    m = m_grid.to(device=device, dtype=torch.float)
    tau = tau_grid.to(device=device, dtype=torch.float)
    m_spacing_inv2 = 1.0 / ((m[1:] - m[:-1]) ** 2)        # (lk-1,)
    t_spacing_inv2 = 1.0 / ((tau[1:] - tau[:-1]) ** 2)    # (lt-1,)
    w_m = m_spacing_inv2.unsqueeze(1).expand(lk - 1, lt).contiguous()
    w_t = t_spacing_inv2.unsqueeze(0).expand(lk, lt - 1).contiguous()
    return SmoothnessOps(w_m=w_m, w_t=w_t, lk=lk, lt=lt)


def discriminator_loss(criterion: nn.BCELoss,
                       disc_real_pred: torch.Tensor,
                       disc_fake_pred: torch.Tensor) -> torch.Tensor:
    """Standard BCE GAN discriminator loss (averaged over real and fake)."""
    real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    return (real_loss + fake_loss) * 0.5


def generator_bce(criterion: nn.BCELoss,
                  disc_fake_pred: torch.Tensor) -> torch.Tensor:
    """Generator BCE loss with the paper's explicit 1/2 factor.

    Returns 0.5 * BCE(D(fake), 1).  Applied identically in gradient matching and
    in the main loop, so the 1/2 is self-consistent (see module docstring).
    """
    return 0.5 * criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))


def smoothness_penalties(fake_surface_flat: torch.Tensor,
                         ops: SmoothnessOps) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched smoothness penalties (moneyness, tau) on the LOG-IV surface.

    fake_surface_flat: (B, lk*lt) -- the *log-IV* surface g_t(m, tau) flattened
        row-major (m, tau).  (The caller must pass log_iv_inc + prev_log_iv, not
        exp(.) of it -- see the module docstring.)
    Returns (m_penalty, t_penalty), each a scalar = per-sample penalty averaged
    over the mini-batch (Monte-Carlo estimate of E[L_m] / E[L_tau]).
    """
    B = fake_surface_flat.shape[0]
    s = fake_surface_flat.view(B, ops.lk, ops.lt)         # (B, lk, lt)

    diff_m = s[:, 1:, :] - s[:, :-1, :]                   # (B, lk-1, lt)
    m_pen = (ops.w_m * (diff_m ** 2)).sum(dim=(1, 2)).mean()

    diff_t = s[:, :, 1:] - s[:, :, :-1]                   # (B, lk, lt-1)
    t_pen = (ops.w_t * (diff_t ** 2)).sum(dim=(1, 2)).mean()

    return m_pen, t_pen
