#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

All training objectives of the conditional DDPM: MSE plus optional smoothness and
static-arbitrage penalties.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

# Pure-tensor Black-Scholes pricing + static-arbitrage kernels. This module has no
# project dependencies, so importing it here introduces no cycle and guarantees
# train == eval penalty math by construction.
from ivs.evaluate.arbitrage import (
    relative_call_price, calendar_penalty, call_penalty, butterfly_penalty,
)

ZERO_TERMS = 5


# Diffusion (denoising) loss -- paper Eq. 10
def diffusion_loss(eps_pred: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Standard DDPM training objective: MSE between predicted and true noise."""
    return nn.functional.mse_loss(eps_pred, eps)


# Target reconstruction (shared, grad-flow safe) -- paper Eq. 8-9
def denorm_x0(
    x0_norm: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    clamp_norm: float = 5.0,
) -> torch.Tensor:
    """Denormalize the predicted x0 back to raw target units.

    x0_norm is clamped to +/- ``clamp_norm`` standard deviations BEFORE the affine
    de-standardization. At high noise levels the x0 estimate can blow up; the clamp
    keeps the downstream surface reconstruction and its gradients well-behaved.
    """
    return x0_norm.clamp(-clamp_norm, clamp_norm) * target_std + target_mean


def reconstruct_log_iv_surface(
    x0_raw: torch.Tensor,
    cond_raw: torch.Tensor,
    surface_dim: int,
) -> torch.Tensor:
    """Rebuild the log-IV surface log sigma_t, flat ``(B, surface_dim)``.

    The model's surface target is the log-IV INCREMENT
        x0_raw[:, -surface_dim:] = log sigma_t - log sigma_{t-1}
    and the conditioning carries the previous-day log-IV surface
        cond_raw[:, -surface_dim:] = log sigma_{t-1}
    so the generated surface is their sum (paper Eq. 9).
    """
    log_iv_prev = cond_raw[:, -surface_dim:]          # log sigma_{t-1}
    log_iv_inc_hat = x0_raw[:, -surface_dim:]         # predicted increment
    return log_iv_prev + log_iv_inc_hat               # log sigma_t


# Finite-difference grid spacings (smoothness denominators)
def grid_spacings(
    m_grid,
    tau_grid,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (d_m, d_tau) node-spacing tensors for the finite differences.

    d_m   : (SURFACE_M - 1,) = m_{i+1}   - m_i
    d_tau : (SURFACE_T - 1,) = tau_{j+1} - tau_j

    These are the (non-uniform) denominators of the Sobolev smoothness penalty.
    ``eps`` guards against a zero spacing (never expected on a strictly increasing
    grid, but keeps the division safe).
    """
    m = torch.as_tensor(m_grid, dtype=dtype, device=device)
    tau = torch.as_tensor(tau_grid, dtype=dtype, device=device)
    d_m = (m[1:] - m[:-1]).clamp_min(eps)
    d_tau = (tau[1:] - tau[:-1]).clamp_min(eps)
    return d_m, d_tau


def smoothness_penalty(
    surf_log: torch.Tensor, d_m: torch.Tensor, d_tau: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(L_m, L_tau) on the LOG-IV surface ``surf_log`` of shape (B, M, T).

    Each first difference is divided by the corresponding node spacing, so the
    penalty is a true squared discrete derivative on the non-uniform grid
    (paper Eq. 11-12). No exp() is applied, keeping the penalty homogeneous with
    the model's log-IV generation target.
    """
    d_m_surf = (surf_log[:, 1:, :] - surf_log[:, :-1, :]) / d_m.view(1, -1, 1)
    d_tau_surf = (surf_log[:, :, 1:] - surf_log[:, :, :-1]) / d_tau.view(1, 1, -1)
    return (d_m_surf ** 2).mean(), (d_tau_surf ** 2).mean()


def arbitrage_penalty(
    surf_log: torch.Tensor,
    m_grid: torch.Tensor,
    tau_grid: torch.Tensor,
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(p_calendar, p_spread, p_butterfly) -- paper Eq. 2-4, batch means.

    exp() the log-IV surface, Black-Scholes-price the relative call surface
    c(m, tau) under the per-example risk-free rate ``r``, then apply the three
    finite-difference violation penalties from the evaluation kernels.
    """
    sigma = torch.exp(surf_log)                                     # (B, M, T)
    c = relative_call_price(sigma, m_grid.float(), tau_grid.float(), r.float())
    p_calendar = calendar_penalty(c, tau_grid.float()).mean()
    p_spread = call_penalty(c, m_grid.float()).mean()               # vertical spread
    p_butterfly = butterfly_penalty(c, m_grid.float()).mean()
    return p_calendar, p_spread, p_butterfly


# The single shared entry used by BOTH train.py and calibrate_lambda.py
def penalties_from_pred(
    x0_norm: torch.Tensor,
    cond_raw: torch.Tensor,
    r: torch.Tensor,
    *,
    surface_m: int,
    surface_t: int,
    surface_dim: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    m_grid: torch.Tensor,
    tau_grid: torch.Tensor,
    d_m: torch.Tensor,
    d_tau: torch.Tensor,
    use_smoothness: bool = True,
    use_arbitrage: bool = True,
    smoothness_mode: str = "logiv_fd",
    clamp_norm: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predicted (normalized) x0 -> (L_m, L_tau, p_calendar, p_spread, p_butterfly).

    The log-IV surface is reconstructed ONCE and fed to whichever penalty families
    the active variant enables:

        x0_norm --clamp/denorm--> x0_raw
                --(+ log sigma_{t-1})--> log-IV surface g_t  (B, M, T)
                |--> finite-difference smoothness on g_t     -> L_m, L_tau
                +--exp--> sigma --Black-Scholes--> c(m, tau) -> p_calendar,
                                                                p_spread,
                                                                p_butterfly

    Disabled families return zero scalars (no graph, no cost), which is what makes
    DDPM_mse exactly the plain denoising objective. ``calibrate_lambda.py`` calls
    this with both families enabled regardless of the variant, because it measures
    every penalty gradient on a penalty-free reference model.

    Args:
        x0_norm  : (B, TARGET_DIM) predicted normalized x0 (= predict_x0_from_eps).
        cond_raw : (B, COND_DIM) RAW condition (carries log sigma_{t-1}).
        r        : (B,) per-example risk-free rate aligned to the target day.
        m_grid, tau_grid : (M,) / (T,) node tensors on the compute device.
        d_m, d_tau       : finite-difference node spacings (``grid_spacings``).

    Returns:
        Five scalar tensors, each a mean over the mini-batch.
    """
    if smoothness_mode != "logiv_fd":
        raise ValueError(
            f"Unsupported smoothness_mode: {smoothness_mode!r}. Only the "
            f"paper-faithful 'logiv_fd' (log-IV finite-difference Sobolev) "
            f"penalty is implemented.")

    zero = x0_norm.new_zeros(())
    if not (use_smoothness or use_arbitrage):
        return (zero,) * ZERO_TERMS

    # Reconstruct in fp32 for stable smoothness + Black-Scholes pricing (the caller
    # disables autocast around this block; .float() also covers a bf16 x0_norm).
    x0_raw = denorm_x0(
        x0_norm.float(), target_mean.float(), target_std.float(), clamp_norm=clamp_norm
    )
    log_surface = reconstruct_log_iv_surface(x0_raw, cond_raw.float(), surface_dim)
    surf_log = log_surface.view(log_surface.shape[0], surface_m, surface_t)

    if use_smoothness:
        L_m, L_tau = smoothness_penalty(surf_log, d_m, d_tau)
    else:
        L_m = L_tau = zero

    if use_arbitrage:
        p_cal, p_spr, p_bfly = arbitrage_penalty(surf_log, m_grid, tau_grid, r)
    else:
        p_cal = p_spr = p_bfly = zero

    return L_m, L_tau, p_cal, p_spr, p_bfly


def total_loss(
    loss_mse: torch.Tensor,
    terms: Tuple[torch.Tensor, ...],
    lambdas: Tuple[float, ...],
) -> torch.Tensor:
    """L = L_mse + sum_k lambda_k * term_k (terms in ``penalties_from_pred`` order)."""
    out = loss_mse.float()
    for lam, term in zip(lambdas, terms):
        if lam:
            out = out + lam * term
    return out
