#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Static-arbitrage penalties (calendar / vertical spread / butterfly) on the IV grid and
the per-ticker arbitrage metrics of Table 2.
"""
from __future__ import annotations

import math

import numpy as np
import torch


_SQRT2 = math.sqrt(2.0)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def relative_call_price(sigma: torch.Tensor,
                        m_grid: torch.Tensor,
                        tau_grid: torch.Tensor,
                        r: torch.Tensor,
                        sigma_floor: float = 1e-4,
                        sigma_cap: float = 5.0) -> torch.Tensor:
    """Black-Scholes relative call c(m, tau) = N(d1) - m e^{-r tau} N(d2).

    Args:
        sigma:    (..., M, T) implied vol surface.
        m_grid:   (M,) moneyness nodes (= K/S).
        tau_grid: (T,) time-to-maturity nodes in years.
        r:        scalar tensor OR (...,) per-state risk-free rate; broadcast
                  to (..., 1, 1).

    Returns:
        c: (..., M, T) relative call price (per unit spot).
    """
    sigma = sigma.clamp(min=sigma_floor, max=sigma_cap)
    # Broadcast helpers
    m = m_grid.view(*([1] * (sigma.dim() - 2)), -1, 1)        # (..., M, 1)
    tau = tau_grid.view(*([1] * (sigma.dim() - 2)), 1, -1)    # (..., 1, T)
    if r.dim() < sigma.dim():
        # promote r -> (..., 1, 1)
        while r.dim() < sigma.dim():
            r = r.unsqueeze(-1)
    sqrt_tau = torch.sqrt(tau)
    log_m = torch.log(m)
    sig_sqrt_tau = sigma * sqrt_tau
    d1 = (-log_m + (r + 0.5 * sigma * sigma) * tau) / sig_sqrt_tau
    d2 = d1 - sig_sqrt_tau
    return _normal_cdf(d1) - m * torch.exp(-r * tau) * _normal_cdf(d2)


def calendar_penalty(c: torch.Tensor, tau_grid: torch.Tensor) -> torch.Tensor:
    """p1: violation of dC/dtau >= 0 (calendar arbitrage), summed over (m, tau)."""
    # c: (..., M, T). diff over the tau axis (last).
    dtau = (tau_grid[1:] - tau_grid[:-1])                     # (T-1,)
    dtau = dtau.view(*([1] * (c.dim() - 1)), -1)              # (..., 1, T-1)
    tau_left = tau_grid[:-1].view(*([1] * (c.dim() - 1)), -1) # (..., 1, T-1)
    # c_j - c_{j+1} along tau
    diff = c[..., :-1] - c[..., 1:]                           # (..., M, T-1)
    term = tau_left * diff / dtau
    return torch.clamp(term, min=0.0).sum(dim=(-2, -1))


def call_penalty(c: torch.Tensor, m_grid: torch.Tensor) -> torch.Tensor:
    """p2: violation of dC/dm <= 0 (call monotonicity), summed."""
    dm = (m_grid[1:] - m_grid[:-1])                           # (M-1,)
    dm = dm.view(*([1] * (c.dim() - 2)), -1, 1)               # (..., M-1, 1)
    diff = c[..., 1:, :] - c[..., :-1, :]                     # (..., M-1, T)
    term = diff / dm
    return torch.clamp(term, min=0.0).sum(dim=(-2, -1))


def butterfly_penalty(c: torch.Tensor, m_grid: torch.Tensor) -> torch.Tensor:
    """p3: violation of d^2C/dm^2 >= 0 (butterfly / convexity)."""
    # forward slope at i:  s_i^+ = (c_{i+1} - c_i) / (m_{i+1} - m_i)
    # backward slope at i: s_i^- = (c_i - c_{i-1}) / (m_i - m_{i-1})
    # convexity:  s_i^+ >= s_i^-  ->  (s_i^- - s_i^+) <= 0
    # penalty   : (s_i^- - s_i^+)^+ summed across interior i, all j.
    dm = (m_grid[1:] - m_grid[:-1])                           # (M-1,)
    dm_view = dm.view(*([1] * (c.dim() - 2)), -1, 1)          # (..., M-1, 1)
    fwd = (c[..., 1:, :] - c[..., :-1, :]) / dm_view          # (..., M-1, T)
    s_minus = fwd[..., :-1, :]                                # (..., M-2, T)
    s_plus = fwd[..., 1:, :]                                  # (..., M-2, T)
    term = s_minus - s_plus
    return torch.clamp(term, min=0.0).sum(dim=(-2, -1))


def arbitrage_phi(sigma: torch.Tensor,
                  m_grid: torch.Tensor,
                  tau_grid: torch.Tensor,
                  r: torch.Tensor) -> dict:
    """Compute (p1, p2, p3, Phi) for arbitrary leading dims.

    Returns a dict of tensors, each with shape sigma.shape[:-2].
    """
    c = relative_call_price(sigma, m_grid, tau_grid, r)
    p1 = calendar_penalty(c, tau_grid)
    p2 = call_penalty(c, m_grid)
    p3 = butterfly_penalty(c, m_grid)
    return {"p1": p1, "p2": p2, "p3": p3, "phi": p1 + p2 + p3}


# Table 2: per-ticker arbitrage metrics
def _adaptive_beta(phi: torch.Tensor, c: float = 500.0, tol: float = 1e-8,
                   max_iter: int = 100) -> torch.Tensor:
    """Per-day beta as the fixed point of beta = c * max_s w_s(beta)."""
    beta = torch.tensor(1e-6, device=phi.device)
    for _ in range(max_iter):
        w = torch.exp(-beta * phi)
        w = w / w.sum()
        beta_new = c * w.max()
        if abs(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta


def arbitrage_metrics(z, device: torch.device, kl_constant: float = 500.0,
                      batch: int = 64) -> dict:
    """Test-period mean calendar / spread / butterfly penalties of the observed
    surfaces and of the generated scenarios, with equal weights (beta = 0) and
    with adaptive arbitrage weights (beta > 0)."""
    m = torch.from_numpy(np.asarray(z["m_grid"], dtype=np.float32)).to(device)
    tau = torch.from_numpy(np.asarray(z["tau_grid"], dtype=np.float32)).to(device)
    out = {}

    real = torch.from_numpy(z["real_iv"].astype(np.float32)).to(device)
    r = arbitrage_phi(real, m, tau, torch.zeros(real.shape[0], device=device))
    for name, key in (("calendar", "p1"), ("spread", "p2"), ("butterfly", "p3")):
        out[f"data_{name}"] = float(r[key].nanmean().cpu())
    out["data_phi"] = float((r["p1"] + r["p2"] + r["p3"]).nanmean().cpu())

    synth = torch.from_numpy(z["synth_iv"].astype(np.float32)).to(device)
    n_states, n_scen = synth.shape[:2]
    per_state = {k: [] for k in ("p1_b0", "p2_b0", "p3_b0", "p1_b", "p2_b", "p3_b")}
    with torch.no_grad():
        for i in range(0, n_states, batch):
            iv = synth[i:i + batch]
            b = iv.shape[0]
            res = arbitrage_phi(iv.reshape(b * n_scen, *iv.shape[2:]), m, tau,
                                torch.zeros(b * n_scen, device=device))
            p = {k: res[k].reshape(b, n_scen) for k in ("p1", "p2", "p3")}
            phi = p["p1"] + p["p2"] + p["p3"]
            for j in range(b):
                if n_scen > 1:
                    beta = _adaptive_beta(phi[j], kl_constant)
                    w = torch.exp(-beta * phi[j])
                    w = w / w.sum()
                else:
                    w = torch.ones_like(phi[j])
                for k in ("p1", "p2", "p3"):
                    per_state[f"{k}_b0"].append(float(p[k][j].mean().cpu()))
                    per_state[f"{k}_b"].append(float((w * p[k][j]).sum().cpu()))

    for suffix, tag in (("b0", "beta0"), ("b", "beta")):
        vals = [float(np.nanmean(per_state[f"{k}_{suffix}"])) for k in ("p1", "p2", "p3")]
        out[f"calendar_{tag}"], out[f"spread_{tag}"], out[f"butterfly_{tag}"] = vals
        out[f"phi_{tag}"] = sum(vals)
    return out
