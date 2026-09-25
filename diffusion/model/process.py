#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

DDPM noise schedule, forward diffusion and reverse sampling.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


# Beta schedules
def make_beta_schedule(
    schedule: str,
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    """Return a (T,) float64 beta schedule tensor."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        # Nichol & Dhariwal (2021) cosine schedule
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_bar = torch.cos((t / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
        return betas.clamp(0.0, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")


# DiffusionSchedule: pre-computed coefficient tensors
class DiffusionSchedule:
    """Pre-computes all schedule tensors for DDPM forward and reverse passes.

    Convention: 1-indexed timesteps t in {1, ..., T}.
    Internal arrays are 0-indexed (index t-1 stores the coefficient for step t).
    """

    def __init__(self, betas: torch.Tensor) -> None:
        # Keep all schedule tensors on CPU; _gather moves them to device on-demand.
        betas = betas.double().cpu()
        T = betas.shape[0]
        self.T = T

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)           # (T,)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        self.alpha_bar_prev = alpha_bar_prev.float()
        self.sqrt_alpha_bar = alpha_bar.sqrt().float()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().float()
        self.sqrt_recip_alpha_bar = (1.0 / alpha_bar).sqrt().float()
        self.sqrt_recip_alpha_bar_m1 = (1.0 / alpha_bar - 1.0).sqrt().float()

        # Posterior q(x_{t-1} | x_t, x0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_var = posterior_var.float()
        self.posterior_log_var_clipped = torch.log(
            torch.cat([posterior_var[1:2], posterior_var[1:]])
        ).float()
        self.posterior_mean_coef1 = (
            betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar)
        ).float()

    _COEF_ATTRS = (
        "betas", "alphas", "alpha_bar", "alpha_bar_prev", "sqrt_alpha_bar",
        "sqrt_one_minus_alpha_bar", "sqrt_recip_alpha_bar", "sqrt_recip_alpha_bar_m1",
        "posterior_var", "posterior_log_var_clipped", "posterior_mean_coef1",
        "posterior_mean_coef2",
    )

    def to(self, device) -> "DiffusionSchedule":
        """Move all coefficient tensors onto `device` ONCE so _gather no longer
        transfers CPU->GPU every step (a large speedup for the 1000-step sampling
        loop). Numerically identical -- same float32 values, different device."""
        for name in self._COEF_ATTRS:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _gather(self, arr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather schedule coefficients at timesteps t (1-indexed)."""
        arr = arr.to(t.device)                          # no-op once .to(device) was called
        return arr[t - 1].view(-1, *([1] * (t.ndim)))   # broadcast-ready shape

    # Forward diffusion
    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x0) at arbitrary t.

        Returns (x_t, noise) where noise ~ N(0, I).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._gather(self.sqrt_alpha_bar, t)
        sqrt_1mab = self._gather(self.sqrt_one_minus_alpha_bar, t)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    # x0 prediction from eps
    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x0_hat from noisy x_t and predicted noise eps_pred."""
        sqrt_recip_ab = self._gather(self.sqrt_recip_alpha_bar, t)
        sqrt_recip_ab_m1 = self._gather(self.sqrt_recip_alpha_bar_m1, t)
        return sqrt_recip_ab * x_t - sqrt_recip_ab_m1 * eps_pred

    # Reverse (DDPM) single step
    def p_sample(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse DDPM step: x_t -> x_{t-1}."""
        eps_pred = model_fn(x_t, t, cond)
        x0_hat = self.predict_x0_from_eps(x_t, t, eps_pred)
        x0_hat = x0_hat.clamp(-10.0, 10.0)   # soft clamp for stability

        c1 = self._gather(self.posterior_mean_coef1, t)
        c2 = self._gather(self.posterior_mean_coef2, t)
        posterior_mean = c1 * x0_hat + c2 * x_t

        log_var = self._gather(self.posterior_log_var_clipped, t)
        noise = torch.randn_like(x_t)
        mask = (t > 1).float().view(-1, *([1] * (x_t.ndim - 1)))
        return posterior_mean + mask * (0.5 * log_var).exp() * noise

    # Full DDPM sampling loop
    @torch.no_grad()
    def ddpm_sample(
        self,
        model_fn: Callable,
        shape: tuple[int, ...],
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Full DDPM reverse loop T -> 1, returns x0_hat.

        Args:
            shape  : (B, x_dim)
            cond   : (B, cond_dim) normalized
            device : torch.device
        """
        x_t = torch.randn(*shape, device=device)
        for step in range(self.T, 0, -1):
            t_tensor = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model_fn, x_t, t_tensor, cond)
        return x_t
