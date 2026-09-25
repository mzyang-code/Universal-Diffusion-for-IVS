#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Based on VolGAN: https://github.com/milenavuletic/VolGAN

VolGAN generator and discriminator (the original paper's MLP architecture).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Strict-original VolGAN generator: 3-layer MLP, Softplus, linear output."""

    def __init__(self, noise_dim: int, cond_dim: int, hidden_dim: int,
                 output_dim: int):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim = noise_dim + cond_dim

        self.linear1 = nn.Linear(self.input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear3 = nn.Linear(hidden_dim * 2, output_dim)
        self.activation1 = nn.Softplus()
        self.activation2 = nn.Softplus()

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """noise: (B, noise_dim), condition: (B, cond_dim) -> (B, output_dim)."""
        x = torch.cat([noise, condition], dim=-1).to(torch.float)
        h = self.activation1(self.linear1(x))
        h = self.activation2(self.linear2(h))
        return self.linear3(h)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, n_per_state: int) -> torch.Tensor:
        """For each row in `condition`, draw n_per_state targets.

        Returns (B, n_per_state, output_dim).
        """
        B = condition.shape[0]
        device = condition.device
        z = torch.randn(B * n_per_state, self.noise_dim, device=device)
        c = condition.repeat_interleave(n_per_state, dim=0).to(torch.float)
        out = self.forward(z, c)
        return out.view(B, n_per_state, -1)


class Discriminator(nn.Module):
    """Strict-original VolGAN discriminator: 2-layer MLP, Softplus, Sigmoid."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_chan: torch.Tensor) -> torch.Tensor:
        """in_chan: concatenated [condition, target_real_or_fake]."""
        h = self.softplus(self.linear1(in_chan.to(torch.float)))
        return self.sigmoid(self.linear2(h))
