#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

FiLM-conditioned MLP denoiser predicting the noise on the return and surface-increment target.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# Sinusoidal time embedding
class SinusoidalTimeEmbedding(nn.Module):
    """t (scalar int) -> (B, time_emb_dim) via sinusoidal frequencies,
    then projected by a small MLP to hidden_dim."""

    def __init__(self, time_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        assert time_emb_dim % 2 == 0
        self.time_emb_dim = time_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float or int
        half = self.time_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)      # (B, time_emb_dim)
        return self.mlp(emb)                                   # (B, hidden_dim)


# Condition encoder
class ConditionEncoder(nn.Module):
    """Encodes normalized cond (B, COND_DIM) -> (B, hidden_dim)."""

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


# FiLM Residual Block
class FiLMResidualBlock(nn.Module):
    """Single FiLM-conditioned residual block.

    h_{l+1} = h_l + F(FiLM(LN(h_l), context))
    FiLM:     h' = (1 + gamma) * LN(h) + beta
    context = t_emb + c_emb  (hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)   # -> gamma, beta
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, h: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor
    ) -> torch.Tensor:
        context = t_emb + c_emb                                # (B, hidden_dim)
        gamma, beta = self.film(context).chunk(2, dim=-1)      # each (B, hidden_dim)
        h_norm = self.norm(h)
        h_film = (1.0 + gamma) * h_norm + beta
        return h + self.ff(h_film)


# Main denoiser
class DiffusionDenoiser(nn.Module):
    """Conditional denoising network for 1D structured data.

    Predicts the noise epsilon given (x_t, t, cond):
        eps_pred = eps_theta(x_t, t, cond)

    Args:
        x_dim     : dimension of the target vector (TARGET_DIM = 100)
        cond_dim  : dimension of the condition     (COND_DIM = 102)
        hidden_dim: width of all hidden layers     (256)
        time_emb_dim: sinusoidal embedding size    (128)
        num_blocks: number of FiLM residual blocks (8)
        dropout   : dropout probability            (0.0)
    """

    def __init__(
        self,
        x_dim: int = 100,
        cond_dim: int = 102,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        num_blocks: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, hidden_dim)
        self.cond_encoder = ConditionEncoder(cond_dim, hidden_dim)
        self.x_proj = nn.Linear(x_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.out = nn.Linear(hidden_dim, x_dim)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t  : (B, x_dim)   noisy target
            t    : (B,)          diffusion timestep (int, 1-indexed)
            cond : (B, cond_dim) normalized condition

        Returns:
            eps_pred: (B, x_dim)
        """
        h = self.x_proj(x_t)                   # (B, hidden_dim)
        t_emb = self.time_embed(t)              # (B, hidden_dim)
        c_emb = self.cond_encoder(cond)         # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb, c_emb)
        return self.out(h)                      # (B, x_dim)


# Exponential Moving Average wrapper
class EMA:
    """Maintain an EMA copy of model parameters (shadow model)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Copy EMA weights into model (for inference)."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore model weights from backup after EMA apply."""
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}
