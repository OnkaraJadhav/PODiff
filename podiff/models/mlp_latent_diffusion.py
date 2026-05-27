from __future__ import annotations
import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=1)
        return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GlobalMLPDiffusion(nn.Module):
    """MLP noise predictor for whole-field POD coefficients.

    Input a_t: (B, K). For SR, cond is LR/interpolated POD coefficients (B, K).
    Output predicted noise: (B, K).
    """
    def __init__(self, k_dim: int, model_dim: int = 512, n_layers: int = 6, time_dim: int = 256, cond_mode: str = 'sr', dropout: float = 0.0):
        super().__init__()
        if cond_mode not in ['none', 'sr']:
            raise ValueError("GlobalMLPDiffusion supports cond_mode='none' or 'sr'.")
        self.k_dim = int(k_dim)
        self.cond_mode = cond_mode
        in_dim = k_dim + time_dim + (k_dim if cond_mode == 'sr' else 0)
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.in_proj = nn.Linear(in_dim, model_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(model_dim, dropout=dropout) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, k_dim))

    def forward(self, a_t: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        te = self.time(t).to(dtype=a_t.dtype)
        if self.cond_mode == 'sr':
            if cond is None:
                raise ValueError("cond must be provided when cond_mode='sr'.")
            x = torch.cat([a_t, cond, te], dim=1)
        else:
            x = torch.cat([a_t, te], dim=1)
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out(h)
