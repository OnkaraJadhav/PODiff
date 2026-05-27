from __future__ import annotations

import torch
import torch.nn.functional as F


def denormalize_minmax(x: torch.Tensor, norm_min: float, norm_max: float) -> torch.Tensor:
    """Invert min-max normalization from [0, 1] back to physical units."""
    return x * (float(norm_max) - float(norm_min)) + float(norm_min)

def masked_rmse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    err2 = (x - y) ** 2
    if mask is None:
        return torch.sqrt(torch.mean(err2) + eps)
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < err2.ndim:
        mask = mask.unsqueeze(0)
    return torch.sqrt((err2 * mask).sum() / mask.sum().clamp_min(1.0) + eps)

def masked_mae(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = torch.abs(x - y)
    if mask is None:
        return torch.mean(err)
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < err.ndim:
        mask = mask.unsqueeze(0)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)

def masked_r2(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        y_mean = y.mean()
        ss_res = ((x - y) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        return 1.0 - ss_res / (ss_tot + eps)
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < y.ndim:
        mask = mask.unsqueeze(0)
    y_mean = (y * mask).sum() / mask.sum().clamp_min(1.0)
    ss_res = (((x - y) ** 2) * mask).sum()
    ss_tot = (((y - y_mean) ** 2) * mask).sum()
    return 1.0 - ss_res / (ss_tot + eps)
