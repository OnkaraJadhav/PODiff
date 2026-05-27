from __future__ import annotations
import os
import torch
from torchvision.utils import make_grid, save_image

def save_grid(x: torch.Tensor, out_path: str, nrow: int = 16) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    grid = make_grid(x, nrow=nrow, pad_value=1.0)
    save_image(grid, out_path)

def save_recon_grid(original: torch.Tensor, recon: torch.Tensor, out_path: str, nrow: int = 16) -> None:
    b = min(original.shape[0], recon.shape[0])
    x = torch.cat([original[:b], recon[:b]], dim=0)
    save_grid(x, out_path, nrow=nrow)
