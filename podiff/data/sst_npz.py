from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class SSTLoaders:
    train: DataLoader
    test: DataLoader
    num_classes: int = 1


def _to_nchw(a: np.ndarray, name: str) -> np.ndarray:
    """Accept (N,H,W), (N,H,W,1), or (N,1,H,W) and return float32 NCHW."""
    if a.ndim == 3:
        a = a[:, None, :, :]
    elif a.ndim == 4:
        if a.shape[-1] == 1:          # NHWC from concatenation.py
            a = np.transpose(a, (0, 3, 1, 2))
        elif a.shape[1] == 1:         # already NCHW
            pass
        else:
            raise ValueError(f"{name} must have one channel. Got shape {a.shape}")
    else:
        raise ValueError(f"{name} must be 3D or 4D. Got shape {a.shape}")
    return a.astype(np.float32, copy=False)


def _load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"SST npz not found: {npz_path}")
    z = np.load(npz_path)
    keys = set(z.files)
    if {"X", "Y"}.issubset(keys):
        x = z["X"]
        y = z["Y"]
    elif {"x", "y"}.issubset(keys):
        x = z["x"]
        y = z["y"]
    elif {"X_train", "Y_train"}.issubset(keys):
        x = z["X_train"]
        y = z["Y_train"]
    else:
        raise KeyError(
            f"Could not find X/Y arrays in {npz_path}. Available keys: {sorted(keys)}. "
            "Expected keys: X and Y."
        )
    x = _to_nchw(x, "X")
    y = _to_nchw(y, "Y")
    if x.shape != y.shape:
        raise ValueError(f"X and Y must have identical shape. Got X={x.shape}, Y={y.shape}")
    if x.shape[1] != 1:
        raise ValueError(f"SST loader expects one channel. Got shape {x.shape}")
    return x, y


def _compute_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    # Static ocean mask: valid in at least one sample. Shape (1,H,W).
    mask = valid.any(axis=0).astype(np.float32)
    if mask.sum() == 0:
        raise ValueError("No finite SST pixels found in X/Y.")
    return mask


def _normalization_stats(x: np.ndarray, y: np.ndarray, mode: str, eps: float = 1e-6) -> tuple[float, float]:
    vals = y[np.isfinite(y)] if mode == "y" else np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if vals.size == 0:
        raise ValueError("Cannot compute SST normalization because no finite values were found.")
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < eps:
        raise ValueError(f"Bad SST normalization range: min={vmin}, max={vmax}")
    return vmin, vmax


def _normalize_and_fill(a: np.ndarray, vmin: float, vmax: float, mask: np.ndarray) -> np.ndarray:
    out = (a - vmin) / (vmax - vmin)
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    return out * mask[None, :, :, :]


class SSTNPZDataset(Dataset):
    """Returns SST tensors from training_data.npz.

    For diffusion SR mode, each item is (X, Y), where X is the LR/interpolated/anomaly
    input and Y is the HR target. Both are normalized to [0, 1] over valid ocean pixels
    and NaN land pixels are filled with zero.

    For POD/non-SR training, use return_pair=False to return (Y, dummy_label),
    matching the ImageFolder calling convention without changing the training loops.
    """

    def __init__(
        self,
        npz_path: str,
        return_pair: bool = False,
        norm_mode: str = "joint",
        norm_min: Optional[float] = None,
        norm_max: Optional[float] = None,
        stats_out: str = "",
    ):
        x, y = _load_npz(npz_path)
        self.mask = _compute_mask(x, y)

        if norm_min is None or norm_max is None:
            vmin, vmax = _normalization_stats(x, y, mode=("y" if norm_mode == "y" else "joint"))
        else:
            vmin, vmax = float(norm_min), float(norm_max)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                raise ValueError(f"Invalid norm_min/norm_max: {vmin}, {vmax}")

        self.x = _normalize_and_fill(x, vmin, vmax, self.mask)
        self.y = _normalize_and_fill(y, vmin, vmax, self.mask)
        self.return_pair = bool(return_pair)
        self.norm_min = vmin
        self.norm_max = vmax
        self.height = int(self.y.shape[2])
        self.width = int(self.y.shape[3])

        if stats_out:
            os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
            with open(stats_out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "npz_path": npz_path,
                        "shape_nchw": list(self.y.shape),
                        "norm_min": self.norm_min,
                        "norm_max": self.norm_max,
                        "norm_mode": norm_mode,
                        "valid_ocean_pixels": int(self.mask.sum()),
                        "total_pixels": int(np.prod(self.mask.shape)),
                    },
                    f,
                    indent=2,
                )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.x[i])
        y = torch.from_numpy(self.y[i])
        if self.return_pair:
            return x, y
        return y, torch.tensor(0, dtype=torch.long)


def build_sst_npz_loaders(
    root: str = "",
    npz_path: str = "",
    batch_size: int = 64,
    num_workers: int = 4,
    fit_samples: Optional[int] = None,
    eval_samples: Optional[int] = None,
    seed: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    return_pair: bool = False,
    norm_mode: str = "joint",
    norm_min: Optional[float] = None,
    norm_max: Optional[float] = None,
    stats_out: str = "",
):
    path = npz_path or root
    if not path:
        raise ValueError("For dataset='sst_npz', set either --npz_path or --root to training_data.npz")

    ds = SSTNPZDataset(
        path,
        return_pair=return_pair,
        norm_mode=norm_mode,
        norm_min=norm_min,
        norm_max=norm_max,
        stats_out=stats_out,
    )
    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_fit = min(fit_samples, n) if fit_samples is not None else n
    remaining = max(0, n - n_fit)
    n_eval = min(eval_samples, remaining) if (eval_samples is not None and remaining > 0) else remaining
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:n_fit + n_eval] if n_eval > 0 else idx[:max(1, min(256, n_fit))]

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers

    train_dl = DataLoader(Subset(ds, fit_idx.tolist()), shuffle=True, drop_last=True, **dl_kwargs)
    test_dl = DataLoader(Subset(ds, eval_idx.tolist()), shuffle=False, drop_last=False, **dl_kwargs)
    return SSTLoaders(train=train_dl, test=test_dl, num_classes=1)


def build_sr_sst_npz_loaders(**kwargs):
    kwargs["return_pair"] = True
    loaders = build_sst_npz_loaders(**kwargs)
    return loaders.train, loaders.test
