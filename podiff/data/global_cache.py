from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from podiff.pod.global_pod import encode_global, decode_global


@dataclass
class GlobalCacheLoaders:
    train: DataLoader
    test: DataLoader


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


def write_global_training_cache(
    dataset,
    art,
    cache_dir: str,
    batch_size: int = 16,
    seed: int = 0,
    save_field_cache: bool = True,
) -> str:
    """Write memory-mapped cache for global PODiff training.

    The original SST npz is large. Loading it inside every GPU job can exceed host RAM.
    This function is intended to run once, at the end of the CPU POD job. It writes:

      x_coeff.npy      (N,K)  conditioning coefficients from X
      y_coeff.npy      (N,K)  target coefficients from Y
      x_norm.npy       (N,C,H,W) normalized conditioning/input fields for diagnostics
      y_norm.npy       (N,C,H,W) normalized target fields for diagnostics/sampling outputs
      pod_projection.npy (N,C,H,W) POD reconstruction of Y for diagnostics
      meta.json        shape/statistics

    The GPU diffusion job trains only on coefficient arrays. The optional field cache is
    used only for saving ground-truth and POD-reconstruction diagnostics during sampling.
    """
    _ensure_dir(cache_dir)
    n = len(dataset)
    K = int(art.k_energy)
    c = int(getattr(art, "channels", 1))
    h = int(getattr(art, "image_height", getattr(dataset, "height", 0)))
    w = int(getattr(art, "image_width", getattr(dataset, "width", 0)))

    x_coeff_path = os.path.join(cache_dir, "x_coeff.npy")
    y_coeff_path = os.path.join(cache_dir, "y_coeff.npy")
    y_norm_path = os.path.join(cache_dir, "y_norm.npy")
    pod_projection_path = os.path.join(cache_dir, "pod_projection.npy")
    meta_path = os.path.join(cache_dir, "meta.json")

    x_coeff = np.lib.format.open_memmap(x_coeff_path, mode="w+", dtype=np.float32, shape=(n, K))
    y_coeff = np.lib.format.open_memmap(y_coeff_path, mode="w+", dtype=np.float32, shape=(n, K))

    x_norm = None
    y_norm = None
    pod_projection = None
    if save_field_cache:
        x_norm_path = os.path.join(cache_dir, "x_norm.npy")
        x_norm = np.lib.format.open_memmap(x_norm_path, mode="w+", dtype=np.float32, shape=(n, c, h, w))
        y_norm = np.lib.format.open_memmap(y_norm_path, mode="w+", dtype=np.float32, shape=(n, c, h, w))
        pod_projection = np.lib.format.open_memmap(pod_projection_path, mode="w+", dtype=np.float32, shape=(n, c, h, w))
    else:
        x_norm_path = ""

    # IMPORTANT: use num_workers=0 here to avoid dataset copies during cache writing.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    start = 0
    for batch in loader:
        # dataset may return (x,y) or (y,label). For global cache we need X and Y.
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]) and batch[0].ndim == 4 and batch[1].ndim == 4:
            xb, yb = batch
        else:
            raise RuntimeError(
                "Global cache requires an SST dataset returning (X,Y). "
                "Build the SST dataset with return_pair=True."
            )
        b = int(yb.shape[0])
        with torch.no_grad():
            ax = encode_global(xb, art, k=K, standardize=True)
            ay = encode_global(yb, art, k=K, standardize=True)
            x_coeff[start:start+b] = _as_numpy(ax)
            y_coeff[start:start+b] = _as_numpy(ay)
            if save_field_cache:
                pod_y = decode_global(ay, art, standardize=True)
                x_norm[start:start+b] = _as_numpy(xb)
                y_norm[start:start+b] = _as_numpy(yb)
                pod_projection[start:start+b] = _as_numpy(pod_y)
        start += b
        if start % max(batch_size * 20, 1) == 0 or start == n:
            print(f"[global-cache] wrote {start}/{n} samples")

    x_coeff.flush(); y_coeff.flush()
    if y_norm is not None:
        x_norm.flush(); y_norm.flush(); pod_projection.flush()

    meta = {
        "n": n,
        "K": K,
        "channels": c,
        "height": h,
        "width": w,
        "has_field_cache": bool(save_field_cache),
        "x_coeff": x_coeff_path,
        "y_coeff": y_coeff_path,
        "x_norm": x_norm_path if save_field_cache else "",
        "y_norm": y_norm_path if save_field_cache else "",
        "pod_projection": pod_projection_path if save_field_cache else "",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[global-cache] saved cache to {cache_dir}")
    return meta_path


class GlobalCoeffDataset(Dataset):
    """Tiny coefficient dataset for global MLP diffusion."""
    def __init__(self, cache_dir: str, cond_mode: str = "sr"):
        self.cache_dir = cache_dir
        self.cond_mode = cond_mode
        self.y = np.load(os.path.join(cache_dir, "y_coeff.npy"), mmap_mode="r")
        self.x = None
        if cond_mode == "sr":
            self.x = np.load(os.path.join(cache_dir, "x_coeff.npy"), mmap_mode="r")
            if self.x.shape != self.y.shape:
                raise ValueError(f"x_coeff and y_coeff shape mismatch: {self.x.shape} vs {self.y.shape}")

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        y = torch.from_numpy(np.array(self.y[i], dtype=np.float32, copy=True))
        if self.cond_mode == "sr":
            x = torch.from_numpy(np.array(self.x[i], dtype=np.float32, copy=True))
            return x, y
        return y, torch.tensor(0, dtype=torch.long)


def _split_dataset(ds: Dataset, fit_samples: Optional[int], eval_samples: Optional[int], seed: int):
    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_fit = min(fit_samples, n) if fit_samples is not None else n
    remaining = max(0, n - n_fit)
    n_eval = min(eval_samples, remaining) if (eval_samples is not None and remaining > 0) else remaining
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:n_fit+n_eval] if n_eval > 0 else idx[:max(1, min(256, n_fit))]
    return Subset(ds, fit_idx.tolist()), Subset(ds, eval_idx.tolist())


def build_global_coeff_loaders(cache_dir: str, batch_size: int, num_workers: int = 0, seed: int = 0, fit_samples: Optional[int] = None, eval_samples: Optional[int] = None, cond_mode: str = "sr"):
    ds = GlobalCoeffDataset(cache_dir, cond_mode=cond_mode)
    train_ds, test_ds = _split_dataset(ds, fit_samples, eval_samples, seed)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False, drop_last=True)
    return GlobalCacheLoaders(
        train=DataLoader(train_ds, shuffle=True, **kwargs),
        test=DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=False, drop_last=False),
    )
