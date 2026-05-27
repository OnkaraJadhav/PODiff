from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class GlobalPODArtifacts:
    mean: np.ndarray                 # (D,)
    components: np.ndarray           # (Kmax, D)
    explained_variance_ratio: np.ndarray
    cum_energy: np.ndarray
    k_energy: int
    image_size: int
    channels: int
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    explained_variance: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None # (C,H,W), 1 ocean, 0 land


def modes_for_energy(cum_energy: np.ndarray, target: float) -> int:
    if cum_energy[-1] < target:
        return int(len(cum_energy))
    return int(np.searchsorted(cum_energy, target) + 1)


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def fit_global_pod(dl_train, image_size: int, n_components_max: int, image_height: Optional[int] = None, image_width: Optional[int] = None) -> GlobalPODArtifacts:
    """Fit whole-field POD on normalized SST tensors.
    """
    from sklearn.decomposition import IncrementalPCA

    x0, _ = next(iter(dl_train))
    c = int(x0.shape[1])
    h = int(image_height if image_height is not None else x0.shape[2])
    w = int(image_width if image_width is not None else x0.shape[3])

    if dl_train.batch_size is None or dl_train.batch_size < n_components_max:
        raise ValueError(f"Need DataLoader batch_size >= n_components_max (got {dl_train.batch_size} < {n_components_max}).")

    ipca = IncrementalPCA(n_components=n_components_max, batch_size=dl_train.batch_size)
    mask_accum = None
    mask_count = 0
    for x, _ in dl_train:
        X = _flatten(x).numpy()
        ipca.partial_fit(X)
        # Valid SST ocean pixels are non-zero after loader masking for normalized data.
        # This is only used to re-apply land mask after decoding.
        m = (x.numpy() != 0.0).any(axis=0).astype(np.float32)
        mask_accum = m if mask_accum is None else np.maximum(mask_accum, m)
        mask_count += 1

    evr = ipca.explained_variance_ratio_
    cum = np.cumsum(evr)
    return GlobalPODArtifacts(
        mean=ipca.mean_.astype(np.float32),
        components=ipca.components_.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        cum_energy=cum.astype(np.float32),
        k_energy=n_components_max,
        image_size=image_size,
        channels=c,
        image_height=h,
        image_width=w,
        explained_variance=ipca.explained_variance_.astype(np.float32),
        mask=mask_accum.astype(np.float32) if mask_accum is not None else None,
    )


def _coeff_scale(art: GlobalPODArtifacts, k: int, device, dtype) -> torch.Tensor:
    if art.explained_variance is None:
        return torch.ones((k,), device=device, dtype=dtype)
    var = torch.from_numpy(np.asarray(art.explained_variance[:k], dtype=np.float32)).to(device=device, dtype=dtype)
    return torch.sqrt(var.clamp_min(1e-8))


def encode_global(x: torch.Tensor, art: GlobalPODArtifacts, k: Optional[int] = None, standardize: bool = True) -> torch.Tensor:
    k = art.k_energy if k is None else k
    X = _flatten(x)
    mu = torch.from_numpy(art.mean).to(X.device, X.dtype)
    U = torch.from_numpy(art.components[:k]).to(X.device, X.dtype)
    coeff = (X - mu) @ U.t()
    if standardize:
        coeff = coeff / _coeff_scale(art, k, X.device, X.dtype)
    return coeff


def decode_global(coeffs: torch.Tensor, art: GlobalPODArtifacts, standardize: bool = True) -> torch.Tensor:
    b, k = coeffs.shape
    coeffs_flat = coeffs
    if standardize:
        coeffs_flat = coeffs_flat * _coeff_scale(art, k, coeffs.device, coeffs.dtype)
    mu = torch.from_numpy(art.mean).to(coeffs.device, coeffs.dtype)
    U = torch.from_numpy(art.components[:k]).to(coeffs.device, coeffs.dtype)
    Xhat = mu + coeffs_flat @ U
    h = int(art.image_height if art.image_height is not None else art.image_size)
    w = int(art.image_width if art.image_width is not None else art.image_size)
    x = Xhat.view(b, art.channels, h, w)
    if art.mask is not None:
        mask = torch.from_numpy(art.mask).to(x.device, x.dtype)
        x = x * mask.unsqueeze(0)
    return x.clamp(0, 1)


def save_npz(art: GlobalPODArtifacts, out_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    kwargs = dict(
        mean=art.mean,
        components=art.components,
        explained_variance_ratio=art.explained_variance_ratio,
        cum_energy=art.cum_energy,
        k_energy=art.k_energy,
        image_size=art.image_size,
        channels=art.channels,
        image_height=art.image_height if art.image_height is not None else art.image_size,
        image_width=art.image_width if art.image_width is not None else art.image_size,
    )
    if art.explained_variance is not None:
        kwargs['explained_variance'] = art.explained_variance
    if art.mask is not None:
        kwargs['mask'] = art.mask
    np.savez_compressed(out_path, **kwargs)


def load_npz(path: str) -> GlobalPODArtifacts:
    z = np.load(path, allow_pickle=False)
    return GlobalPODArtifacts(
        mean=z['mean'],
        components=z['components'],
        explained_variance_ratio=z['explained_variance_ratio'],
        cum_energy=z['cum_energy'],
        k_energy=int(z['k_energy']),
        image_size=int(z['image_size']),
        channels=int(z['channels']),
        image_height=int(z['image_height']) if 'image_height' in z.files else int(z['image_size']),
        image_width=int(z['image_width']) if 'image_width' in z.files else int(z['image_size']),
        explained_variance=(z['explained_variance'] if 'explained_variance' in z.files else None),
        mask=(z['mask'] if 'mask' in z.files else None),
    )
