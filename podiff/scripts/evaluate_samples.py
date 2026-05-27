from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from podiff.utils.io import load_json, save_json
from podiff.utils.metrics import denormalize_minmax, masked_mae, masked_r2, masked_rmse


def apply_json_config(args, parser):
    if not args.config:
        return args
    cfg = load_json(args.config)
    valid = {a.dest for a in parser._actions}
    unknown = sorted(set(cfg) - valid)
    if unknown:
        raise ValueError(f"Unknown keys in config {args.config}: {unknown}")
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


def _load_npy(path: Path) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim == 3:
        arr = arr[None, :, :, :]  # C,H,W -> 1,C,H,W
    if arr.ndim != 4:
        raise ValueError(f"Expected npy array with shape HxW, CxHxW, or NxCxHxW. Got {arr.shape} from {path}")
    return torch.from_numpy(arr.astype(np.float32, copy=False))


def _image_ids(npy_dir: Path) -> list[int]:
    ids = []
    for p in npy_dir.glob("gt_*.npy"):
        m = re.match(r"gt_(\d+)\.npy", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def _sample_paths(npy_dir: Path, image_id: int) -> list[Path]:
    return sorted(npy_dir.glob(f"image_{image_id:06d}_sample_*.npy"))


def _load_ensemble(npy_dir: Path, image_id: int) -> torch.Tensor:
    paths = _sample_paths(npy_dir, image_id)
    if not paths:
        raise FileNotFoundError(f"No ensemble samples found for image {image_id:06d} in {npy_dir}")
    xs = [_load_npy(p).squeeze(0) for p in paths]  # C,H,W
    return torch.stack(xs, dim=0)  # M,C,H,W


def _stats_from_json(stats_json: str) -> tuple[float | None, float | None]:
    if not stats_json:
        return None, None
    p = Path(stats_json)
    if not p.exists():
        return None, None
    with open(p, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return float(stats["norm_min"]), float(stats["norm_max"])


def _mask_from_pod(pod_npz: str) -> torch.Tensor | None:
    if not pod_npz:
        return None
    p = Path(pod_npz)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if "mask" not in z.files:
        return None
    mask = torch.from_numpy(z["mask"].astype(np.float32, copy=False))  # C,H,W
    return mask


def _mask_from_gt(gt: torch.Tensor) -> torch.Tensor:
    # Normalized SST land is usually zero-filled. This fallback treats nonzero finite pixels as valid.
    m = torch.isfinite(gt) & (gt != 0.0)
    return m.squeeze(0).float()


def _denorm_if_needed(x: torch.Tensor, args, norm_min, norm_max) -> torch.Tensor:
    if args.denormalize and norm_min is not None and norm_max is not None:
        return denormalize_minmax(x, norm_min, norm_max)
    return x


def _empirical_coverage(ens: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, levels: Iterable[float]) -> dict[str, float]:
    # ens: M,C,H,W; gt: 1,C,H,W; mask: C,H,W
    out = {}
    m = mask.bool().unsqueeze(0)  # 1,C,H,W
    for level in levels:
        alpha = (1.0 - float(level)) / 2.0
        lo = torch.quantile(ens, alpha, dim=0, keepdim=True)
        hi = torch.quantile(ens, 1.0 - alpha, dim=0, keepdim=True)
        inside = (gt >= lo) & (gt <= hi)
        out[f"coverage_{level:.2f}"] = float(inside[m].float().mean().item())
    return out


def _crps_ensemble(ens: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, max_pixels: int, rng: np.random.Generator) -> float:
    # Flatten only valid pixels. ens_valid: M,P; gt_valid: P
    valid = mask.bool().flatten()
    ens_flat = ens.reshape(ens.shape[0], -1)[:, valid]
    gt_flat = gt.reshape(-1)[valid]
    p = int(gt_flat.numel())
    if p == 0:
        return float("nan")
    if max_pixels and p > max_pixels:
        idx = torch.from_numpy(rng.choice(p, size=max_pixels, replace=False)).long()
        ens_flat = ens_flat[:, idx]
        gt_flat = gt_flat[idx]
    m = ens_flat.shape[0]
    term1 = torch.mean(torch.abs(ens_flat - gt_flat.unsqueeze(0)), dim=0)
    xs, _ = torch.sort(ens_flat, dim=0)
    # 1/M^2 * sum_{i,j}|x_i-x_j| = 2/M^2 * sum_i (2i-M-1)x_(i), one-indexed i.
    i = torch.arange(1, m + 1, dtype=xs.dtype, device=xs.device).view(m, 1)
    mean_pair_abs = (2.0 / (m * m)) * torch.sum((2.0 * i - m - 1.0) * xs, dim=0)
    crps = term1 - 0.5 * mean_pair_abs
    return float(crps.mean().item())


def _downsample_like(x: torch.Tensor, target: torch.Tensor, scale: int) -> torch.Tensor:
    # x and target are N,C,H,W. If scale=1, compare on their native shapes.
    if x.shape[-2:] == target.shape[-2:]:
        return x
    if scale and scale > 1:
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser(description="Evaluate PODiff ensemble samples saved by multisample_global_sr.py")
    ap.add_argument("--config", default="")
    ap.add_argument("--samples_dir", default="outputs/samples")
    ap.add_argument("--stats_json", default="outputs/sst_stats.json")
    ap.add_argument("--pod_npz", default="artifacts/pod_sst.npz")
    ap.add_argument("--out_json", default="outputs/evaluation_metrics.json")
    ap.add_argument("--denormalize", action="store_true")
    ap.add_argument("--mask_mode", choices=["pod", "gt", "none"], default="pod")
    ap.add_argument("--coverage_levels", type=float, nargs="+", default=[0.5, 0.7, 0.9, 0.95])
    ap.add_argument("--crps_max_pixels", type=int, default=50000)
    ap.add_argument("--lr_consistency", action="store_true")
    ap.add_argument("--lr_scale", type=int, default=1, help="Optional average-pooling factor before LR consistency comparison.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args = apply_json_config(args, ap)

    samples_dir = Path(args.samples_dir)
    npy_dir = samples_dir / "npy"
    if not npy_dir.exists():
        raise FileNotFoundError(f"Expected npy directory at {npy_dir}")

    image_ids = _image_ids(npy_dir)
    if not image_ids:
        raise FileNotFoundError(f"No gt_*.npy files found in {npy_dir}")

    norm_min, norm_max = _stats_from_json(args.stats_json)
    pod_mask = _mask_from_pod(args.pod_npz) if args.mask_mode == "pod" else None
    rng = np.random.default_rng(args.seed)

    per_image = []
    for image_id in image_ids:
        gt = _load_npy(npy_dir / f"gt_{image_id:06d}.npy")  # 1,C,H,W
        ens = _load_ensemble(npy_dir, image_id)              # M,C,H,W
        mean = ens.mean(dim=0, keepdim=True)
        std = ens.std(dim=0, keepdim=True, unbiased=False)

        if args.mask_mode == "pod" and pod_mask is not None:
            mask = pod_mask
        elif args.mask_mode == "gt":
            mask = _mask_from_gt(gt)
        else:
            mask = torch.ones_like(gt.squeeze(0))

        gt_eval = _denorm_if_needed(gt, args, norm_min, norm_max)
        mean_eval = _denorm_if_needed(mean, args, norm_min, norm_max)
        ens_eval = _denorm_if_needed(ens, args, norm_min, norm_max)

        row = {
            "image_id": image_id,
            "n_samples": int(ens.shape[0]),
            "rmse": float(masked_rmse(mean_eval, gt_eval, mask).item()),
            "mae": float(masked_mae(mean_eval, gt_eval, mask).item()),
            "r2": float(masked_r2(mean_eval, gt_eval, mask).item()),
            "mean_predictive_std": float(masked_mae(std, torch.zeros_like(std), mask).item()),
            "crps": _crps_ensemble(ens_eval, gt_eval, mask, args.crps_max_pixels, rng),
        }

        row.update(_empirical_coverage(ens_eval, gt_eval, mask, args.coverage_levels))

        if args.lr_consistency:
            input_path = npy_dir / f"input_{image_id:06d}.npy"
            # Backward compatibility with old sample folders.
            if not input_path.exists():
                input_path = npy_dir / f"lr_{image_id:06d}.npy"
            if input_path.exists():
                inp = _load_npy(input_path)
                inp_eval = _denorm_if_needed(inp, args, norm_min, norm_max)
                pred_lr = _downsample_like(mean_eval, inp_eval, args.lr_scale)
                inp_mask = mask if pred_lr.shape[-2:] == mask.shape[-2:] else torch.ones_like(inp_eval.squeeze(0))
                row["lr_consistency_rmse"] = float(masked_rmse(pred_lr, inp_eval, inp_mask).item())
                row["lr_consistency_mae"] = float(masked_mae(pred_lr, inp_eval, inp_mask).item())
                row["lr_consistency_r2"] = float(masked_r2(pred_lr, inp_eval, inp_mask).item())
        per_image.append(row)

    keys = [k for k in per_image[0].keys() if k != "image_id"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in per_image if k in r and np.isfinite(r[k])], dtype=np.float64)
        if vals.size:
            summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}

    # Mean absolute calibration error over requested levels.
    mace_terms = []
    for level in args.coverage_levels:
        key = f"coverage_{level:.2f}"
        vals = np.array([r[key] for r in per_image if key in r], dtype=np.float64)
        if vals.size:
            mace_terms.append(abs(float(vals.mean()) - float(level)))
    summary["mace"] = float(np.mean(mace_terms)) if mace_terms else float("nan")

    result = {
        "samples_dir": str(samples_dir),
        "n_images": len(per_image),
        "denormalized_to_physical_units": bool(args.denormalize and norm_min is not None and norm_max is not None),
        "norm_min": norm_min,
        "norm_max": norm_max,
        "mask_mode": args.mask_mode,
        "coverage_levels": list(map(float, args.coverage_levels)),
        "summary": summary,
        "per_image": per_image,
    }
    save_json(args.out_json, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
