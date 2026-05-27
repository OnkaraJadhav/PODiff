from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import torch

try:
    from torchvision.utils import save_image
except Exception:
    def save_image(tensor, fp):
        from PIL import Image
        x = tensor.detach().cpu().clamp(0.0, 1.0)
        if x.ndim == 3:
            x = x[0] if x.shape[0] == 1 else x.permute(1, 2, 0)
        arr = (x.numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(fp)

from podiff.utils.seed import set_seed
from podiff.utils.device import get_device
from podiff.utils.io import load_json
from podiff.pod.global_pod import load_npz, encode_global, decode_global
from podiff.diffusion.ddpm import make_ddpm_constants
from podiff.models.mlp_latent_diffusion import GlobalMLPDiffusion


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


@torch.no_grad()
def p_sample_global(model, xt, t, const, cond):
    betas_t = const.betas[t].view(-1, 1)
    sqrt_1m = const.sqrt_1mabar[t].view(-1, 1)
    sqrt_recip_alpha = torch.sqrt(1.0 / const.alphas[t]).view(-1, 1)
    eps = model(xt, t, cond)
    mean = sqrt_recip_alpha * (xt - betas_t * eps / sqrt_1m)
    noise = torch.randn_like(xt)
    var = const.post_var[t].view(-1, 1)
    mask = (t != 0).float().view(-1, 1)
    return mean + mask * torch.sqrt(var) * noise


@torch.no_grad()
def p_sample_loop_global(model, shape, const, cond, device):
    xt = torch.randn(shape, device=device)
    for i in reversed(range(const.betas.shape[0])):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        xt = p_sample_global(model, xt, t, const, cond)
    return xt


@torch.no_grad()
def ddim_sample_loop_global(model, shape, const, cond, device, sample_steps: int = 100, eta: float = 0.0):
    """Reduced-step DDIM sampler for PODiff.

    The diffusion model is trained with T timesteps, but inference can use S<T
    denoising steps by traversing a uniformly spaced subset of timesteps.
    eta=0 gives deterministic DDIM updates; eta>0 adds stochasticity.
    """
    total_steps = int(const.betas.shape[0])
    sample_steps = int(sample_steps)
    if sample_steps <= 0 or sample_steps >= total_steps:
        return p_sample_loop_global(model, shape, const, cond, device)

    step_ids = torch.linspace(0, total_steps - 1, sample_steps, device=device).long().unique()
    step_ids = torch.flip(step_ids, dims=[0])
    xt = torch.randn(shape, device=device)

    for j, i in enumerate(step_ids):
        t = torch.full((shape[0],), int(i.item()), device=device, dtype=torch.long)
        eps = model(xt, t, cond)

        alpha_bar_t = const.abar[t].view(-1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0 = (xt - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

        if j == len(step_ids) - 1:
            xt = x0
            continue

        prev_i = int(step_ids[j + 1].item())
        prev_t = torch.full((shape[0],), prev_i, device=device, dtype=torch.long)
        alpha_bar_prev = const.abar[prev_t].view(-1, 1)

        if eta > 0.0:
            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(xt)
            direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)) * eps
            xt = torch.sqrt(alpha_bar_prev) * x0 + direction + sigma * noise
        else:
            xt = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps

    return xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='')
    ap.add_argument('--dataset', choices=['sst_npz'], default='sst_npz')
    ap.add_argument('--root', default='')
    ap.add_argument('--npz_path', default='')
    ap.add_argument('--norm_mode', choices=['joint','y'], default='joint')
    ap.add_argument('--norm_min', type=float, default=None)
    ap.add_argument('--norm_max', type=float, default=None)
    ap.add_argument('--stats_out', default='')
    ap.add_argument('--pod_npz', default='artifacts/globalpod_sst.npz')
    ap.add_argument('--cache_dir', default='')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--cond_mode', choices=['none','sr'], default='sr')
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--sample_steps', type=int, default=100)
    ap.add_argument('--ddim_eta', type=float, default=0.0)
    ap.add_argument('--beta_schedule', choices=['cosine','linear'], default='cosine')
    ap.add_argument('--n_test_images', type=int, default=100)
    ap.add_argument('--n_ensemble', type=int, default=100)
    ap.add_argument('--sample_batch_size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--samples_out', default='outputs/global_ensemble_samples')
    ap.add_argument('--model_dim', type=int, default=512)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--time_dim', type=int, default=256)
    args = ap.parse_args()
    args = apply_json_config(args, ap)

    set_seed(args.seed)
    device = get_device(args.device)
    art = load_npz(args.pod_npz)
    if art.explained_variance is None:
        raise ValueError('Global POD artifact missing explained_variance.')
    K = art.k_energy

    ck = torch.load(args.ckpt, map_location=device)
    ck_cfg = ck.get('cfg', {}) or {}
    model_dim = int(ck_cfg.get('model_dim', args.model_dim))
    n_layers = int(ck_cfg.get('n_layers', args.n_layers))
    time_dim = int(ck_cfg.get('time_dim', args.time_dim))
    dropout = float(ck_cfg.get('dropout', args.dropout))
    cond_mode = ck_cfg.get('cond_mode', args.cond_mode)
    if cond_mode != args.cond_mode:
        print(f'[warn] Sampling cond_mode={args.cond_mode} but checkpoint was trained with cond_mode={cond_mode}. Using checkpoint cond_mode.')
        args.cond_mode = cond_mode
    print(f'[load] diffusion architecture from checkpoint/config: model_dim={model_dim}, n_layers={n_layers}, time_dim={time_dim}, dropout={dropout}, cond_mode={args.cond_mode}')
    model = GlobalMLPDiffusion(k_dim=K, model_dim=model_dim, n_layers=n_layers, time_dim=time_dim, cond_mode=args.cond_mode, dropout=dropout).to(device)
    state = ck['ema_model'] if ck.get('ema_model') is not None else ck['model']
    model.load_state_dict(state, strict=True)
    model.eval()

    const = make_ddpm_constants(args.timesteps, schedule=args.beta_schedule, device=device)

    out_dir = Path(args.samples_out)
    gt_dir = out_dir / 'gt'
    input_dir = out_dir / 'input'
    pod_proj_dir = out_dir / 'pod_projection'
    ens_dir = out_dir / 'ensemble'
    npy_dir = out_dir / 'npy'
    for d in [gt_dir, input_dir, pod_proj_dir, ens_dir, npy_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Memory-safe path: use the cache written by fit_global_pod.py.
    # This avoids loading training_data.npz again during sampling.
    if args.cache_dir:
        import os, json
        meta_path = os.path.join(args.cache_dir, 'meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'cache_dir was set but meta.json was not found: {meta_path}')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        x_coeff = np.load(os.path.join(args.cache_dir, 'x_coeff.npy'), mmap_mode='r')
        y_norm = np.load(os.path.join(args.cache_dir, 'y_norm.npy'), mmap_mode='r') if os.path.exists(os.path.join(args.cache_dir, 'y_norm.npy')) else None
        x_norm = np.load(os.path.join(args.cache_dir, 'x_norm.npy'), mmap_mode='r') if os.path.exists(os.path.join(args.cache_dir, 'x_norm.npy')) else None
        # Backward compatibility for older caches that used coarse_y.npy.
        pod_projection_path = os.path.join(args.cache_dir, 'pod_projection.npy')
        if not os.path.exists(pod_projection_path):
            pod_projection_path = os.path.join(args.cache_dir, 'coarse_y.npy')
        pod_projection = np.load(pod_projection_path, mmap_mode='r') if os.path.exists(pod_projection_path) else None
        N = min(args.n_test_images, int(x_coeff.shape[0]))
        all_cond = torch.from_numpy(np.array(x_coeff[:N], dtype=np.float32, copy=True)) if args.cond_mode == 'sr' else None
        # Save diagnostics without holding the full dataset in RAM.
        for i in range(N):
            if y_norm is not None:
                gt_i = torch.from_numpy(np.array(y_norm[i], dtype=np.float32, copy=True))
                save_image(gt_i.clamp(0, 1), gt_dir / f'image_{i:06d}.png')
                np.save(npy_dir / f'gt_{i:06d}.npy', gt_i.numpy())
            if x_norm is not None:
                inp_i = torch.from_numpy(np.array(x_norm[i], dtype=np.float32, copy=True))
                save_image(inp_i.clamp(0, 1), input_dir / f'image_{i:06d}.png')
                np.save(npy_dir / f'input_{i:06d}.npy', inp_i.numpy())
            if pod_projection is not None:
                pp_i = torch.from_numpy(np.array(pod_projection[i], dtype=np.float32, copy=True))
                save_image(pp_i.clamp(0, 1), pod_proj_dir / f'image_{i:06d}.png')
                np.save(npy_dir / f'pod_projection_{i:06d}.npy', pp_i.numpy())
    else:
        # Fallback path for older configs. Use num_workers=0 to avoid duplicating full SST arrays.
        from podiff.data.sst_npz import build_sr_sst_npz_loaders
        _, test_dl = build_sr_sst_npz_loaders(root=args.root, npz_path=args.npz_path, batch_size=args.sample_batch_size, num_workers=0, seed=args.seed, eval_samples=args.n_test_images, norm_mode=args.norm_mode, norm_min=args.norm_min, norm_max=args.norm_max, stats_out=args.stats_out)

        all_lr, all_gt = [], []
        for lr_batch, gt_batch in test_dl:
            all_lr.append(lr_batch)
            all_gt.append(gt_batch)
            if sum(x.shape[0] for x in all_lr) >= args.n_test_images:
                break
        all_lr = torch.cat(all_lr, dim=0)[:args.n_test_images]
        all_gt = torch.cat(all_gt, dim=0)[:args.n_test_images]
        N = all_lr.shape[0]

        for i in range(N):
            save_image(all_gt[i].clamp(0, 1), gt_dir / f'image_{i:06d}.png')
            save_image(all_lr[i].clamp(0, 1), input_dir / f'image_{i:06d}.png')
            np.save(npy_dir / f'gt_{i:06d}.npy', all_gt[i].numpy())
            np.save(npy_dir / f'input_{i:06d}.npy', all_lr[i].numpy())

        with torch.no_grad():
            all_cond = encode_global(all_lr.to(device), art, k=K, standardize=True).cpu() if args.cond_mode == 'sr' else None

    img_batches = math.ceil(N / args.sample_batch_size)
    print(f'[ensemble] Generating {args.n_ensemble} samples for each of {N} SST fields')
    for m in range(args.n_ensemble):
        set_seed(args.seed + m)
        for b in range(img_batches):
            start = b * args.sample_batch_size
            end = min(start + args.sample_batch_size, N)
            cur_bs = end - start
            cond = all_cond[start:end].to(device) if all_cond is not None else None
            with torch.no_grad():
                coeff = ddim_sample_loop_global(model, shape=(cur_bs, K), const=const, cond=cond, device=device, sample_steps=args.sample_steps, eta=args.ddim_eta)
                x = decode_global(coeff, art, standardize=True)
            x = x.detach().cpu().clamp(0.0, 1.0)
            for i in range(cur_bs):
                img_idx = start + i
                img_ens_dir = ens_dir / f'image_{img_idx:06d}'
                img_ens_dir.mkdir(exist_ok=True)
                save_image(x[i], img_ens_dir / f'sample_{m:03d}.png')
                np.save(npy_dir / f'image_{img_idx:06d}_sample_{m:03d}.npy', x[i].numpy())
        print(f'[ensemble] {m+1:3d}/{args.n_ensemble} done')

    print(f'[done] Samples saved to {out_dir}')


if __name__ == '__main__':
    main()
