from __future__ import annotations
import argparse, copy, os, time
import torch
from torch.amp import autocast, GradScaler

from podiff.utils.seed import set_seed
from podiff.utils.device import get_device
from podiff.utils.io import ensure_dir, save_json, append_jsonl, load_json
from podiff.pod.global_pod import load_npz, encode_global
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


def q_sample_global(x0, t, const, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    return const.sqrt_abar[t].view(-1, 1) * x0 + const.sqrt_1mabar[t].view(-1, 1) * noise


def update_ema(ema_model, model, decay: float):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            v.copy_(v * decay + msd[k].detach() * (1.0 - decay))


def save_ckpt(path: str, model, ema_model, opt, step: int, cfg: dict, best_ema_loss: float):
    ensure_dir(os.path.dirname(path) or '.')
    torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict() if ema_model is not None else None, 'opt': opt.state_dict(), 'step': step, 'cfg': cfg, 'best_ema_loss': best_ema_loss}, path)


def train_one(args, dl_train, out_dir: str):
    device = get_device(args.device)
    ensure_dir(out_dir)
    art = load_npz(args.pod_npz)
    if art.explained_variance is None:
        raise ValueError('Global POD artifact is missing explained_variance. Rerun fit_global_pod.py.')
    K = art.k_energy

    model = GlobalMLPDiffusion(k_dim=K, model_dim=args.model_dim, n_layers=args.n_layers, time_dim=args.time_dim, cond_mode=args.cond_mode, dropout=args.dropout).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device=('cuda' if device.type == 'cuda' else 'cpu'), enabled=(device.type == 'cuda' and args.amp))
    const = make_ddpm_constants(args.timesteps, schedule=args.beta_schedule, device=device)

    cfg = vars(args).copy()
    cfg.update({'K': K, 'coeff_standardized': True, 'arch': 'global_mlp_diffusion'})
    save_json(os.path.join(out_dir, 'run_config.json'), cfg)
    save_json(os.path.join(out_dir, 'env.json'), {'torch_version': torch.__version__, 'device': str(device), 'cuda_available': bool(torch.cuda.is_available()), 'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0})
    log_path = os.path.join(out_dir, 'train_log.jsonl')

    step = 0
    model.train()
    t0 = time.time()
    ema_loss = None
    best_ema_loss = float('inf')

    while step < args.n_steps:
        for batch in dl_train:
            if step >= args.n_steps:
                break
            if args.cond_mode == 'sr':
                lr, x = batch
                lr = lr.to(device, non_blocking=(device.type == 'cuda'))
                x = x.to(device, non_blocking=(device.type == 'cuda'))
                with torch.no_grad():
                    cond = encode_global(lr, art, k=K, standardize=True)
            else:
                x, _ = batch
                x = x.to(device, non_blocking=(device.type == 'cuda'))
                cond = None

            with torch.no_grad():
                a0 = encode_global(x, art, k=K, standardize=True)

            t = torch.randint(0, args.timesteps, (a0.shape[0],), device=device).long()
            noise = torch.randn_like(a0)
            at = q_sample_global(a0, t, const, noise=noise)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == 'cuda' and args.amp)):
                pred = model(at, t, cond)
                loss = torch.mean((pred - noise) ** 2)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            update_ema(ema_model, model, args.ema_decay)

            l = float(loss.item())
            ema_loss = l if ema_loss is None else args.loss_ema * ema_loss + (1.0 - args.loss_ema) * l
            elapsed = time.time() - t0
            if step % args.log_every == 0:
                print(f'[step {step:07d}] loss={l:.6f} ema={ema_loss:.6f} mins={elapsed/60:.1f}')
                append_jsonl(log_path, {'step': step, 'loss': l, 'ema_loss': ema_loss, 'mins': elapsed / 60.0})
            if step % args.save_every == 0 and step > 0:
                save_ckpt(os.path.join(out_dir, f'step_{step:07d}.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
                save_ckpt(os.path.join(out_dir, 'latest.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
            if ema_loss < best_ema_loss:
                best_ema_loss = ema_loss
                save_ckpt(os.path.join(out_dir, 'best.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
            step += 1

    save_ckpt(os.path.join(out_dir, 'latest.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
    print(f'[done] saved latest.pt and best.pt to {out_dir}')


def train_one_cached_coeffs(args, dl_train, out_dir: str):
    """Train MLP diffusion directly on cached global POD coefficients.

    This avoids loading full 640x480 SST fields inside the GPU job.
    """
    device = get_device(args.device)
    ensure_dir(out_dir)
    art = load_npz(args.pod_npz)
    if art.explained_variance is None:
        raise ValueError('Global POD artifact is missing explained_variance. Rerun fit_global_pod.py.')
    K = int(art.k_energy)

    model = GlobalMLPDiffusion(k_dim=K, model_dim=args.model_dim, n_layers=args.n_layers, time_dim=args.time_dim, cond_mode=args.cond_mode, dropout=args.dropout).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device=('cuda' if device.type == 'cuda' else 'cpu'), enabled=(device.type == 'cuda' and args.amp))
    const = make_ddpm_constants(args.timesteps, schedule=args.beta_schedule, device=device)

    cfg = vars(args).copy()
    cfg.update({'K': K, 'coeff_standardized': True, 'arch': 'global_mlp_diffusion_cached_coeffs'})
    save_json(os.path.join(out_dir, 'run_config.json'), cfg)
    save_json(os.path.join(out_dir, 'env.json'), {'torch_version': torch.__version__, 'device': str(device), 'cuda_available': bool(torch.cuda.is_available()), 'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0})
    log_path = os.path.join(out_dir, 'train_log.jsonl')

    step = 0
    model.train()
    t0 = time.time()
    ema_loss = None
    best_ema_loss = float('inf')

    while step < args.n_steps:
        for batch in dl_train:
            if step >= args.n_steps:
                break
            if args.cond_mode == 'sr':
                cond, a0 = batch
                cond = cond.to(device, non_blocking=False)
                a0 = a0.to(device, non_blocking=False)
            else:
                a0, _ = batch
                a0 = a0.to(device, non_blocking=False)
                cond = None

            t = torch.randint(0, args.timesteps, (a0.shape[0],), device=device).long()
            noise = torch.randn_like(a0)
            at = q_sample_global(a0, t, const, noise=noise)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == 'cuda' and args.amp)):
                pred = model(at, t, cond)
                loss = torch.mean((pred - noise) ** 2)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            update_ema(ema_model, model, args.ema_decay)

            l = float(loss.item())
            ema_loss = l if ema_loss is None else args.loss_ema * ema_loss + (1.0 - args.loss_ema) * l
            elapsed = time.time() - t0
            if step % args.log_every == 0:
                print(f'[step {step:07d}] loss={l:.6f} ema={ema_loss:.6f} mins={elapsed/60:.1f}')
                append_jsonl(log_path, {'step': step, 'loss': l, 'ema_loss': ema_loss, 'mins': elapsed / 60.0})
            if step % args.save_every == 0 and step > 0:
                save_ckpt(os.path.join(out_dir, f'step_{step:07d}.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
                save_ckpt(os.path.join(out_dir, 'latest.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
            if ema_loss < best_ema_loss:
                best_ema_loss = ema_loss
                save_ckpt(os.path.join(out_dir, 'best.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
            step += 1

    save_ckpt(os.path.join(out_dir, 'latest.pt'), model, ema_model, opt, step, cfg, best_ema_loss)
    print(f'[done] saved latest.pt and best.pt to {out_dir}')

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
    ap.add_argument('--fit_samples', type=int, default=None)
    ap.add_argument('--eval_samples', type=int, default=None)
    ap.add_argument('--cond_mode', choices=['none','sr'], default='sr')
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--beta_schedule', choices=['cosine','linear'], default='cosine')
    ap.add_argument('--n_steps', type=int, default=200000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--loss_ema', type=float, default=0.98)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model_dim', type=int, default=512)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--time_dim', type=int, default=256)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--out_dir', default='outputs/global_train')
    args = ap.parse_args()
    args = apply_json_config(args, ap)
    set_seed(args.seed)
    device = get_device(args.device)
    print(f'[device] {device}')
    pin_memory = False  # SST fields are large; avoid extra host-memory copies
    persistent_workers = False

    if args.cache_dir:
        from podiff.data.global_cache import build_global_coeff_loaders
        print(f'[global-cache] using coefficient cache: {args.cache_dir}')
        loaders = build_global_coeff_loaders(
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            fit_samples=args.fit_samples,
            eval_samples=args.eval_samples,
            cond_mode=args.cond_mode,
        )
        train_one_cached_coeffs(args, loaders.train, args.out_dir)
    else:
        from podiff.data.sst_npz import build_sst_npz_loaders
        loaders = build_sst_npz_loaders(root=args.root, npz_path=args.npz_path, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed, pin_memory=pin_memory, persistent_workers=persistent_workers, norm_mode=args.norm_mode, norm_min=args.norm_min, norm_max=args.norm_max, stats_out=args.stats_out, return_pair=(args.cond_mode == 'sr'))
        train_one(args, loaders.train, args.out_dir)


if __name__ == '__main__':
    main()
