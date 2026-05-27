from __future__ import annotations
import argparse
from podiff.utils.seed import set_seed
from podiff.utils.io import load_json
from podiff.pod.global_pod import fit_global_pod, modes_for_energy, save_npz


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='')
    ap.add_argument('--dataset', choices=['sst_npz'], default='sst_npz')
    ap.add_argument('--root', default='')
    ap.add_argument('--npz_path', default='')
    ap.add_argument('--image_size', type=int, default=640)
    ap.add_argument('--image_height', type=int, default=0)
    ap.add_argument('--image_width', type=int, default=0)
    ap.add_argument('--norm_mode', choices=['joint','y'], default='joint')
    ap.add_argument('--norm_min', type=float, default=None)
    ap.add_argument('--norm_max', type=float, default=None)
    ap.add_argument('--stats_out', default='')
    ap.add_argument('--n_components_max', type=int, default=128)
    ap.add_argument('--fit_samples', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--energy', type=float, default=0.99)
    ap.add_argument('--pod_out', default='artifacts/globalpod_sst.npz')
    ap.add_argument('--cache_dir', default='')
    ap.add_argument('--write_cache', action='store_true')
    ap.add_argument('--cache_batch_size', type=int, default=8)
    args = ap.parse_args()
    args = apply_json_config(args, ap)
    set_seed(args.seed)

    if args.batch_size < args.n_components_max:
        raise ValueError('IncrementalPCA requires batch_size >= n_components_max.')

    from podiff.data.sst_npz import build_sst_npz_loaders
    # For fitting POD, return_pair=False is enough.
    loaders = build_sst_npz_loaders(root=args.root, npz_path=args.npz_path, batch_size=args.batch_size, num_workers=args.num_workers, fit_samples=args.fit_samples, seed=args.seed, norm_mode=args.norm_mode, norm_min=args.norm_min, norm_max=args.norm_max, stats_out=args.stats_out, return_pair=False)
    dl = loaders.train
    ds = dl.dataset.dataset if hasattr(dl.dataset, 'dataset') else dl.dataset
    image_height = ds.height
    image_width = ds.width
    image_size = image_height if image_height == image_width else max(image_height, image_width)

    art = fit_global_pod(dl_train=dl, image_size=image_size, n_components_max=args.n_components_max, image_height=image_height, image_width=image_width)
    art.k_energy = int(modes_for_energy(art.cum_energy, args.energy))
    print('\n=== Global-POD Fit Summary ===')
    print(f'dataset: {args.dataset}')
    print(f'image_height={image_height}, image_width={image_width}')
    print(f'n_components_max={args.n_components_max}, cum[-1]={art.cum_energy[-1]:.4f}')
    print(f'energy={args.energy} -> K={art.k_energy}')
    save_npz(art, args.pod_out)
    print(f'[saved] {args.pod_out}')

    if args.write_cache or args.cache_dir:
        if not args.cache_dir:
            raise ValueError('--write_cache was set but --cache_dir is empty')
        print(f'[global-cache] building training cache in {args.cache_dir}')
        # Rebuild dataset as (X,Y). This happens only in the CPU POD job.
        from podiff.data.sst_npz import SSTNPZDataset
        from podiff.data.global_cache import write_global_training_cache
        npz_path = args.npz_path or args.root
        ds_pair = SSTNPZDataset(
            npz_path,
            return_pair=True,
            norm_mode=args.norm_mode,
            norm_min=args.norm_min,
            norm_max=args.norm_max,
            stats_out=args.stats_out,
        )
        write_global_training_cache(
            ds_pair,
            art,
            cache_dir=args.cache_dir,
            batch_size=args.cache_batch_size,
            seed=args.seed,
            save_field_cache=True,
        )


if __name__ == '__main__':
    main()
