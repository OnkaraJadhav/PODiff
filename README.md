# PODiff: POD-space diffusion for scientific super-resolution

This repository contains the official implementation of **PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution** (ICML2026)

The code focuses on the paper pipeline:

1. Fit a global POD basis on high-resolution scientific fields.
2. Cache POD coefficients for memory-safe GPU training.
3. Train a conditional diffusion model in POD coefficient space.
4. Generate ensemble super-resolution samples.
5. Evaluate reconstruction and uncertainty metrics.

---

## Repository structure

```text
podiff/
  configs/
    fit_pod.json                    # POD fitting + coefficient cache config
    train_diffusion.json             # MLP latent diffusion training config
    multisample_generation_gpu.json  # ensemble generation config
    evaluate_samples.json            # evaluation config
  data/
    sst_npz.py                       # SST NPZ loader
    global_cache.py                  # cached POD coefficient dataset
  diffusion/
    ddpm.py
    schedule.py
  models/
    mlp_latent_diffusion.py          # PODiff denoising network
  pod/
    global_pod.py                    # POD fit/encode/decode utilities
  scripts/
    fit_global_pod.py
    train_global_diffusion.py
    multisample_global_sr.py
    evaluate_samples.py
  utils/

experimental/
  data/, models/                     # unused experimental files kept outside default pipeline
```

---

## Data format

The loader expects a `.npz` file with arrays named either:

```text
X, Y
```

or

```text
x, y
```

where:

- `X` is the low-resolution/interpolated/conditioning field on the high-resolution grid.
- `Y` is the high-resolution target field.
- Accepted shapes are `(N,H,W)`, `(N,H,W,1)`, or `(N,1,H,W)`.
- NaNs are allowed and are converted to zero after normalization; valid ocean pixels are tracked through a mask.

The paper-style training file is:
```text
data/training_data.npz
```
or update the `npz_path` field in the JSON configs.

> Note: The paper uses temporal splits for the SST experiments: training on 1998–2009, validation on 2010, and testing on 2011. These scripts assume that the user supplies the appropriate pre-split `.npz` files or adjusts the configs accordingly.

---

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The provided SLURM scripts assume the `pytorch/2.7.1-rocm6.3.3` module for GPU jobs.

---

## Step 1: Fit POD basis and write coefficient cache

Edit paths in:

```text
podiff/configs/fit_pod.json
```

Then run:

```bash
export PYTHONPATH=$PWD
python -m podiff.scripts.fit_global_pod --config podiff/configs/fit_pod.json
```

This creates:

```text
artifacts/pod_sst.npz
outputs/sst_stats.json
cache/x_coeff.npy
cache/y_coeff.npy
cache/x_norm.npy
cache/y_norm.npy
cache/pod_projection.npy
cache/meta.json
```

The coefficient cache avoids loading full `640 x 480` fields inside every GPU training job.

---

## Step 2: Train PODiff diffusion model

Edit:

```text
podiff/configs/train_diffusion.json
```

Then run:

```bash
export PYTHONPATH=$PWD
python -m podiff.scripts.train_global_diffusion --config podiff/configs/train_diffusion.json
```

Default settings include:

```text
K = 40
T = 1000 diffusion steps
S = 100 DDIM sampling steps at inference
MLP width = 256
MLP layers = 4
learning rate = 2e-4
```

Checkpoints are written to:

```text
outputs/diffusion_model/
```

---

## Step 3: Generate ensemble samples

Edit:

```text
podiff/configs/multisample_generation_gpu.json
```

Then run:

```bash
export PYTHONPATH=$PWD
python -m podiff.scripts.multisample_global_sr --config podiff/configs/multisample_generation_gpu.json
```

Outputs are written to:

```text
outputs/samples/
  gt/                 # target images
  input/              # normalized conditioning/input fields, if cached
  pod_projection/     # deterministic POD projection diagnostic, if cached
  ensemble/           # PNG samples grouped by image
  npy/                # gt/input/sample arrays for evaluation
```

---

## Step 4: Evaluate reconstruction and uncertainty

Run:

```bash
export PYTHONPATH=$PWD
python -m podiff.scripts.evaluate_samples --config podiff/configs/evaluate_samples.json
```

The evaluation script reports:

- RMSE
- MAE
- R²
- optional input/LR consistency metrics for `input_*.npy`

Results are saved to:

```text
outputs/evaluation_metrics.json
```

By default, metrics are denormalized using `outputs/sst_stats.json`, so RMSE/MAE are reported in physical units if the input data were normalized from SST values.

---

## SLURM usage

The repository includes scripts:

```bash
sbatch python_pod.sh
sbatch python_gpu_train_diffusion.sh
sbatch python_multisample_generation.sh
```

or submit the full dependency chain:

```bash
bash submit_global_cpu_then_parallel_gpu_all.sh
```

Before submitting, edit or export:

```bash
export REPO_DIR=/path/to/PODiff_codebase
export RESULT_ROOT=/path/to/PODiff_codebase
```
---

## Notes on reproducibility
- This is a code release for the PODiff method: POD fitting, diffusion training, ensemble generation, and evaluation.
- For exact paper reproduction, use the same temporal splits, normalization range, latent dimension, sampling steps, and ensemble size described in the paper.

---

## Citation

Please cite the PODiff paper as:
```bibtex
@misc{jadhav2026podifflatentdiffusionproper,
      title={PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution}, 
      author={Onkar Jadhav and Tim French and Matthew Rayson and Nicole L. Jones},
      year={2026},
      eprint={2605.03399},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.03399}, 
}
```