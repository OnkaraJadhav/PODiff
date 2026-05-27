#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --job-name=podiff_sst_gpu_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --export=NONE

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3

# Edit these paths if running outside the repository root.
REPO_DIR="${REPO_DIR:-$PWD}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_DIR}/podiff/configs/train_diffusion.json}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}}"

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/artifacts" "${RESULT_ROOT}/outputs" "${RESULT_ROOT}/cache"

CPU_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -V
which python
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name:', torch.cuda.get_device_name(0))
PY

python -m podiff.scripts.train_global_diffusion --config "${TRAIN_CONFIG}"
