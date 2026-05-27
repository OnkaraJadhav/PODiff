#!/bin/bash -l
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=podiff_sst_cpu_fit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --export=NONE

set -euo pipefail

# Edit these two paths for your system before submitting.
REPO_DIR="${REPO_DIR:-$PWD}"
FIT_CONFIG="${FIT_CONFIG:-${REPO_DIR}/podiff/configs/fit_pod.json}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}}"

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/artifacts" "${RESULT_ROOT}/outputs" "${RESULT_ROOT}/cache"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -V
which python
python -m podiff.scripts.fit_global_pod --config "${FIT_CONFIG}"
