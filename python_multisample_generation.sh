#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=podiff_sst_samples
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --export=NONE

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3

REPO_DIR="${REPO_DIR:-$PWD}"
SAMPLE_CONFIG="${SAMPLE_CONFIG:-${REPO_DIR}/podiff/configs/multisample_generation_gpu.json}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}}"

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/outputs" "${RESULT_ROOT}/cache"

export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -m podiff.scripts.multisample_global_sr --config "${SAMPLE_CONFIG}"
