#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "${REPO_DIR}"
mkdir -p logs artifacts outputs cache

CPU_JOB=$(sbatch python_pod.sh | awk '{print $4}')
echo "Submitted CPU POD fit/cache job: ${CPU_JOB}"

DIFFUSION_JOB=$(sbatch --dependency=afterok:${CPU_JOB} python_gpu_train_diffusion.sh | awk '{print $4}')
echo "Submitted GPU diffusion training job: ${DIFFUSION_JOB} (afterok:${CPU_JOB})"

SAMPLE_JOB=$(sbatch --dependency=afterok:${DIFFUSION_JOB} python_multisample_generation.sh | awk '{print $4}')
echo "Submitted GPU sampling job: ${SAMPLE_JOB} (afterok:${DIFFUSION_JOB})"
