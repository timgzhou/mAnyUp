#!/bin/bash
#SBATCH --job-name=ys_prepare
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
# Build the packed YieldSAT store (exp/yieldsat/prepare.py): index -> pack -> verify.
# CPU only. Needs data/YieldSAT/{preprocessed-24-ts, original-preprocessed} extracted from
# Both.zip (see docs/RUNBOOK.md). Every step is resumable; rerun the same command after a
# timeout.
#
#   sbatch scripts/slurm/yieldsat/prepare.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repo root}"
source env_setup/env_olmo.sh
export TQDM_DISABLE=1
[ -f data/YieldSAT/packed/index.json ] || python -u -m exp.yieldsat.prepare index
python -u -m exp.yieldsat.prepare pack --workers 32
python -u -m exp.yieldsat.prepare verify --workers 4
