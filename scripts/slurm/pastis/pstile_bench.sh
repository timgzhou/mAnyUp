#!/bin/bash
#SBATCH --job-name=pstile_bench
#SBATCH --account=aip-gpleiss
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pastis/pstile_bench_%j.out
set -euo pipefail
export TQDM_DISABLE=1
cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh
python -u -m exp.bench.olmo_throughput --configs 1:1,2:2,4:4 --image_size 64 --iters 4 --warmup 2 \
    --out results/bench/olmo_throughput_pstile.csv
