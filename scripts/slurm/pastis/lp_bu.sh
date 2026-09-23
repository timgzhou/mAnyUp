#!/bin/bash
#SBATCH --job-name=lp_bu
#SBATCH --account=aip-gpleiss
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/pastis/lp_bu_%j.out
set -euo pipefail
export TQDM_DISABLE=1
PS="${PS:?}"
cd "$SLURM_SUBMIT_DIR"; source env_setup/env_olmo.sh
python -u -m exp.pastis.lp_cached_features \
    --features "oe_base_s2_ps${PS}_tile64" --head_mode lp_bu_px2px --epochs 32 \
    --out_root "$HOME/projects/aip-gpleiss/timz/features" --data_splits data/pastis_olmoearth \
    --save_head checkpoints/lp_heads --max_ram_gb 40 \
    --results_csv results/pastis/lp_olmoearth_pastis.csv
