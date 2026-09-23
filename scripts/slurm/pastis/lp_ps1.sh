#!/bin/bash
#SBATCH --job-name=lp_ps1
#SBATCH --account=aip-gpleiss
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=logs/pastis/lp_ps1_%j.out
set -euo pipefail
export TQDM_DISABLE=1
cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh
# ps1 is the 684 GB cache: force the per-sample disk path rather than a preload that cannot fit.
python -u -m exp.pastis.lp_cached_features \
    --features oe_base_s2_ps1_tile64 --head_mode lp_pa2px --epochs 32 \
    --out_root "$HOME/projects/aip-gpleiss/timz/features" --data_splits data/pastis_olmoearth \
    --max_ram_gb 0 --num_workers 8 --save_head checkpoints/lp_heads \
    --results_csv results/pastis/lp_olmoearth_pastis.csv
