#!/bin/bash
#SBATCH --job-name=lp_half
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pastis/lp_half_%j.out
# Leakage control: train the upsampler on the FIRST half of the train ids and its pa2px probe
# on the SECOND half, so the probe never sees samples the upsampler was fitted on. Compare
# against the same cell trained on all 5820 for both stages.
set -euo pipefail
export TQDM_DISABLE=1
LR_PS="${LR_PS:?}"; HR_PS="${HR_PS:?}"; TD="${TD:?}"
cd "$SLURM_SUBMIT_DIR"; source env_setup/env_olmo.sh
LR_CFG="oe_base_s2_ps${LR_PS}_tile64"; HR_CFG="oe_base_s2_ps${HR_PS}_tile64"
python -u -m exp.upsamplers.train_manyup \
    --lr_cfg "$LR_CFG" --hr_cfg "$HR_CFG" --id_half first \
    --arch manyup --transform_depth "$TD" --window_ratio 1.0 --down_reg 0 --no-proj_head \
    --epochs 32 --out_dir checkpoints/manyup_half
CK="checkpoints/manyup_half/manyup_${TD}transform_${LR_CFG}_to_${HR_CFG}_manyup_w1_dr0_ep31.pth"
python -u -m exp.pastis.lp_cached_features \
    --features "$LR_CFG" --manyup_ckpt "$CK" --manyup_native_out --id_half second --epochs 32 \
    --out_root "$HOME/projects/aip-gpleiss/timz/features" --data_splits data/pastis_olmoearth \
    --save_head checkpoints/lp_heads_half \
    --results_csv results/pastis/lp_olmoearth_pastis.csv
