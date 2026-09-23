#!/bin/bash
#SBATCH --job-name=mu_sweep
#SBATCH --account=aip-gpleiss
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pastis/mu_sweep_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=END,FAIL

# Train ONE shared-probe-eligible upsampler: transform_depth=0 + no projector, so the output
# stays a pure attention-weighted sum of the LR features and an lp_bu_px2px probe fitted on
# bilinear-upsampled LR features can be reused on it FROZEN. window_ratio=1.0 lets every query
# attend to all LR tokens; down_reg=0 drops the consistency loss.
#
#   MODS   s2 | s1 | s2s1     LR_PS  16|8     HR_PS  4|2|1
set -euo pipefail
export TQDM_DISABLE=1
MODS="${MODS:?set MODS}"; LR_PS="${LR_PS:?set LR_PS}"; HR_PS="${HR_PS:-4}"
EPOCHS="${EPOCHS:-32}"
# The ps2/ps1 caches were extracted at their own tile sizes (ps2 -> tile32, ps1 -> tile64), so
# the HR config name is not always <mods>_ps<N>_tile64. HR_CFG overrides it outright.
HR_CFG="${HR_CFG:-oe_base_${MODS}_ps${HR_PS}_tile64}"
# Depth of mAnyUp's post-attention transform. TD=0 keeps the output a pure attention-weighted
# sum of the LR features, which is what makes a shared px2px probe valid (see
# CachedManyUpSharedProbe); TD>0 adds nonlinear ResBlocks and is NOT shared-probe eligible.
TD="${TD:-0}"
cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh

python -u -m exp.upsamplers.train_manyup \
    --lr_cfg "oe_base_${MODS}_ps${LR_PS}_tile64" \
    --hr_cfg "$HR_CFG" \
    --arch manyup --transform_depth $TD --window_ratio 1.0 --down_reg 0 --no-proj_head \
    --epochs "$EPOCHS" --out_dir checkpoints/manyup
