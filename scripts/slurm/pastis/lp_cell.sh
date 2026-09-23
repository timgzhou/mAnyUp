#!/bin/bash
#SBATCH --job-name=lp_cell
#SBATCH --account=aip-gpleiss
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pastis/lp_cell_%j.out
# One cell of the mAnyUp comparison table.
#   MODE=own    train a pa2px head on this upsampler          (needs LR_PS, HR_PS, TD)
#   MODE=shared borrow a frozen bu_px2px probe, no training   (+ SIDE=lr|hr)
set -euo pipefail
export TQDM_DISABLE=1
LR_PS="${LR_PS:?}"; HR_PS="${HR_PS:?}"; TD="${TD:?}"; MODE="${MODE:?}"; SIDE="${SIDE:-lr}"
cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh
CK="checkpoints/manyup/manyup_${TD}transform_oe_base_s2_ps${LR_PS}_tile64_to_oe_base_s2_ps${HR_PS}_tile64_manyup_w1_dr0_ep31.pth"
COMMON="--features oe_base_s2_ps${LR_PS}_tile64 --manyup_ckpt $CK --manyup_native_out
        --out_root $HOME/projects/aip-gpleiss/timz/features --data_splits data/pastis_olmoearth
        --results_csv results/pastis/lp_olmoearth_pastis.csv"
if [ "$MODE" = "own" ]; then
    python -u -m exp.pastis.lp_cached_features $COMMON --epochs 32 --save_head checkpoints/lp_heads
else
    # LR-side probe = fitted on the LR config; HR-side = fitted on the target-resolution config.
    if [ "$SIDE" = "hr" ]; then P="oe_base_s2_ps${HR_PS}_tile64"; else P="oe_base_s2_ps${LR_PS}_tile64"; fi
    EXTRA=""; [ "$SIDE" = "lr" ] && [ "$TD" != "0" ] && EXTRA="--manyup_allow_probe_mismatch"
    python -u -m exp.pastis.lp_cached_features $COMMON \
        --manyup_shared_probe "checkpoints/lp_heads/lphead_${P}_lp_bu_px2px.pth" \
        --manyup_shared_probe_side "$SIDE" $EXTRA
fi
