#!/bin/bash
#SBATCH --job-name=lp_hctrl
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/pastis/lp_hctrl_%j.out
# Two controls for the half-split experiment, both fitting the PROBE on half the train split:
#   MODE=baseline  a plain head (lp_pa2px / lp_bu_px2px) on frozen features, so the baselines
#                  are matched to the half-split mAnyUp runs in probe-training data.
#   MODE=leak      the FULL-data upsampler with its probe on half the ids it already saw --
#                  isolates data volume from leakage (vs the disjoint half-split).
set -euo pipefail
export TQDM_DISABLE=1
MODE="${MODE:?}"; PS="${PS:?}"; HALF="${HALF:-second}"
cd "$SLURM_SUBMIT_DIR"; source env_setup/env_olmo.sh
CFG="oe_base_s2_ps${PS}_tile64"
COMMON="--out_root $HOME/projects/aip-gpleiss/timz/features --data_splits data/pastis_olmoearth
        --epochs 32 --id_half $HALF --max_ram_gb 40
        --results_csv results/pastis/lp_olmoearth_pastis.csv"
if [ "$MODE" = "baseline" ]; then
    python -u -m exp.pastis.lp_cached_features --features "$CFG" --head_mode "${HEAD:?}" $COMMON
else
    TD="${TD:?}"; HR_PS="${HR_PS:?}"
    CK="checkpoints/manyup/manyup_${TD}transform_${CFG}_to_oe_base_s2_ps${HR_PS}_tile64_manyup_w1_dr0_ep31.pth"
    python -u -m exp.pastis.lp_cached_features --features "$CFG" --manyup_ckpt "$CK" \
        --manyup_native_out $COMMON
fi
