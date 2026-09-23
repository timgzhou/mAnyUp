#!/bin/bash
#SBATCH --job-name=lp_mu_pair
#SBATCH --account=aip-gpleiss
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pastis/lp_mu_pair_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=END,FAIL

# pa2px LP on one LR feature cfg, WITH and WITHOUT a mAnyUp upsampler, so the pair is
# directly comparable (same features, same head, same epochs -- only the upsampler differs).
#
#   MODS      s2 | s1 | s2s1        (required)
#   LR_PS     LR patch size  (default 8)
#   HR_PS     mAnyUp target  (default 4)
#   EPOCHS    default 32
#   MU_EPOCH  which mAnyUp ckpt epoch to probe (default 31)
#   ARCH      "" for stock anyup ckpts, "_manyup" for the nonlinear arch
#
# sbatch --export=ALL,MODS=s1 scripts/slurm/pastis/lp_manyup_pair.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1
MODS="${MODS:?set MODS}"
LR_PS="${LR_PS:-8}"; HR_PS="${HR_PS:-4}"
EPOCHS="${EPOCHS:-32}"; MU_EPOCH="${MU_EPOCH:-31}"; ARCH="${ARCH:-}"
OUT_ROOT="${OUT_ROOT:-$HOME/projects/aip-gpleiss/timz/features}"
CSV="${CSV:-results/pastis/lp_olmoearth_pastis.csv}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/pastis
source env_setup/env_olmo.sh

LR_CFG="oe_base_${MODS}_ps${LR_PS}_tile64"
HR_CFG="oe_base_${MODS}_ps${HR_PS}_tile64"
CKPT="checkpoints/manyup/${LR_CFG}__to__${HR_CFG}/manyup_${LR_CFG}_to_${HR_CFG}${ARCH}_ep${MU_EPOCH}.pth"

echo "=== baseline: pa2px on ${LR_CFG} (no upsampling) ==="
python -u -m exp.pastis.lp_cached_features --features "$LR_CFG" --head_mode lp_pa2px \
    --out_root "$OUT_ROOT" --data_splits data/pastis_olmoearth \
    --epochs "$EPOCHS" --results_csv "$CSV"
BASE=$?

echo "=== mAnyUp: ${LR_CFG} -> ${HR_CFG}, pa2px on the upsampled map ==="
if [ -f "$CKPT" ]; then
    python -u -m exp.pastis.lp_cached_features --features "$LR_CFG" \
        --manyup_ckpt "$CKPT" --manyup_native_out \
        --out_root "$OUT_ROOT" --data_splits data/pastis_olmoearth \
        --epochs "$EPOCHS" --results_csv "$CSV"
    MU=$?
else
    echo "ERROR: no mAnyUp checkpoint at $CKPT"; MU=-1
fi

LOG="logs/pastis/lp_mu_pair_${SLURM_JOB_ID}.out"
{
    echo "mods=$MODS  ps${LR_PS} -> ps${HR_PS}  arch='${ARCH:-anyup}'"
    echo "baseline status=$BASE   manyup status=$MU"
    echo "--- results ---"
    grep -aE "^TEST" "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID] lp pair $MODS ps${LR_PS}->ps${HR_PS}" "$EMAIL"
