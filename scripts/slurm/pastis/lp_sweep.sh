#!/bin/bash
#SBATCH --job-name=oe_lpsweep
#SBATCH --account=aip-gpleiss
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pastis/oe_lpsweep_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=END,FAIL

# Re-probe existing cached features at several (epochs, lr) settings.
#
# Why: at a FIXED batch_size the img128 prep takes 4x fewer optimizer steps per epoch than
# img64 (1455 vs 5820 samples), while delivering the same number of token-updates. Every
# img128 mIoU came in 0.01-0.02 BELOW its img64 twin and the val curves were still rising at
# epoch 32, so the gap may be under-training rather than a real property of the tiling.
# This sweeps epochs (more steps) and lr (bigger steps) to separate the two.
#
#   sbatch --export=ALL,FEATURES=oe_base_s2_ps4_tile64_img128 scripts/slurm/pastis/lp_sweep.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1

FEATURES="${FEATURES:?set FEATURES}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
GRID="${GRID:-32,64,128:0.01 128:0.02 128:0.005}"   # unused placeholder; see SETTINGS
SETTINGS="${SETTINGS:-32:0.01 64:0.01 128:0.01 128:0.02 128:0.005}"
OUT_ROOT="${OUT_ROOT:-$HOME/projects/aip-gpleiss/timz/features}"
RESULTS_CSV="${RESULTS_CSV:-results/pastis/lp_olmoearth_pastis.csv}"

if [ "$IMAGE_SIZE" = "128" ]; then
    DATA_SPLITS="${DATA_SPLITS:-data/pastis128_olmoearth}"
else
    DATA_SPLITS="${DATA_SPLITS:-data/pastis_olmoearth}"
fi

cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh

for s in $SETTINGS; do
    EP="${s%%:*}"; LR="${s##*:}"
    echo "=== $FEATURES epochs=$EP lr=$LR ==="
    python -u -m exp.pastis.lp_cached_features \
        --features "$FEATURES" --out_root "$OUT_ROOT" --data_splits "$DATA_SPLITS" \
        --head_mode lp_pa2px --epochs "$EP" --lr "$LR" --results_csv "$RESULTS_CSV"
done

LOG="logs/pastis/oe_lpsweep_${SLURM_JOB_ID}.out"
{
    echo "features: $FEATURES  settings: $SETTINGS"
    grep -aE "^BEST val miou|^TEST " "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID] lp sweep $FEATURES" "$EMAIL"
