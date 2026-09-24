#!/bin/bash
#SBATCH --job-name=oe_pstile
#SBATCH --account=aip-gpleiss
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/pastis/oe_pstile_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# One (patch_size, tile_size) arm of the ps x tile study: extract features, run the
# lp_pa2px linear probe, then DELETE the features unless this arm is one we keep.
#
# Keep policy (KEEP_FEATURES, default auto):
#   auto -> keep iff PATCH_SIZE is 1 or 8. ps1 is the most expensive to recompute, and ps8
#           is the arm later work builds on; every other arm is cheap enough to redo, and
#           the full 15-arm grid would otherwise be far more disk than we need.
#   1/0  -> force keep / force delete.
#
# Env vars:
#   PATCH_SIZE  (required)
#   TILE_SIZE   (required)
#   IMAGE_SIZE  (default 64; use 128 with DATA_SPLITS=data/pastis128_olmoearth)
#   MODALITIES  (default sentinel2_l2a)
#   EPOCHS      (default 32)  lp_cached_features epochs
#   KEEP_FEATURES (default auto)
#
# Examples:
#   sbatch --export=ALL,PATCH_SIZE=8,TILE_SIZE=16 scripts/slurm/pastis/pstile_arm.sh
#   sbatch --export=ALL,PATCH_SIZE=4,TILE_SIZE=128,IMAGE_SIZE=128,DATA_SPLITS=data/pastis128_olmoearth scripts/slurm/pastis/pstile_arm.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1

PATCH_SIZE="${PATCH_SIZE:?set PATCH_SIZE}"
TILE_SIZE="${TILE_SIZE:?set TILE_SIZE}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
MODALITIES="${MODALITIES:-sentinel2_l2a}"
MODEL_SIZE="${MODEL_SIZE:-base}"
EPOCHS="${EPOCHS:-32}"
KEEP_FEATURES="${KEEP_FEATURES:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_ROOT="${OUT_ROOT:-$HOME/projects/aip-gpleiss/timz/features}"
RESULTS_CSV="${RESULTS_CSV:-results/pastis/lp_olmoearth_pastis.csv}"

# DATA_SPLITS must match IMAGE_SIZE: the 128 prep has its own sample count and indices, and
# pointing 128 data at a 64-derived cache silently misaligns every sample.
if [ "$IMAGE_SIZE" = "128" ]; then
    DATA_SPLITS="${DATA_SPLITS:-data/pastis128_olmoearth}"
else
    DATA_SPLITS="${DATA_SPLITS:-data/pastis_olmoearth}"
fi

if [ "$KEEP_FEATURES" = "auto" ]; then
    case "$PATCH_SIZE" in
        1|8) KEEP=1 ;;
        *)   KEEP=0 ;;
    esac
else
    KEEP="$KEEP_FEATURES"
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/pastis results/pastis
source env_setup/env_olmo.sh

# Feature dir name must match extract_features.cfg_name(): oe_<size>_<mods>_ps<P>_tile<T>[_img<N>]
MODS=$(echo "$MODALITIES" | sed 's/sentinel2_l2a/s2/g; s/sentinel1/s1/g; s/,//g')
IMGSUF=""
[ "$IMAGE_SIZE" != "64" ] && IMGSUF="_img${IMAGE_SIZE}"
FEATURES="oe_${MODEL_SIZE}_${MODS}_ps${PATCH_SIZE}_tile${TILE_SIZE}${IMGSUF}"
TAG="ps${PATCH_SIZE} tile${TILE_SIZE} img${IMAGE_SIZE}"

echo "=== config $FEATURES (keep_features=$KEEP) ==="
echo "extract ps=$PATCH_SIZE tile=$TILE_SIZE img=$IMAGE_SIZE splits=$DATA_SPLITS" \
    | mail -s "[START job $SLURM_JOB_ID] pstile $TAG" "$EMAIL"

python -u -m exp.pastis.extract_features \
    --model_size "$MODEL_SIZE" --modalities "$MODALITIES" \
    --patch_size "$PATCH_SIZE" --tile_size "$TILE_SIZE" --image_size "$IMAGE_SIZE" \
    --data_splits "$DATA_SPLITS" --batch_size "$BATCH_SIZE" --out_root "$OUT_ROOT"
EXTRACT_STATUS=$?

LP_STATUS=-1
if [ $EXTRACT_STATUS -eq 0 ]; then
    echo "=== lp_pa2px on $FEATURES ==="
    python -u -m exp.pastis.lp_cached_features \
        --features "$FEATURES" --out_root "$OUT_ROOT" --data_splits "$DATA_SPLITS" \
        --head_mode lp_pa2px --epochs "$EPOCHS" --results_csv "$RESULTS_CSV"
    LP_STATUS=$?
fi

# Drop the features only when the probe SUCCEEDED -- deleting after a failed LP would throw
# away the expensive part and leave nothing to retry from.
if [ "$KEEP" = "0" ] && [ $LP_STATUS -eq 0 ]; then
    echo "=== removing $OUT_ROOT/$FEATURES (keep policy) ==="
    rm -rf "${OUT_ROOT:?}/${FEATURES:?}"
elif [ "$KEEP" = "0" ]; then
    echo "=== KEEPING $FEATURES despite policy: lp status=$LP_STATUS ==="
fi

LOG="logs/pastis/oe_pstile_${SLURM_JOB_ID}.out"
{
    echo "features: $FEATURES"
    echo "extract status: $EXTRACT_STATUS   lp status: $LP_STATUS   kept: $KEEP"
    echo "--- lp result ---"
    tail -n 3 "$RESULTS_CSV" 2>/dev/null
    echo "--- last 10 log lines ---"
    tail -n 10 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID ex=$EXTRACT_STATUS lp=$LP_STATUS] pstile $TAG" "$EMAIL"
