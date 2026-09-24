#!/bin/bash
#SBATCH --job-name=extract_chain
#SBATCH --account=aip-gpleiss
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/pastis/extract_chain_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=END,FAIL

# Extract a PASTIS feature cache too slow for one job's walltime (default: the original
# case, oe_base_s2_ps1_tile128_img128).
# ps1/tile128 is a 128x128 token grid (196,608 tokens in ONE attention call): it OOMs above
# batch 1, and at batch 1 runs ~75 s/sample, so 2433 samples needs ~51 GPU-hours.
#
# Self-chaining: each job extracts until it is nearly out of walltime, then submits its own
# successor and exits. exp/pastis/extract_features.py skips batches whose outputs already
# exist (shuffle=False makes indices stable), so a successor resumes instead of restarting.
# The chain stops when extraction completes, and the LAST job runs the LP probe.
#
#   sbatch scripts/slurm/pastis/extract_chain.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1

# Defaults target the original ps1/tile128 arm; override via --export for any other slow
# config (ps1/tile64 needs ~11 h, just over a single job's walltime).
PATCH_SIZE="${PATCH_SIZE:-1}"
TILE_SIZE="${TILE_SIZE:-128}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
DATA_SPLITS="${DATA_SPLITS:-data/pastis128_olmoearth}"
IMGSUF=""
[ "$IMAGE_SIZE" != "64" ] && IMGSUF="_img${IMAGE_SIZE}"
FEATURES="oe_base_s2_ps${PATCH_SIZE}_tile${TILE_SIZE}${IMGSUF}"
OUT_ROOT="${OUT_ROOT:-$HOME/projects/aip-gpleiss/timz/features}"
RESULTS_CSV="results/pastis/lp_olmoearth_pastis.csv"
MAX_LINKS="${MAX_LINKS:-8}"          # safety stop so a bug cannot chain forever
LINK="${LINK:-1}"

cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh

# sample counts differ per prep: 128 keeps whole patches, 64 quarters them
if [ "$IMAGE_SIZE" = "128" ]; then
    NEED_TRAIN=1455; NEED_VALID=482; NEED_TEST=496
else
    NEED_TRAIN=5820; NEED_VALID=1928; NEED_TEST=1984
fi
count() { ls "$OUT_ROOT/$FEATURES/pastis_r_$1"/*.pt 2>/dev/null | wc -l; }
total_have() { echo $(( $(count train) + $(count valid) + $(count test) )); }
NEED_TOTAL=$(( NEED_TRAIN + NEED_VALID + NEED_TEST ))

HAVE_START=$(total_have)
echo "=== link $LINK/$MAX_LINKS : $HAVE_START/$NEED_TOTAL samples present ==="

# Queue the successor NOW (afterany: runs whether we finish, fail, or time out). If this
# link turns out to be the last one, we scancel it below.
NEXT_ID=""
if [ "$HAVE_START" -lt "$NEED_TOTAL" ] && [ "$LINK" -lt "$MAX_LINKS" ]; then
    NEXT_ID=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
                     --export=ALL,LINK=$((LINK + 1)),MAX_LINKS=$MAX_LINKS "$0")
    echo "=== queued successor link $((LINK + 1)) as job $NEXT_ID ==="
fi

python -u -m exp.pastis.extract_features \
    --model_size base --modalities sentinel2_l2a \
    --patch_size $PATCH_SIZE --tile_size $TILE_SIZE --image_size $IMAGE_SIZE \
    --data_splits "$DATA_SPLITS" --batch_size 1 --out_root "$OUT_ROOT"
EXTRACT_STATUS=$?

HAVE=$(total_have)
echo "=== after link $LINK: $HAVE/$NEED_TOTAL (extract status $EXTRACT_STATUS) ==="

if [ "$HAVE" -ge "$NEED_TOTAL" ] && [ $EXTRACT_STATUS -eq 0 ]; then
    # Done: drop the successor we optimistically queued.
    [ -n "$NEXT_ID" ] && scancel "$NEXT_ID" && echo "=== cancelled queued successor $NEXT_ID ==="
    echo "=== extraction COMPLETE -> lp_pa2px ==="
    python -u -m exp.pastis.lp_cached_features \
        --features "$FEATURES" --out_root "$OUT_ROOT" --data_splits "$DATA_SPLITS" \
        --head_mode lp_pa2px --epochs 32 --results_csv "$RESULTS_CSV"
    LP=$?
    { echo "features: $FEATURES"; echo "lp status: $LP"; tail -n 3 "$RESULTS_CSV"; } \
        | mail -s "[DONE chain] ps1 tile128 lp=$LP" "$EMAIL"
elif [ "$LINK" -ge "$MAX_LINKS" ]; then
    echo "=== reached MAX_LINKS=$MAX_LINKS with $HAVE/$NEED_TOTAL; stopping ==="
    echo "$HAVE/$NEED_TOTAL after $MAX_LINKS links" \
        | mail -s "[STOP chain] ps1 tile128 incomplete" "$EMAIL"
else
    echo "=== link $LINK done at $HAVE/$NEED_TOTAL; successor $NEXT_ID already queued ==="
fi
