#!/bin/bash
#SBATCH --job-name=oe_ups16
#SBATCH --account=aip-gpleiss
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/oe_ups16_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Run eval_upsamplers_pa2pa.py on a COARSER feature grid than the ps4 run in
# upsampler_pa2pa.csv, to test the hypothesis that ps4 (16x16 tokens) is already fine enough
# that guided upsampling has nothing left to recover -- which would explain why upma never
# beats lr_bilinear there. ps16 gives a 4x4 token grid, a 16x upsample to 64x64, so if the
# hypothesis holds the guided upsamplers should show their advantage here.
#
# Does a smoke run (--limit_test 64) first and aborts if it fails, so a broken config costs
# ~2 minutes instead of the full ~1.5h eval.
#
# IMPORTANT: --retrain is passed on BOTH runs. eval_upsamplers_pa2pa.py reuses a cached head
# if --head_ckpt exists, and the head is a 1x1 conv whose shape depends only on embed_dim --
# so a ps4-trained head would load against ps16 features WITHOUT error and silently produce
# garbage. The ckpt path is also distinct from checkpoints/pa2pa_head.pt so the ps4 head
# survives. On the full run --retrain additionally avoids inheriting the smoke run's head,
# which was trained on the same schedule but is overwritten here for clarity.
#
# Configurable knobs (env vars from --export override these defaults):
#   FEATURES     cached feature set to evaluate (default oe_base_s2_ps16_tile64)
#   HEAD_CKPT    where to cache this run's probe (default checkpoints/pa2pa_head_ps16.pt)
#   SMOKE_N      images for the smoke run (default 64; set 0 to skip straight to full)
#   METHODS      comma-separated eval paths (default all four)
#
# Examples:
#   sbatch scripts/slurm/upsamplers/upsampler_ps16.sh
#   sbatch --dependency=afterok:4701460 upsampler_ps16.sh    # wait for the ps16 extraction
#   sbatch --export=ALL,FEATURES=oe_base_s2_ps8_tile64 upsampler_ps16.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1   # silence tqdm progress bars in the batch log

FEATURES="${FEATURES:-oe_base_s2_ps16_tile64}"
HEAD_CKPT="${HEAD_CKPT:-checkpoints/pa2pa_head_ps16.pt}"
SMOKE_N="${SMOKE_N:-64}"
METHODS="${METHODS:-lr_bilinear,upa,upma,anyup}"

cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh

FEAT_DIR="$HOME/projects/aip-gpleiss/timz/features/$FEATURES"
LOG="logs/oe_ups16_${SLURM_JOB_ID}.out"

# Email at start.
echo "features: $FEATURES | head_ckpt: $HEAD_CKPT | smoke: $SMOKE_N | methods: $METHODS" \
    | mail -s "[START job $SLURM_JOB_ID] upsampler eval $FEATURES" "$EMAIL"

# Fail fast if the extraction did not actually finish. meta.json is written last by
# exp/pastis/extract_features.py, so its presence means the cache is complete.
if [ ! -f "$FEAT_DIR/meta.json" ]; then
    echo "ERROR: $FEAT_DIR/meta.json missing -- extraction incomplete or wrong FEATURES." >&2
    echo "ERROR: $FEAT_DIR/meta.json missing -- extraction incomplete." \
        | mail -s "[FAIL job $SLURM_JOB_ID] upsampler eval $FEATURES" "$EMAIL"
    exit 1
fi
echo "=== feature meta ==="
cat "$FEAT_DIR/meta.json"

# ---- step 2: smoke run, 64 images. Confirms the 4x4 -> 64x64 upsample path works. ----
STATUS_SMOKE=0
if [ "$SMOKE_N" -gt 0 ]; then
    echo "=== SMOKE features=$FEATURES limit_test=$SMOKE_N ==="
    python -u -m exp.upsamplers.eval_pa2pa \
        --features "$FEATURES" \
        --head_ckpt "$HEAD_CKPT" \
        --retrain \
        --methods "$METHODS" \
        --limit_test "$SMOKE_N"
    STATUS_SMOKE=$?
    echo "=== EXIT smoke status=$STATUS_SMOKE ==="

    if [ $STATUS_SMOKE -ne 0 ]; then
        echo "ERROR: smoke run failed, skipping the full eval." >&2
        {
            echo "features: $FEATURES"
            echo "smoke exit status: $STATUS_SMOKE -- full eval SKIPPED"
            echo "--- last 20 log lines ---"
            tail -n 20 "$LOG"
        } | mail -s "[FAIL job $SLURM_JOB_ID] upsampler eval $FEATURES" "$EMAIL"
        exit $STATUS_SMOKE
    fi
fi

# ---- step 3: full eval, all 1984 test images x 4 methods (~1.5h at ps4 timings) ----
echo "=== FULL features=$FEATURES ==="
python -u -m exp.upsamplers.eval_pa2pa \
    --features "$FEATURES" \
    --head_ckpt "$HEAD_CKPT" \
    --retrain \
    --methods "$METHODS"
STATUS=$?
echo "=== EXIT full status=$STATUS ==="

# Email the outcome: the per-method summary table + log tail.
{
    echo "features: $FEATURES"
    echo "smoke exit status: $STATUS_SMOKE"
    echo "full exit status: $STATUS"
    echo "--- results (mIoU / delta vs lr_bilinear) ---"
    grep -aE "^=== (SMOKE|FULL|EXIT)|mIoU=|^method |^lr_bilinear|^upa |^upma |^anyup " "$LOG" \
        || echo "(no metrics found; see log)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] upsampler eval $FEATURES" "$EMAIL"

exit $STATUS
