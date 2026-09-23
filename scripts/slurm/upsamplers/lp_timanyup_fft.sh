#!/bin/bash
#SBATCH --job-name=lp_ta_fft
#SBATCH --account=aip-gpleiss
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/lp_ta_fft_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# LP a timAnyUp checkpoint on FINE-TUNED (FFT) feature caches.
#
# The stored transform head maps F_hrlc into F_lrhc_up space for ONE backbone. Under a
# fine-tuned encoder both endpoints move, so that alignment is stale and must be refit --
# zero-shot transfer is not expected to work. The upsampler and query head stay frozen: they
# read guidance and the LR map, not F_hrlc, so they transfer.
#
#   CKPT        timAnyUp checkpoint
#   FEATURES    FFT F_lrhc cache
#   HRLC        FFT F_hrlc cache
#   TF_EPOCHS   0 = refit transform jointly with the probe; N>0 = two-stage (transform only
#               for N epochs, then probe only)
#   MODES       space-separated probe modes (default "timanyup timanyup_t" -- both, to match
#               the pretrained LP sweep)
#   EPOCHS      total LP epochs (default 32)

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1
# BOTH probe modes by default. The pretrained LP sweep runs timanyup and timanyup_t, so an
# FFT run that quietly did only one left the comparison lopsided -- the two modes differ by
# ~+0.07 mIoU, far more than the effects being measured here.
MODES="${MODES:-timanyup timanyup_t}"
EPOCHS="${EPOCHS:-32}"
TF_EPOCHS="${TF_EPOCHS:-0}"
# AnyUp's LearnedFeatureUnification builds a (B*T, out_ch*C, H, W) tensor -- with C=768 and a
# 32x32 target grid that passes 2^31 elements at the LP default batch of 32, and CUDA's
# 32-bit indexed kernels fail ("canUse32BitIndexMath"). Scale the batch to the TARGET grid
# instead of remembering it per run: 4 is measured-safe for a 32x32 target (ps2), 8 for
# 16x16 (ps4) and coarser.
if [ -z "${BATCH_SIZE:-}" ]; then
    case "$HRLC" in
        *_ps2_*) BATCH_SIZE=4 ;;
        *)       BATCH_SIZE=8 ;;
    esac
fi

cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh

LOG="logs/lp_ta_fft_${SLURM_JOB_ID}.out"
STATUS=0
for MODE in $MODES; do
    echo "=== FFT LP :: features=$FEATURES hrlc=$HRLC mode=$MODE tf_epochs=$TF_EPOCHS ==="
    python -u -m exp.pastis.lp_cached_features \
        --features "$FEATURES" --head_mode "$MODE" --timanyup_ckpt "$CKPT" \
        --hrlc_features "$HRLC" --timanyup_train_transform \
        --timanyup_transform_epochs "$TF_EPOCHS" --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" ${EXTRA:-}
    [ $? -ne 0 ] && STATUS=1
done
{
    echo "features=$FEATURES hrlc=$HRLC modes=$MODES tf_epochs=$TF_EPOCHS batch=$BATCH_SIZE status=$STATUS"
    grep -aE "^TEST \{|refitting|NOTE:" "$LOG" | tail -10
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] LP timAnyUp FFT $TF_EPOCHS-stage" "$EMAIL"
