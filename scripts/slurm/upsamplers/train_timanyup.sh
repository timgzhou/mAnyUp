#!/bin/bash
#SBATCH --job-name=timanyup
#SBATCH --account=aip-gpleiss
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/timanyup_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Train a timAnyUp upsampler: two CHEAP feature maps (low-res/high-context + high-res/low-
# context) -> one EXPENSIVE high-res/high-context target. Configure via --export env vars:
#   LRHC_CFG    low-res HIGH-context input   (default oe_base_s2_ps16_tile64)
#   HRLC_CFG    high-res LOW-context input   (default oe_base_s2_ps4_tile4_single)
#   HRHC_CFG    high-res HIGH-context TARGET (default oe_base_s2_ps4_tile64)
#   K           per-sample lookup budget     (default 512)
#   EPOCHS      (default 20)
#   BATCH_SIZE  (default 4; effective batch is BATCH_SIZE*T frames, T=12 -> 48)
#   LR          learning rate (default 1e-3)
#   QUERY_INPUT bilinear|upsampled (default bilinear)
#   EXTRA       extra args passed verbatim (e.g. "--lambda_query 0.5")
# Features are staged to $SLURM_TMPDIR automatically. Emails at start and finish.
#
# Examples:
#   sbatch scripts/slurm/upsamplers/train_timanyup.sh
#   sbatch --export=ALL,K=256 scripts/slurm/upsamplers/train_timanyup.sh
#   sbatch --export=ALL,QUERY_INPUT=upsampled scripts/slurm/upsamplers/train_timanyup.sh   # ablation

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1

LRHC_CFG="${LRHC_CFG:-oe_base_s2_ps16_tile64}"
HRLC_CFG="${HRLC_CFG:-oe_base_s2_ps4_tile4_single}"
HRHC_CFG="${HRHC_CFG:-oe_base_s2_ps4_tile64}"
K="${K:-512}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
QUERY_INPUT="${QUERY_INPUT:-bilinear}"
EXTRA="${EXTRA:-}"

cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh

# Per-run output dir keyed by the arms + budget so parallel jobs don't clobber each other.
# BATCH_SIZE is in the path too: it changes the effective batch (BATCH_SIZE*T frames) and so
# the LR schedule, i.e. a genuinely different run -- two batch sizes must not share a dir.
# OUT_DIR_SUFFIX appends anything else that distinguishes a run (set it for one-off variants).
# QUERY_INPUT is in the path as well as the run tag: it is a different SELECTOR, not a tuning
# knob, so bilinear and upsampled runs of the same arms must be separable on disk at a glance.
OUT_DIR="checkpoints/timanyup/${LRHC_CFG}__k${K}__bs${BATCH_SIZE}__q${QUERY_INPUT}__to__${HRHC_CFG}${OUT_DIR_SUFFIX:-}"

ARGS="--lrhc_cfg $LRHC_CFG --hrlc_cfg $HRLC_CFG --hrhc_cfg $HRHC_CFG --k $K \
--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --query_input $QUERY_INPUT \
--stage_to_tmpdir --out_dir $OUT_DIR $EXTRA"
TAG="${LRHC_CFG} + ${HRLC_CFG} -> ${HRHC_CFG} (k=$K)"

LOG="logs/timanyup_${SLURM_JOB_ID}.out"
python -u -m exp.upsamplers.train_timanyup $ARGS
STATUS=$?

# Email the outcome. The lines that matter are the [test] comparisons: fused vs upsample-only
# (does the side channel earn its cost) and vs random-k (did the MASK learn anything).
{
    echo "args: $ARGS"
    echo "exit status: $STATUS"
    echo "out_dir: $OUT_DIR"
    echo "--- epoch summaries ---"
    grep -aE "^== epoch|^   \[test|^   \[train" "$LOG" | tail -30 || echo "(no summaries; see log)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] timAnyUp $TAG" "$EMAIL"
