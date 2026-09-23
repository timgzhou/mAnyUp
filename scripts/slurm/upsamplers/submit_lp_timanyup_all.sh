#!/bin/bash
# Submit ONE lp_timanyup job per timAnyUp checkpoint, so the sweep runs in parallel.
#
# A full 32-epoch LP takes ~66 min per probe mode, so all checkpoints x both modes in a single
# job would blow the time limit. One job per checkpoint keeps each under ~2.5h and finishes the
# whole sweep in roughly that wall-clock time instead of ~15h serially.
#
#   bash scripts/slurm/upsamplers/submit_lp_timanyup_all.sh
#   CKPT_ROOT=checkpoints/timanyup EPOCHS=16 bash scripts/slurm/upsamplers/submit_lp_timanyup_all.sh
set -u
CKPT_ROOT="${CKPT_ROOT:-checkpoints/timanyup}"
EPOCHS="${EPOCHS:-32}"
MODES="${MODES:-timanyup timanyup_t}"

# mindepth 2: the bare CKPT_ROOT/*.pth files are stray one-off smoke checkpoints.
CKPTS=$(find "$CKPT_ROOT" -mindepth 2 -name "*.pth" | sort)
[ -z "$CKPTS" ] && { echo "no checkpoints under $CKPT_ROOT"; exit 1; }

n=0
for CK in $CKPTS; do
    JID=$(sbatch --parsable --export=ALL,CKPTS="$CK",EPOCHS="$EPOCHS",MODES="$MODES" \
          scripts/slurm/upsamplers/lp_timanyup.sh)
    echo "submitted $JID  $(basename "$CK")"
    n=$((n+1))
done
echo "$n jobs submitted ($MODES, $EPOCHS epochs each)"
