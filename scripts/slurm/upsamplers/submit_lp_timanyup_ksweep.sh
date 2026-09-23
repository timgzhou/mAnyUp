#!/bin/bash
# Sweep the timAnyUp lookup budget k for ONE checkpoint: one job per k, run in parallel.
#
# k is the method's central knob -- how many F_hrlc patches are fetched per sample. The two
# ends are controls, not just sweep points:
#   k=0     no lookups at all, so the blend is an identity on the upsampled map. This is plain
#           per-timestep mAnyUp and should line up with the anyup/ps4 numbers.
#   k=3072  T*H*W = every location, i.e. F_hrlc everywhere the mask could possibly pick.
# Everything between measures what the budget actually buys.
#
#   bash scripts/slurm/upsamplers/submit_lp_timanyup_ksweep.sh <ckpt.pth>
#   KS="0 512 3072" MODES=timanyup_t bash scripts/slurm/upsamplers/submit_lp_timanyup_ksweep.sh <ckpt>
set -u
CK="${1:?usage: $0 <timanyup checkpoint.pth>}"
KS="${KS:-0 64 256 512 1024 2048 3072}"
MODES="${MODES:-timanyup timanyup_t}"
EPOCHS="${EPOCHS:-32}"

[ -f "$CK" ] || { echo "no such checkpoint: $CK"; exit 1; }

n=0
for K in $KS; do
    JID=$(sbatch --parsable \
        --export=ALL,CKPTS="$CK",EPOCHS="$EPOCHS",MODES="$MODES",EXTRA="--timanyup_k $K" \
        scripts/slurm/upsamplers/lp_timanyup.sh)
    echo "submitted $JID  k=$K"
    n=$((n+1))
done
echo "$n jobs ($MODES, $EPOCHS epochs, ckpt $(basename "$CK"))"
