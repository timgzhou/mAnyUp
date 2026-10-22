#!/bin/bash
# Self-entropy all-months (--score_src entropy): windows ranked by the entropy of the IN-TRAINING
# model's own crop-free prediction (as the label oracle scores its own loss), every month cropped,
# warm from the location oracle. Replaces the frozen LR-only --scorer of the _lrent_ arms. One
# change each, then combined:
#   base        self-entropy, no residual                       run names _sent_srall_warm_
#   hr          + residual r from each window's raw monthly pixels
#   xattn       + r from those pixels as queries over the model's own season tokens
#   xattn_conv  + r from xattn + the conv residual's inputs (probs, entropy, margin, tokens)
# residuals trained e2e (task loss through the crop gate + Gumbel), r = 0 at the start.
# PASTIS, both setups. Run names _sent_rese2e{arch}_srall_warm_. Mail off.
#   [DRY=1] bash scripts/slurm/self_entropy_arms.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <after job or ""> <args...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
T=pastis; EPOCHS=50
for S in s16m8 s8m4; do
if [ "$S" = s16m8 ]; then NS="8 16 32"; GEO=(--lr_patch 16 --crop_px 8 --loc_stride 8); LTAG=_cpx8_st8_poscenter_lr
else NS="32 64 128"; GEO=(--lr_patch 8 --crop_px 4); LTAG=_poscenter_lrps8_lr; fi
for N in $NS; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')"); KX=$(python3 -c "print($N / 256)")
LOC="inject_fft_${T}_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_ep${EPOCHS}"
FT=(exp.upsamplers.finetune_inject --task $T --epochs $EPOCHS --num_workers 8 --max_hours 2.5
    --selector oracle_bothalf --select_unit location --inject_layer 0 --readout native_set
    --pos center --k_frac "$KX" "${GEO[@]}" --score_src entropy
    --season_rule all --init_ckpt "$LOC")
prev=""; for i in 1 2 3; do prev=$(sub "ftsent_${S}_n${N}_$i" "$prev" "${FT[@]}"); done
for AR in hr xattn xattn_conv; do
  prev=""; for i in 1 2 3; do
    prev=$(sub "ftsent${AR}_${S}_n${N}_$i" "$prev" "${FT[@]}" --res_mode e2e --res_arch $AR); done
done
echo "$S n$N queued"
done
done
