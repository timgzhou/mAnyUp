#!/bin/bash
# Three arms on top of the frozen better-half oracle + learned rank head (learned_select.sh),
# one change at a time, then combined; same 18 (task, setup, n) as learned_select.sh:
#   A  finetune the model (warm start from the oracle checkpoint, full epochs) with the
#      LINEAR head's picks; head frozen          -> finetune_inject run name ..._rklinear_warm_...
#   B  CONV rank head (3 x 3 token neighbourhood + hidden layer) on the frozen model
#                                                -> learned_select.csv rows with arch = conv
#   C  B's conv head + A's finetuning            -> ..._rkconv_warm_...; starts after B's chain
# Mail off.   [DRY=1] bash scripts/slurm/learned_select_arms.sh
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
declare -A EP=([pastis]=50 [biomassters]=10 [yieldsat_raw]=50)
for T in pastis biomassters yieldsat_raw; do
LINKS=$([ "$T" = yieldsat_raw ] && echo 5 || echo 3)
for S in s16m8 s8m4; do
if [ "$S" = s16m8 ]; then NS="8 16 32"; GEO=(--lr_patch 16 --crop_px 8 --loc_stride 8); TAG=_cpx8_st8_poscenter_lr
else NS="32 64 128"; GEO=(--lr_patch 8 --crop_px 4); TAG=_poscenter_lrps8_lr; fi
for N in $NS; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')")
CK="inject_fft_${T}_tiny_oracle_bothalf_L0_k${KF}_loc_native_set${TAG}0.001_ep${EP[$T]}"
FT=(exp.upsamplers.finetune_inject --task "$T" --epochs "${EP[$T]}" --num_workers 8 --max_hours 2.5
    --selector oracle_bothalf --select_unit location --inject_layer 0 --readout native_set
    --pos center --k_frac "$(python3 -c "print($N / 256)")" "${GEO[@]}" --init_ckpt "$CK" --score_src learned)
# A
prev=""; for i in $(seq 1 "$LINKS"); do prev=$(sub "armA_${T:0:3}_${S}_n${N}_$i" "$prev" "${FT[@]}" --rank_head "${CK}_rankhead" --rank_arch linear); done
# B
b=""; for i in 1 2; do b=$(sub "armB_${T:0:3}_${S}_n${N}_$i" "$b" exp.upsamplers.learned_select --task "$T" --setup "$S" --n "$N" --arch conv); done
# C (after B)
prev=$b; for i in $(seq 1 "$LINKS"); do prev=$(sub "armC_${T:0:3}_${S}_n${N}_$i" "$prev" "${FT[@]}" --rank_head "${CK}_rankhead_conv" --rank_arch conv); done
echo "$T $S n$N: A/B/C queued"
done
done
done
