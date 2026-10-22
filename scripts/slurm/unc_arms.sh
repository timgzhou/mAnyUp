#!/bin/bash
# Uncertainty selectors from the per-season predictions (UncHead): spatially the windows of
# highest total uncertainty, temporally each window's lowest-uncertainty half of seasons.
#   unc_ent   untrained: entropy (seg) / variance of the prediction inside each crop (regr.)
#   unc0      untrained: season disagreement (mutual information / variance of season means)
#   unc_nll   trained copy of the head, per-season proper scoring rule (CE / Gaussian NLL)
#   unc_rank  trained copy of the head, ListNet ranking of the label losses
#   unc_both  both losses
# each: learned_select --arch X on the frozen better-half ckpt, then FT with that head's picks
# (run names _rk{X}_warm_). Mail off.   [DRY=1] bash scripts/slurm/unc_arms.sh
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
KF=$(python3 -c "print(f'{$N / 256:.3g}')"); KX=$(python3 -c "print($N / 256)")
CK="inject_fft_${T}_tiny_oracle_bothalf_L0_k${KF}_loc_native_set${TAG}0.001_ep${EP[$T]}"
FT=(exp.upsamplers.finetune_inject --task "$T" --epochs "${EP[$T]}" --num_workers 8 --max_hours 2.5
    --selector oracle_bothalf --select_unit location --inject_layer 0 --readout native_set
    --pos center --k_frac "$KX" "${GEO[@]}" --init_ckpt "$CK" --score_src learned)
for AR in unc_ent unc0 unc_nll unc_rank unc_both; do
  b=""; for i in $(seq 1 $([ "${AR}" = unc_ent ] || [ "${AR}" = unc0 ] && echo 1 || echo 2)); do
    b=$(sub "rk${AR}_${T:0:3}_${S}_n${N}_$i" "$b" exp.upsamplers.learned_select --task "$T" --setup "$S" --n "$N" --arch "$AR"); done
  prev=$b; for i in $(seq 1 "$LINKS"); do prev=$(sub "ft${AR}_${T:0:3}_${S}_n${N}_$i" "$prev" "${FT[@]}" --rank_head "${CK}_rankhead_${AR}" --rank_arch "$AR"); done
done
echo "$T $S n$N queued"
done
done
done
