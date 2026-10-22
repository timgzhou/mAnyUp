#!/bin/bash
# Selectors that process a window's seasons together (one change at a time, then combined),
# each: rank head trained on the FROZEN better-half oracle checkpoint (learned_select --arch X),
# then the model fine-tuned with that frozen head's picks (as arms B -> C):
#   sattn   self-attention over [location token, season tokens]            (vs linear: S)
#   mattn   each month's raw patch embedding queries the season tokens      (vs linear: M)
#   smattn  both                                                            (S + M)
# plus select_diag (location vs season ranking, oracle/learned mixes on frozen + arm A) and
# seeds 1, 2 of arm A and season grid at the lowest budget. Mail off.
#   [DRY=1] bash scripts/slurm/rank_attn_arms.sh
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
LOW=${NS%% *}
for N in $NS; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')"); KX=$(python3 -c "print($N / 256)")
CK="inject_fft_${T}_tiny_oracle_bothalf_L0_k${KF}_loc_native_set${TAG}0.001_ep${EP[$T]}"
FT=(exp.upsamplers.finetune_inject --task "$T" --epochs "${EP[$T]}" --num_workers 8 --max_hours 2.5
    --selector oracle_bothalf --select_unit location --inject_layer 0 --readout native_set
    --pos center --k_frac "$KX" "${GEO[@]}" --init_ckpt "$CK" --score_src learned)
for AR in sattn mattn smattn; do
  b=""; for i in 1 2; do b=$(sub "rk${AR}_${T:0:3}_${S}_n${N}_$i" "$b" exp.upsamplers.learned_select --task "$T" --setup "$S" --n "$N" --arch "$AR"); done
  prev=$b; for i in $(seq 1 "$LINKS"); do prev=$(sub "ft${AR}_${T:0:3}_${S}_n${N}_$i" "$prev" "${FT[@]}" --rank_head "${CK}_rankhead_${AR}" --rank_arch "$AR"); done
done
sub "diag_${T:0:3}_${S}_n${N}" "" exp.upsamplers.select_diag --task "$T" --setup "$S" --n "$N" >/dev/null
if [ "$N" = "$LOW" ]; then
  if [ "$S" = s16m8 ]; then SG=(--crop_px 8); else SG=(--lr_patch 8 --crop_px 4); fi
  for SD in 1 2; do
    prev=""; for i in $(seq 1 "$LINKS"); do prev=$(sub "seedA_${T:0:3}_${S}_n${N}_s${SD}_$i" "$prev" "${FT[@]}" --rank_head "${CK}_rankhead" --rank_arch linear --seed "$SD"); done
    prev=""; for i in $(seq 1 "$LINKS"); do prev=$(sub "seedG_${T:0:3}_${S}_n${N}_s${SD}_$i" "$prev" exp.upsamplers.finetune_inject --task "$T" --epochs "${EP[$T]}" --num_workers 8 --max_hours 2.5 --selector season_grid --inject_layer 0 --select_unit location "${SG[@]}" --k_frac "$KX" --readout native_set --pos center --seed "$SD"); done
  done
fi
echo "$T $S n$N queued"
done
done
done
