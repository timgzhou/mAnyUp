#!/bin/bash
# Learned windows (arm A's frozen linear head), fixed season rule -- one change each vs arm A:
#   all      every season of each picked window (n windows x all months)
#   prior    each window's half of seasons the oracle usually picks (per-season-index prior
#            from the train split, exp/upsamplers/season_rule.py)
#   random   a random half (control)
# per (task, setup, n): season_rule.py (prior + frozen test, learned/oracle windows x rules),
# then FT with each rule (warm from the better-half ckpt; run names _rklinear_sr{rule}_warm_).
# Mail off.   [DRY=1] bash scripts/slurm/season_rule_arms.sh
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
    --pos center --k_frac "$KX" "${GEO[@]}" --init_ckpt "$CK" --score_src learned
    --rank_head "${CK}_rankhead" --rank_arch linear)
ev=$(sub "sr_${T:0:3}_${S}_n${N}" "" exp.upsamplers.season_rule --task "$T" --setup "$S" --n "$N")
for R in all prior random; do
  extra=(); [ "$R" = prior ] && extra=(--season_prior "${CK}_season_prior")
  prev=$ev; for i in $(seq 1 "$LINKS"); do prev=$(sub "ftsr${R}_${T:0:3}_${S}_n${N}_$i" "$prev" "${FT[@]}" --season_rule "$R" "${extra[@]}"); done
done
echo "$T $S n$N queued"
done
done
done
