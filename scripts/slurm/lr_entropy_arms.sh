#!/bin/bash
# Label-free picks from the LR-only model's prediction entropy (--score_src lr_entropy): windows =
# highest entropy of its pooled prediction, seasons = each window's lowest-entropy half of its own
# per-season maps. PASTIS, both setups. Per (setup, n): lr_entropy_eval.py on the frozen
# better-half ckpt (oracle / lr_entropy / the two mixes + overlap), then FT warm from that ckpt
# with the lr_entropy picks (run names _lrent_warm_). Arm "all": the same windows, every month
# cropped (--season_rule all, no season choice), warm from the location oracle (true-loss windows,
# every month; run names _lrent_srall_warm_). EVAL=0 skips the frozen evaluation. Mail off.
# (2026-10-19: the first FT round never set score_src -- it trained on oracle picks; fixed.)
#   [DRY=1] bash scripts/slurm/lr_entropy_arms.sh
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
if [ "$S" = s16m8 ]; then NS="8 16 32"; GEO=(--lr_patch 16 --crop_px 8 --loc_stride 8); TAG=_cpx8_st8_poscenter_lr
  LTAG=_cpx8_st8_poscenter_lr
  SC=inject_fft_pastis_tiny_none_native_set_lr0.001_ep50
else NS="32 64 128"; GEO=(--lr_patch 8 --crop_px 4); TAG=_poscenter_lrps8_lr
  LTAG=_poscenter_lrps8_lr
  SC=inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50; fi
for N in $NS; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')"); KX=$(python3 -c "print($N / 256)")
CK="inject_fft_${T}_tiny_oracle_bothalf_L0_k${KF}_loc_native_set${TAG}0.001_ep${EPOCHS}"
LOC="inject_fft_${T}_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_ep${EPOCHS}"
[ "${EVAL:-1}" = 1 ] && sub "lreval_${S}_n${N}" "" exp.upsamplers.lr_entropy_eval --task $T --setup $S --n $N >/dev/null
FT=(exp.upsamplers.finetune_inject --task $T --epochs $EPOCHS --num_workers 8 --max_hours 2.5
    --selector oracle_bothalf --select_unit location --inject_layer 0 --readout native_set
    --pos center --k_frac "$KX" "${GEO[@]}" --score_src lr_entropy --scorer "$SC")
prev=""; for i in 1 2 3; do prev=$(sub "ftlrent_${S}_n${N}_$i" "$prev" "${FT[@]}" --init_ckpt "$CK"); done
prev=""; for i in 1 2 3; do prev=$(sub "ftlrall_${S}_n${N}_$i" "$prev" "${FT[@]}" --init_ckpt "$LOC" --season_rule all); done
echo "$S n$N queued"
done
done
