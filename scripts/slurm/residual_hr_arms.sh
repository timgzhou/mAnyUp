#!/bin/bash
# HR-input residuals (ResidualScorer --res_arch) on the LR-entropy all-months arm (windows =
# z(LR-only scorer's window entropy) + r, every month cropped, warm from the location oracle; r
# starts at 0 = the plain arm). The conv residual (residual_arms.sh) only saw the scorer's own maps
# and tokens and changed nothing. One change each, then combined:
#   hr          r from each window's raw monthly pixels (the HR input the crops would encode)
#   xattn       those pixels as queries cross-attending to the scorer's season tokens
#   xattn_conv  xattn + the conv residual's inputs
# each trained e2e (task loss through the crop gate + Gumbel) and both (+ the oracle loss, weight
# 1 -> 0 over the first half). PASTIS, both setups. Run names _lrent_res{mode}{arch}_srall_warm_.
# Mail off.
#   [DRY=1] bash scripts/slurm/residual_hr_arms.sh
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
  SC=inject_fft_pastis_tiny_none_native_set_lr0.001_ep50
else NS="32 64 128"; GEO=(--lr_patch 8 --crop_px 4); LTAG=_poscenter_lrps8_lr
  SC=inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50; fi
for N in $NS; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')"); KX=$(python3 -c "print($N / 256)")
LOC="inject_fft_${T}_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_ep${EPOCHS}"
FT=(exp.upsamplers.finetune_inject --task $T --epochs $EPOCHS --num_workers 8 --max_hours 2.5
    --selector oracle_bothalf --select_unit location --inject_layer 0 --readout native_set
    --pos center --k_frac "$KX" "${GEO[@]}" --score_src lr_entropy --scorer "$SC"
    --season_rule all --init_ckpt "$LOC")
for AR in hr xattn xattn_conv; do for MD in e2e both; do
  prev=""; for i in 1 2 3; do
    prev=$(sub "ftres${MD}${AR}_${S}_n${N}_$i" "$prev" "${FT[@]}" --res_mode $MD --res_arch $AR); done
done; done
echo "$S n$N queued"
done
done
