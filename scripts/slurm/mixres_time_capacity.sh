#!/bin/bash
# Capacity of the season residual (residual.SeasonResidual), one change at a time on the best
# time arm (xattn_conv, oracle ListNet on the actual season-loss ranking, entropy prior: .523 /
# .540 at s16m8), then all four combined:
#   pconv   pixel encoder: 3x3 conv stem over the window patch instead of one linear map
#   mattn   months -> season: learned attention over the season's present months, not the mean
#   h256d2  head width 64 -> 256, one more 3x3 conv layer
#   sm2     2 transformer layers across each location's seasons before the final layer
#   all     the four together
# keep one / half, PASTIS s16m8 + s8m4, same checkpoints and budgets, 3-link chains. Mail off.
# PRIOR=none drops the entropy term (score = r alone; the better setting at s16m8).
#   [DRY=1] [ARMS="pconv sm2"] [PRIOR=none] bash scripts/slurm/mixres_time_capacity.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
arm() {   # arm <name> -> the overrides
  case $1 in
    pconv)  echo select.residual.pix_enc=conv ;;
    mattn)  echo select.residual.month_pool=attn ;;
    h256d2) echo select.residual.hidden=256 select.residual.depth=2 ;;
    sm2)    echo select.residual.season_mix=2 ;;
    all)    echo select.residual.pix_enc=conv select.residual.month_pool=attn \
                 select.residual.hidden=256 select.residual.depth=2 select.residual.season_mix=2 ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}
for S in s16m8 s8m4; do
if [ "$S" = s16m8 ]; then
  GEO=(select.crop_px=8 select.loc_stride=8 model.lr_patch=16)
  LOC=inject_fft_pastis_tiny_oracle_L0_k0.125_loc_native_set_cpx8_st8_poscenter_lr0.001_ep50
else
  GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8)
  LOC=inject_fft_pastis_tiny_oracle_L0_k0.5_loc_native_set_poscenter_lrps8_lr0.001_ep50
fi
[ -f "checkpoints/$LOC.pt" ] || { echo "missing $LOC" >&2; exit 1; }
COMMON=(task=pastis select.n=0 "${GEO[@]}" train.epochs=50 train.max_hours=2.5 train.init_ckpt=$LOC
        select.scorer=season_residual select.residual.arch=xattn_conv select.residual.train=oracle
        select.allocator=every_location ${PRIOR:+select.residual.prior=$PRIOR} ${SMOKE:+train.limit_batches=3 train.seed=999})
for K in half one; do
for A in ${ARMS:-pconv mattn h256d2 sm2 all}; do
  prev=""; for i in $(seq 1 ${LINKS:-3}); do
    prev=$(sub "mxtcap${PRIOR:+np}${A}_${K}_${S}_$i" "$prev" "${COMMON[@]}" select.keep=$K $(arm $A)); done
done
done
echo "$S queued"
done
