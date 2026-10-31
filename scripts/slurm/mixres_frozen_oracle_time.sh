#!/bin/bash
# The time label-oracle models (label-every_location, trained with each location's lowest-loss
# season(s)), FROZEN; only the temporal scorer is trained on them (train.freeze_model):
#   learned   season_residual xattn_conv, oracle ListNet on the frozen model's per-season loss,
#             prior none (score = r alone), 50 epochs, the scorer's loss only
# and eval-only references on the same frozen model (train.epochs=0):
#   oracle (the ceiling, = the run's own test) / random seasons / lowest entropy / month grid
# keep one / half, PASTIS s16m8 + s8m4. Mail off.
#   [DRY=1] [SMOKE=1] bash scripts/slurm/mixres_frozen_oracle_time.sh
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
for S in s16m8 s8m4; do
if [ "$S" = s16m8 ]; then GEO=(select.crop_px=8 select.loc_stride=8 model.lr_patch=16); PS=16; G=c8s8
else GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8); PS=8; G=c4; fi
for K in one half; do
  ORC=$(ls checkpoints | grep -E "^mixres_pastis_ps${PS}_label-every_location-${K}_n0_${G}_warm_seed0_[0-9a-f]{8}\.pt$" | sed 's/\.pt$//')
  [ -n "$ORC" ] || { echo "missing oracle $S $K" >&2; exit 1; }
  COMMON=(task=pastis select.n=0 "${GEO[@]}" train.init_ckpt=$ORC train.freeze_model=true
          ${SMOKE:+train.limit_batches=3 train.seed=999})
  STEP=$([ "$K" = one ] && echo 4 || echo 2)
  prev=""; for i in 1 2; do
    prev=$(sub "mxfrz_learned_${K}_${S}_$i" "$prev" "${COMMON[@]}" train.epochs=${SMOKE:+2}${SMOKE:-50} \
           train.max_hours=2.5 select.scorer=season_residual select.allocator=every_location \
           select.keep=$K select.residual.arch=xattn_conv select.residual.train=oracle \
           select.residual.prior=none); done
  sub "mxfrz_oracle_${K}_${S}" "" "${COMMON[@]}" train.epochs=0 select.scorer=label \
      select.allocator=every_location select.keep=$K >/dev/null
  sub "mxfrz_random_${K}_${S}" "" "${COMMON[@]}" train.epochs=0 select.scorer=season_random \
      select.allocator=every_location select.keep=$K >/dev/null
  sub "mxfrz_entropy_${K}_${S}" "" "${COMMON[@]}" train.epochs=0 select.scorer=entropy \
      select.allocator=every_location select.keep=$K >/dev/null
  sub "mxfrz_grid_${K}_${S}" "" "${COMMON[@]}" train.epochs=0 select.scorer=none \
      select.allocator=month_grid select.month_step=$STEP >/dev/null
done
echo "$S queued"
done
