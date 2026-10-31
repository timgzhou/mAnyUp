#!/bin/bash
# Space then time on the FROZEN time label-oracle models, evaluation only (train.epochs=0):
#   ent_learned  n highest-entropy windows, each with the seasons of the learned temporal scorer
#                trained on that frozen model (mixres_frozen_oracle_time.sh; train.init_selector)
#   ent_oracle   the same windows, the true lowest-loss season(s) (the time ceiling)
#   ent_random   the same windows, random season(s)
#   rand_random  n random windows, random season(s)
# keep one / half (each on its own oracle model), PASTIS s16m8 n=8/16/32, s8m4 n=32/64/128.
# ent_learned waits for the learned scorer's chain (AFTER_<keep>_<setup> = its last job id).
#   [DRY=1] bash scripts/slurm/mixres_frozen_spacetime.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <after job or ""> <overrides...>
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; else "${cmd[@]}" >/dev/null; fi
}
RES=(select.residual.arch=xattn_conv select.residual.train=oracle select.residual.prior=none)
for S in s16m8 s8m4; do
if [ "$S" = s16m8 ]; then GEO=(select.crop_px=8 select.loc_stride=8 model.lr_patch=16); PS=16; G=c8s8
  NS="8 16 32"
else GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8); PS=8; G=c4; NS="32 64 128"; fi
for K in one half; do
  ORC=$(ls checkpoints | grep -E "^mixres_pastis_ps${PS}_label-every_location-${K}_n0_${G}_warm_seed0_[0-9a-f]{8}\.pt$" | sed 's/\.pt$//')
  LRN=$(ls checkpoints | grep -E "^mixres_pastis_ps${PS}_season_residual-every_location-${K}-resoraclexattn_conv-noneprior_n0_${G}_warm_frozen_seed0_[0-9a-f]{8}(_state)?\.pt$" | head -1 | sed -E 's/(_state)?\.pt$//')
  [ -n "$ORC" ] && [ -n "$LRN" ] || { echo "missing oracle / learned $S $K" >&2; exit 1; }
  var="AFTER_${K}_${S}"; AFTER=${!var:-}
  for N in $NS; do
    C=(task=pastis select.n=$N "${GEO[@]}" train.init_ckpt=$ORC train.freeze_model=true
       train.epochs=0 select.allocator=top select.time_allocator=every_location select.keep=$K)
    sub "mxfst_entlearn_${K}_${S}_n$N" "$AFTER" "${C[@]}" select.scorer=entropy \
        select.time_scorer=season_residual "${RES[@]}" train.init_selector=$LRN
    sub "mxfst_entorc_${K}_${S}_n$N" "" "${C[@]}" select.scorer=entropy select.time_scorer=label
    sub "mxfst_entrand_${K}_${S}_n$N" "" "${C[@]}" select.scorer=entropy \
        select.time_scorer=season_random
    sub "mxfst_randrand_${K}_${S}_n$N" "" "${C[@]}" select.scorer=random \
        select.time_scorer=season_random
  done
done
echo "$S queued"
done
