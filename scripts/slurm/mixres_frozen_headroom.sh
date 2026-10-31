#!/bin/bash
# Spatial headroom on the FROZEN time label-oracle models (evaluation only, train.epochs=0), next
# to mixres_frozen_spacetime.sh's entropy-space arms:
#   orc_orc     oracle locations (highest pooled loss) + oracle seasons: the ceiling
#   orc_learn   oracle locations + the learned temporal scorer of that frozen model
#   rand_orc    random locations + oracle seasons
# keep one / half, PASTIS s16m8 n=8/16/32, s8m4 n=32/64/128. Mail off.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {
  local name=$1; shift
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; else "${cmd[@]}" >/dev/null; fi
}
RES=(select.residual.arch=xattn_conv select.residual.train=oracle select.residual.prior=none)
for S in s16m8 s8m4; do
if [ "$S" = s16m8 ]; then GEO=(select.crop_px=8 select.loc_stride=8 model.lr_patch=16); PS=16; G=c8s8
  NS="8 16 32"
else GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8); PS=8; G=c4; NS="32 64 128"; fi
for K in one half; do
  ORC=$(ls checkpoints | grep -E "^mixres_pastis_ps${PS}_label-every_location-${K}_n0_${G}_warm_seed0_[0-9a-f]{8}\.pt$" | sed 's/\.pt$//')
  LRN=$(ls checkpoints | grep -E "^mixres_pastis_ps${PS}_season_residual-every_location-${K}-resoraclexattn_conv-noneprior_n0_${G}_warm_frozen_seed0_[0-9a-f]{8}\.pt$" | sed 's/\.pt$//')
  [ -n "$ORC" ] && [ -n "$LRN" ] || { echo "missing oracle / learned $S $K" >&2; exit 1; }
  for N in $NS; do
    C=(task=pastis select.n=$N "${GEO[@]}" train.init_ckpt=$ORC train.freeze_model=true
       train.epochs=0 select.allocator=top select.time_allocator=every_location select.keep=$K)
    sub "mxhead_orcorc_${K}_${S}_n$N" "${C[@]}" select.scorer=label select.time_scorer=label
    sub "mxhead_orclearn_${K}_${S}_n$N" "${C[@]}" select.scorer=label \
        select.time_scorer=season_residual "${RES[@]}" train.init_selector=$LRN
    sub "mxhead_randorc_${K}_${S}_n$N" "${C[@]}" select.scorer=random select.time_scorer=label
  done
done
echo "$S queued"
done
