#!/bin/bash
# Matched content-blind controls for the entropy arms, all warm from the same location-oracle
# checkpoint as the entropy arms (so training length is not a confound), via exp.mixres.train:
#   grid          n lattice windows x every month        <- matches entropy all-months
#                 (legacy --selector grid --select_unit location, so far only trained cold)
#   season_grid   in every season n lattice windows (shifted per season) x that season's months
#   season_top    in every season its own n highest-entropy windows x that season's months
#                 (the season grid's allocation, windows chosen by entropy; new)
# Budget n windows x present months for all. PASTIS s16m8 n=8/16/32, s8m4 n=32/64/128, 3-link
# chains. Mail off.
#   [DRY=1] bash scripts/slurm/mixres_grid_controls.sh
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
if [ "$S" = s16m8 ]; then NS="8 16 32"; GEO=(select.crop_px=8 model.lr_patch=16); STRIDE=8
  LTAG=_cpx8_st8_poscenter_lr
else NS="32 64 128"; GEO=(select.crop_px=4 model.lr_patch=8); STRIDE=none
  LTAG=_poscenter_lrps8_lr; fi
for N in $NS; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')")
LOC="inject_fft_pastis_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_ep50"
[ -f "checkpoints/$LOC.pt" ] || { echo "missing $LOC" >&2; exit 1; }
COMMON=(task=pastis select.n=$N "${GEO[@]}" train.epochs=50 train.max_hours=2.5 train.init_ckpt=$LOC)
declare -A ARM=(
  [grid]="select.scorer=grid select.allocator=top select.loc_stride=none"
  [sgrid]="select.scorer=season_grid select.allocator=pairs select.loc_stride=none"
  [stop]="select.scorer=entropy select.allocator=season_top select.loc_stride=$STRIDE")
for A in grid sgrid stop; do
  prev=""; for i in 1 2 3; do
    prev=$(sub "mx${A}_${S}_n${N}_$i" "$prev" "${COMMON[@]}" ${ARM[$A]}); done
done
echo "$S n$N queued"
done
done
