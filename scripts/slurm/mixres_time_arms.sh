#!/bin/bash
# Time selection with space fixed (the time counterpart of mixres_grid_controls.sh): EVERY spatial
# location gets crops; the arms differ only in which months, all warm from the location oracle at
# the same crop budget (s16m8 n=32, s8m4 n=128):
#   oracle    each location's lowest-loss seasons (the better-half oracle at every location)
#   entropy   each location's lowest-entropy seasons (the in-training model's own season maps)
#   attention each location's seasons the readout attends to most (selectors.ReadoutAttention)
#   random    random seasons
#   grid      every other (half) / every 4th (one) present month
# keep half: half the seasons (~6 of 12 months) | one: the single best season (~3 months).
# PASTIS s16m8 (64 ps8 locations) and s8m4 (256 ps4 locations), 3-link chains. Mail off.
#   [DRY=1] [ARMS="oracle entropy attention random grid"] bash scripts/slurm/mixres_time_arms.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
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
COMMON=(task=pastis select.n=0 "${GEO[@]}" train.epochs=50 train.max_hours=2.5 train.init_ckpt=$LOC)
for K in half one; do
STEP=$([ $K = half ] && echo 2 || echo 4)
declare -A ARM=(
  [oracle]="select.scorer=label select.allocator=every_location select.keep=$K"
  [entropy]="select.scorer=entropy select.allocator=every_location select.keep=$K"
  [attention]="select.scorer=attention select.allocator=every_location select.keep=$K"
  [random]="select.scorer=season_random select.allocator=every_location select.keep=$K"
  [grid]="select.scorer=none select.allocator=month_grid select.month_step=$STEP")
for A in ${ARMS:-oracle entropy attention random grid}; do
  prev=""; for i in 1 2 3; do
    prev=$(sub "mxt${A}_${K}_${S}_$i" "$prev" "${COMMON[@]}" ${ARM[$A]}); done
done
done
echo "$S queued"
done
