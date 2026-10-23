#!/bin/bash
# Time selection with space fixed, the HIGH direction (companion of mixres_time_arms.sh, same
# checkpoints and budgets): every spatial location gets crops of
#   entropy   its highest-entropy seasons (the in-training model's own season maps)
#   oracle    its highest-loss seasons (the worse-half oracle at every location)
# keep half (two seasons) / one. PASTIS s16m8 + s8m4, 3-link chains. Mail off.
#   [DRY=1] bash scripts/slurm/mixres_time_high.sh
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
declare -A ARM=(
  [oracle]="select.scorer=label select.allocator=every_location select.keep=$K select.season_order=high"
  [entropy]="select.scorer=entropy select.allocator=every_location select.keep=$K select.season_order=high")
for A in oracle entropy; do
  prev=""; for i in 1 2 3; do
    prev=$(sub "mxth${A}_${K}_${S}_$i" "$prev" "${COMMON[@]}" ${ARM[$A]}); done
done
done
echo "$S queued"
done
