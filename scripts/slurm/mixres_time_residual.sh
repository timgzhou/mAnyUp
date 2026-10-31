#!/bin/bash
# Learned residual on the season-entropy ranking (selectors.SeasonResidualScorer, scorer
# season_residual): season score = z(window entropy over the location's seasons) + r, r per
# (season, location) from the spatial residual's input sets, per season:
#   conv        the season's head-map features + its token
#   hr          the raw pixels of the season's months at the window
#   xattn       those pixels as queries over the tile's season tokens
#   xattn_conv  xattn + conv
# trained by MODES (default e2e): e2e = the task loss through each picked crop's gate
# logsigmoid(-r) - logsigmoid(0) + Gumbel exploration; oracle = a ListNet ranking of each
# location's seasons by their actual loss (lowest first), no gate; both = oracle fading into e2e.
# r = 0 at the start = the lowest-entropy time arm (exp.mixres.check_season_residual).
#   MODES=oracle bash scripts/slurm/mixres_time_residual.sh
# PRIOR=none drops the entropy term: score = r alone (ties broken at random at the start).
#   MODES=oracle PRIOR=none bash scripts/slurm/mixres_time_residual.sh
# keep one / half, PASTIS s16m8 + s8m4, same checkpoints and budgets, 3-link chains. Mail off.
#   [DRY=1] bash scripts/slurm/mixres_time_residual.sh
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
for MD in ${MODES:-e2e}; do
TAG=$([ "$MD" = e2e ] && echo "" || echo "$MD")$([ "${PRIOR:-entropy}" = entropy ] && echo "" || echo "np")
for AR in conv hr xattn xattn_conv; do
  prev=""; for i in 1 2 3; do
    prev=$(sub "mxtres${TAG}${AR}_${K}_${S}_$i" "$prev" "${COMMON[@]}" select.scorer=season_residual \
           select.residual.arch=$AR select.residual.train=$MD select.allocator=every_location \
           select.keep=$K ${PRIOR:+select.residual.prior=$PRIOR}); done
done
done
done
echo "$S queued"
done
