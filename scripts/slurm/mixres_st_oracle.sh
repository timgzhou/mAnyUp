#!/bin/bash
# Space + time label-oracle models (selectors.SpaceTime: label locations, each with its lowest-
# loss season(s)), the base for learning the selectors on a frozen model (mixres_st_learn.sh).
# keep one / half, PASTIS s16m8 n=8/16/32, s8m4 n=32/64/128, warm from the location oracle at
# k = n/256 (as mixres_spacetime.sh), 3-link chains; prints "<setup> <keep> <n> <last job id>".
#   [DRY=1] bash scripts/slurm/mixres_st_oracle.sh
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
if [ "$S" = s16m8 ]; then NS="8 16 32"; GEO=(select.crop_px=8 select.loc_stride=8 model.lr_patch=16)
  LTAG=_cpx8_st8_poscenter_lr
else NS="32 64 128"; GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8)
  LTAG=_poscenter_lrps8_lr; fi
for N in $NS; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')")
LOC="inject_fft_pastis_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_ep50"
[ -f "checkpoints/$LOC.pt" ] || { echo "missing $LOC" >&2; exit 1; }
for K in one half; do
  prev=""; for i in 1 2 3; do
    prev=$(sub "mxsto_${K}_${S}_n${N}_$i" "$prev" task=pastis select.n=$N "${GEO[@]}" \
           train.epochs=50 train.max_hours=2.5 train.init_ckpt=$LOC select.scorer=label \
           select.allocator=top select.time_scorer=label select.time_allocator=every_location \
           select.keep=$K); done
  echo "$S $K $N $prev"
done
done
done
