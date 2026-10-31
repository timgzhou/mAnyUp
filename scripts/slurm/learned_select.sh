#!/bin/bash
# Learned (season, window) selection on the FROZEN better-half oracle checkpoints
# (exp.upsamplers.learned_select): train a linear rank head, then test the same weights with
# label / season_sum / learned / entropy / season_random scoring -> results/<task>/learned_select.csv.
# s16m8 = ps16 seasons + ps8 crops at stride 8 (n = 8 / 16 / 32); s8m4 = ps8 seasons + ps4 crops
# (n = 32 / 64 / 128). Two chained <= 3 h jobs per run (a rerun reuses the saved head and skips
# logged modes). Mail off. FIT=val: fit the head on half of the val split instead of train (more
# epochs: the split is small), rows tagged fit_split=val.
#   [FIT=val] [DRY=1] bash scripts/slurm/learned_select.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
for T in ${TASKS:-pastis biomassters yieldsat_raw}; do
for S in s16m8 s8m4; do
for N in $([ "$S" = s16m8 ] && echo "8 16 32" || echo "32 64 128"); do
prev=""
for i in 1 2; do
  cmd=(sbatch --parsable -J "lsel${FIT:+$FIT}_${T:0:3}_${S}_n${N}_$i" --time=3:00:00 --cpus-per-task=9
       --exclude=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088 ${prev:+--dependency=afterany:$prev}
       scripts/slurm/run.sh exp.upsamplers.learned_select --task "$T" --setup "$S" --n "$N"
       ${FIT:+--fit_split $FIT --epochs 60})
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; prev="DRY"; else prev=$("${cmd[@]}"); fi
done
echo "$T $S n$N: ending $prev"
done
done
done
