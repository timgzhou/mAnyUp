#!/bin/bash
# Every model trained at a (task, setup, n) -- better-half oracle, the fine-tuned learned-selector
# arms (linear/conv/sattn/mattn/smattn), season grid (+ seeds at the lowest n) -- tested with oracle,
# learned, season_grid and no picks (exp/upsamplers/pick_matrix.py). One resumable job per
# (task, setup, n), chained twice (BioMassters evals ~6 min each). Mail off.
#   [DRY=1] bash scripts/slurm/pick_matrix.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
for T in pastis biomassters yieldsat_raw; do
for S in s16m8 s8m4; do
NS=$([ "$S" = s16m8 ] && echo "8 16 32" || echo "32 64 128")
for N in $NS; do
  prev=""
  for i in 1 2; do
    cmd=(sbatch --parsable -J "pmx_${T:0:3}_${S}_n${N}_$i" --time=3:00:00 --cpus-per-task=9
         --exclude=$EXC ${prev:+--dependency=afterany:$prev}
         scripts/slurm/run.sh exp.upsamplers.pick_matrix --task "$T" --setup "$S" --n "$N")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; prev=DRY; else prev=$("${cmd[@]}"); fi
  done
  echo "$T $S n$N queued"
done; done; done
