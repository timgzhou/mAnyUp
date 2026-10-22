#!/bin/bash
# Ranking-noise tolerance of the better-half seasons oracle (exp/upsamplers/oracle_noise.py): its
# frozen checkpoint tested with label scores corrupted at rho in {.9,.75,.5,.25,0} on the location
# ranking, the season ranking, or both, + no crops / season_random / season_grid on the same
# weights. One job per (task, setup, n). Mail off.
#   [DRY=1] bash scripts/slurm/oracle_noise.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
for T in pastis biomassters yieldsat_raw; do
for S in s16m8 s8m4; do
NS=$([ "$S" = s16m8 ] && echo "8 16 32" || echo "32 64 128")
for N in $NS; do
  cmd=(sbatch --parsable -J "noise_${T:0:3}_${S}_n${N}" --time=3:00:00 --cpus-per-task=9
       --exclude=$EXC scripts/slurm/run.sh exp.upsamplers.oracle_noise --task "$T" --setup "$S" --n "$N")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else echo "$T $S n$N: $("${cmd[@]}")"; fi
done; done; done
