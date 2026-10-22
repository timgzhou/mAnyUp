#!/bin/bash
# Oracle lead = high-res detail or the pick pattern? exp/upsamplers/crop_content_test.py: oracle,
# arm A (oracle + learned picks) and season grid models with the picked crops' pixels real / band
# mean / random window / other sample. One resumable job per (task, setup, n), chained twice.
# Mail off.   [DRY=1] bash scripts/slurm/crop_content.sh
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
    cmd=(sbatch --parsable -J "ccx_${T:0:3}_${S}_n${N}_$i" --time=3:00:00 --cpus-per-task=9
         --exclude=$EXC ${prev:+--dependency=afterany:$prev}
         scripts/slurm/run.sh exp.upsamplers.crop_content_test --task "$T" --setup "$S" --n "$N")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; prev=DRY; else prev=$("${cmd[@]}"); fi
  done
  echo "$T $S n$N queued"
done; done; done
