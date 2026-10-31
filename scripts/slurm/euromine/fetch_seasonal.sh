#!/bin/bash
# Seasonal S2 L2A composites for every EuroMineNet site (exp.euromine.fetch_seasonal): 132 sites x
# 38 quarters, sharded over CPU jobs, each a chain of 3 h resumable links. ~90 GB in
# data/EuroMineNet/seasonal. Mail off.
#   [DRY=1] [SHARDS=24] [LINKS=3] bash scripts/slurm/euromine/fetch_seasonal.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
export EMAIL=
SHARDS=${SHARDS:-24} LINKS=${LINKS:-3}
for s in $(seq 0 $((SHARDS - 1))); do
  dep=""
  for i in $(seq 1 "$LINKS"); do
    cmd=(sbatch --parsable -J "em_fetch_s${s}_$i" --gres=none --cpus-per-task=4 --mem=32G
         --time=3:00:00 ${dep:+--dependency=afterany:$dep}
         scripts/slurm/run.sh exp.euromine.fetch_seasonal --shard "$s" --num_shards "$SHARDS"
         --threads 12)
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; id=DRY; else id=$("${cmd[@]}"); fi
    dep=$id
  done
done
echo "$SHARDS shards x $LINKS links queued"
