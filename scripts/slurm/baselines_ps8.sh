#!/bin/bash
# Fixed-ps8 OlmoEarth tiny FFT baselines (exp.olmoearth.train), the compute-matched fixed arm for
# mixed resolution, as chains of <=3 h resumable jobs (lr 1e-3: best for every tiny FFT config):
#
#   [DRY=1] [LINKS=n] bash scripts/slurm/baselines_ps8.sh <pastis|biomassters|yieldsat_ml24>
#
#   ps8ts64_ts  series, 8 px patches over the 64 px tile   (12 x 64 x 4 = 3072 tokens, one sequence)
#   ps8ts8_ts   series, each 8 px tile its own sequence    (64 sequences of 12 x 4 tokens)
#   ps8ts8_pd   per timestep, each 8 px tile and month alone
set -euo pipefail
TASK="${1:?usage: bash scripts/slurm/baselines_ps8.sh <pastis|biomassters|yieldsat_ml24>}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$TASK" in
  pastis)      EP=50; LINKS="${LINKS:-5}"; CPUS=9 ;;
  biomassters) EP=10; LINKS="${LINKS:-8}"; CPUS=17 ;;
  yieldsat_ml24) EP=50; LINKS="${LINKS:-3}"; CPUS=9 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
for cfg in "series 8 64 ps8ts64_ts" "series 8 8 ps8ts8_ts" "per_timestep 8 8 ps8ts8_pd"; do
  set -- $cfg
  prev=""
  for i in $(seq 1 "$LINKS"); do
    cmd=(sbatch --parsable -J "fix_${TASK:0:3}_$4_$i" --time=3:00:00 --cpus-per-task="$CPUS"
         --exclude=kn117,kn104,kn101 ${prev:+--dependency=afterany:$prev}
         scripts/slurm/run.sh exp.olmoearth.train --task "$TASK" --temporal_mode "$1"
         --patch_size "$2" --tile_size "$3" --mode fft --lrs 1e-3 --epochs "$EP"
         --model_size tiny --batch_size 16 --num_workers $((CPUS - 1)) --max_hours 2.5)
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; prev="DRY$i"; else prev=$("${cmd[@]}"); fi
  done
  echo "$4: chain of $LINKS ending $prev"
done
