#!/bin/bash
# Sampling ablation (exp.mixres.sample_eval) on every trained entropy + learned space-then-time
# checkpoint: top-k -> sampling P ~ exp(z / tau), tau 0.5 / 1 / 2 / inf, in space only, time
# only and both. One job per checkpoint -> results/pastis/mixres_sampling.csv. Mail off.
#   [DRY=1] bash scripts/slurm/mixres_sample_eval.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
for f in checkpoints/mixres_pastis_ps*_entropy-top+season_residual-*_seed0_????????.pt; do
  run=$(basename "$f" .pt)
  n=$(echo "$run" | grep -oE "_n[0-9]+_"); k=$(echo "$run" | grep -oE "location-(one|half)" | cut -d- -f2)
  ps=$(echo "$run" | grep -oE "_ps[0-9]+_")
  cmd=(sbatch --parsable -J "mxsamp${ps}${k}${n%_}" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
       scripts/slurm/run.sh exp.mixres.sample_eval --run "$run")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
done
