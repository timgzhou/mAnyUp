#!/bin/bash
# Predictability of the oracle's selection quantity (exp.mixres.lossup_probe): LossUp stats /
# bilinear / manyup trained 10 epochs on frozen co-trained models, plus the untrained baselines
# (mode none: entropy and the learned season scorer). PASTIS s16m8 + s8m4. Mail off.
#   [DRY=1] bash scripts/slurm/mixres_lossup_probe.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
for S in s16m8 s8m4; do for M in none stats bilinear manyup; do
  cmd=(sbatch --parsable -J "mxlossup_${S}_$M" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
       scripts/slurm/run.sh exp.mixres.lossup_probe --setup $S --mode $M --epochs ${EPOCHS:-10})
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
done; done
