#!/bin/bash
# Measured inference MACs for every arm (exp.bench.flop_trace; finetune_inject --flops).
# results/flops/commands.txt holds one "module args" line per arm; job i of N runs the lines
# whose index is i mod N, each as its own short process (a failure is logged, not fatal).
#
#   sbatch [...] scripts/slurm/flops.sh <i> <N> [commands file]
#   -> rows in results/<task>/flops.csv (fixed baselines) and results/<task>/inject_flops.csv
#SBATCH --account=aip-gpleiss
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source env_setup/env_olmo.sh
export PYTHONPATH=$PWD:$PYTHONPATH
I="$1"; N="$2"; CMDS="${3:-results/flops/commands.txt}"; n=0
while IFS= read -r line; do
  if [ $((n % N)) -eq "$I" ]; then
    python -u -m $line 2>&1 | grep -a "^FLOPS\|Error\|Traceback" || echo "FAILED: $line"
  fi
  n=$((n + 1))
done < "$CMDS"
