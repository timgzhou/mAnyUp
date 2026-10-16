#!/bin/bash
# Fixed-resolution OlmoEarth grid for the performance-vs-FLOPs scatter: every (patch_size,
# tile_size) with tile_size in {patch_size, 32, 64} that nests into the 64 px native sample
# (tile_size % patch_size == 0, 64 % tile_size == 0), beyond what scripts/slurm/sweep.sh and
# baselines_ps8.sh already cover. tiny model, FFT only, lr 1e-3 (baselines_ps8.sh: best for
# every tiny FFT config). Small-ps/large-ts (ps2ts32, ps2ts64) are launched anyway -- they are
# expected to OOM on some tasks; a failure there is itself a data point (no fixed-res number
# fits at that compute), not a bug to avoid.
#
#   [DRY=1] [ONLY=<regex>] bash scripts/slurm/ps_ts_grid.sh <pastis|biomassters|geoidflood|yieldsat_ml24>
#
# Run from the repo root, NOT under sbatch. Submits one chain (<=3h links) per (mode, ps, ts).
# train.py now records samples_s and peak_mem_gb at the best epoch (results/<task>/olmoearth.csv).
set -euo pipefail
TASK="${1:?usage: bash scripts/slurm/ps_ts_grid.sh <pastis|biomassters|geoidflood|yieldsat_ml24>}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }

case "$TASK" in
  pastis)        EP=50; CPUS=9 ;;
  biomassters)   EP=10; CPUS=17 ;;
  geoidflood)    EP=10; CPUS=9 ;;
  yieldsat_ml24) EP=50; CPUS=9 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
MODEL=tiny
LR=1e-3
EXCLUDE="--exclude=kn117,kn104,kn101"

# Full grid, minus combos scripts/slurm/sweep.sh (ps4/16 x ts64) and baselines_ps8.sh
# (ps8 x ts{8,64} series + ts8 per_timestep) already ran.
GRID=("2 2" "2 32" "2 64" "4 32" "8 32" "16 16" "16 32")
# geoidflood never ran ps8 at all (series or per_timestep); pastis/biomassters/yieldsat_ml24
# ran ps8 series but not ps8ts64 per_timestep (baselines_ps8.sh only did ps8ts8 per_timestep).
EXTRA_SERIES=()
EXTRA_PD=("8 64")
if [[ "$TASK" == geoidflood ]]; then EXTRA_SERIES+=("8 8" "8 64"); EXTRA_PD+=("8 8"); fi

# Batch size: halve the task's usual ps4/ps8 batch for the token-heavy small-ps/large-ts arms
# so a CUDA OOM (if any) comes from the arm itself, not from reusing a batch tuned for ts64/ps4.
bs() {   # bs <patch> <tile>
  case "$TASK:$1:$2" in
    *:2:32|*:2:64)   echo 4 ;;
    *:4:32)          echo 8 ;;
    pastis:8:*|biomassters:8:*|geoidflood:8:*)        echo 16 ;;
    yieldsat_ml24:8:*)                                echo 8 ;;
    *:16:*)          echo 32 ;;
    *)               echo 8 ;;
  esac
}

submit() {   # submit <name> <hours> <module> [args...]
  local name=$1 h=$2; shift 2
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then return; fi
  local cmd=(sbatch --parsable -J "$name" --time="$h:00:00" --cpus-per-task="$CPUS" $EXCLUDE
             ${AFTER:+--dependency=afterany:$AFTER} scripts/slurm/run.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo ""; else "${cmd[@]}"; fi
}

submit_chain() {   # submit_chain <name> <module> [args...]: chain of <=3h links, 4 links
  local name=$1; shift
  local prev="" i
  for i in 1 2 3 4; do
    prev=$(AFTER=$prev submit "${name}_$i" 3 "$@" --max_hours 2.5)
    if [ -n "${DRY:-}" ]; then prev="DRY_${name}_$i"; fi
  done
}

tag() { echo "${MODEL}_${TASK:0:3}_${1:0:3}_ps$2ts$3"; }   # tag <mode> <patch> <tile>

submit_cfg() {   # submit_cfg <mode> <patch> <tile>
  local t=$(tag "$1" "$2" "$3")
  submit_chain "oe_${t}_fft_lr$LR" exp.olmoearth.train --task "$TASK" --model_size "$MODEL" \
      --temporal_mode "$1" --patch_size "$2" --tile_size "$3" --mode fft --lrs "$LR" \
      --epochs "$EP" --batch_size "$(bs "$2" "$3")" --num_workers $((CPUS - 1))
}

for cfg in "${GRID[@]}"; do set -- $cfg; submit_cfg series "$1" "$2"; done
for cfg in "${GRID[@]}"; do set -- $cfg; submit_cfg per_timestep "$1" "$2"; done
for cfg in "${EXTRA_SERIES[@]:-}"; do [ -n "$cfg" ] && { set -- $cfg; submit_cfg series "$1" "$2"; }; done
for cfg in "${EXTRA_PD[@]:-}"; do [ -n "$cfg" ] && { set -- $cfg; submit_cfg per_timestep "$1" "$2"; }; done
echo "Submitted ps/ts grid for $TASK ($MODEL, fft, lr=$LR)"
