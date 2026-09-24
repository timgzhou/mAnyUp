#!/bin/bash
# Submit the OlmoEarth LP / FFT sweep and the UTAE baselines for one task
# (exp/common/tasks.py). Run from the repo root, NOT under sbatch:
#
#   bash scripts/slurm/sweep.sh <pastis|biomassters|geoidflood> [olmoearth|utae|extract|all]
#   DRY=1 bash scripts/slurm/sweep.sh pastis        # print the sbatch lines only
#
# OlmoEarth: 2 temporal modes x 3 (patch, tile) configs x {LP, FFT} x 3 learning rates.
#   LP trains one head per learning rate on the same frozen forward pass -> 6 LP jobs;
#   FFT trains the encoder -> one job per learning rate, 18 jobs. = 36 results per task.
# UTAE: one job per learning rate.
# extract (not in "all"): frozen-feature caches, series encoding for the 3 configs plus
#   per_timestep for ps4ts4. PASTIS keeps every timestep (upsamplers train on it);
#   BioMassters / GEOID store only what the head reads, since every timestep would need
#   ~2.5 TB / ~0.7 TB.
# LP jobs that read a cache must run after "extract" has finished.
set -euo pipefail
TASK="${1:?usage: bash scripts/slurm/sweep.sh <task> [olmoearth|utae|extract|all]}"
WHAT="${2:-all}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }

LRS=(1e-3 1e-4 3e-5)
CONFIGS=("4 4" "4 64" "16 64")
MODES=(series per_timestep)
EXCLUDE="--exclude=kn117,kn104"      # nodes with GPU ECC errors

# per task: OlmoEarth epochs, UTAE epochs, CPUs (loader workers + 1)
case "$TASK" in
  pastis)      OE_EPOCHS=50; UTAE_EPOCHS=100; CPUS=9;  STORE=per_timestep; EXTRACT_TIME=4:00:00 ;;
  biomassters) OE_EPOCHS=10; UTAE_EPOCHS=50;  CPUS=17; STORE=reduced;      EXTRACT_TIME=12:00:00 ;;
  geoidflood)  OE_EPOCHS=10; UTAE_EPOCHS=20;  CPUS=9;  STORE=reduced;      EXTRACT_TIME=6:00:00 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac

# Walltimes: ~1.5x the benchmark (L40S, FFT samples/s after unfreezing, eval included):
#   pastis  ps4 5.5-9.5 /s  (~22 min/epoch)   ps16 ~37 /s (~4 min)
#   bio     ps4 6.0-10.6 /s (~3.4 h/epoch)    ps16 ~26 /s (~1 h, data-loading bound)
#   geoid   ps4ts4 60-74 /s (~1 h/epoch)      ps4ts64 ~200 /s (~20 min)   ps16 ~370 /s (~12 min)
# LP runs the encoder forward only (~1/3 of FFT), or reads a cache (disk bound).
walltime() {   # walltime <task> <patch> <tile> <fft|lp|lp_cache|utae>
  case "$1:$2:$3:$4" in
    pastis:4:*:fft)        echo 1-04:00:00 ;;  pastis:16:*:fft)       echo 6:00:00 ;;
    pastis:4:*:lp)         echo 12:00:00 ;;    pastis:16:*:lp)        echo 4:00:00 ;;
    biomassters:4:*:fft)   echo 2-06:00:00 ;;  biomassters:16:*:fft)  echo 16:00:00 ;;
    biomassters:4:*:lp)    echo 1-00:00:00 ;;  biomassters:16:*:lp)   echo 12:00:00 ;;
    geoidflood:4:4:fft)    echo 16:00:00 ;;    geoidflood:4:64:fft)   echo 6:00:00 ;;
    geoidflood:16:*:fft)   echo 4:00:00 ;;     geoidflood:*:lp)       echo 6:00:00 ;;
    *:lp_cache)            echo 8:00:00 ;;
    pastis:*:utae)         echo 8:00:00 ;;     biomassters:*:utae)    echo 12:00:00 ;;
    geoidflood:*:utae)     echo 8:00:00 ;;
    *) echo "no walltime for $1:$2:$3:$4" >&2; exit 1 ;;
  esac
}
# LP reads the exp.olmoearth.extract cache where one was made (see "extract" below)
has_cache() {   # has_cache <mode> <patch> <tile>
  [[ "$1" == series || "$1:$2:$3" == per_timestep:4:4 ]]
}

submit() {   # submit <job name> <walltime> <module> [args...]; $AFTER = job id to wait for
  local name=$1 time=$2; shift 2
  local cmd=(sbatch -J "$name" --time="$time" --cpus-per-task="$CPUS" $EXCLUDE
             ${AFTER:+--dependency=afterok:$AFTER} scripts/slurm/run.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

if [[ "$WHAT" == olmoearth || "$WHAT" == all ]]; then
  for mode in "${MODES[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      set -- $cfg; ps=$1; ts=$2
      common=(exp.olmoearth.train --task "$TASK" --temporal_mode "$mode" --patch_size "$ps"
              --tile_size "$ts" --epochs "$OE_EPOCHS" --num_workers $((CPUS - 1)))
      tag="${TASK:0:3}_${mode:0:3}_ps${ps}ts${ts}"
      if has_cache "$mode" "$ps" "$ts"; then
        # wait for the extraction job if it is still queued or running
        AFTER=$(squeue -u "$USER" -h -n "ext_${TASK:0:3}_${mode:0:3}_ps${ps}ts${ts}" -o %i | head -1) \
        submit "oe_${tag}_lp" "$(walltime "$TASK" "$ps" "$ts" lp_cache)" \
            "${common[@]}" --mode lp --lrs "$(IFS=,; echo "${LRS[*]}")" --from_cache "$STORE"
      else
        submit "oe_${tag}_lp" "$(walltime "$TASK" "$ps" "$ts" lp)" \
            "${common[@]}" --mode lp --lrs "$(IFS=,; echo "${LRS[*]}")"
      fi
      for lr in "${LRS[@]}"; do
        submit "oe_${tag}_fft_lr${lr}" "$(walltime "$TASK" "$ps" "$ts" fft)" \
            "${common[@]}" --mode fft --lrs "$lr"
      done
    done
  done
fi

if [[ "$WHAT" == extract ]]; then
  for cfg in "4 4 series" "4 64 series" "16 64 series" "4 4 per_timestep"; do
    set -- $cfg
    submit "ext_${TASK:0:3}_${3:0:3}_ps$1ts$2" "$EXTRACT_TIME" exp.olmoearth.extract \
        --task "$TASK" --temporal_mode "$3" --patch_size "$1" --tile_size "$2" \
        --store "$STORE" --num_workers $((CPUS - 1))
  done
fi

if [[ "$WHAT" == utae || "$WHAT" == all ]]; then
  for lr in "${LRS[@]}"; do
    submit "utae_${TASK:0:3}_lr${lr}" "$(walltime "$TASK" 0 0 utae)" \
        exp.utae.train --task "$TASK" --lr "$lr" --epochs "$UTAE_EPOCHS" --num_workers $((CPUS - 1))
  done
fi
