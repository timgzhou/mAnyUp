#!/bin/bash
# Submit the feature extraction, the OlmoEarth LP / FFT sweep or the UTAE baselines for one
# task (exp/common/tasks.py). Run from the repo root, NOT under sbatch:
#
#   [MODEL=base|tiny] bash scripts/slurm/sweep.sh <pastis|biomassters|geoidflood|yieldsat_ml24> <what>
#       what: extract | olmoearth | utae
#   DRY=1 ...                                        # print the sbatch lines only
#   ONLY=<regex> ...                                 # submit only jobs whose name matches
#   LRS="1e-2 5e-3" ...                             # learning rates (default 1e-3 1e-4 3e-5)
#   STORE=per_timestep|reduced ...                   # cache store (default: per task, below)
#   MAX_JOB_H=3 ...                                  # chain links of <= 3 h (default 23)
#
# extract: frozen-feature caches, series encoding for the 3 (patch, tile) configs plus
#   per_timestep for ps4ts4. PASTIS keeps every timestep (upsamplers train on it);
#   BioMassters / GEOID store only what the head reads (every timestep: ~2.5 / ~0.7 TB at base).
# olmoearth: 2 temporal modes x 3 configs x {LP, FFT} x 3 learning rates = 36 results.
#   LP trains one head per learning rate on the same frozen forward pass -> 6 LP jobs; it
#   reads the cache where "extract" made one (and waits for that extraction job).
#   FFT: one job per learning rate. A run longer than MAX_JOB_H is a CHAIN of jobs: each
#   resumes the saved state and stops cleanly before its walltime (exp/olmoearth/train.py).
# utae: one job per learning rate. MODEL does not apply.
set -euo pipefail
TASK="${1:?usage: [MODEL=base|tiny] bash scripts/slurm/sweep.sh <task> <extract|olmoearth|utae>}"
WHAT="${2:?what: extract | olmoearth | utae}"
MODEL="${MODEL:-base}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }

read -ra LRS <<< "${LRS:-1e-3 1e-4 3e-5}"   # override: LRS="1e-2 5e-3"
CONFIGS=("4 4" "4 64" "16 64")
MODES=(series per_timestep)
EXCLUDE="--exclude=kn117,kn104"      # nodes with GPU ECC errors
MAX_JOB_H="${MAX_JOB_H:-23}"         # longer runs are chained (24 h+ L40S jobs rarely start)

# per task: OlmoEarth epochs, UTAE epochs, CPUs (loader workers + 1), cache store
case "$TASK" in
  pastis)      OE_EPOCHS=50; UTAE_EPOCHS=100; CPUS=9;  DEFAULT_STORE=per_timestep ;;
  biomassters) OE_EPOCHS=10; UTAE_EPOCHS=50;  CPUS=17; DEFAULT_STORE=reduced ;;
  geoidflood)  OE_EPOCHS=10; UTAE_EPOCHS=20;  CPUS=9;  DEFAULT_STORE=reduced ;;
  yieldsat_ml24) OE_EPOCHS=50; UTAE_EPOCHS=100; CPUS=9; DEFAULT_STORE=reduced ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
STORE="${STORE:-$DEFAULT_STORE}"

# Hours per job: ~1.5x the L40S benchmark (FFT samples/s after unfreezing, eval included)
#   base  pastis ps4 5.5-9.5/s (~22 min/epoch), ps16 ~37/s | bio ps4 6-10.6/s (~3.4 h/epoch),
#         ps16 ~26/s (data-loading bound) | geoid ps4ts4 60-74/s, ps4ts64 ~200/s, ps16 ~370/s
#   tiny  pastis ps4 ~20/s | bio ps4 21-23/s, ps16 ~26/s (data-loading bound) | geoid ps4ts4 ~290/s
#   base  yieldsat_ml24 (8703 train tiles, batch 4) ps4ts4 8.9/s (series and per_timestep),
#         ps4ts64 12.3/s, ps16ts64 82/s -> 50 epochs: ps4 ~12-17 min/epoch, ps16 ~2 min/epoch
# LP runs the encoder forward only (~1/3 of FFT) or reads a cache (disk bound).
hours() {   # hours <patch> <tile> <fft|lp|lp_cache|extract|utae>
  case "$MODEL:$TASK:$1:$2:$3" in
    *:yieldsat_ml24:*:*:lp_cache) echo 2 ;;   # reduced cache: 10.9k tiles x one token map
    *:*:*:*:lp_cache)             echo 8 ;;
    *:pastis:*:*:utae)            echo 8 ;;
    *:biomassters:*:*:utae)       echo 12 ;;
    *:geoidflood:*:*:utae)        echo 8 ;;
    *:yieldsat_ml24:*:*:utae)     echo 8 ;;
    base:pastis:*:*:extract)      echo 4 ;;
    base:biomassters:*:*:extract) echo 12 ;;
    base:geoidflood:*:*:extract)  echo 6 ;;
    base:yieldsat_ml24:*:*:extract) echo 2 ;;
    base:pastis:4:*:fft)          echo 28 ;;  base:pastis:16:*:fft)       echo 6 ;;
    base:pastis:4:*:lp)           echo 12 ;;  base:pastis:16:*:lp)        echo 4 ;;
    base:biomassters:4:*:fft)     echo 54 ;;  base:biomassters:16:*:fft)  echo 16 ;;
    base:biomassters:4:*:lp)      echo 24 ;;  base:biomassters:16:*:lp)   echo 12 ;;
    base:geoidflood:4:4:fft)      echo 16 ;;  base:geoidflood:4:64:fft)   echo 6 ;;
    base:geoidflood:16:*:fft)     echo 4 ;;   base:geoidflood:*:*:lp)     echo 6 ;;
    base:yieldsat_ml24:4:*:fft)   echo 22 ;;  base:yieldsat_ml24:16:*:fft) echo 3 ;;
    base:yieldsat_ml24:4:*:lp)    echo 8 ;;   base:yieldsat_ml24:16:*:lp)  echo 2 ;;
    tiny:pastis:*:*:extract)      echo 3 ;;
    tiny:biomassters:*:*:extract) echo 12 ;;
    tiny:geoidflood:*:*:extract)  echo 4 ;;
    tiny:pastis:4:*:fft)          echo 8 ;;   tiny:pastis:16:*:fft)       echo 4 ;;
    tiny:pastis:*:*:lp)           echo 4 ;;
    tiny:biomassters:*:*:fft)     echo 16 ;;  tiny:biomassters:*:*:lp)    echo 12 ;;
    tiny:geoidflood:4:4:fft)      echo 5 ;;   tiny:geoidflood:*:*:fft)    echo 4 ;;
    tiny:geoidflood:*:*:lp)       echo 4 ;;
    tiny:yieldsat_ml24:*:*:extract) echo 1 ;;
    tiny:yieldsat_ml24:4:*:fft)   echo 8 ;;   tiny:yieldsat_ml24:16:*:fft) echo 2 ;;
    tiny:yieldsat_ml24:*:*:lp)    echo 3 ;;
    *) echo "no walltime for $MODEL:$TASK:$1:$2:$3" >&2; exit 1 ;;
  esac
}
# LP reads the exp.olmoearth.extract cache where one was made (see extract below)
has_cache() {   # has_cache <mode> <patch> <tile>
  [[ "$1" == series || "$1:$2:$3" == per_timestep:4:4 ]]
}

submit() {   # submit <job name> <hours> <module> [args...] -> job id; $AFTER = dependency
  local name=$1 h=$2; shift 2
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then echo ""; return; fi
  local cmd=(sbatch --parsable -J "$name" --time="$h:00:00" --cpus-per-task="$CPUS" $EXCLUDE
             ${AFTER:+--dependency=$AFTER} scripts/slurm/run.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY_${name}"; else "${cmd[@]}"; fi
}

submit_chain() {   # submit_chain <job name> <hours> <module> [args...]: <= MAX_JOB_H per job
  local name=$1 h=$2; shift 2
  if (( h <= MAX_JOB_H )); then submit "$name" "$h" "$@" >/dev/null; return; fi
  local n=$(( (h + MAX_JOB_H - 1) / MAX_JOB_H )) prev="" i
  for (( i = 1; i <= n; i++ )); do
    # each link stops cleanly before its walltime and the next resumes; afterany, so a link
    # that stopped early or ran out of time still hands over (a finished run exits at once)
    prev=$(AFTER=${prev:+afterany:$prev} submit "${name}_$i" "$MAX_JOB_H" "$@" \
           --max_hours $((MAX_JOB_H - 1)))
  done
}

tag() { echo "${MODEL}_${TASK:0:3}_${1:0:3}_ps$2ts$3"; }   # tag <mode> <patch> <tile>

case "$WHAT" in
  extract)
    for cfg in "4 4 series" "4 64 series" "16 64 series" "4 4 per_timestep"; do
      set -- $cfg
      submit "ext_$(tag "$3" "$1" "$2")" "$(hours "$1" "$2" extract)" exp.olmoearth.extract \
          --task "$TASK" --model_size "$MODEL" --temporal_mode "$3" --patch_size "$1" \
          --tile_size "$2" --store "$STORE" --num_workers $((CPUS - 1)) >/dev/null
    done ;;
  olmoearth)
    for mode in "${MODES[@]}"; do
      for cfg in "${CONFIGS[@]}"; do
        set -- $cfg; ps=$1; ts=$2; t=$(tag "$mode" "$ps" "$ts")
        common=(exp.olmoearth.train --task "$TASK" --model_size "$MODEL" --temporal_mode "$mode"
                --patch_size "$ps" --tile_size "$ts" --epochs "$OE_EPOCHS"
                --num_workers $((CPUS - 1)))
        lp=(--mode lp --lrs "$(IFS=,; echo "${LRS[*]}")")
        if has_cache "$mode" "$ps" "$ts"; then
          ext=$(squeue -u "$USER" -h -n "ext_$t" -o %i | head -1)   # still extracting?
          AFTER=${ext:+afterok:$ext} submit "oe_${t}_lp" "$(hours "$ps" "$ts" lp_cache)" \
              "${common[@]}" "${lp[@]}" --from_cache "$STORE" >/dev/null
        else
          submit_chain "oe_${t}_lp" "$(hours "$ps" "$ts" lp)" "${common[@]}" "${lp[@]}"
        fi
        for lr in "${LRS[@]}"; do
          submit_chain "oe_${t}_fft_lr$lr" "$(hours "$ps" "$ts" fft)" \
              "${common[@]}" --mode fft --lrs "$lr"
        done
      done
    done ;;
  utae)
    for lr in "${LRS[@]}"; do
      submit "utae_${TASK:0:3}_lr$lr" "$(hours 0 0 utae)" exp.utae.train --task "$TASK" \
          --lr "$lr" --epochs "$UTAE_EPOCHS" --num_workers $((CPUS - 1)) >/dev/null
    done ;;
  *) echo "what: extract | olmoearth | utae" >&2; exit 1 ;;
esac
