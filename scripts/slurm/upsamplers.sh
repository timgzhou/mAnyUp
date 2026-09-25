#!/bin/bash
# Submit the feature-upsampler comparison for one task (exp/upsamplers), from the repo root:
#
#   [MODEL=tiny] [DRY=1] [ONLY=<regex>] bash scripts/slurm/upsamplers.sh <pastis|biomassters|geoidflood>
#
#   LP (exp.upsamplers.lp), 5 learning rates per job:
#     bilinear, bicubic, anyup                 after the ps16ts64 cache exists
#     manyup, timanyup                         after their training (exp.upsamplers.train),
#                                              which waits for the ps16ts64 / ps4ts64 (and, for
#                                              timanyup, ps4ts4 per_timestep) caches
#     upa, upma                                test set only with the bilinear run's heads
#                                              (--shared_probe), split over SHARDS jobs, then
#                                              merged
# Dependencies on extraction jobs are found by name (ext_<model>_<tsk>_<mode>_ps<p>ts<t>, as
# scripts/slurm/sweep.sh names them) among your queued jobs.
set -euo pipefail
TASK="${1:?usage: bash scripts/slurm/upsamplers.sh <pastis|biomassters|geoidflood>}"
MODEL="${MODEL:-tiny}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }

LRS="1e-2,5e-3,1e-3,1e-4,3e-5"
EXCLUDE="--exclude=kn117,kn104"
# per task: LP epochs (as the OlmoEarth LP), upsampler training epochs (mAnyUp, timAnyUp;
# ~ the legacy PASTIS sample budget), UPA/UPMA test shards (~0.8 s per fitted image), hours
case "$TASK" in
  pastis)      LP_EP=50; MU_EP=32; TA_EP=20; SHARDS=1;  LP_H=6;  TR_H=4; SH_H=2 ;;
  biomassters) LP_EP=10; MU_EP=4;  TA_EP=3;  SHARDS=8;  LP_H=12; TR_H=8; SH_H=3 ;;
  geoidflood)  LP_EP=10; MU_EP=2;  TA_EP=2;  SHARDS=16; LP_H=12; TR_H=8; SH_H=3 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
CPUS=9

submit() {   # submit <job name> <hours> <module> [args...] -> job id; $AFTER = dependency
  local name=$1 h=$2; shift 2
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then echo ""; return; fi
  local cmd=(sbatch --parsable -J "$name" --time="$h:00:00" --cpus-per-task="$CPUS" $EXCLUDE
             ${AFTER:+--dependency=$AFTER} scripts/slurm/run.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY_${name}"; else "${cmd[@]}"; fi
}

ext() {   # queued extraction job of a cache config, as an afterok clause (empty if none)
  local id
  id=$(squeue -u "$USER" -h -n "ext_${MODEL}_${TASK:0:3}_$1" -o %i | head -1)
  echo "${id:+afterok:$id}"
}
deps() {   # join the non-empty dependency clauses with "," (all must hold)
  local out="" x
  for x in "$@"; do [ -n "$x" ] && out="${out:+$out,}$x"; done
  echo "$out"
}

LR_CACHE=$(ext ser_ps16ts64)
TG_CACHE=$(ext ser_ps4ts64)
HR_CACHE=$(ext per_ps4ts4)
T="${MODEL}_${TASK:0:3}"
common=(--task "$TASK" --model_size "$MODEL" --num_workers $((CPUS - 1)))

# training-free upsamplers
declare -A LP
for u in bilinear bicubic anyup; do
  LP[$u]=$(AFTER=$LR_CACHE submit "up_${T}_${u}_lp" "$LP_H" exp.upsamplers.lp "${common[@]}" \
           --upsampler "$u" --lrs "$LRS" --epochs "$LP_EP")
done

# trained upsamplers, then their LP
for m in manyup timanyup; do
  ep=$([ $m = manyup ] && echo $MU_EP || echo $TA_EP)
  need=$(deps "$LR_CACHE" "$TG_CACHE" $([ $m = timanyup ] && echo "$HR_CACHE"))
  tr=$(AFTER=$need submit "up_${T}_${m}_train" "$TR_H" exp.upsamplers.train "${common[@]}" \
       --method "$m" --epochs "$ep")
  AFTER=${tr:+afterok:$tr} submit "up_${T}_${m}_lp" "$LP_H" exp.upsamplers.lp "${common[@]}" \
      --upsampler "$m" --ckpt "checkpoints/upsamplers/${m}_${TASK}_${MODEL}_ep${ep}.pt" \
      --lrs "$LRS" --epochs "$LP_EP" >/dev/null
done

# UPA / UPMA: bilinear's heads on the test set, sharded, then merged
probe="up_${TASK}_${MODEL}_bilinear_lp_ep${LP_EP}"
for u in upa upma; do
  shards=()
  for (( i = 0; i < SHARDS; i++ )); do
    shards+=($(AFTER=${LP[bilinear]:+afterok:${LP[bilinear]}} submit "up_${T}_${u}_s$i" "$SH_H" \
               exp.upsamplers.lp "${common[@]}" --upsampler "$u" --lrs "$LRS" \
               --epochs "$LP_EP" --shared_probe "$probe" --shard "$i/$SHARDS"))
  done
  AFTER=afterok:$(IFS=:; echo "${shards[*]}") submit "up_${T}_${u}_merge" 1 exp.upsamplers.lp \
      "${common[@]}" --upsampler "$u" --lrs "$LRS" --epochs "$LP_EP" --shared_probe "$probe" \
      --merge "$SHARDS" >/dev/null
done
