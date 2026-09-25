#!/bin/bash
# Submit the feature-upsampler comparison for one task (exp/upsamplers), from the repo root:
#
#   [MODEL=tiny] [DRY=1] [ONLY=<regex>] [SKIP=<regex>] bash scripts/slurm/upsamplers.sh <task>
#       task: pastis | biomassters | geoidflood; ONLY / SKIP filter the job names
#
#   LP (exp.upsamplers.lp), 5 learning rates per job, features standardized (--head_norm):
#     bilinear, bicubic, anyup                 after the ps16ts64 cache exists
#     manyup, manyup_t, timanyup               after their training (exp.upsamplers.train),
#                                              which waits for the ps16ts64 / ps4ts64 (and, for
#                                              timanyup, ps4ts4 per_timestep) caches; a trained
#                                              run exits at once, so rerunning is cheap
#     timanyup k=0                             the same timAnyUp without lookups (control)
#     timanyup_2stage (+ k=0)                  timAnyUp on the trained manyup_t, frozen
#     upa, upma                                after their kernel fields are fitted and cached
#                                              for every split (exp.upsamplers.fit_jbu)
#     OlmoEarth LP ps16ts64 / ps4ts64          floor / ceiling, same head (exp.olmoearth.train)
# Dependencies on extraction jobs are found by name (ext_<model>_<tsk>_<mode>_ps<p>ts<t>, as
# scripts/slurm/sweep.sh names them) among your queued jobs.
set -euo pipefail
TASK="${1:?usage: bash scripts/slurm/upsamplers.sh <pastis|biomassters|geoidflood>}"
MODEL="${MODEL:-tiny}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }

LRS="1e-1,3e-2,1e-2,5e-3,1e-3"     # standardized features; 1e-2 was best on the old grid
EXCLUDE="--exclude=kn117,kn104"
# per task: LP epochs (as the OlmoEarth LP), upsampler training epochs (mAnyUp, timAnyUp and
# manyup_t; ~ the legacy PASTIS sample budget), hours, UPA/UPMA kernel-fit shards
declare -A FIT_SHARDS          # UPA/UPMA kernel-fit jobs per split (~30k fitted images each)
case "$TASK" in
  pastis)      LP_EP=50; MU_EP=32; TA_EP=20; LP_H=6;  TR_H=4
               FIT_SHARDS=([train]=1 [val]=1 [test]=1) ;;
  biomassters) LP_EP=10; MU_EP=4;  TA_EP=3;  LP_H=12; TR_H=8
               FIT_SHARDS=([train]=3 [val]=1 [test]=2) ;;
  geoidflood)  LP_EP=10; MU_EP=2;  TA_EP=2;  LP_H=12; TR_H=8
               FIT_SHARDS=([train]=15 [val]=3 [test]=4) ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
FIT_H=3
CPUS=9

submit() {   # submit <job name> <hours> <module> [args...] -> job id; $AFTER = dependency
  local name=$1 h=$2; shift 2
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then echo ""; return; fi
  if [[ -n "${SKIP:-}" && "$name" =~ $SKIP ]]; then echo ""; return; fi
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
lp=(exp.upsamplers.lp "${common[@]}" --lrs "$LRS" --epochs "$LP_EP" --head_norm)

# OlmoEarth LP floor (ps16ts64) and ceiling (ps4ts64), same normalized head
for cfg in "16 $LR_CACHE" "4 $TG_CACHE"; do
  set -- $cfg
  AFTER=${2:-} submit "up_${T}_oe_ps${1}ts64_lp" "$LP_H" exp.olmoearth.train "${common[@]}" \
      --temporal_mode series --patch_size "$1" --tile_size 64 --mode lp --lrs "$LRS" \
      --epochs "$LP_EP" --from_cache per_timestep --head_norm >/dev/null
done

# training-free upsamplers
declare -A LP
for u in bilinear bicubic anyup; do
  LP[$u]=$(AFTER=$LR_CACHE submit "up_${T}_${u}_lp" "$LP_H" "${lp[@]}" --upsampler "$u")
done

# trained upsamplers, then their LP
# where the task's head needs every timestep (GEOID), mAnyUp is already per timestep: it is
# manyup_t there, and it is the frozen upsampler of the two-stage timAnyUp
if [ "$TASK" = geoidflood ]; then PER_T=manyup; PER_T_EP=$MU_EP; METHODS=(manyup timanyup)
else PER_T=manyup_t; PER_T_EP=$TA_EP; METHODS=(manyup manyup_t timanyup); fi
declare -A TR
for m in "${METHODS[@]}"; do
  ep=$([ $m = manyup ] && echo $MU_EP || echo $TA_EP)
  need=$(deps "$LR_CACHE" "$TG_CACHE" $([ $m = timanyup ] && echo "$HR_CACHE"))
  tr=$(AFTER=$need submit "up_${T}_${m}_train" "$TR_H" exp.upsamplers.train "${common[@]}" \
       --method "$m" --epochs "$ep")
  TR[$m]=$tr
  ck="checkpoints/upsamplers/${m}_${TASK}_${MODEL}_ep${ep}.pt"
  AFTER=${tr:+afterok:$tr} submit "up_${T}_${m}_lp" "$LP_H" "${lp[@]}" --upsampler "$m" \
      --ckpt "$ck" >/dev/null
  if [ $m = timanyup ]; then
    AFTER=${tr:+afterok:$tr} submit "up_${T}_${m}_k0_lp" "$LP_H" "${lp[@]}" --upsampler "$m" \
        --ckpt "$ck" --k 0 >/dev/null
  fi
done

# two-stage timAnyUp: the trained manyup_t, frozen, as its upsampler; the query head reads
# that upsampler's output. Its k=0 LP must reproduce manyup_t's (a check of the setup).
need=$(deps "${TR[$PER_T]:+afterok:${TR[$PER_T]}}" "$LR_CACHE" "$TG_CACHE" "$HR_CACHE")
tr=$(AFTER=$need submit "up_${T}_timanyup_2stage_train" "$TR_H" exp.upsamplers.train \
     "${common[@]}" --method timanyup --epochs "$TA_EP" --query_input upsampled \
     --upsampler_ckpt "checkpoints/upsamplers/${PER_T}_${TASK}_${MODEL}_ep${PER_T_EP}.pt")
ck="checkpoints/upsamplers/timanyup_2stage_${TASK}_${MODEL}_ep${TA_EP}.pt"
for k in "" 0; do
  AFTER=${tr:+afterok:$tr} submit "up_${T}_timanyup_2stage${k:+_k$k}_lp" "$LP_H" "${lp[@]}" \
      --upsampler timanyup --ckpt "$ck" ${k:+--k $k} >/dev/null
done

# UPA / UPMA: fit every sample's kernel field once (exp.upsamplers.fit_jbu, ~0.2 s per image
# at batch 128), sharded so a job stays under ~2 h, then train an LP on them like the others.
# The cache is reused by every later LP; a finished shard exits at once.
for u in upa upma; do
  fits=()
  for s in train val test; do
    for (( i = 0; i < ${FIT_SHARDS[$s]}; i++ )); do
      fits+=($(AFTER=$LR_CACHE submit "up_${T}_${u}_fit_${s:0:2}$i" "$FIT_H" \
               exp.upsamplers.fit_jbu --task "$TASK" --model_size "$MODEL" \
               --num_workers $((CPUS - 1)) --method "$u" --split "$s" \
               --shard "$i/${FIT_SHARDS[$s]}"))
    done
  done
  fits=(${fits[@]})                                             # drop filtered-out (empty)
  AFTER=${fits[0]:+afterok:$(IFS=:; echo "${fits[*]}")} submit "up_${T}_${u}_lp" "$LP_H" \
      "${lp[@]}" --upsampler "$u" >/dev/null
done
