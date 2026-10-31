#!/bin/bash
# EuroMineNet on our own quarterly S2 L2A composites (exp.euromine.fetch_seasonal, packed by
# exp.euromine.prepare pack_seasonal): fixed-resolution arms, every step joint, no crops, 30 epochs
# cold (the recipe of mixres.sh's fix_ps*), 64 px tiles, the release's labels:
#   euromine_seasonal  38 quarters 2015 Q3 .. 2024 Q4, year t scored on its Jul-Sep quarter
#   euromine_q3        only the 10 Jul-Sep composites (isolates the extra quarters from the source
#                      change: release = top-of-atmosphere August mosaics, ours = L2A medians)
# at ps16 and ps8; then exp.euromine.eval_protocols -> results/euromine/protocols.csv. Mail off.
# The seasonal arms read the euromine_seasonal tile store (38 frames a tile from the site arrays
# runs at ~8 samples/s), built first here (2 resumable CPU links, afterok) unless STORE_JOB=<id>|none.
#   [DRY=1] [STORE_JOB=<id>|none] bash scripts/slurm/euromine/seasonal.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <dependency or ""> <module> <args...> -> prints the job id
  local name=$1 dep=$2 mod=$3; shift 3
  local res=(--cpus-per-task=9)
  [[ "$name" == store_* ]] && res=(--gres=none --cpus-per-task=17)
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 "${res[@]}" --mem=96G --exclude=$EXC
             ${dep:+--dependency=$dep} scripts/slurm/run.sh "$mod" "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
run_name() {
  source env_setup/env_olmo.sh >/dev/null 2>&1
  python -c "import sys; from manyup.mixres.config import RunConfig, apply_overrides; \
print(apply_overrides(RunConfig(), sys.argv[1:]).name())" "$@" 2>/dev/null
}
if [ -z "${STORE_JOB:-}" ]; then
  S1=$(sub store_em_seasonal_1 "" exp.common.build_tile_store --task euromine_seasonal --num_workers 16 --check 8)
  STORE_JOB=$(sub store_em_seasonal_2 "afterany:$S1" exp.common.build_tile_store --task euromine_seasonal --num_workers 16 --check 8)
  echo "tile store: $S1 -> $STORE_JOB" >&2
fi
[ "$STORE_JOB" = none ] && STORE_JOB=""
RUNS=() WAIT=""
for ARM in "seasonal 16 32 2" "seasonal 8 8 3" "q3 16 32 2" "q3 8 32 2"; do
  set -- $ARM; S=$1 P=$2 BS=$3 LINKS=$4
  ARGS=(task=euromine_$S select.scorer=season_random select.allocator=pairs select.n=0
        model.lr_patch=$P train.batch_size=$BS train.epochs=30)
  RUNS+=("$(run_name "${ARGS[@]}")")
  dep="" id=""
  [ "$S" = seasonal ] && dep=${STORE_JOB:+afterok:$STORE_JOB}
  for i in $(seq 1 "$LINKS"); do
    id=$(sub "em_${S}_ps${P}_$i" "$dep" exp.mixres.train train.num_workers=8 train.max_hours=2.5 "${ARGS[@]}")
    dep="afterany:$id"
  done
  WAIT="${WAIT:+$WAIT:}$id"
  echo "${S}_ps$P: $LINKS links ending $id" >&2
done
printf '  %s\n' "${RUNS[@]}" >&2
id=$(sub em_protocols_seasonal "afterany:$WAIT" exp.euromine.eval_protocols --batch_size 8 "${RUNS[@]}")
echo "protocol eval: $id" >&2
