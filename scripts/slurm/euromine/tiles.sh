#!/bin/bash
# EuroMineNet with more spatial context: fixed-resolution arms (no crops, every year joint, 30 epochs
# cold, the recipe of mixres.sh's fix_ps*) on 128 px tiles at ps8 / 16 / 32 (= the token counts of
# ps4 / 8 / 16 on 64 px tiles) and 256 px tiles at ps16 / 32; then exp.euromine.eval_protocols
# scores them on the stitched sites -> results/euromine/protocols.csv. Mail off.
#   [DRY=1] bash scripts/slurm/euromine/tiles.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <dependency or ""> <module> <args...> -> prints the job id
  local name=$1 dep=$2 mod=$3; shift 3
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${dep:+--dependency=$dep} scripts/slurm/run.sh "$mod" "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
run_name() {
  source env_setup/env_olmo.sh >/dev/null 2>&1
  python -c "import sys; from manyup.mixres.config import RunConfig, apply_overrides; \
print(apply_overrides(RunConfig(), sys.argv[1:]).name())" "$@" 2>/dev/null
}
RUNS=() WAIT=""
for ARM in "128 8 8 3" "128 16 16 2" "128 32 32 2" "256 16 8 3" "256 32 16 2"; do
  set -- $ARM; T=$1 P=$2 BS=$3 LINKS=$4
  ARGS=(task=euromine_t$T select.scorer=season_random select.allocator=pairs select.n=0
        model.lr_patch=$P train.batch_size=$BS train.epochs=30)
  RUNS+=("$(run_name "${ARGS[@]}")")
  dep="" id=""
  for i in $(seq 1 "$LINKS"); do
    id=$(sub "em_t${T}_ps${P}_$i" "$dep" exp.mixres.train train.num_workers=8 train.max_hours=2.5 "${ARGS[@]}")
    dep="afterany:$id"
  done
  WAIT="${WAIT:+$WAIT:}$id"
  echo "t${T}_ps$P: $LINKS links ending $id" >&2
done
printf '  %s\n' "${RUNS[@]}" >&2
id=$(sub em_protocols_tiles "afterany:$WAIT" exp.euromine.eval_protocols "${RUNS[@]}")
echo "protocol eval: $id" >&2
