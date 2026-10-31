#!/bin/bash
# EuroMineNet with every year encoded ALONE (model.independent_steps: one sequence per (tile,
# year), no attention across years) -- the paper's setting, a single-image segmenter run on each
# year -- at ps16 / ps8 / ps4, the same recipe as mixres.sh's fix_ps* (30 epochs cold, no crops;
# ps4 at batch 8). Then exp.euromine.eval_protocols scores these and the joint fix_ps* / cont_ps*
# runs under the paper's averaging rules -> results/euromine/protocols.csv. Mail off.
#
#   [DRY=1] [AFTER=<job ids, colon separated>] bash scripts/slurm/euromine/independent.sh
# AFTER: jobs the protocol eval also waits for (the joint fixed chains still running).
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
print(apply_overrides(RunConfig(), sys.argv[1:]).name())" task=euromine "$@" 2>/dev/null
}

NOCROP=(select.scorer=season_random select.allocator=pairs select.n=0)
RUNS=() WAIT=${AFTER:-}
for P in 16 8 4; do
  for INDEP in true false; do
    ARGS=("${NOCROP[@]}" model.lr_patch=$P train.epochs=30)
    [ $P = 4 ] && ARGS+=(train.batch_size=8)
    [ $INDEP = true ] && ARGS+=(model.independent_steps=true)
    RUNS+=("$(run_name "${ARGS[@]}")")
    [ $INDEP = false ] && continue          # the joint runs: mixres.sh's fix_ps*
    dep="" id=""
    for i in $(seq 1 "$([ $P = 4 ] && echo 3 || echo 2)"); do
      id=$(sub "em_indep_ps${P}_$i" "$dep" exp.mixres.train task=euromine train.num_workers=8 \
           train.max_hours=2.5 "${ARGS[@]}")
      dep="afterany:$id"
    done
    WAIT="${WAIT:+$WAIT:}$id"
    echo "indep_ps$P: chain ending $id" >&2
  done
done
for P in 16 8; do      # the 20-epoch continuations of the joint fixed runs
  RUNS+=("$(run_name "${NOCROP[@]}" model.lr_patch=$P train.epochs=20 \
            train.init_ckpt="$(run_name "${NOCROP[@]}" model.lr_patch=$P train.epochs=30)" \
            $([ $P = 8 ] && echo select.crop_px=4 select.loc_stride=4))")
done
printf '  %s\n' "${RUNS[@]}" >&2
id=$(sub em_protocols "afterany:$WAIT" exp.euromine.eval_protocols "${RUNS[@]}")
echo "protocol eval: $id (after $WAIT)" >&2
