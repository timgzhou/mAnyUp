#!/bin/bash
# EuroMineNet (task euromine: one S2 image + one mine mask PER YEAR, 2015-2024) with OlmoEarth
# tiny in manyup.mixres, one low-res token per (location, year), a map decoded per year. Every
# chain is <= 3 h links (train.max_hours 2.5, afterany); the mixed arms wait for the fixed run
# they warm-start from (afterok on its chain). Mail off.
#
#   [DRY=1] [ONLY=<regex>] bash scripts/slurm/euromine/mixres.sh
#
# fixed       fix_ps{16,8,4}         every year at one patch size, no crops, 30 epochs cold
#             cont_ps{16,8}          the fixed ps16 / ps8 model 20 more epochs, no crops (the
#                                    mixed arms' warm start + the same extra training)
# Two setups: s16m8 = ps16 years + ps8 crops (64 tiling windows), n = 8 / 16 / 32
#             s8m4  = ps8 years + ps4 crops (256 cells),          n = 32 / 64 / 128
# all mixed arms warm from fix_ps16 (s16m8) / fix_ps8 (s8m4), 20 epochs; budget fixed per row:
#   pairs     space x time: the n x 10 best (year, window) pairs, highest first: entropy (the
#             year token's own map) | season_random | season_grid (n per year, shifted coset
#             per year) | label (oracle: the year token's loss) -> n x 10 crops for all
#   top       space only (s16m8): n windows x every year: entropy (mean over years) | random |
#             label -> n x 10 crops
#   time      every window, its years chosen: keep one (1 year) | half (5 years):
#             entropy high / low, label high / low, season_random, month_grid (every 10th / 2nd)
#             -> 64 / 320 crops (s16m8), 256 / 1280 (s8m4)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088

sub() {   # sub <job name> <dependency or ""> <overrides...> -> prints the job id
  local name=$1 dep=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${dep:+--dependency=$dep} scripts/slurm/run.sh exp.mixres.train task=euromine
             train.num_workers=8 train.max_hours=2.5 "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
chain() {   # chain <name> <links> <first dependency or ""> <overrides...> -> last job id
  local name=$1 links=$2 dep=$3 id i; shift 3
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then echo ""; return; fi
  for i in $(seq 1 "$links"); do
    id=$(sub "em_${name}_$i" "$dep" "$@"); dep="afterany:$id"
  done
  echo "$name: $links links ending $id" >&2
  echo "$id"
}
run_name() {   # the run name exp.mixres.train gives these overrides
  source env_setup/env_olmo.sh >/dev/null 2>&1
  python -c "import sys; from manyup.mixres.config import RunConfig, apply_overrides; \
print(apply_overrides(RunConfig(), sys.argv[1:]).name())" task=euromine "$@"
}

NOCROP=(select.scorer=season_random select.allocator=pairs select.n=0)
declare -A FIX_JOB FIX_NAME
for P in 16 8 4; do
  ARGS=("${NOCROP[@]}" model.lr_patch=$P train.epochs=30)
  [ $P = 4 ] && ARGS+=(train.batch_size=8)
  FIX_NAME[$P]=$(run_name "${ARGS[@]}")
  FIX_JOB[$P]=$(chain "fix_ps$P" "$(case $P in 16) echo 2 ;; 8) echo 2 ;; 4) echo 3 ;; esac)" "" "${ARGS[@]}")
done

for S in s16m8 s8m4; do
  if [ $S = s16m8 ]; then
    P=16; GEO=(select.crop_px=8 select.loc_stride=8 model.lr_patch=16); NS=(8 16 32); LINKS=2
  else
    P=8; GEO=(select.crop_px=4 select.loc_stride=4 model.lr_patch=8); NS=(32 64 128); LINKS=3
  fi
  DEP=${FIX_JOB[$P]:+afterok:${FIX_JOB[$P]}}
  WARM=("${GEO[@]}" train.epochs=20 train.init_ckpt=${FIX_NAME[$P]})
  chain "cont_ps$P" 2 "$DEP" "${NOCROP[@]}" "${WARM[@]}" >/dev/null
  for N in "${NS[@]}"; do
    for SC in entropy season_random season_grid label; do
      chain "${S}_pairs_${SC}_n$N" $LINKS "$DEP" "${WARM[@]}" select.scorer=$SC \
        select.allocator=pairs select.n=$N >/dev/null
    done
    if [ $S = s16m8 ]; then
      for SC in entropy random label; do
        chain "${S}_top_${SC}_n$N" $LINKS "$DEP" "${WARM[@]}" select.scorer=$SC \
          select.allocator=top select.n=$N >/dev/null
      done
    fi
  done
  for K in one half; do
    STEP=$([ $K = half ] && echo 2 || echo 10)
    for O in high low; do
      for SC in entropy label; do
        chain "${S}_time_${SC}${O}_$K" $LINKS "$DEP" "${WARM[@]}" select.scorer=$SC \
          select.allocator=every_location select.keep=$K select.season_order=$O >/dev/null
      done
    done
    chain "${S}_time_random_$K" $LINKS "$DEP" "${WARM[@]}" select.scorer=season_random \
      select.allocator=every_location select.keep=$K >/dev/null
    chain "${S}_time_grid_$K" $LINKS "$DEP" "${WARM[@]}" select.scorer=none \
      select.allocator=month_grid select.month_step=$STEP >/dev/null
  done
done
