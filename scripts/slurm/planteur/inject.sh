#!/bin/bash
# Submit the fixed- and mixed-resolution arms on PLANTEUR (task planteur: S2 + S1, calendar-2019
# months; OlmoEarth tiny, exp.upsamplers.finetune_inject), from the repo root, each as a chain of
# <= 3 h resumable jobs (links run afterany; a finished run exits at once). The same flags as
# the PASTIS rounds they come from (scripts/slurm/inject.sh), so the run names line up and the
# PASTIS figure scripts read results/planteur/ unchanged. Every chain first waits for the tile
# store build (afterok), submitted here too unless STORE_JOB=<id> or STORE_JOB=none.
#
#   [DRY=1] [ONLY=<regex>] [STORE_JOB=<id>|none] bash scripts/slurm/planteur/inject.sh
#
# Arms (2026-10-22, the arms of exp.viz.res_pairs_tradeoff / oracle_season_tradeoff):
#   fixed endpoints    fix_ps{16,8,4}       seasons only at one patch size (PASTIS round 43)
#                      sg_ps{16,8,4}_base   monthly at one patch size (round 47; ps4 = S4Tmonth)
#   per-month curves   ded_{16-8,8-4}_n<K>  a model dedicated to K crops on the shared lattice,
#                                           K = 64 .. 2048 (round 44)
#   oracles            bothalf_st8_n<N>_L0_set        ps16 seasons + ps8 crops, better-half season
#                                                     oracle, stride 8, N = 8/16/32 windows (r54)
#                      s8m4_oracle_bothalf_n<N>_L0_set  ps8 seasons + ps4 crops, N = 32/64/128 (r55)
#   controls           {ran,grid}_n<N>_L0_set         location random / lattice, ps16 + ps8 (r49/52)
#                      s8m4_{random,grid}_n<N>_L0_set location random / lattice, ps8 + ps4 (r56)
#   No-crop baselines of the oracles: fix_ps16 / fix_ps8.
# Links: PASTIS's x 1.3, rounded up (7584 vs 5820 train tiles, the same tokens per tile).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
TASK=planteur
EP=50
CPUS=9
UNIT=slot
export EMAIL=              # no job emails (run.sh mails unless EMAIL is empty)

# the 64 px tile store (exp.common.build_tile_store): two resumable 3 h links on CPU
if [ -z "${STORE_JOB:-}" ]; then
  S1=$( [ -n "${DRY:-}" ] && echo DRYS1 || sbatch --parsable -J store_planteur_1 --gres=none \
        --cpus-per-task=17 --mem=64G --time=3:00:00 \
        scripts/slurm/run.sh exp.common.build_tile_store --task "$TASK" --num_workers 16 )
  STORE_JOB=$( [ -n "${DRY:-}" ] && echo DRYS2 || sbatch --parsable -J store_planteur_2 --gres=none \
        --cpus-per-task=17 --mem=64G --time=3:00:00 --dependency=afterany:"$S1" \
        scripts/slurm/run.sh exp.common.build_tile_store --task "$TASK" --num_workers 16 )
  echo "tile store: $S1 -> $STORE_JOB"
fi
[ "$STORE_JOB" = none ] && STORE_JOB=""

chain() {   # chain <name> <PASTIS links> [finetune_inject args...]
  local name=$1 links=$(( ($2 * 13 + 9) / 10 )); shift 2
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then return; fi
  # centre-aligned crop positions (2026-10-01 default for new runs), not on no-crop chains
  if [[ " $* " != *" --pos "* && " $* " != *" --selector none "* ]]; then
    set -- "$@" --pos center
  fi
  local dep="${STORE_JOB:+afterok:$STORE_JOB}" id i
  for i in $(seq 1 "$links"); do
    local cmd=(sbatch --parsable -J "inj_pla_${name}_$i" --time=3:00:00
               --cpus-per-task="$CPUS" --exclude=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
               ${dep:+--dependency=$dep}
               scripts/slurm/run.sh exp.upsamplers.finetune_inject --task "$TASK" --epochs "$EP"
               --num_workers $((CPUS - 1)) --max_hours 2.5 "$@")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; id="DRY$i"; else id=$("${cmd[@]}"); fi
    dep="afterany:$id"
  done
  echo "$name: chain of $links ending $id"
}

# ---- fixed endpoints
for P in 16 8 4; do
  chain "fix_ps$P" "$([ "$P" = 4 ] && echo 8 || echo 3)" --selector none --readout native_set --lr_patch "$P"
  chain "sg_ps${P}_base" "$([ "$P" = 4 ] && echo 10 || echo 5)" \
    --lr_steps all --readout native_set --batch_size 8 --lr_patch "$P" --selector none
done

# ---- per-month crop curves: dedicated models on the shared lattice
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --grid_mode shared)
for PAIR in "16 8" "8 4"; do
  set -- $PAIR
  for K in 64 128 256 512 1024 2048; do
    chain "ded_$1-$2_n$K" "$(case $K in 2048) echo 12 ;; 1024) echo 7 ;; 512) echo 5 ;; *) echo 4 ;; esac)" \
      "${FR[@]}" --lr_patch "$1" --crop_px "$2" --crop_counts "$K"
  done
done

# ---- better-half season oracle and location controls
LOC=(--inject_layer 0 --select_unit location --readout native_set)
for N in 8 16 32; do          # ps16 seasons + ps8 crops, N ps8 windows (x present months)
  KF=$(python3 -c "print($N / 256)")
  chain "bothalf_st8_n${N}_L0_set" 3 --selector oracle_bothalf "${LOC[@]}" --crop_px 8 --loc_stride 8 --k_frac "$KF"
  chain "ran_n${N}_L0_set"         3 --selector random         "${LOC[@]}" --crop_px 8 --k_frac "$KF"
  chain "grid_n${N}_L0_set"        3 --selector grid           "${LOC[@]}" --crop_px 8 --k_frac "$KF"
done
for N in 32 64 128; do        # ps8 seasons + ps4 crops, N ps4 cells (x present months)
  KF=$(python3 -c "print($N / 256)")
  L=$(case $N in 32) echo 4 ;; 64) echo 6 ;; *) echo 8 ;; esac)
  for SEL in oracle_bothalf random grid; do
    chain "s8m4_${SEL}_n${N}_L0_set" "$L" --selector "$SEL" "${LOC[@]}" --lr_patch 8 --crop_px 4 --k_frac "$KF"
  done
done
