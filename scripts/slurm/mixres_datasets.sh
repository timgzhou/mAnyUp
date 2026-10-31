#!/bin/bash
# The committed PASTIS method on the other datasets: high-res crops where the low-res model is
# uncertain (train-set = global threshold, val-refit at test), each in the season the readout
# attends to most (select.quadrant skip/global/crop/score, as mixres_quadrant_rep.sh ARMS=global)
#   classification (planteur)          spatial uncertainty = pooled prediction entropy
#   regression (biomassters, yieldsat_raw)  = variance of the used seasons' own maps
#   geoidseries_flood (S1 6-month series: the season tokens are calendar months, the crops single
#                      acquisition dates) = planteur's rule; time = the dates of the month the
#                      readout attends to most at that window (hs one) / of half the months
#   per-step (euromine, a label per year)    selectors.steps.StepThreshold: (year, window) pairs
#                                            by the entropy of that year's prediction (space and
#                                            time jointly, one global threshold)
# Arms per dataset and setup (s16m8: ps16 seasons + ps8 crops; s8m4: ps8 seasons + ps4 crops):
#   global   keep one | half, q_s 1/8, 1/4, 1/2
#   grid / random   controls at the matched n (8/16/32 | 32/64/128), one season per location
#            (grid + month_grid every 4th month; random + season_random), as mixres_spacetime.sh
# Warm starts as on PASTIS: the location oracle at that budget (biomassters / yieldsat_raw: the
# legacy oracle_L0_k*_loc checkpoints; planteur / geoidseries_flood have none -> trained here
# first, label top n, every present month (date), cold, and the arms wait on it). geoidseries_flood
# (212k train chips): oracles 8 epochs, arms 6, 5-link chains, grid every 6th date (~one of the
# window's 6 months), plus fixed baselines (quadrant sample q_s 0 = no crops): month tokens and
# per-date tokens (task geoidseries_flood_dates) at ps16 / 8 / 4 (FIXED="m16 m8 m4 d16 d8 d4"). euromine: from fix_ps16 / fix_ps8
# (scripts/slurm/euromine/mixres.sh, 20 epochs; its pairs / time controls already ran).
# ATTN=cross_hr: every mixed arm (ours and the controls, not the oracles / fixed models) with
# model.attn (crop tokens read the season tokens and the sample's other crops, season tokens never
# read crops; inference reuses the crop-free pass, so ours costs ~ the controls): the PASTIS
# "ours" (spacetime_tradeoff XHR). With ATTN set, euromine also gets its (year, window) pair
# controls (season_random / season_grid pairs at n, as scripts/slurm/euromine/mixres.sh).
# SETUPS=s32m16 (euromine only): ps32 year tokens + ps16 crops, warm from fix_ps32 (trained here
# first if missing, 30 epochs cold as the other fix_ps*).
# KEEPS="one": ours with one season per location only (default "one half").
# SCHED=1: the selector schedule in ONE run, no warm start (as mixres_quadrant_rep.sh SCHED, the
# PASTIS s1o50 runs): the oracle's epochs with the location oracle at the arm's n (label top n,
# every present month, model.attn joint), then the arm's epochs with its selector -- for ours AND
# the grid / random controls; no oracle runs. euromine has no oracle stage (warm from fix_ps*)
# and keeps its recipe. SEEDS="0 1 2": train.seed (every arm and GEOID fixed model; default 0).
# Mail off. <= 3 h links (train.max_hours 2.5, afterany).
#   [SCHED=1] [SEEDS="0 1 2"] [ATTN=cross_hr] [KEEPS=one] [DRY=1] [SMOKE=1] [DATASETS="planteur biomassters yieldsat_raw euromine geoidseries_flood"] [SETUPS=..]
#   [ARMS="global grid random"] bash scripts/slurm/mixres_datasets.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source env_setup/env_olmo.sh >/dev/null 2>&1
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
sub() {   # sub <job name> <dependency or ""> <overrides...> -> prints the job id
  local name=$1 dep=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${dep:+--dependency=$dep} scripts/slurm/run.sh exp.mixres.train
             train.max_hours=2.5 "$@" ${SMOKE:+train.limit_batches=3 train.seed=999})
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
MIX=(${ATTN:+model.attn=$ATTN})   # the mixed arms' attention
AT=${ATTN:+${ATTN}_}               # job name tag
chain() {   # chain <name> <links> <first dependency or ""> <overrides...> -> last job id
  local name=$1 links=$2 dep=$3 id i; shift 3
  for i in $(seq 1 "$links"); do id=$(sub "mxds_${name}${SCHED:+_s1o}_seed${SEED}_$i" "$dep" "$@" train.seed=$SEED)
    dep="afterany:$id"; done
  echo "$id"
}
run_name() {   # the run name exp.mixres.train gives these overrides
  python -c "import sys; from manyup.mixres.config import RunConfig, apply_overrides; \
print(apply_overrides(RunConfig(), sys.argv[1:]).name())" "$@"
}
QUAD=(select.n=0 select.scorer=entropy select.allocator=top select.quadrant.lowspace=skip
      select.quadrant.mode=global select.quadrant.space=crop select.quadrant.rank=score
      select.quadrant.q_t=0.5)
arm() {   # arm <name> <keep> <n> <stride> -> the overrides
  case $1 in
    global) echo "${QUAD[@]}" select.quadrant.hs_ht=$2 select.quadrant.hs_lt=$2 ;;
    grid)   echo select.n=$3 select.scorer=grid select.allocator=top select.loc_stride=none \
                 select.time_scorer=none select.time_allocator=month_grid select.month_step=$MSTEP ;;
    random) echo select.n=$3 select.scorer=random select.allocator=top select.loc_stride=$4 \
                 select.time_scorer=season_random select.time_allocator=every_location \
                 select.keep=one ;;
  esac
}
n=0
for SEED in ${SEEDS:-0}; do
for D in ${DATASETS:-planteur biomassters yieldsat_raw euromine}; do
  EP=50; OEP=50; LINKS=3; EPS=ep50; MSTEP=4
  case $D in
    biomassters) EP=10; OEP=10; EPS=ep10 ;;
    planteur)    LINKS=4 ;;              # ~1.3 x PASTIS
    geoidseries_flood) EP=6; OEP=8; LINKS=5; MSTEP=6
      # fixed models, no crops: month tokens (geoidseries_flood) and every date its own token
      # (geoidseries_flood_dates) at ps16 / 8 / 4, the same 8 x 212k training samples (dates ps4:
      # 24 epochs of 70k, so an epoch fits a link)
      for F in ${FIXED:-m16 m8 m4 d16 d8 d4}; do
        T=$([ ${F:0:1} = m ] && echo $D || echo ${D}_dates); P=${F:1}
        FX=(task=$T model.lr_patch=$P select.crop_px=$((P / 2)) select.loc_stride=none select.n=0
            select.scorer=entropy select.allocator=top select.quadrant.mode=sample
            select.quadrant.q_s=0)
        # (d4 at batch 16: 64 runs out of GPU memory once the backbone unfreezes)
        if [ $F = d4 ]; then FX+=(train.epochs=24 train.epoch_samples=70000 train.batch_size=16); FL=18
        else FX+=(train.epochs=$OEP); FL=$([ $P = 16 ] && echo $LINKS || echo 8); fi
        [ -f checkpoints/$(run_name "${FX[@]}" train.seed=$SEED).pt ] && continue
        chain "${D}_fixed_$F" $FL "" "${FX[@]}" >/dev/null
        n=$((n + 1))
      done ;;
  esac
  for S in ${SETUPS:-s16m8 s8m4}; do
    if [ $S = s16m8 ]; then P=16; C=8; NS=(8 16 32); STRIDE=8; LTAG=_cpx8_st8_poscenter_lr
    elif [ $S = s32m16 ]; then P=32; C=16; NS=(2 4 8)      # euromine only (2 x 2 year tokens)
      [ $D = euromine ] || continue
    else P=8; C=4; NS=(32 64 128); STRIDE=none; LTAG=_poscenter_lrps8_lr
      [ $D = euromine ] && continue       # euromine: ps4 hurts even at fixed resolution
    fi
    QS=(0.125 0.25 0.5)
    if [ $D = euromine ]; then      # per-step: the tiling windows (stride = crop), no controls
      FX=(task=euromine select.scorer=season_random select.allocator=pairs select.n=0
          model.lr_patch=$P train.epochs=30)
      FIX=$(run_name "${FX[@]}"); EDEP=""
      if [ ! -f checkpoints/$FIX.pt ]; then
        [ $P = 32 ] || { echo "missing $FIX" >&2; exit 1; }   # fix_ps32: trained here first
        EDEP="afterok:$(chain em_fix_ps32 2 "" "${FX[@]}")"; n=$((n + 1)); fi
      EL=$([ $S = s8m4 ] && echo 3 || echo 2)
      EW=(task=euromine model.lr_patch=$P select.crop_px=$C select.loc_stride=$C train.epochs=20
          train.init_ckpt=$FIX "${MIX[@]}")
      for i in 0 1 2; do
        Q=${QS[$i]}; N=${NS[$i]}
        if [[ " ${ARMS:-global grid random} " == *" global "* ]]; then
          chain "em_${AT}global_qs${Q}_$S" $EL "$EDEP" "${EW[@]}" "${QUAD[@]}" select.quadrant.q_s=$Q >/dev/null
          n=$((n + 1)); fi
        [ -n "${ATTN:-}" ] || continue
        for A in ${ARMS:-global grid random}; do
          [ $A = global ] && continue
          SC=$([ $A = grid ] && echo season_grid || echo season_random)
          chain "em_${AT}${A}_n${N}_$S" $EL "$EDEP" "${EW[@]}" select.scorer=$SC select.allocator=pairs \
            select.n=$N >/dev/null
          n=$((n + 1))
        done
      done
      continue
    fi
    for i in 0 1 2; do
      N=${NS[$i]}; Q=${QS[$i]}; DEP=""
      KF=$(python3 -c "print(f'{$N / 256:.3g}')")
      if [ -n "${SCHED:-}" ]; then        # one run: the oracle stage inside each arm's run
        DEP=""; LOC=""; SL=$([ $D = geoidseries_flood ] && echo 16 || echo $((LINKS * 2)))
      elif [ $D = planteur ] || [ $D = geoidseries_flood ]; then   # its location oracle, trained here
        OR=(task=$D model.lr_patch=$P select.crop_px=$C select.loc_stride=$C select.n=$N
            select.scorer=label select.allocator=top train.epochs=$OEP)
        LOC=$(run_name "${OR[@]}")
        if [ -f checkpoints/$LOC.pt ]; then DEP=""
        else DEP="afterok:$(chain "${D}_oracle_n${N}_$S" $LINKS "" "${OR[@]}")"; n=$((n + 1)); fi
      else
        LOC="inject_fft_${D}_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_${EPS}"
        [ -f checkpoints/$LOC.pt ] || { echo "missing $LOC" >&2; exit 1; }
      fi
      COMMON=(task=$D model.lr_patch=$P select.crop_px=$C train.epochs=$EP train.init_ckpt=$LOC
              "${MIX[@]}")
      if [ -n "${SCHED:-}" ]; then
        COMMON=(task=$D model.lr_patch=$P select.crop_px=$C train.epochs=$((OEP + EP))
                train.stage1_epochs=$OEP train.stage1_n=$N "${MIX[@]}"); fi
      CL=${SL:-$LINKS}
      for A in ${ARMS:-global grid random}; do
        [ $A = none ] && continue          # ARMS=none: the fixed models / oracles only
        if [ $A = global ]; then
          for K in ${KEEPS:-one half}; do
            chain "${D}_${AT}global_${K}_qs${Q}_$S" $CL "$DEP" "${COMMON[@]}" \
              select.loc_stride=none $(arm global $K) select.quadrant.q_s=$Q >/dev/null
            n=$((n + 1))
          done
        else
          chain "${D}_${AT}${A}_n${N}_$S" $CL "$DEP" "${COMMON[@]}" $(arm $A one $N $STRIDE) >/dev/null
          n=$((n + 1))
        fi
      done
    done
  done
done
done
echo "$n chains queued"
