#!/bin/bash
# Space then time (selectors.SpaceTime): pick n locations, then at each only one season / two
# seasons of its months. The best spatial and temporal selectors combined, vs content-blind
# controls at the same budget (n locations x ~1/4 or ~1/2 of the present months):
#   ent_learn  space: highest-entropy windows (the model's own uniform pass); time: the learned
#              season score alone (season_residual xattn_conv, oracle ListNet, prior none),
#              each location's lowest-score season(s)
#   grid       space: n-window lattice; time: every 4th (one) / 2nd (two) present month
#   random     space: n random windows; time: random season(s) per location
#   ent_attn   space by entropy; time: each location's season(s) the readout attends to most
#              (selectors.ReadoutAttention: no labels, no extra network)
#   ent_ent    space and time by entropy (each location's lowest-entropy season(s))
#   ent_T      + a per-season temperature on the season entropy (fitted online, NLL)
#   ent_sl     + a per-season loss on every season token's own map (into the backbone)
#   ent_sh     + per-season heads (copies of the shared head), trained by that loss
#   ent_all    ent_sh + ent_T
#   ent_Tv / ent_allv   ent_T / ent_all with the temperature fitted on the val split each epoch
#   ent_shd    per-season heads trained by the per-season loss on DETACHED season tokens (the
#              backbone learns from the task alone); ent_shdv + the val-fitted temperature
#   (ARMS="ent_ent ent_T ent_sl ent_sh ent_all" for these)
#   ran_grid / ran_ent / ran_attn   space: n random windows; time: every 4th (one) / 2nd (two)
#              present month, lowest entropy, most attended (with arm random: time selectors alone)
# keep one / half(=two seasons), PASTIS s16m8 n=8/16/32, s8m4 n=32/64/128, warm from the
# location oracle at k = n/256 (as the spatial table), 3-link chains. Mail off.
# INIT=lr: warm from the fixed-resolution model of that LR patch instead (no crops; one start for
# every n, run names _warmlr).
#   [INIT=lr] [DRY=1] [SMOKE=1 LINKS=1] [ARMS=..] [KEEPS=..] [NLIST=..] bash scripts/slurm/mixres_spacetime.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=8 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
arm() {   # arm <name> <keep> <stride> -> the overrides
  local step=$([ "$2" = one ] && echo 4 || echo 2)
  case $1 in
    ent_learn) echo select.scorer=entropy select.allocator=top select.loc_stride=$3 \
                    select.time_scorer=season_residual select.time_allocator=every_location \
                    select.keep=$2 select.residual.arch=xattn_conv select.residual.train=oracle \
                    select.residual.prior=none ;;
    grid)      echo select.scorer=grid select.allocator=top select.loc_stride=none \
                    select.time_scorer=none select.time_allocator=month_grid select.month_step=$step ;;
    random)    echo select.scorer=random select.allocator=top select.loc_stride=$3 \
                    select.time_scorer=season_random select.time_allocator=every_location \
                    select.keep=$2 ;;
    # random locations, then each time selector (random: the random arm above)
    ran_grid)  echo select.scorer=random select.allocator=top select.loc_stride=$3 \
                    select.time_scorer=none select.time_allocator=month_grid select.month_step=$step ;;
    ran_ent)   echo select.scorer=random select.allocator=top select.loc_stride=$3 \
                    select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 ;;
    ran_attn)  echo select.scorer=random select.allocator=top select.loc_stride=$3 \
                    select.time_scorer=attention select.time_allocator=every_location select.keep=$2 ;;
    # entropy for time too, made season-comparable (per-season supervision / temperature):
    ent_attn)  echo select.scorer=entropy select.allocator=top select.loc_stride=$3 \
                    select.time_scorer=attention select.time_allocator=every_location select.keep=$2 ;;
    ent_ent)   echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 ;;
    ent_T)     echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 select.entropy_T=true ;;
    ent_sl)    echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 train.season_loss=1 ;;
    ent_sh)    echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 model.season_heads=4 train.season_loss=1 ;;
    ent_Tv)    echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 select.entropy_T=true select.entropy_T_fit=val ;;
    ent_allv)  echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 model.season_heads=4 train.season_loss=1 select.entropy_T=true select.entropy_T_fit=val ;;
    ent_shd)   echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 model.season_heads=4 train.season_loss=1 train.season_loss_detach=true ;;
    ent_shdv)  echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 model.season_heads=4 train.season_loss=1 train.season_loss_detach=true select.entropy_T=true select.entropy_T_fit=val ;;
    ent_all)   echo select.scorer=entropy select.allocator=top select.loc_stride=$3 select.time_scorer=entropy select.time_allocator=every_location select.keep=$2 model.season_heads=4 train.season_loss=1 select.entropy_T=true ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}
for S in s16m8 s8m4; do
if [ "$S" = s16m8 ]; then NS="8 16 32"; GEO=(select.crop_px=8 model.lr_patch=16); STRIDE=8
  LTAG=_cpx8_st8_poscenter_lr; LRCK=inject_fft_pastis_tiny_none_native_set_lr0.001_ep50
else NS="32 64 128"; GEO=(select.crop_px=4 model.lr_patch=8); STRIDE=none
  LTAG=_poscenter_lrps8_lr; LRCK=inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50; fi
NV=NLIST_$S
for N in ${!NV:-${NLIST:-$NS}}; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')")
LOC="inject_fft_pastis_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_ep50"
[ "${INIT:-}" = lr ] && LOC=$LRCK
[ -f "checkpoints/$LOC.pt" ] || { echo "missing $LOC" >&2; exit 1; }
COMMON=(task=pastis select.n=$N "${GEO[@]}" train.epochs=50 train.max_hours=2.5 train.init_ckpt=$LOC
        ${SMOKE:+train.limit_batches=3 train.seed=999})
for K in ${KEEPS:-one half}; do
for A in ${ARMS:-ent_learn grid random}; do
  prev=""; for i in $(seq 1 ${LINKS:-3}); do
    prev=$(sub "mxst${A}${INIT:+_$INIT}_${K}_${S}_n${N}_$i" "$prev" "${COMMON[@]}" $(arm $A $K $STRIDE)); done
done
done
echo "$S n$N queued"
done
done
