#!/bin/bash
# Quadrant replicating entropy-top + attention-every_location, then one change at a time:
#   rep       lowspace skip, mode sample, space crop, rank score, hs_ht = hs_lt = one | half
#             (same crops as mxstent_attn_{one,half}: selection checked identical on the test set)
#   cell      rep, u_s per low-res cell instead of per crop window
#   weight    rep, seasons ranked by attention weight (score + log months)
#   global    rep, the spatial threshold of the training distribution
#   top2      rep one, but high-temporal-uncertainty cells get the top-2 attended seasons
#             (hs_ht half), q_t 1/4 | 1/2
#   top2g     top2 + global (the combined arm)
#   [QSS="0.0625 0.1875 0.375 0.75" ARMS=global: the adaptive-budget curve at finer q_s; a q_s
#   without its own location oracle warms from the nearest one]
# PASTIS s16m8 + s8m4, q_s = 1/8, 1/4, 1/2 (n 8/16/32 | 32/64/128), warm from the location oracle
# at that n, 3-link chains. Mail off.
# INIT=lr: warm from the fixed-resolution model of that LR patch instead (no crops; one start for
# every q_s, run names _warmlr). KEEPS_ONLY=one: only that keep.
# ATTN=cross | cross_hr: model.attn (crops read only the season tokens / + the sample's other
# crops; the season tokens never read crops, inference reuses the crop-free pass).
# EXIT=L: model.exit_layer (the crop-free pass the selector reads exits after block L), its exit
# head from the green run's exp.mixres.early_exit_probe heads at that q_s (distill_L), distilled
# online from the full pass while training.
# LOC_K=k: every q_s warms from the location oracle at k (one shared start, e.g. 0.25) instead of
# the one at its own budget (whose val/test gaps differ).
# SETUPS=s6m3 (with SCHED): ps6 seasons + ps3 month crops (padded frame, 11 x 11 windows).
# SETUPS=s4m2 (with SCHED): ps4 seasons + ps2 month crops (4 px windows, four ps2 tokens each).
# SLINKS=n: links per SCHED chain (default 4). BS=b: train.batch_size (s4m2 q_s >= 1/4 OOM at 32).
# OFFGRID=1: select.quadrant.offgrid (as many windows as the grid gives, at any pixel offset,
# sequential-greedy NMS in batched rounds).
# RANK=oracle: seasons per crop window by the label (each season map's loss there; oracle time).
# SEED=s: train.seed (seed repeats; default 0).
# SCHED=E: the selector schedule in ONE run, no warm start: E epochs with the location oracle
# at the same budget (q_s x the crop windows, every present month; model.attn S1ATTN, default
# joint), then 50 with the arm's selector (and model.attn ATTN). 4-link chains.
# UNC=msr: spatial uncertainty = softmax response (1 - max p) instead of entropy.
#   [UNC=msr] [SEED=1] [SCHED=50 [S1ATTN=cross_hr]] [LOC_K=0.25] [EXIT=4] [ATTN=cross] [INIT=lr] [KEEPS_ONLY=one] [DRY=1] [SMOKE=1] [ARMS="rep top2"] bash scripts/slurm/mixres_quadrant_rep.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source env_setup/env_olmo.sh >/dev/null 2>&1
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
LINKS=${LINKS:-3}
arm() {   # arm <name> -> "<keep list>|<q_t list>|<quadrant overrides>"
  local R="select.quadrant.space=crop select.quadrant.rank=score"
  case $1 in
    rep)    echo "one half|0.5|$R select.quadrant.mode=sample" ;;
    cell)   echo "one half|0.5|select.quadrant.rank=score select.quadrant.mode=sample" ;;
    weight) echo "one half|0.5|select.quadrant.space=crop select.quadrant.mode=sample" ;;
    global) echo "one half|0.5|$R select.quadrant.mode=global" ;;
    top2)   echo "top2|0.25 0.5|$R select.quadrant.mode=sample" ;;
    top2g)  echo "top2|0.25 0.5|$R select.quadrant.mode=global" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}
n=0
for S in ${SETUPS:-s16m8 s8m4}; do
if [ "$S" = s16m8 ]; then GEO=(select.crop_px=8 select.loc_stride=none model.lr_patch=16); LTAG=_cpx8_st8_poscenter_lr; KD=4
  LRCK=inject_fft_pastis_tiny_none_native_set_lr0.001_ep50
elif [ "$S" = s8m4 ]; then GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8); LTAG=_poscenter_lrps8_lr; KD=1
  LRCK=inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50
elif [ "$S" = s6m3 ]; then   # ps6 seasons (the tile padded to 66 px), 6 px windows as four
  # ps3 crops each (SCHED only)
  [ -n "${SCHED:-}" ] || { echo "s6m3: SCHED only" >&2; exit 1; }
  GEO=(select.crop_px=6 select.loc_stride=none model.lr_patch=6 model.crop_patch=3); LTAG=; KD=1
else   # s4m2: ps4 seasons, the selector's 4 px windows each tokenized as four ps2 crops (no
  # location oracle at ps4: SCHED only)
  [ -n "${SCHED:-}" ] || { echo "s4m2: SCHED only" >&2; exit 1; }
  GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=4 model.crop_patch=2); LTAG=; KD=1
  LRCK=inject_fft_pastis_tiny_none_native_set_lrps4_lr0.001_ep50; fi
loc() {   # the location oracle at that n, else the nearest one (log scale) that exists
  python3 - "$1" "$KD" "$LTAG" <<'PY'
import math, os, sys
q, kd, tag = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
name = lambda k: f"inject_fft_pastis_tiny_oracle_L0_k{k:.3g}_loc_native_set{tag}0.001_ep50"
have = [k for k in (2 ** -i for i in range(8)) if os.path.exists(f"checkpoints/{name(k)}.pt")]
if os.environ.get("LOC_K"):
    have = [float(os.environ["LOC_K"])]
print(name(min(have, key=lambda k: abs(math.log(k * kd / q)))))
PY
}
for A in ${ARMS:-rep cell weight global top2 top2g}; do
  IFS='|' read -r KEEPS QTS OV <<<"$(arm $A)"
  for K in ${KEEPS_ONLY:-$KEEPS}; do
    if [ "$K" = top2 ]; then KQ=(select.quadrant.hs_ht=half select.quadrant.hs_lt=one)
    else KQ=(select.quadrant.hs_ht=$K select.quadrant.hs_lt=$K); fi
    C=(task=pastis select.n=0 "${GEO[@]}" select.scorer=entropy select.allocator=top
       select.quadrant.lowspace=skip $OV "${KQ[@]}" train.max_hours=2.5 ${ATTN:+model.attn=$ATTN} ${UNC:+select.quadrant.unc=$UNC}
       ${SEED:+train.seed=$SEED} ${BS:+train.batch_size=$BS} ${OFFGRID:+select.quadrant.offgrid=true} ${RANK:+select.quadrant.rank=$RANK} ${SMOKE:+train.limit_batches=3 train.seed=999})
    EP=${SMOKE:+2}; EP=${EP:-50}
    for QS in ${QSS:-0.125 0.25 0.5}; do for QT in $QTS; do
      EX=()
      if [ -n "${EXIT:-}" ]; then
        G=$(ls checkpoints | grep -E "^mixres_pastis_ps${GEO[2]#model.lr_patch=}_quadg-skip-qs${QS}-qt0\.5-hhone-crop-score_n0_c[48]_warm_seed0_[0-9a-f]+_exit\.pt$" | head -1)
        [ -n "$G" ] || { echo "no exit heads for $S qs$QS" >&2; exit 1; }
        EX=(model.exit_layer=$EXIT train.init_exit=${G%.pt}:distill_$EXIT); fi
      if [ -n "${SCHED:-}" ]; then
        NW=$(case $S in s16m8) echo 64 ;; s6m3) echo 121 ;; *) echo 256 ;; esac)  # windows / sample
        EX+=(train.stage1_epochs=$SCHED train.stage1_n=$(python3 -c "print(round($QS * $NW))")
             train.stage1_attn=${S1ATTN:-joint})
      fi
      if [ -z "${SCHED:-}" ]; then
        L=$(loc $QS); [ "${INIT:-}" = lr ] && L=$LRCK; [ -f "checkpoints/$L.pt" ] || { echo "missing $L" >&2; exit 1; }; fi
      prev=""; for i in $(seq 1 $([ -n "${SCHED:-}" ] && echo ${SLINKS:-4} || echo $LINKS)); do
        INITC=(train.init_ckpt=${L:-}); EPC=$EP
        [ -n "${SCHED:-}" ] && { INITC=(); EPC=$((SCHED + EP)); }
        prev=$(sub "mxqrep_${A}${INIT:+_$INIT}${ATTN:+_$ATTN}${EXIT:+_ex$EXIT}${LOC_K:+_k$LOC_K}${SCHED:+_s1o$SCHED${S1ATTN:+$S1ATTN}}${SEED:+_seed$SEED}${UNC:+_$UNC}${OFFGRID:+_og}${RANK:+_$RANK}_${K}_qs${QS}_qt${QT}_${S}_$i" "$prev" "${C[@]}" train.epochs=$EPC \
               "${INITC[@]}" select.quadrant.q_s=$QS select.quadrant.q_t=$QT "${EX[@]}"); done
      n=$((n + 1))
    done; done
  done
done; done
echo "$n chains queued"
