#!/bin/bash
# The MiSTR simplification round (doc: MiSTR simplification ideas), ours at q_s 1/4, S16+S8 and
# S8+S4, one-run schedule, cross_hr, as plotted. One change at a time + all three together:
#   base   the plotted method (a run that already has its checkpoint is skipped)
#   nht    model.hr_type=false       no learned crop embedding
#   tm     model.time_step=month     the month index as every token's time step
#   npr    model.months_prior=false  no log-months prior in the readout
#   all    nht + tm + npr
#   p2     model.second_pass=true    the crop tokens' own copy of the blocks from stage 2 (not a
#                                    simplification: twice the parameters, the same compute)
#   tt4, tt2  select.quadrant.time_thr 0.4 / 0.2: every season at that readout attention or more
#             (at least the top one) trained in stage 2
#   simp   nht + npr, the two kept simplifications
#   simpw  simp, warm from the plotted base checkpoint (trained with the crop embedding and the
#          prior; the embedding zeroed on load): stage 2 only, half the stage 2 epochs
#   lrl    train.lr_loss=1           + the crop-free output's own loss in stage 2
#   dec    lrl + model.detach_lr     season tokens learn from that loss alone
#   decp2  dec + second pass         no shared blocks: low-res and high-res fully apart
#   hlr    train.head_lr=0.001       head and readout at 10 x the stage 2 rate
#   onep   model.one_pass=true       stage 2: the selector reads the gradient pass (drop path on)
# Fixed seasonal models for the ideas that change them (ps16, ps8; no crops, q_s 0): fixed_base,
# fixed_tm, fixed_npr, fixed_all (tm + npr), all through this code path.
# Mail off. <= 3 h links (train.max_hours 2.5, afterany).
#   [DATASETS="pastis biomassters yieldsat_raw"] [SETUPS="s16m8 s8m4"] [SEEDS="0 1 2"]
#   [ARMS="base nht tm npr all p2"] [FIXED="base tm npr all"] [DRY=1] [SMOKE=1]
#   bash scripts/slurm/mixres_simplify.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source env_setup/env_olmo.sh >/dev/null 2>&1
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010,kn029
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=8 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train
             train.max_hours=2.5 "$@" ${SMOKE:+train.limit_batches=3 train.seed=999})
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
run_name() {   # the run name exp.mixres.train gives these overrides
  python -W ignore -c "import sys; from manyup.mixres.config import RunConfig, apply_overrides; \
print(apply_overrides(RunConfig(), sys.argv[1:]).name())" "$@" 2>/dev/null
}
chain() {   # chain <job name> <links> <overrides...>: skipped when the run has its checkpoint
  local name=$1 links=$2 prev="" i; shift 2
  if [ -f "checkpoints/$(run_name "$@").pt" ]; then echo "have $name" >&2; return; fi
  for i in $(seq 1 "$links"); do prev=$(sub "${name}_$i" "$prev" "$@"); done
  n=$((n + 1))
}
flags() {   # flags <arm> -> its overrides
  case $1 in
    base) ;;
    nht) echo model.hr_type=false ;;
    tm)  echo model.time_step=month ;;
    npr) echo model.months_prior=false ;;
    all) echo model.hr_type=false model.time_step=month model.months_prior=false ;;
    p2)  echo model.second_pass=true ;;
    simp) echo model.hr_type=false model.months_prior=false ;;
    lrl) echo train.lr_loss=1 ;;
    dec) echo train.lr_loss=1 model.detach_lr=true ;;
    decp2) echo train.lr_loss=1 model.detach_lr=true model.second_pass=true ;;
    hlr) echo train.head_lr=0.001 ;;
    onep) echo model.one_pass=true ;;
    tt4) echo select.quadrant.time_thr=0.4 ;;
    tt2) echo select.quadrant.time_thr=0.2 ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}
QUAD=(select.n=0 select.loc_stride=none select.scorer=entropy select.allocator=top
      select.quadrant.lowspace=skip select.quadrant.mode=global select.quadrant.space=crop
      select.quadrant.rank=score select.quadrant.hs_ht=one select.quadrant.hs_lt=one
      select.quadrant.q_t=0.5 select.quadrant.q_s=0.25 model.attn=cross_hr train.stage1_attn=joint)
n=0
for D in ${DATASETS:-pastis biomassters yieldsat_raw}; do
  E=$([ $D = biomassters ] && echo 10 || echo 50)
  for S in ${SETUPS:-s16m8 s8m4}; do
    if [ $S = s16m8 ]; then P=16; C=8; N=16; else P=8; C=4; N=64; fi
    for SD in ${SEEDS:-0 1 2}; do
      for A in ${ARMS:-base nht tm npr all p2}; do
        if [ $A = simpw ]; then
          CK=$(ls checkpoints | grep -E "^mixres_${D}_ps${P}_quadg-skip-qs0\.25-qt0\.5-hhone-crop-score-xhr-s1o${E}_n0_c${C}_seed${SD}_[0-9a-f]+\.pt$" | head -1)
          [ -n "$CK" ] || { echo "no base checkpoint $D $S seed $SD" >&2; continue; }
          chain "mxsimp_${D}_${S}_${A}_seed$SD" 3 task=$D model.lr_patch=$P select.crop_px=$C \
            "${QUAD[@]}" train.epochs=$((E / 2)) train.init_ckpt=${CK%.pt} train.seed=$SD \
            $(flags simp)
          continue
        fi
        chain "mxsimp_${D}_${S}_${A}_seed$SD" 6 task=$D model.lr_patch=$P select.crop_px=$C \
          "${QUAD[@]}" train.epochs=$((2 * E)) train.stage1_epochs=$E train.stage1_n=$N \
          train.seed=$SD $(flags $A)
      done
      for A in ${FIXED:-base tm npr all}; do
        F=$([ $A = all ] && echo "model.time_step=month model.months_prior=false" || flags $A)
        chain "mxsimp_${D}_${S}_fixed_${A}_seed$SD" $([ $P = 16 ] && echo 3 || echo 4) task=$D \
          model.lr_patch=$P select.crop_px=$C select.loc_stride=none select.n=0 \
          select.scorer=entropy select.allocator=top select.quadrant.mode=sample \
          select.quadrant.q_s=0 train.epochs=$E train.seed=$SD $F
      done
    done
  done
done
echo "$n chains queued"
