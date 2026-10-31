#!/bin/bash
# Mixed resolution on a NON-FlexiViT backbone: Prithvi-EO-2.0 tiny-TL (manyup.encoders.prithvi:
# its 16 px kernel PI-resized per patch size, physical sin-cos positions), PASTIS S2 only
# (task pastis_s2: Prithvi reads 6 HLS-equivalent S2 bands). Every model goes through
# exp.mixres.train (same readout, same recipe as the OlmoEarth PASTIS runs):
#   fixed seasonal  task pastis_s2, the season tokens alone (quadrant sample q_s 0 = no crops),
#                   ps FIXED_PS, 50 epochs cold
#   fixed monthly   task pastis_s2_mon (each month its own low-res step), the same
#   ours            the committed method (mixres_quadrant_rep.sh SCHED=50 ATTN=cross_hr
#                   ARMS=global KEEPS_ONLY=one): 50 epochs location oracle (joint), then 50 with
#                   entropy space (global train-set threshold) + readout-attention time, one
#                   season, model.attn cross_hr; SETUPS s16m8 / s8m4 / s6m3, q_s 1/8, 1/4, 1/2
# Mail off, <= 3 h links (train.max_hours 2.5, afterany).
# CROP_TIME=frac: model.crop_time (a month crop at its season slot + its offset in the season:
# Prithvi's crops otherwise cannot tell the months of one season apart).
#   [CROP_TIME=frac] [MODEL=prithvi_tiny] [WHAT="fixed ours"] [FIXED_PS="16 10 8 6 4"] [SETUPS="s16m8 s8m4 s6m3"]
#   [QSS="0.125 0.25 0.5"] [SEED=0] [DRY=1] bash scripts/slurm/mixres_prithvi.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source env_setup/env_olmo.sh >/dev/null 2>&1
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
MODEL=${MODEL:-prithvi_tiny}; SEED=${SEED:-0}
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train
             model.model_size=$MODEL train.seed=$SEED train.max_hours=2.5
             ${CROP_TIME:+model.crop_time=$CROP_TIME} "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
chain() {   # chain <name> <links> <overrides...>
  local name=$1 links=$2 prev="" i; shift 2
  for i in $(seq 1 "$links"); do prev=$(sub "mxpr_${MODEL#prithvi_}${CROP_TIME:+_ct$CROP_TIME}_${name}_seed${SEED}_$i" "$prev" "$@"); done
}
n=0
if [[ " ${WHAT:-fixed ours} " == *" fixed "* ]]; then
  for P in ${FIXED_PS:-16 10 8 6 4}; do
    for T in pastis_s2 pastis_s2_mon; do
      TAG=$([ $T = pastis_s2 ] && echo sea || echo mon)
      L=$([ $P -le 6 ] && echo 4 || echo 3)
      chain fix_${TAG}${P} $L task=$T model.lr_patch=$P select.crop_px=$((P / 2)) \
        select.loc_stride=none select.n=0 select.scorer=entropy select.allocator=top \
        select.quadrant.mode=sample select.quadrant.space=crop select.quadrant.q_s=0 train.epochs=50
      n=$((n + 1))
    done
  done
fi
if [[ " ${WHAT:-fixed ours} " == *" ours "* ]]; then
  for S in ${SETUPS:-s16m8 s8m4 s6m3}; do
    case $S in
      s16m8) GEO=(select.crop_px=8 model.lr_patch=16); NW=64 ;;
      s8m4)  GEO=(select.crop_px=4 model.lr_patch=8); NW=256 ;;
      s6m3)  GEO=(select.crop_px=6 model.lr_patch=6 model.crop_patch=3); NW=121 ;;
      s4m2)  GEO=(select.crop_px=4 model.lr_patch=4 model.crop_patch=2); NW=256 ;;
      *) echo "unknown setup $S" >&2; exit 1 ;;
    esac
    for QS in ${QSS:-0.125 0.25 0.5}; do
      chain ours_${S}_qs$QS 4 task=pastis_s2 select.n=0 select.loc_stride=none "${GEO[@]}" \
        select.scorer=entropy select.allocator=top select.quadrant.lowspace=skip \
        select.quadrant.space=crop select.quadrant.rank=score select.quadrant.mode=global \
        select.quadrant.hs_ht=one select.quadrant.hs_lt=one select.quadrant.q_s=$QS \
        select.quadrant.q_t=0.5 model.attn=cross_hr train.epochs=100 train.stage1_epochs=50 \
        train.stage1_n=$(python3 -c "print(round($QS * $NW))") train.stage1_attn=joint
      n=$((n + 1))
    done
  done
fi
echo "$n chains queued"
