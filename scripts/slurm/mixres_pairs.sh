#!/bin/bash
# More season / month patch-size pairs for ours (entropy space with the training-set threshold +
# attention time, ONE season per location, cross attention: the PASTIS "ours"), as the selector
# schedule in one run (train.stage1_*: E epochs with the location oracle at the same budget, joint
# attention, then E with ours and cross attention; no warm start, so no oracle checkpoint per
# pair). q_s 1/8, 1/4, 1/2 per pair:
#   s16m12  ps16 seasons + 12 px crops      s8m6   ps8 seasons + 6 px crops
#   s10m8   ps10 seasons + 8 px crops       s6m4   ps6 seasons + 4 px crops
#   s10m5   ps10 seasons, 10 px windows as four ps5 crops (model.crop_patch)
#   s6m3    ps6 seasons, 6 px windows as four ps3 crops (PASTIS: already run by
#           mixres_quadrant_rep.sh SETUPS=s6m3)
#   s14m7   ps14 seasons (70 px frame), 14 px windows as four ps7 crops
#   s12m6   ps12 seasons (72 px frame), 12 px windows as four ps6 crops
# SEEDS="0 1 2": train.seed (job names gain _seedS for S != 0; default 0).
# A patch that does not divide the 64 px tile tiles it padded to whole patches (ps10 seasons: 70
# px, 12 px windows: 72); a crop window that does not tile its season cell reads the time
# decision of the season cell under its centre (selectors.Quadrant). E = 50 (biomassters 10).
# Mail off. <= 3 h links (train.max_hours 2.5, afterany).
#   [DATASETS="pastis planteur biomassters yieldsat_raw"] [PAIRS="s16m12 s10m8 s10m5 s8m6 s6m4 s6m3"]
#   [QSS="0.125 0.25 0.5"] [SEEDS=0] [DRY=1] [SMOKE=1] bash scripts/slurm/mixres_pairs.sh
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
pair() {   # pair <name> -> "<season patch> <window px> <crop patch or 0> <links>"
  case $1 in
    s16m12) echo 16 12 0 4 ;;  s10m8) echo 10 8 0 4 ;;  s10m5) echo 10 10 5 4 ;;
    s8m6)   echo 8 6 0 4 ;;    s6m4)  echo 6 4 0 7 ;;   s6m3)  echo 6 6 3 7 ;;
    s14m7)  echo 14 14 7 4 ;;  s12m6) echo 12 12 6 4 ;;
    *) echo "unknown pair $1" >&2; exit 1 ;;
  esac
}
QUAD=(select.n=0 select.loc_stride=none select.scorer=entropy select.allocator=top
      select.quadrant.lowspace=skip select.quadrant.mode=global select.quadrant.space=crop
      select.quadrant.rank=score select.quadrant.hs_ht=one select.quadrant.hs_lt=one
      select.quadrant.q_t=0.5 model.attn=cross_hr train.stage1_attn=joint)
n=0
for D in ${DATASETS:-pastis planteur biomassters yieldsat_raw}; do
  E=$([ $D = biomassters ] && echo 10 || echo 50)
  for S in ${PAIRS:-s16m12 s10m8 s10m5 s8m6 s6m4 s6m3}; do
    [ $D = pastis ] && [ $S = s6m3 ] && continue         # (mixres_quadrant_rep.sh SETUPS=s6m3)
    read -r P C CP L <<<"$(pair $S)"
    [ $D = planteur ] && L=$((L + 1))                    # (~1.3 x PASTIS per epoch)
    NW=$(python3 -c "import math; print(math.ceil(64 / $C) ** 2)")   # crop windows per sample
    for Q in ${QSS:-0.125 0.25 0.5}; do for SD in ${SEEDS:-0}; do
      SX=$([ $SD = 0 ] || echo _seed$SD)
      BS=()
      [[ $P = 6 && $Q = 0.5 ]] && BS=(train.batch_size=16)   # (ps4+ps2 / ps6+ps3 OOM at 32)
      prev=""
      for i in $(seq 1 $L); do
        prev=$(sub "mxpair_${D}_${S}_qs${Q}${SX}_$i" "$prev" task=$D model.lr_patch=$P \
               select.crop_px=$C $([ $CP != 0 ] && echo model.crop_patch=$CP) "${QUAD[@]}" \
               select.quadrant.q_s=$Q train.epochs=$((2 * E)) train.stage1_epochs=$E \
               train.stage1_n=$(python3 -c "print(round($Q * $NW))") "${BS[@]}" train.seed=$SD)
      done
      n=$((n + 1))
    done; done
  done
done
echo "$n chains queued"
