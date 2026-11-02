#!/bin/bash
# Attention-threshold seasons (select.quadrant.time_thr), no retraining: the plotted seed 0 models
# at q_s 1/4 evaluated with every season above the threshold (at least the top one). thr 0 = the
# one-season control, which must repeat the plotted run.
#   [DATASETS=..] [SETUPS=..] [THRS="0 0.4 0.2"] [DRY=1] bash scripts/slurm/mixres_time_thr_eval.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010,kn029
for D in ${DATASETS:-pastis biomassters yieldsat_raw}; do
  E=$([ $D = biomassters ] && echo 10 || echo 50)
  for S in ${SETUPS:-s16m8 s8m4}; do
    if [ $S = s16m8 ]; then P=16; C=8; else P=8; C=4; fi
    CK=$(ls checkpoints | grep -E "^mixres_${D}_ps${P}_quadg-skip-qs0.25-qt0\.5-hhone-crop-score-xhr-s1o${E}_n0_c${C}_seed0_[0-9a-f]+\.pt$")
    for T in ${THRS:-0 0.4 0.2}; do
      cmd=(sbatch --parsable -J "mxtt_${D}_${S}_thr$T" --time=1:00:00 --cpus-per-task=8 --exclude=$EXC
           scripts/slurm/run.sh exp.mixres.train task=$D model.lr_patch=$P select.crop_px=$C select.n=0
           select.loc_stride=none select.scorer=entropy select.allocator=top
           select.quadrant.lowspace=skip select.quadrant.mode=global select.quadrant.space=crop
           select.quadrant.rank=score select.quadrant.hs_ht=one select.quadrant.hs_lt=one
           select.quadrant.q_t=0.5 select.quadrant.q_s=0.25 select.quadrant.time_thr=$T
           model.attn=cross_hr train.init_ckpt=${CK%.pt} train.epochs=0 train.freeze_model=true)
      if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
    done
  done
done
