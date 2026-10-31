#!/bin/bash
# The LossUp scorer (selectors.LossUpScorer: mAnyUp-guided predicted loss maps, trained by the
# space / time ListNet rankings + per-pixel regression), next to the residual scorers:
#   frozen (on the space + time oracle models, after their chains; IDS as mixres_st_learn.sh),
#   50 epochs, 2-link chains:
#     lu_space  lossup locations + oracle seasons
#     lu_time   oracle locations + lossup seasons
#     lu_both   lossup locations + lossup seasons (one network for both)
#   co-trained with the model (as mixres_spacetime.sh's ent_learn, warm from the location oracle
#   at k = n/256), 50 epochs, 3-link chains:
#     lu_cotrain  lossup locations + lossup seasons
# keep one / half, PASTIS s16m8 n=8/16/32, s8m4 n=32/64/128. Mail off.
#   IDS=<file> [DRY=1] bash scripts/slurm/mixres_st_lossup.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source env_setup/env_olmo.sh >/dev/null 2>&1
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
name() { python -c "import sys; from manyup.mixres.config import *; \
c = apply_overrides(RunConfig(), sys.argv[1:]); c.check('classification'); print(c.name())" "$@" 2>/dev/null; }
RES=(select.residual.arch=manyup select.residual.train=oracle)
while read -r S K N AFTER; do
  if [ "$S" = s16m8 ]; then GEO=(select.crop_px=8 select.loc_stride=8 model.lr_patch=16)
    LTAG=_cpx8_st8_poscenter_lr
  else GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8); LTAG=_poscenter_lrps8_lr; fi
  KF=$(python3 -c "print(f'{$N / 256:.3g}')")
  LOC="inject_fft_pastis_tiny_oracle_L0_k${KF}_loc_native_set${LTAG}0.001_ep50"
  BASE=(task=pastis select.n=$N "${GEO[@]}" select.allocator=top select.time_allocator=every_location
        select.keep=$K)
  ORC=$(name "${BASE[@]}" train.epochs=50 train.max_hours=2.5 train.init_ckpt=$LOC select.scorer=label \
        select.time_scorer=label)
  T="${K}_${S}_n$N"
  for A in space time both; do
    case $A in
      space) X=(select.scorer=lossup select.time_scorer=label) ;;
      time)  X=(select.scorer=label select.time_scorer=lossup) ;;
      both)  X=(select.scorer=lossup select.time_scorer=lossup) ;;
    esac
    prev=$AFTER; for i in 1 2; do
      prev=$(sub "mxlu_${A}_${T}_$i" "$prev" "${BASE[@]}" train.init_ckpt=$ORC train.freeze_model=true \
             train.epochs=50 train.max_hours=2.5 "${X[@]}" "${RES[@]}")
    done
  done
  prev=""; for i in 1 2 3; do
    prev=$(sub "mxlu_cotrain_${T}_$i" "$prev" "${BASE[@]}" train.init_ckpt=$LOC train.epochs=50 \
           train.max_hours=2.5 select.scorer=lossup select.time_scorer=lossup "${RES[@]}")
  done
  echo "$T queued"
done < "${IDS:?IDS=<file of setup keep n last-job-id lines>}"
