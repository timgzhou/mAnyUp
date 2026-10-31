#!/bin/bash
# Learn the selectors on the FROZEN space + time oracle models (mixres_st_oracle.sh), one change
# at a time and combined, then fine-tune jointly. Per setup / keep / n:
#   frozen, 50 epochs, the scorers' ranking losses only (train.freeze_model), 2-link chains:
#     space   learned locations (residual xattn_conv, prior none, ListNet on the pooled window
#             loss) + oracle seasons
#     time    oracle locations + learned seasons (season_residual, same settings)
#     both    learned locations + learned seasons
#   frozen, eval only: oracle + oracle (sanity), entropy + oracle, random + random
#   joint fine-tune, 10 epochs, lr 1e-4 (model x 0.1 = 1e-5, scorers 1e-4):
#     ft      from the oracle model + the frozen-trained "both" scorers (train.init_selector)
#     ft0     the same with the scorers from scratch (the matched co-training baseline)
# Waits for each oracle chain: IDS = file of "<setup> <keep> <n> <last job id>" lines.
#   IDS=<file> [DRY=1] bash scripts/slurm/mixres_st_learn.sh
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
RES=(select.residual.arch=xattn_conv select.residual.train=oracle select.residual.prior=none)
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
  F=("${BASE[@]}" train.init_ckpt=$ORC train.freeze_model=true)
  T="${K}_${S}_n$N"
  for A in space time both; do
    case $A in
      space) X=(select.scorer=residual select.time_scorer=label "${RES[@]}") ;;
      time)  X=(select.scorer=label select.time_scorer=season_residual "${RES[@]}") ;;
      both)  X=(select.scorer=residual select.time_scorer=season_residual "${RES[@]}") ;;
    esac
    prev=$AFTER; for i in 1 2; do
      prev=$(sub "mxstl_${A}_${T}_$i" "$prev" "${F[@]}" train.epochs=50 train.max_hours=2.5 "${X[@]}")
    done
    [ "$A" = both ] && BOTH_AFTER=$prev
  done
  sub "mxstl_orcorc_$T" "$AFTER" "${F[@]}" train.epochs=0 select.scorer=label select.time_scorer=label >/dev/null
  sub "mxstl_entorc_$T" "$AFTER" "${F[@]}" train.epochs=0 select.scorer=entropy select.time_scorer=label >/dev/null
  sub "mxstl_randrand_$T" "$AFTER" "${F[@]}" train.epochs=0 select.scorer=random \
      select.time_scorer=season_random >/dev/null
  BOTH=$(name "${F[@]}" train.epochs=50 train.max_hours=2.5 select.scorer=residual \
         select.time_scorer=season_residual "${RES[@]}")
  J=("${BASE[@]}" train.init_ckpt=$ORC train.epochs=10 train.lr=1e-4 train.max_hours=2.5
     select.scorer=residual select.time_scorer=season_residual "${RES[@]}")
  sub "mxstl_ft_$T" "$BOTH_AFTER" "${J[@]}" train.init_selector=$BOTH >/dev/null
  sub "mxstl_ft0_$T" "$AFTER" "${J[@]}" >/dev/null
  echo "$T queued (oracle $ORC)"
done < "${IDS:?IDS=<file of setup keep n last-job-id lines>}"
