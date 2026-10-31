#!/bin/bash
# The quadrant policy (selectors.Quadrant): per low-res cell, spatial uncertainty (pooled
# entropy) x temporal uncertainty (attention entropy over seasons) -> all-month high-res crops /
# most-attended-season crops / low-res month tokens (lowspace lr_all) or nothing (skip) / nothing.
#   dedicated   one model per (q_s, q_t) in {1/8, 1/4, 1/2} x {1/4, 1/2}, 3-link chains
#   random      one model trained with q_s ~ U(0, 1/2), q_t ~ U(0, 1) per batch, then evaluated
#               (train.epochs=0) at every grid point
# x threshold mode sample | global, x lowspace skip | lr_all, PASTIS s16m8 + s8m4. q_s = 1/8,
# 1/4, 1/2 of the cells = n 8 / 16 / 32 (s16m8) or 32 / 64 / 128 (s8m4) crop windows: warm from
# the location oracle at that n (random: the middle). Mail off.
#   [DRY=1] [SMOKE=1] bash scripts/slurm/mixres_quadrant.sh
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
LINKS=${LINKS:-3}
for S in ${SETUPS:-s16m8 s8m4}; do
if [ "$S" = s16m8 ]; then GEO=(select.crop_px=8 select.loc_stride=none model.lr_patch=16); LTAG=_cpx8_st8_poscenter_lr; KD=4
else GEO=(select.crop_px=4 select.loc_stride=none model.lr_patch=8); LTAG=_poscenter_lrps8_lr; KD=1; fi
loc() { echo "inject_fft_pastis_tiny_oracle_L0_k$(python3 -c "print(f'{$1 / $KD:.3g}')")_loc_native_set${LTAG}0.001_ep50"; }
for LS in ${LOWS:-skip lr_all}; do
for MD in ${MODES:-sample global}; do
  Q=(select.quadrant.mode=$MD select.quadrant.lowspace=$LS select.scorer=entropy select.allocator=top)
  C=(task=pastis select.n=0 "${GEO[@]}" "${Q[@]}" train.max_hours=2.5 ${SMOKE:+train.limit_batches=3 train.seed=999})
  EP=${SMOKE:+2}; EP=${EP:-50}
  for QS in ${QSS:-0.125 0.25 0.5}; do for QT in ${QTS:-0.25 0.5}; do
    L=$(loc $QS); [ -f "checkpoints/$L.pt" ] || { echo "missing $L" >&2; exit 1; }
    prev=""; for i in $(seq 1 $LINKS); do
      prev=$(sub "mxquad_${LS}_${MD}_qs${QS}_qt${QT}_${S}_$i" "$prev" "${C[@]}" train.epochs=$EP \
             train.init_ckpt=$L select.quadrant.q_s=$QS select.quadrant.q_t=$QT); done
  done; done
  [ -n "${NO_RANDOM:-}" ] && continue
  L=$(loc 0.25)
  R=("${C[@]}" train.epochs=$EP train.init_ckpt=$L select.quadrant.q_s=0.25 select.quadrant.q_t=0.5
     select.quadrant.random=true)
  prev=""; for i in $(seq 1 $LINKS); do prev=$(sub "mxquad_${LS}_${MD}_rand_${S}_$i" "$prev" "${R[@]}"); done
  RN=$(name "${R[@]}")
  for QS in ${QSS:-0.125 0.25 0.5}; do for QT in ${QTS:-0.25 0.5}; do
    sub "mxquad_${LS}_${MD}_randeval_qs${QS}_qt${QT}_${S}" "$prev" "${C[@]}" train.epochs=0 \
        train.init_ckpt=$RN train.init_selector=$RN select.quadrant.q_s=$QS \
        select.quadrant.q_t=$QT >/dev/null
  done; done
  echo "$S $LS $MD queued (random model $RN)"
done; done; done
