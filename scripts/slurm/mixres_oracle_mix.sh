#!/bin/bash
# Oracle mixing instead of the two-stage selector schedule (train.oracle_mix): one cold run where
# the quadrant's high-space decision comes from the label oracle with probability p (per batch
# or per crop window), else from the entropy rule; seasons by attention throughout. Compared
# with the one-run schedule SCHED=50 in mixres_quadrant_rep.sh (location oracle 50 epochs, then
# the selector 50; seeds 0-2 at ps8) at the same 100 epochs:
#   crop    per crop window, p 1 -> 0 over the first half
#   batch   per batch, p 1 -> 0 over the first half
#   const   per batch, p 0.3 throughout
# PASTIS s8m4 (ps8 seasons + ps4 crops), model.attn cross_hr, one season, global threshold,
# q_s 1/8, 1/4, 1/2, seeds 0-2. <= 3 h links (max_hours 2.5), 5-link chains. Mail off.
#   [DRY=1] [ARMS="crop batch const"] [SEEDS="0 1 2"] bash scripts/slurm/mixres_oracle_mix.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
arm() {
  case $1 in
    crop)  echo train.oracle_mix=crop train.oracle_p=1 train.oracle_anneal=0.5 ;;
    batch) echo train.oracle_mix=batch train.oracle_p=1 train.oracle_anneal=0.5 ;;
    const) echo train.oracle_mix=batch train.oracle_p=0.3 train.oracle_anneal=0 ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}
C=(task=pastis model.lr_patch=8 select.crop_px=4 select.loc_stride=none model.attn=cross_hr
   select.n=0 select.scorer=entropy select.allocator=top select.quadrant.lowspace=skip
   select.quadrant.mode=global select.quadrant.space=crop select.quadrant.rank=score
   select.quadrant.q_t=0.5 select.quadrant.hs_ht=one select.quadrant.hs_lt=one
   train.epochs=100 train.max_hours=2.5)
n=0
for A in ${ARMS:-crop batch const}; do
  for QS in 0.125 0.25 0.5; do
    for S in ${SEEDS:-0 1 2}; do
      prev=""; for i in 1 2 3 4 5; do
        prev=$(sub "mxomix_${A}_qs${QS}_s${S}_$i" "$prev" "${C[@]}" $(arm $A) \
               select.quadrant.q_s=$QS train.seed=$S); done
      n=$((n + 1))
    done
  done
done
echo "$n chains queued"
