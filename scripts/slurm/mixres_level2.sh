#!/bin/bash
# A second, finer layer of month crops on the label-free space + time selector (selectors.Level2
# over SpaceTime ent_attn): ps16 seasons -> ps8 month crops (layer 1: entropy windows, each its
# one / two most-attended seasons) -> ps4 month crops. The layer-1 model's own prediction WITH
# its crops (no grad) picks n2 ps4 cells under layer-1 crops and, at each, half of the months
# the first layer cropped there (legacy round 59, but label-free):
#   e    entropy where (highest prediction entropy) + attention when (the months whose crop
#        token the readout scores highest)
#   er   entropy where + a random half of the months         (one axis at a time)
#   ra   random cells + attention when                        (one axis at a time)
#   r    random cells + random months                         (control)
#   e0   e from the crop-free pass layer 1 already ran (no second pass): entropy where, the
#        months of the most-attended cropped season when (ARMS="e0 e0r", round 2 2026-10-26)
#   e0r  e0's cells + a random half of the months             (one axis at a time)
#   none the layer-1 checkpoint fine-tuned again without level 2 (the extra epochs alone)
# Warm from the ent_attn checkpoint at n1, full FT 50 epochs, 3-link chains. n1 = 8 / 16 / 32,
# n2 = n1/2, n1, 2 n1 for e and r; er / ra at n2 = n1. Mail off.
#   [DRY=1] [KEEPS=..] [NLIST=..] bash scripts/slurm/mixres_level2.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
chain() {   # chain <job name> <overrides...>
  local name=$1 prev=""; shift
  for i in $(seq 1 ${LINKS:-3}); do prev=$(sub "${name}_$i" "$prev" "$@"); done
}
declare -A PICK=([e]=entropy [er]=entropy_r [ra]=r_attention [r]=random [e0]=entropy0
                 [e0r]=entropy0_r)
ARMS=${ARMS:-e r er ra}
for K in ${KEEPS:-one half}; do
for N in ${NLIST:-8 16 32}; do
CK=$(ls -tr checkpoints | grep -E "^mixres_pastis_ps16_entropy-top\+attention-every_location-${K}_n${N}_c8s8_warm_seed0_[0-9a-f]{8}\.pt$" | sed 's/\.pt$//' | head -1)   # oldest: the ent_attn run, not a later mxl2none
[ -n "$CK" ] || { echo "missing ent_attn $K n$N" >&2; exit 1; }
C=(task=pastis select.n=$N select.crop_px=8 model.lr_patch=16 select.scorer=entropy
   select.allocator=top select.loc_stride=8 select.time_scorer=attention
   select.time_allocator=every_location select.keep=$K train.epochs=50 train.max_hours=2.5
   train.init_ckpt=$CK)
[[ " $ARMS " == *" e "* ]] && chain "mxl2none_${K}_n${N}" "${C[@]}"
for N2 in $((N / 2)) $N $((2 * N)); do
for A in $ARMS; do
  [ "$N2" != "$N" ] && [[ " er ra e0r " == *" $A "* ]] && continue
  chain "mxl2${A}_${K}_n${N}_${N2}" "${C[@]}" select.level2.patch=4 select.level2.n=$N2 \
        select.level2.pick=${PICK[$A]}
done
done
echo "$K n$N queued (layer 1 $CK)"
done
done
