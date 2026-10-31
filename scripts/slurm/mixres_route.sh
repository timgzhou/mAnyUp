#!/bin/bash
# Routing between resolutions (selectors.Route), PASTIS s16m8: n crops, each either a ps8 window
# or a ps4 cell, never overlapping, picked greedily from ONE per-pixel entropy map of the
# crop-free pass (ps8 window value = its entropy mass, ps4 cell = alpha x its mass); each crop
# then its season(s) the readout attends to most (as ent_attn). Same token count (and GMACs) as
# ent_attn at the same n; alpha 1 == ent_attn (ps8 only, the baseline, not rerun). Entropy is
# spatially smooth, so the ps4 share only moves for alpha in (2, 4) -- measured on val (ent_attn
# half n16 model), share of crops at ps4 for n = 8 / 16 / 32:
#   2.5: .03/.07/.17   3: .24/.33/.44   3.25: .46/.54/.62   3.5: .70/.76/.81   inf: 1
# keep one / half, n = 8 / 16 / 32, warm from the location oracle at k = n/256, 50 epochs, 3-link
# chains (as mixres_spacetime.sh). Mail off.
#   [DRY=1] [ALPHAS=..] [KEEPS=..] [NLIST=..] bash scripts/slurm/mixres_route.sh
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
for N in ${NLIST:-8 16 32}; do
KF=$(python3 -c "print(f'{$N / 256:.3g}')")
LOC="inject_fft_pastis_tiny_oracle_L0_k${KF}_loc_native_set_cpx8_st8_poscenter_lr0.001_ep50"
[ -f "checkpoints/$LOC.pt" ] || { echo "missing $LOC" >&2; exit 1; }
for K in ${KEEPS:-one half}; do
for A in ${ALPHAS:-2.5 3 3.25 3.5 inf}; do
  prev=""; for i in $(seq 1 ${LINKS:-3}); do
    prev=$(sub "mxroute${A}_${K}_n${N}_$i" "$prev" task=pastis select.n=$N select.crop_px=8 \
      model.lr_patch=16 select.scorer=entropy select.allocator=top select.loc_stride=8 \
      select.time_scorer=attention select.time_allocator=every_location select.keep=$K \
      select.route_alpha=$A train.epochs=50 train.max_hours=2.5 train.init_ckpt=$LOC); done
done
done
echo "n$N queued"
done
