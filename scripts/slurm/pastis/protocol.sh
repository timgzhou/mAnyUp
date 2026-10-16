#!/bin/bash
# PASTIS under the benchmark's protocol, and the train-from-scratch reference (2026-10-06).
#
#   [DRY=1] [ONLY=<regex>] bash scripts/slurm/pastis/protocol.sh
#
# Benchmark protocol (Garnot & Landrieu 2021, utae-paps/train_semantic.py; what U-TAE 63.1,
# TSViT 65.4, Exchanger+Mask2Former 67.9 report): Sentinel-2 only, 5 protocol folds (fold f
# trains on dataset folds f..f+2, validates on f+3, tests on f+4), 20 classes with the void
# label ignored, mIoU of the SUM of the 5 test confusion matrices. Our earlier runs are their
# fold 1 with S1A added. Here: --task pastis_s2 (S2 only) x --fold 1..5; the test confusion
# matrix of each inject run is saved to results/pastis_s2/conf/<run>.pt.
#   proto_{tiny,base}_grid33_f{1..5}   pooled month crops, staggered lattice at 1/3 of the slots
#   proto_{tiny,base}_all_f{1..5}      pooled month crops, every slot
#   proto_{tiny,base}_ps4ts64_f{1..5}  fixed ps4 series (exp.olmoearth.train)
# Still different from the benchmark: 64 px tiles (4 per 128 px patch, same pixels scored),
# 50 epochs, pretrained OlmoEarth.
#
# Train from scratch (--no_pretrain: random init, no frozen phase, 5 warmup epochs), on the
# S2 + S1A task and fold 1 so the numbers sit next to every earlier `pastis` run (tiny):
#   nopre_all_ep{50,150}[_lr3e-4]      pooled month crops, every slot
#   nopre_ps4ts64_ep{50,150}[_lr3e-4]  fixed ps4 series
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
CPUS=9

chain() {   # chain <name> <links> <module> [args...]: <links> jobs of <=3h, each resuming the last
  local name=$1 links=$2; shift 2
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then return; fi
  local prev="" id i
  for i in $(seq 1 "$links"); do
    local cmd=(sbatch --parsable -J "pas_${name}_$i" --time=3:00:00 --cpus-per-task="$CPUS"
               --exclude=kn117,kn104,kn101,kn001 ${prev:+--dependency=afterany:$prev}
               scripts/slurm/run.sh "$@" --num_workers $((CPUS - 1)) --max_hours 2.5)
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; id="DRY$i"; else id=$("${cmd[@]}"); fi
    prev=$id
  done
  echo "$name: chain of $links ending $prev"
}

# pooled month crops (learned weights over each month's acquisitions), injected before block 3
INJ=(exp.upsamplers.finetune_inject --select_unit slot --readout native_set --inject_layer 3
     --frac_slot --acq_pool attn --pos center)
GRID33=(--selector grid --k_frac 0.3333)
ALL=(--selector random --k_frac 1)
PS4=(exp.olmoearth.train --temporal_mode series --patch_size 4 --tile_size 64 --mode fft)

# ---- benchmark protocol: S2 only, 5 folds, pretrained --------------------------------------
for F in 1 2 3 4 5; do
  P=(--task pastis_s2 --fold "$F" --epochs 50)
  chain "proto_tiny_grid33_f$F"  3 "${INJ[@]}" "${GRID33[@]}" "${P[@]}" --model_size tiny
  chain "proto_tiny_all_f$F"     4 "${INJ[@]}" "${ALL[@]}"    "${P[@]}" --model_size tiny --batch_size 8
  chain "proto_tiny_ps4ts64_f$F" 3 "${PS4[@]}" "${P[@]}" --model_size tiny --lrs 1e-3 --batch_size 8
  chain "proto_base_grid33_f$F"  6 "${INJ[@]}" "${GRID33[@]}" "${P[@]}" --model_size base --batch_size 8
  chain "proto_base_all_f$F"    12 "${INJ[@]}" "${ALL[@]}"    "${P[@]}" --model_size base --batch_size 3
  chain "proto_base_ps4ts64_f$F" 7 "${PS4[@]}" "${P[@]}" --model_size base --lrs 1e-3 --batch_size 8
done

# ---- train from scratch: S2 + S1A, fold 1, tiny ---------------------------------------------
N=(--task pastis --model_size tiny --no_pretrain --warmup_epochs 5 --batch_size 8)
chain nopre_all_ep50            5 "${INJ[@]}" "${ALL[@]}" "${N[@]}" --epochs 50
chain nopre_all_ep150          14 "${INJ[@]}" "${ALL[@]}" "${N[@]}" --epochs 150
chain nopre_all_ep150_lr3e-4   14 "${INJ[@]}" "${ALL[@]}" "${N[@]}" --epochs 150 --lr 3e-4
chain nopre_ps4ts64_ep50        3 "${PS4[@]}" "${N[@]}" --epochs 50  --lrs 1e-3
chain nopre_ps4ts64_ep150       7 "${PS4[@]}" "${N[@]}" --epochs 150 --lrs 1e-3
chain nopre_ps4ts64_ep150_lr3e-4 7 "${PS4[@]}" "${N[@]}" --epochs 150 --lrs 3e-4
