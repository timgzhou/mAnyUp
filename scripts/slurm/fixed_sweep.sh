#!/bin/bash
# The full fixed-resolution sweep PASTIS has, for PLANTEUR / BioMassters / YieldSAT (their
# mixed-vs-fixed panels in exp.viz.datasets_tradeoff): every patch size PASTIS ran.
#   seasonal  ps14 / 12 / 10 / 6 / 2: exp.upsamplers.finetune_inject --selector none --readout
#             native_set --lr_patch P (as PASTIS's inj_pas_fix_ps*; ps2 at batch 8)
#   monthly   ps14 / 12 / 10 / 6: finetune_inject --lr_steps all, the mixed models' head
#             (native_set = attn_temporal_pool) as the seasonal runs; MON_OE=1: OlmoEarth's series
#             fine-tune at the padded tile (70 / 72 / 70 / 66 px) instead (its own head)
# A run with a checkpoint / result row already is skipped. Then one job traces every
# configuration's GMACs (finetune_inject --flops / exp.bench.flop_trace -> inject_flops.csv /
# flops.csv). <= 3 h links (max_hours 2.5, afterany). Mail off.
# Seed repeats: SEEDS="1 2" (--seed; the run names gain _seedS), PS_SEA / PS_MON the patch
# sizes (default the sweep's; "16 14 12 10 8 6 4 2" / "16 14 12 10 8 6 4" for every point
# plotted), TASKS may include pastis (monthly = OlmoEarth series, as biomassters). NO_FLOPS=1: no
# GMAC trace job (the traces do not depend on the seed).
#   [DRY=1] [SEEDS=0] [PS_SEA=..] [PS_MON=..] [NO_FLOPS=1] [TASKS="planteur biomassters yieldsat_raw"] bash scripts/slurm/fixed_sweep.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
sub() {   # sub <job name> <dependency or ""> <cpus> <module + args...> -> prints the job id
  local name=$1 dep=$2 cpus=$3; shift 3
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=$cpus --exclude=$EXC
             ${dep:+--dependency=$dep} scripts/slurm/run.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
chain() {   # chain <name> <links> <cpus> <module + args...> -> last job id
  local name=$1 links=$2 cpus=$3 dep="" id i; shift 3
  for i in $(seq 1 "$links"); do id=$(sub "${name}_$i" "$dep" "$cpus" "$@"); dep="afterany:$id"; done
  echo "$id"
}
mkdir -p results/flops
FLOPS=results/flops/fixed_sweep_commands.txt     # one GMAC trace per configuration
: > $FLOPS
ids=()
n=0
for S in ${SEEDS:-0}; do
SX=$([ $S = 0 ] || echo _seed$S)                 # run-name / job-name suffix
for T in ${TASKS:-planteur biomassters yieldsat_raw}; do
  case $T in
    planteur)     AB=pl; EP=50; CPUS=9 ;;
    biomassters)  AB=bio; EP=10; CPUS=17 ;;
    yieldsat_raw) AB=ys; EP=50; CPUS=9 ;;
    pastis)       AB=pas; EP=50; CPUS=9 ;;
  esac
  INJ=(exp.upsamplers.finetune_inject --task $T --epochs $EP --selector none --readout native_set)
  for P in ${PS_SEA:-14 12 10 6 2}; do            # seasonal tokens
    L=$(case $P in 2) echo 18 ;; 4|6) echo 8 ;; 16) echo 3 ;; *) echo 4 ;; esac)
    [ $T = biomassters ] && L=$(( (L + 1) / 2 + 1 ))
    BS=(); [ $P = 2 ] && BS=(--batch_size 8)
    echo "${INJ[*]} --lr_patch $P --num_workers 2 --batch_size 2 --flops 4" >> "$FLOPS"
    PT=$([ $P = 16 ] || echo _lrps$P)
    [ -f checkpoints/inject_fft_${T}_tiny_none_native_set${PT}_lr0.001_ep${EP}$SX.pt ] && continue
    ids+=($(chain inj_${AB}_fix_ps$P$SX $L $CPUS "${INJ[@]}" --num_workers 8 --max_hours 2.5 \
            --lr_patch $P "${BS[@]}" --seed $S)); n=$((n + 1))
  done
  for P in ${PS_MON:-14 12 10 6}; do              # monthly tokens
    TS=$(( (64 + P - 1) / P * P ))
    if [ -n "${MON_OE:-}" ]; then        # (MON_OE=1: OlmoEarth's own head, not the mixed models')
      echo "exp.bench.flop_trace --task $T --temporal_mode series --patch_size $P --tile_size $TS" >> "$FLOPS"
      grep -q "oe_${T}_tiny_series_ps${P}ts${TS}_fft_ep${EP}$SX," results/$T/olmoearth.csv 2>/dev/null && continue
      BS=$(case $P in 2) echo 4 ;; 4) echo 8 ;; 6) echo 8 ;; *) echo 32 ;; esac)
      L=$(case $P in 2) echo 18 ;; 4) echo 8 ;; 6) echo 6 ;; *) echo 3 ;; esac)
      ids+=($(chain oe_tiny_${AB}_ser_ps${P}ts${TS}_fft_lr1e-3$SX $L $CPUS exp.olmoearth.train \
              --task $T --model_size tiny --temporal_mode series --patch_size $P --tile_size $TS \
              --mode fft --lrs 1e-3 --epochs $EP --batch_size $BS --num_workers 8 --max_hours 2.5 \
              --seed $S))
    else
      echo "${INJ[*]} --lr_steps all --lr_patch $P --num_workers 2 --batch_size 2 --flops 4" >> "$FLOPS"
      PT=$([ $P = 16 ] || echo _lrps$P)
      [ -f checkpoints/inject_fft_${T}_tiny_none_native_set_lrall${PT}_lr0.001_ep${EP}$SX.pt ] && continue
      L=$(case $P in 4) echo 16 ;; 6) echo 8 ;; *) echo 5 ;; esac)
      ids+=($(chain inj_${AB}_mon_ps$P$SX $L $CPUS "${INJ[@]}" --num_workers 8 --max_hours 2.5 \
              --lr_steps all --lr_patch $P --seed $S))
    fi
    n=$((n + 1))
  done
done
done
[ "${NO_FLOPS:-}" = 1 ] && { echo "$n chains queued"; exit 0; }
# GMACs of every configuration (no training needed: traces run on fresh weights)
if [ -n "${DRY:-}" ]; then echo "flops: sbatch scripts/slurm/flops.sh 0 1 $FLOPS" >&2
else sbatch --parsable -J fixed_sweep_flops --time=3:00:00 --cpus-per-task=9 --exclude=$EXC \
       scripts/slurm/flops.sh 0 1 $FLOPS; fi
echo "$n chains queued, $(wc -l < $FLOPS) GMAC traces"
