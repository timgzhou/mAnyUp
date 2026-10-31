#!/bin/bash
# The published baselines against MiSTR (docs/baselines.md), compute-controlled, on PASTIS and
# BioMassters. Run from the repo root; results in results/<task>/mixres.csv, tag upsample-* /
# tldr-* / apt-*.
#   calib      exp.baselines.calibrate_apt (APT's stretch + thresholds), first
#   upsample   linear probe on a fixed model's upsampled tokens, per lr: bilinear, AnyUp
#   anyup_ft   AnyUp, the backbone fine-tuned through it (warm lr x 0.1)
#   upa        Upsample Anything, test-time: bilinear's best head, kernels fit per test frame;
#              submit after the bilinear runs have finished (ARMS=upa)
#   tldr       vid-TLDR on the fixed monthly models, evaluation only, one sequence over the months
#   apt        APT at the MiSTR pairs, keep share of coarse patches, warm from the fixed model
# Mail off, <= 3 h links, 8 CPUs.
#   [SEEDS="0 1 2"] [TASKS="pastis biomassters"] [ARMS=...] [DRY=1] bash scripts/slurm/baselines.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source env_setup/env_olmo.sh >/dev/null 2>&1
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088,kn010
sub() {   # sub <job name> <dependency or ""> <module> <args...> -> prints the job id
  local name=$1 dep=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=8 --exclude=$EXC
             ${dep:+--dependency=$dep} scripts/slurm/run.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo DRY; else "${cmd[@]}"; fi
}
chain() {  # chain <name> <links> <dependency or ""> <overrides...>
  local name=$1 links=$2 dep=$3 id i; shift 3
  for i in $(seq 1 "$links"); do
    id=$(sub "bl_${name}_$i" "$dep" exp.mixres.train train.max_hours=2.5 "$@"); dep="afterany:$id"
  done
}
fixed() {  # fixed <task> <months|seasons> <patch> <ep> <seed> -> the checkpoint name
  local all="" tail="" s=""
  [ "$2" = months ] && all=_lrall
  [ "$3" = 16 ] || tail=_lrps$3
  [ "$5" = 0 ] || s=_seed$5
  local n=inject_fft_$1_tiny_none_native_set$all${tail}_lr0.001_$4$s
  if [ -e "checkpoints/$n.pt" ]; then echo "$n"; else echo "${n%_seed*}"; fi
}
ARMS=${ARMS:-calib upsample anyup_ft tldr apt}
for T in ${TASKS:-pastis biomassters}; do
  case $T in
    pastis)      EPS=ep50; E=50; LINKS=3; PS="16 14 12 10 8";   SHARDS=1 ;;
    biomassters) EPS=ep10; E=10; LINKS=5; PS="16 14 12 10 8 6"; SHARDS=4 ;;
  esac
  C=(task=$T select.n=0 train.epochs=$E)
  dep=""
  if [[ " $ARMS " == *" calib "* ]]; then
    id=$(sub "bl_calib_$T" "" exp.baselines.calibrate_apt --task "$T"); dep="afterok:$id"
  fi
  for SEED in ${SEEDS:-0}; do
    for P in $PS; do
      if [[ " $ARMS " == *" upsample "* ]]; then for U in bilinear anyup; do for F in seasons months; do
        for LR in 1e-2 1e-3 1e-4; do
          chain "up_${T}_${U}_${F}_p${P}_lr${LR}_s$SEED" $LINKS "" "${C[@]}" model.lr_patch=$P \
            train.init_ckpt=$(fixed $T $F $P $EPS $SEED) train.seed=$SEED train.head_only=1 train.lr=$LR \
            train.batch_size=8 baseline.method=upsample baseline.tokens=$F baseline.upsampler=$U
        done; done; done
      fi
      if [[ " $ARMS " == *" anyup_ft "* ]]; then
        chain "anyft_${T}_p${P}_s$SEED" $LINKS "" "${C[@]}" model.lr_patch=$P \
          train.init_ckpt=$(fixed $T seasons $P $EPS $SEED) train.seed=$SEED train.batch_size=8 \
          baseline.method=upsample baseline.tokens=seasons baseline.upsampler=anyup
      fi
      if [[ " $ARMS " == *" upa "* ]]; then
        ids=""
        for i in $(seq 0 $((SHARDS - 1))); do
          id=$(sub "bl_upa_${T}_p${P}_${i}of$SHARDS" "" exp.baselines.eval_upa --task $T --patch $P \
               --seed $SEED --shard $i/$SHARDS); ids="$ids:$id"
        done
        sub "bl_upa_${T}_p${P}_merge" "afterok$ids" exp.baselines.eval_upa --task $T --patch $P \
            --seed $SEED --merge $SHARDS >/dev/null
      fi
      if [[ " $ARMS " == *" tldr "* ]]; then
        # upstream's schedules as shares of its 2352 tokens (b16 MSVD, MSRVTT), lighter and heavier
        for R in 0.1 0.19,0.1,0.085 0.25,0.085,0.043 0.4,0.1 0.5,0.25,0.125; do
          chain "tldr_${T}_p${P}_r${R}_s$SEED" 1 "" task=$T select.n=0 model.lr_patch=$P \
            train.init_ckpt=$(fixed $T months $P $EPS $SEED) train.seed=$SEED train.epochs=0 \
            train.freeze_model=1 train.batch_size=8 baseline.method=tldr baseline.tokens=months baseline.r=$R
        done
      fi
      if [[ " $ARMS " == *" apt "* ]] && [ "$P" != 8 ]; then
        for F in seasons months; do for K in 0.25 0.5 0.75 0.9; do
          chain "apt_${T}_${F}_p${P}_k${K}_s$SEED" $LINKS "$dep" "${C[@]}" model.lr_patch=$P \
            select.crop_px=$((P / 2)) train.init_ckpt=$(fixed $T $F $P $EPS $SEED) train.seed=$SEED \
            train.batch_size=8 train.compute_split=test train.compute_batches=16 \
            baseline.method=apt baseline.tokens=$F baseline.keep=$K
        done; done
      fi
    done
  done
done
