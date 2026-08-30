#!/bin/bash
#SBATCH --job-name=oe_bench
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/bench/oe_bench_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Benchmark OlmoEarth feature-extraction throughput (exp/bench/olmo_throughput.py).
# Configure via --export env vars:
#   ARM       (default s2)  one of s2 | s1 | s2s1 -- ONE arm per job
#   CONFIGS   (default: full ps{1,2,4,8,16} x tile{16,64,128} grid for IMAGE_SIZE)
#   IMAGE_SIZE(default 64)  sample side; tile 128 needs IMAGE_SIZE=128
#   ITERS     (default 5)   timed steps per config
#   WARMUP    (default 2)
#   MODEL_SIZE(default base)
#
# One arm per job on purpose: the arms are independent, so running them as three jobs
# gets them in parallel AND means a failure/timeout in one does not cost the others.
# Each job writes its OWN csv (results/bench/olmo_throughput_<arm>.csv); merge with
# exp/viz/plot_olmo_throughput.py, which globs them.
#
# The GPU must be exclusive: the benchmark auto-tunes batch size to fill GPU memory, so
# anything else resident on the card both perturbs the timings and shrinks the batch found.
#
# Examples:
#   sbatch --export=ALL,ARM=s2,IMAGE_SIZE=64   scripts/slurm/bench/throughput.sh
#   sbatch --export=ALL,ARM=s2,IMAGE_SIZE=128  scripts/slurm/bench/throughput.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1

ARM="${ARM:-s2}"
CONFIGS="${CONFIGS:-}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
MODEL_SIZE="${MODEL_SIZE:-base}"
OUT="results/bench/olmoearth_ps-tile_speed_${ARM}_img${IMAGE_SIZE}.csv"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/bench logs/bench
source env_setup/env_olmo.sh

ARGS="--model_size $MODEL_SIZE --arms $ARM --image_size $IMAGE_SIZE --iters $ITERS --warmup $WARMUP --out $OUT"
[ -n "$CONFIGS" ] && ARGS="$ARGS --configs $CONFIGS"
TAG="${MODEL_SIZE} arm=${ARM} img${IMAGE_SIZE}"

echo "exp/bench/olmo_throughput.py $ARGS" \
    | mail -s "[START job $SLURM_JOB_ID] bench $TAG" "$EMAIL"

LOG="logs/bench/oe_bench_${SLURM_JOB_ID}.out"
python -u -m exp.bench.olmo_throughput $ARGS
STATUS=$?

{
    echo "args: $ARGS"
    echo "exit status: $STATUS"
    echo "--- results ---"
    grep -aE "samp/s|OOM at batch|^Wrote " "$LOG" || echo "(no results found; see log)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] bench $TAG" "$EMAIL"
