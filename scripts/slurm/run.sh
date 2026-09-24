#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# Run ONE exp module under Slurm, forwarding every argument after the module name:
#
#   sbatch [sbatch options] scripts/slurm/run.sh <module> [module args...]
#
# Resources above are defaults; override them on the sbatch line, and name the job with -J so
# the log (logs/<job-name>_<jobid>.out) says what ran:
#
#   sbatch -J extract --time=9:00:00 scripts/slurm/run.sh exp.pastis.extract_features \
#       --modalities sentinel2_l2a --patch_size 4 --tile_size 64
#   sbatch -J prep --gres=none --mem=256G scripts/slurm/run.sh exp.pastis.prepare_data
#   sbatch -J mu --time=12:00:00 scripts/slurm/run.sh exp.upsamplers.train_manyup \
#       --lr_cfg oe_base_s2_ps16_tile64 --hr_cfg oe_base_s2_ps4_tile64 --stage_to_tmpdir
#
# Submit from the repo root (Slurm runs a copy of this script, so the repo is found via
# $SLURM_SUBMIT_DIR). Emails EMAIL at start and end; set EMAIL= to skip.
# See docs/RUNBOOK.md for the command behind every experiment.
set -uo pipefail
MODULE="${1:?usage: sbatch scripts/slurm/run.sh <module> [args...]}"
shift
EMAIL="${EMAIL-tiange.zhou@outlook.com}"
export TQDM_DISABLE=1   # progress bars only clutter a batch log

cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repo root}"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
mkdir -p logs
source env_setup/env_olmo.sh

TAG="$MODULE $*"
[ -n "$EMAIL" ] && echo "$TAG" | mail -s "[START job $SLURM_JOB_ID] $SLURM_JOB_NAME" "$EMAIL"

python -u -m "$MODULE" "$@"
STATUS=$?

if [ -n "$EMAIL" ]; then
    LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
    {
        echo "$TAG"
        echo "exit status: $STATUS"
        echo "--- summary ---"
        grep -aE "^(Run:|BEST|TEST|Wrote |run tag:|== epoch|   \[test)|samp/s" "$LOG" | tail -30
        echo "--- last 10 log lines ---"
        tail -n 10 "$LOG"
    } | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] $SLURM_JOB_NAME" "$EMAIL"
fi
exit $STATUS
