#!/bin/bash
#SBATCH --job-name=lp_timanyup
#SBATCH --account=aip-gpleiss
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/lp_timanyup_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Linear-probe every available timAnyUp checkpoint, in BOTH probe modes:
#   timanyup    fused map mean-pooled over T -> one shared probe
#   timanyup_t  an independent probe per timestep, logits averaged (temporal ensemble)
# Runs sequentially on one GPU; each run appends a row to --results_csv.
#
# Env vars:
#   CKPTS     newline/space separated checkpoint paths (default: every *.pth under CKPT_ROOT).
#             One checkpoint per job is the intended use: a full 32-epoch LP is ~66 min per
#             mode, so a 12-run sweep in ONE job would exceed the time limit. Submit one job
#             per checkpoint (see submit_lp_timanyup_all.sh) and let them run in parallel.
#   CKPT_ROOT root scanned when CKPTS is unset (default checkpoints/timanyup)
#   MODES     probe modes to run (default "timanyup timanyup_t")
#   EPOCHS    LP epochs (default 32, the lp_cached_features default)
#   EXTRA     extra args passed verbatim
#
#   sbatch scripts/slurm/upsamplers/lp_timanyup.sh
#   sbatch --export=ALL,MODES=timanyup_t,EPOCHS=16 scripts/slurm/upsamplers/lp_timanyup.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1

CKPT_ROOT="${CKPT_ROOT:-checkpoints/timanyup}"
MODES="${MODES:-timanyup timanyup_t}"
EPOCHS="${EPOCHS:-32}"
EXTRA="${EXTRA:-}"

cd "$SLURM_SUBMIT_DIR"
source env_setup/env_olmo.sh

# Only checkpoints in a per-run subdir: the bare CKPT_ROOT/*.pth are stray one-off smoke files.
if [ -z "$CKPTS" ]; then
    CKPTS=$(find "$CKPT_ROOT" -mindepth 2 -name "*.pth" | sort)
fi

LOG="logs/lp_timanyup_${SLURM_JOB_ID}.out"
N_OK=0; N_FAIL=0
for CK in $CKPTS; do
    # --features is the F_lrhc arm the checkpoint was trained on; lp_cached_features reads it
    # (and F_hrlc) out of the checkpoint and REFUSES a mismatch, so we pass the trained arm.
    LRHC=$(python -c "
import torch,sys
print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('args',{}).get('lrhc_cfg',''))" "$CK")
    if [ -z "$LRHC" ]; then
        echo "SKIP (no lrhc_cfg): $CK"; continue
    fi
    for MODE in $MODES; do
        echo "=== LP $MODE :: $(basename $CK) (features=$LRHC) ==="
        python -u -m exp.pastis.lp_cached_features \
            --features "$LRHC" --head_mode "$MODE" --timanyup_ckpt "$CK" \
            --epochs "$EPOCHS" $EXTRA
        if [ $? -eq 0 ]; then N_OK=$((N_OK+1)); else N_FAIL=$((N_FAIL+1)); fi
    done
done

{
    echo "checkpoints: $(echo "$CKPTS" | wc -w)   modes: $MODES   epochs: $EPOCHS"
    echo "runs ok=$N_OK failed=$N_FAIL"
    echo "--- test metrics ---"
    grep -aE "^TEST \{|=== LP " "$LOG" | tail -40
} | mail -s "[DONE job $SLURM_JOB_ID] LP timAnyUp ($N_OK ok, $N_FAIL failed)" "$EMAIL"
