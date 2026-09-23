#!/bin/bash
#SBATCH --job-name=knn_mu
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/pastis/knn_mu_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=END,FAIL

# Probe-FREE comparison: KNN on frozen head.features(). A linear probe makes
# "bilinear-then-px2px" and "px2px-then-bilinear" the SAME function (1x1 conv and bilinear
# commute -- disjoint axes), so lp_pa2px / lp_pa2pa_bu cannot show what mAnyUp's upsampling
# adds. KNN is non-parametric: it reads the feature geometry the reconstruction loss actually
# optimizes, with no probe capacity to launder the difference.
#
# Three arms per modality, all ending at 64x64 features:
#   bilinear : lp_pa2px.features()  = tokens bilinear-upsampled       (the AnyUp-paper baseline)
#   manyup   : frozen mAnyUp upsample of the SAME tokens
#   real_hr  : the actual finer-patch features (upper bound / oracle)
#
#   MODS   s2 | s1 | s2s1   (default s2s1)
#   LR_PS  LR patch size    (default 16)
#   HR_PS  mAnyUp target    (default 4)

export TQDM_DISABLE=1
MODS="${MODS:-s2s1}"; LR_PS="${LR_PS:-16}"; HR_PS="${HR_PS:-4}"
K="${K:-20}"
OUT_ROOT="${OUT_ROOT:-$HOME/projects/aip-gpleiss/timz/features}"
CSV="${CSV:-results/pastis/lp_olmoearth_pastis.csv}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/pastis
source env_setup/env_olmo.sh

LR_CFG="oe_base_${MODS}_ps${LR_PS}_tile64"
HR_CFG="oe_base_${MODS}_ps${HR_PS}_tile64"
CKPT="checkpoints/manyup/${LR_CFG}__to__${HR_CFG}/manyup_${LR_CFG}_to_${HR_CFG}_ep31.pth"

COMMON="--knn --knn_k $K --out_root $OUT_ROOT --data_splits data/pastis_olmoearth --results_csv $CSV"

echo "=== [1/3] bilinear baseline: KNN on ${LR_CFG} tokens (bilinear-upsampled) ==="
python -u -m exp.pastis.lp_cached_features --features "$LR_CFG" --head_mode lp_pa2px $COMMON
A=$?

echo "=== [2/3] mAnyUp: KNN on ${LR_CFG} -> ${HR_CFG} upsampled features ==="
if [ -f "$CKPT" ]; then
    python -u -m exp.pastis.lp_cached_features --features "$LR_CFG" --manyup_ckpt "$CKPT" $COMMON
    B=$?
else
    echo "ERROR: no mAnyUp checkpoint at $CKPT"; B=-1
fi

echo "=== [3/3] oracle: KNN on REAL ${HR_CFG} features ==="
python -u -m exp.pastis.lp_cached_features --features "$HR_CFG" --head_mode lp_pa2px $COMMON
C=$?

echo "exit codes: bilinear=$A manyup=$B real_hr=$C"
