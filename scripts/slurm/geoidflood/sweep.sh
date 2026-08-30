#!/bin/bash
#SBATCH --job-name=geoid_ps_sweep
#SBATCH --account=aip-gpleiss
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/geoid_ps_sweep_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#
# GEOID-Flood: OlmoEarth patch-size study -- ps8tile128 vs ps4tile64 vs ps1tile16.
#
# All three configs give a 16x16 token grid, so the probe sees the same token count and
# has the same parameter count in every arm. The only thing that varies is how much
# ground each token covers (80 m / 40 m / 10 m) and how much context the chip spans
# (1.28 km / 640 m / 160 m). That is the comparison.
#
#   sbatch scripts/slurm/geoidflood/sweep.sh
#   bash   scripts/slurm/geoidflood/sweep.sh     # inside a GPU salloc
#
# SMOKE=1 caps the source tiles and epochs so the whole three-arm sweep runs in minutes
# on a fraction of the data -- use it to confirm the pipeline end-to-end on REAL data
# before committing a 12h GPU job. Smoke results go to a separate CSV so they can never
# be mistaken for the real numbers.
#
#   SMOKE=1 bash scripts/slurm/geoidflood/sweep.sh
set -e
# Resolve the repo root from THIS script's location, not $SLURM_SUBMIT_DIR: a login shell
# can already have SLURM_SUBMIT_DIR set (e.g. to /scratch/timz) from an earlier job, which
# would cd somewhere that has no env_setup/.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export TQDM_DISABLE=1
source env_setup/env_olmo.sh

if [[ -n "${SMOKE:-}" ]]; then
  CSV=results/geoidflood/lp_smoke.csv
  PREP_EXTRA="--limit 40"
  LP_EXTRA="--epochs 2"
  echo "### SMOKE MODE: --limit 40 source tiles, 2 epochs -> $CSV (NOT real results)"
else
  CSV=results/geoidflood/lp.csv
  PREP_EXTRA=""
  LP_EXTRA=""
fi

# "<patch_size> <tile_size>" -- the three arms of the study.
CONFIGS=("8 128" "4 64" "1 16")

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; PS=$1; TILE=$2
  echo "=== PREP tile=$TILE ==="
  python -u -m exp.geoidflood.prep_tiles --splits train,val --tile_size "$TILE" $PREP_EXTRA

  FEAT="geoid_base_s1_ps${PS}_res10_t${TILE}"
  echo "=== EXTRACT ps=$PS tile=$TILE -> $FEAT ==="
  python -u -m exp.geoidflood.extract_features --splits train,val \
      --tile_size "$TILE" --patch_size "$PS"

  for HEAD in concat diff; do
    echo "=== LP $FEAT head=$HEAD ==="
    python -u -m exp.geoidflood.lp --features "$FEAT" --head "$HEAD" \
        --weighted_ce --results_csv "$CSV" $LP_EXTRA
  done
done

echo "=== DONE. Results: $CSV ; visualizations: results/geoidflood/ ==="
