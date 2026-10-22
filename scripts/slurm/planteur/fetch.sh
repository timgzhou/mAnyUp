#!/bin/bash
#SBATCH --job-name=planteur_fetch
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
# Fetch the PLANTEUR files mAnyUp reads (S2, S1, labels, metadata; ~120 GB of the 258 GB
# release: no Landsat, monthly mosaics or embeddings) into data/PLANTEUR. Plain curl, 12 in
# parallel: huggingface_hub's snapshot_download stalled on this repo's 13.8k small files
# (2026-10-22), and the Hub allows 3000 resolves / 5 min, so this takes >= 25 min. Resumable:
# finished files are skipped; rerun after a timeout. Prints FAILED <path> per failed file.
#
#   bash scripts/slurm/planteur/fetch.sh          # or sbatch from the repo root
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DEST=data/PLANTEUR; B=https://huggingface.co/datasets/allenai/planteur-benchmark/resolve/main
{
  for f in metadata.geojson README.md NORM_S2_patch.json NORM_S1_patch.json splits/443.json \
           splits/loio.json splits/fewshot_pixels.json documentation/class_map.json \
           documentation/label_names.json documentation/colormap.txt \
           ANNOTATIONS_GEOTIFF/_crs_summary.json; do echo "$f"; done
  for i in $(seq 0 2749); do
    echo "ANNOTATIONS/TARGET_$i.npy"; echo "ANNOTATIONS_GEOTIFF/TARGET_$i.tif"
    echo "ANNOTATIONS_BALANCED/TARGET_$i.npy"; echo "DATA_S1/S1_$i.npy"; echo "DATA_S2/S2_$i.npy"
  done
} | xargs -P 12 -I{} sh -c '
  [ -s "$0/{}" ] && exit 0
  mkdir -p "$(dirname "$0/{}")"
  curl -sfL --retry 20 --retry-all-errors --retry-delay 5 -o "$0/{}.part" "$1/{}" \
    && mv "$0/{}.part" "$0/{}" || echo "FAILED {}"' "$DEST" "$B"
echo DONE
