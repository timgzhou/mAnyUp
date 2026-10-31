#!/bin/bash
# Download + pack EuroMineNet (RODARE record 4656, 18.4 GB zip, single stream: the server takes no
# range requests, ~25 MB/s) into data/EuroMineNet/packed (exp.euromine.prepare), from the repo
# root. ~15 min download, ~4 min unzip, <1 min pack; verify re-reads every GeoTIFF.
#   bash scripts/slurm/euromine/fetch.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAW=data/EuroMineNet/raw
mkdir -p "$RAW"
curl -sSL --retry 5 -C - -o "$RAW/EuroMineNet.zip" \
  https://rodare.hzdr.de/api/files/a5bd97d7-85d3-45df-9c1c-a2b6c69233c5/EuroMineNet.zip
[ -d "$RAW/EuroMineNet" ] || unzip -q "$RAW/EuroMineNet.zip" -d "$RAW"
source env_setup/env_olmo.sh
python -u -m exp.euromine.prepare pack --workers 8
python -u -m exp.euromine.prepare verify --workers 8
