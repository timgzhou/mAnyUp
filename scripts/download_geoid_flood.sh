#!/bin/bash
# Download the GEOID-Flood layers needed for the OlmoEarth patch-size study.
#
# We take s1grd (pre+post VV/VH -- the T=2 change-detection signal), label (the 3-class
# target) and validity (the per-pixel "imaged and mapped" mask). We deliberately SKIP
# s1rtc (192GB) and s2l2a (177GB): the full 584GB does not fit, and s1grd+label+validity
# (~205GB) is all the S1-only pre/post pipeline needs.
#
# NOTE: this only needs huggingface_hub + tqdm, NOT the full OlmoEarth stack. Sourcing
# env_olmo.sh here would rebuild the shared /tmp/env_olmo venv and race any interactive
# shell already using it (seen as half-written pip/torch files). So we just load the
# python module and use the user-site huggingface_hub.
set -e
cd /scratch/timz/mAnyUp
module load python/3.12 scipy-stack >/dev/null 2>&1
export HF_HUB_DISABLE_XET=1
export TQDM_DISABLE=1   # keep the log readable; progress bars spam megabytes
export PATH="$HOME/.local/bin:$PATH"
python -u data/GEOID-Flood/get_data.py \
  --dest data/GEOID-Flood-full \
  --layer s1grd label validity \
  --workers 4
