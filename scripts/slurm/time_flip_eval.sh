#!/bin/bash
# exp.mixres.time_flip_eval on the four oracle-trained time models (no training). Mail off.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
for R in mixres_pastis_ps16_label-every_location-one_n0_c8s8_warm_seed0_613f740f \
         mixres_pastis_ps16_label-every_location-half_n0_c8s8_warm_seed0_c82194cb \
         mixres_pastis_ps8_label-every_location-one_n0_c4_warm_seed0_9d93609c \
         mixres_pastis_ps8_label-every_location-half_n0_c4_warm_seed0_6cfc876e; do
  sbatch --parsable -J "tflip_${R:14:3}_$(echo $R | grep -o 'one\|half')" --time=1:30:00 --cpus-per-task=9 \
    --exclude=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088 \
    scripts/slurm/run.sh exp.mixres.time_flip_eval --run "$R"
done
