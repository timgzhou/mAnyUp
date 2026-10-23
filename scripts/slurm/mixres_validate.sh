#!/bin/bash
# End-to-end check of exp.mixres.train against the legacy arm it ports: self-entropy windows, all
# months, warm from the location oracle (legacy: scripts/slurm/self_entropy_arms.sh, base arm
# _sent_srall_warm_), PASTIS s16m8 n=8 and s8m4 n=32, 3-link chains. Mail off.
#   [DRY=1] bash scripts/slurm/mixres_validate.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
sub() {   # sub <job name> <after job or ""> <overrides...> -> prints the job id
  local name=$1 after=$2; shift 2
  local cmd=(sbatch --parsable -J "$name" --time=3:00:00 --cpus-per-task=9 --exclude=$EXC
             ${after:+--dependency=afterany:$after} scripts/slurm/run.sh exp.mixres.train "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY"; else "${cmd[@]}"; fi
}
COMMON=(task=pastis select.scorer=entropy select.allocator=halves select.keep=all
        train.epochs=50 train.max_hours=2.5)
S16=(select.n=8 select.crop_px=8 select.loc_stride=8 model.lr_patch=16
     train.init_ckpt=inject_fft_pastis_tiny_oracle_L0_k0.0312_loc_native_set_cpx8_st8_poscenter_lr0.001_ep50)
S8=(select.n=32 select.crop_px=4 select.loc_stride=none model.lr_patch=8
    train.init_ckpt=inject_fft_pastis_tiny_oracle_L0_k0.125_loc_native_set_poscenter_lrps8_lr0.001_ep50)
prev=""; for i in 1 2 3; do prev=$(sub "mxsent_s16m8_n8_$i" "$prev" "${COMMON[@]}" "${S16[@]}"); done
prev=""; for i in 1 2 3; do prev=$(sub "mxsent_s8m4_n32_$i" "$prev" "${COMMON[@]}" "${S8[@]}"); done
echo queued
