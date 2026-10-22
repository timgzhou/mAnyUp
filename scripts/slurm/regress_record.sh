#!/bin/bash
# Reference outputs for the manyup.mixres port (docs/restructure_mixed_res.md, step 3): each
# trained checkpoint below rebuilt by the LEGACY finetune_inject (tag legacy-finetune-inject) with
# --regress_out: its picks + logits on the first 4 test batches and its full test metrics ->
# cache/regress/<key>.pt. The ported modules must reproduce these. Mail off.
#   [DRY=1] bash scripts/slurm/regress_record.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMAIL=
EXC=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088
mkdir -p cache/regress
rec() {   # rec <key> <finetune_inject args...>
  local key=$1; shift
  local cmd=(sbatch --parsable -J "regress_$key" --time=1:00:00 --cpus-per-task=9 --exclude=$EXC
             scripts/slurm/run.sh exp.upsamplers.finetune_inject --num_workers 8
             --regress_out "cache/regress/$key.pt" "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else echo "$key $("${cmd[@]}")"; fi
}
S16=(--select_unit location --inject_layer 0 --readout native_set --crop_px 8 --pos center
     --k_frac 0.03125)
S8=(--select_unit location --inject_layer 0 --readout native_set --lr_patch 8 --crop_px 4
    --pos center --k_frac 0.125)
P=(--task pastis --epochs 50)
LOC16=inject_fft_pastis_tiny_oracle_L0_k0.0312_loc_native_set_cpx8_st8_poscenter_lr0.001_ep50
BH16=inject_fft_pastis_tiny_oracle_bothalf_L0_k0.0312_loc_native_set_cpx8_st8_poscenter_lr0.001_ep50
ENT=(--score_src lr_entropy --scorer inject_fft_pastis_tiny_none_native_set_lr0.001_ep50
     --season_rule all --init_ckpt "$LOC16")

rec loc_oracle_s16m8_n8      "${P[@]}" --selector oracle "${S16[@]}" --loc_stride 8
rec bothalf_s16m8_n8         "${P[@]}" --selector oracle_bothalf "${S16[@]}" --loc_stride 8
rec bothalf_s8m4_n32         "${P[@]}" --selector oracle_bothalf "${S8[@]}"
rec season_grid_s16m8_n8     "${P[@]}" --selector season_grid "${S16[@]}"
rec random_s8m4_n32          "${P[@]}" --selector random "${S8[@]}"
rec level2_m4_s16m8_n8       "${P[@]}" --selector oracle_bothalf "${S16[@]}" --loc_stride 8 \
                             --init_ckpt "$BH16" --level2 m4 --l2_n 8
rec lrent_all_s16m8_n8       "${P[@]}" --selector oracle_bothalf "${S16[@]}" --loc_stride 8 "${ENT[@]}"
rec lrent_rese2e_s16m8_n8    "${P[@]}" --selector oracle_bothalf "${S16[@]}" --loc_stride 8 "${ENT[@]}" \
                             --res_mode e2e
rec yieldsat_bothalf_s16m8_n8 --task yieldsat_raw --epochs 50 --selector oracle_bothalf "${S16[@]}" \
                             --loc_stride 8
