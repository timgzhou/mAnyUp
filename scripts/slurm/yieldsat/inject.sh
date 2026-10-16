#!/bin/bash
# Submit the mixed-resolution (injected-token) arms on YieldSAT's full time series
# (task yieldsat_raw: every acquisition; OlmoEarth tiny, exp.upsamplers.finetune_inject), from
# the repo root, each as a chain of <= 3 h resumable jobs (links run afterany; a finished run
# exits at once). The fixed psN-tileM baselines on the ML-ready 24 steps are
# `MAX_JOB_H=3 bash scripts/slurm/sweep.sh yieldsat_ml24 olmoearth`.
#
#   [DRY=1] [ONLY=<regex>] bash scripts/slurm/yieldsat/inject.sh
#
# Arms (naming as in results tables: resolution x time per level, then selector + budget):
#   ps16mon_ts_native         fixed ps16 over the 24 month slots, through this pipeline
#   ps4mon_ts                 fixed ps4 over the 24 month slots (the costly reference)
#   ps16sea_ts_native_set     8 ps16 seasons, no crops
#   ps16sea->ps4mon rand k    + k random (month, ps4 cell) crops, k = 1/6, 1/3, 1/2 of slots
#   ps16sea->ps4mon->ps4date  + 2 random single-acquisition ps4 crops per month pick
# L40S throughput (tiny, 8703 train tiles): months arms 21.5 samples/s (~7 min/epoch);
# date-level arms 6.9 samples/s at batch 16 / 4 workers (~21 min/epoch): every raw
# acquisition of a tile (up to 125) travels through the loader, and batch 32 with 8 workers
# exceeds the 64 GB job memory.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
TASK=yieldsat_raw
EP=50
CPUS=9

LAST=""
chain() {   # chain <name> <links> [finetune_inject args...]
  local name=$1 links=$2; shift 2
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then return; fi
  # crop positions (2026-10-01): new submissions are centre-aligned (--pos center, run names
  # gain _poscenter). Kept as they are: a chain that names --pos, a no-crop chain, and a warm
  # start (its checkpoint was trained on corner positions). CROP_POS=corner submits the old way.
  if [[ "${CROP_POS:-center}" != corner && " $* " != *" --pos "* && " $* " != *" --init_ckpt "*
        && " $* " != *" --selector none "* ]]; then
    set -- "$@" --pos "${CROP_POS:-center}"
  fi
  local prev="${AFTER:-}" id i     # AFTER=<job id>: start after another chain's last link
  for i in $(seq 1 "$links"); do
    local cmd=(sbatch --parsable -J "inj_ys_${name}_$i" --time=3:00:00
               --cpus-per-task="$CPUS" --exclude=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088 ${MEM:+--mem=$MEM}
               ${prev:+--dependency=afterany:$prev}
               scripts/slurm/run.sh exp.upsamplers.finetune_inject --task "$TASK" --epochs "$EP"
               --max_hours 2.5 "$@")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; id="DRY$i"; else id=$("${cmd[@]}"); fi
    prev=$id
  done
  echo "$name: chain of $links ending $prev"
  LAST=$prev                        # the chain's last job id, for a dependent chain's AFTER
}

W=(--num_workers $((CPUS - 1)))
NS=(--inject_layer 0 --select_unit slot --readout native_set)
# references: the fixed series in this pipeline (same head, loss, epochs)
chain ps16mon_ts_native        3  "${W[@]}" --lr_steps all --selector none --readout native
chain ps4mon_ts                6  "${W[@]}" --lr_steps all --lr_patch 4 --selector none --batch_size 8
chain ps8mon_ts                4  "${W[@]}" --lr_steps all --lr_patch 8 --selector none   # 2026-10-09
chain ps16sea_ts_native_set    3  "${W[@]}" --selector none --readout native_set
# season -> month crops
chain rand_L0_native_set       4  "${W[@]}" --selector random "${NS[@]}"; R16=$LAST
chain rand_L0_k33_native_set   5  "${W[@]}" --selector random "${NS[@]}" --k_frac 0.3333; R33=$LAST
chain rand_L0_k50_native_set   4  "${W[@]}" --selector random "${NS[@]}" --k_frac 0.5
# season -> month -> single acquisition (the level only the raw release has)
D=(--num_workers 4 --batch_size 16 "${NS[@]}" --frac_slot --date_level free --date_per_month 2)
chain rand_L0_frac_free2_native_set      8  --selector random "${D[@]}"
chain rand_L0_k33_frac_free2_native_set  9  --selector random "${D[@]}" --k_frac 0.3333
# 2026-09-30: the current best candidates from BioMassters.
#   random ps8 / ps4 half-half (res0-1-1) and all-ps8 (res0-1-0) at 1/6
#   JOINT selector + backbone (--joint), warm start from the random run of the same budget
#   (20 more epochs; control = the same warm start, random picks), at 1/6 and 1/3; the joint
#   selector also choosing ps4 or ps8 (--joint_res, warm start from res0-1-1); and joint from
#   scratch at 1/6. The warm chains wait for their starting checkpoint (AFTER).
R=(--selector random "${NS[@]}")
chain rand_L0_native_set_res0-1-0  3  "${W[@]}" "${R[@]}" --res_mix 0,1,0
chain rand_L0_native_set_res0-1-1  4  "${W[@]}" "${R[@]}" --res_mix 0,1,1; R011=$LAST
chain scratch_joint_1-0.3          4  "${W[@]}" "${R[@]}" --joint --eps_start 1 --eps_end 0.3
CK="inject_fft_${TASK}_tiny_random_L0"
WA=("${W[@]}" "${R[@]}" --epochs 20)
AFTER=${R16:-} chain warm_rand            2 "${WA[@]}" --init_ckpt "${CK}_k0.167_native_set_lr0.001_ep${EP}"
AFTER=${R16:-} chain warm_joint_1-0.3     3 "${WA[@]}" --init_ckpt "${CK}_k0.167_native_set_lr0.001_ep${EP}" --joint --eps_start 1 --eps_end 0.3
AFTER=${R16:-} chain warm_joint_1-0.3_seed1 3 "${WA[@]}" --init_ckpt "${CK}_k0.167_native_set_lr0.001_ep${EP}" --joint --eps_start 1 --eps_end 0.3 --seed 1
AFTER=${R16:-} chain warm_rand_seed1      2 "${WA[@]}" --init_ckpt "${CK}_k0.167_native_set_lr0.001_ep${EP}" --seed 1
AFTER=${R33:-} chain warm_k33_rand        3 "${WA[@]}" --k_frac 0.3333 --init_ckpt "${CK}_k0.333_native_set_lr0.001_ep${EP}"
AFTER=${R33:-} chain warm_k33_joint_1-0.3 3 "${WA[@]}" --k_frac 0.3333 --init_ckpt "${CK}_k0.333_native_set_lr0.001_ep${EP}" --joint --eps_start 1 --eps_end 0.3
AFTER=${R011:-} chain warm_res011_rand    2 "${WA[@]}" --res_mix 0,1,1 --init_ckpt "${CK}_k0.167_native_set_res0-1-1_lr0.001_ep${EP}"
AFTER=${R011:-} chain warm_jres_1-0.3     3 "${WA[@]}" --init_ckpt "${CK}_k0.167_native_set_res0-1-1_lr0.001_ep${EP}" --joint --joint_res --eps_start 1 --eps_end 0.3
# 2026-10-01: from scratch with a random-pick phase first (--eps_hold): eps 1 for 50 epochs, then
# 1 -> 0.3 over 10 and held, 70 epochs (= the warm arms' 50 + 20)
chain scratch_hold_joint_1-0.3 6 "${W[@]}" "${R[@]}" --epochs 70 --joint --eps_start 1 --eps_end 0.3 --eps_hold 0.714 --eps_anneal 0.143
# 2026-10-01: DENSE SELF-TEACHER, feature space (--self_teach 4, tsplat): the teacher reads 4
# random RAW acquisitions densely at ps4 (task loss); the mixed-resolution splat read at those
# times must reconstruct its features; --joint values = reconstruction gain. From scratch, 70
# epochs. The teacher's 4 acquisitions are drawn in the loader workers (WithDates teach_n), so
# the teach arms run at the default batch with 8 workers (~33 tiles/s, ~4.5 min/epoch).
TS=(--selector random --inject_layer 0 --select_unit slot --readout tsplat --epochs 70)
JH=(--joint --eps_start 1 --eps_end 0.3 --eps_hold 0.714 --eps_anneal 0.143)
chain scratch_ts_rand                          5  "${W[@]}" "${TS[@]}"
chain scratch_ts_teach4                        5 "${W[@]}" "${TS[@]}" --self_teach 4
chain scratch_ts_teach4_hold_joint_1-0.3       5 "${W[@]}" "${TS[@]}" --self_teach 4 "${JH[@]}"
chain scratch_ts_teach4_hold_joint_1-0.3_seed1 5 "${W[@]}" "${TS[@]}" --self_teach 4 "${JH[@]}" --seed 1
chain scratch_ts_k33_rand                      6  "${W[@]}" "${TS[@]}" --k_frac 0.3333
chain scratch_ts_k33_teach4_hold_joint_1-0.3   5 "${W[@]}" "${TS[@]}" --k_frac 0.3333 --self_teach 4 "${JH[@]}"
# 2026-10-01: dense self-teacher with the LOGIT-SPACE task readout (native_set) and a separate
# reconstruction splat; warm start as the joint arms above (WA); values: reconstruction, or both
J=(--joint --eps_start 1 --eps_end 0.3)
NSC=(--init_ckpt "${CK}_k0.167_native_set_lr0.001_ep${EP}")
chain warm_ns_teach4                    3 "${WA[@]}" "${NSC[@]}" --self_teach 4
chain warm_ns_teach4_joint_1-0.3        3 "${WA[@]}" "${NSC[@]}" --self_teach 4 "${J[@]}"
chain warm_ns_teach4_joint_1-0.3_seed1  3 "${WA[@]}" "${NSC[@]}" --self_teach 4 "${J[@]}" --seed 1
chain warm_ns_teach4_vboth_joint_1-0.3  3 "${WA[@]}" "${NSC[@]}" --self_teach 4 "${J[@]}" --teach_value both
# 2026-10-01: --teach_detach recon / all (see scripts/slurm/inject.sh round 28)
for DT in recon all; do
chain "warm_ns_teach4_dt${DT}"                   3 "${WA[@]}" "${NSC[@]}" --self_teach 4 --teach_detach "$DT"
chain "warm_ns_teach4_dt${DT}_joint_1-0.3"       3 "${WA[@]}" "${NSC[@]}" --self_teach 4 --teach_detach "$DT" "${J[@]}"
chain "warm_ns_teach4_dt${DT}_joint_1-0.3_seed1" 3 "${WA[@]}" "${NSC[@]}" --self_teach 4 --teach_detach "$DT" "${J[@]}" --seed 1
done
# 2026-10-02: the injection depth L with and without --head_attn oneway (crops read the cached
# lr keys / values in blocks [0, L), joint attention from L on); random ps4 month crops at 1/6,
# native_set, centre positions (L0 repeated). And a cross-attention selector on the lr tokens at
# depth L.
SW=("${W[@]}" --select_unit slot --readout native_set)
chain sweep_rand_L0_native_set 4 --selector random "${SW[@]}" --inject_layer 0
for LY in 2 3 4 6 8; do
chain "sweep_rand_L${LY}_native_set"          4 --selector random "${SW[@]}" --inject_layer "$LY"
chain "sweep_rand_L${LY}_native_set_haoneway" 5 --selector random "${SW[@]}" --inject_layer "$LY" --head_attn oneway
done
for LY in 4 8; do
chain "sweep_xattn_L${LY}_native_set_haoneway" 5 --selector xattn --score_norm "${SW[@]}" --inject_layer "$LY" --head_attn oneway
done
# 2026-10-02: the joint selector without its 12-block pre-look (--joint_feat lrL; see
# scripts/slurm/inject.sh round 30), warm start from the random L{4,6} one-way centre runs
for LY in 2 3 4 6; do
JF=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer "$LY"
    --head_attn oneway --pos center --epochs 20
    --init_ckpt "${CK%_L0}_L${LY}_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
chain "jf_warm_L${LY}_rand"          2 "${JF[@]}"
chain "jf_warm_L${LY}_joint_prelook" 3 "${JF[@]}" --joint --eps_start 1 --eps_end 0.3
chain "jf_warm_L${LY}_joint_lrL"     3 "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
done
# 2026-10-02: the layer-3 selector trained through the task loss with DPS's perturbed top-k
# (--joint_grad dps / both; see scripts/slurm/inject.sh round 31); 2 seeds
JF=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs 20 --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --init_ckpt "${CK%_L0}_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for JG in dps both; do
chain "dps_warm_L3_lrL_${JG}"       3 "${JF[@]}" --joint_grad "$JG"
chain "dps_warm_L3_lrL_${JG}_seed1" 3 "${JF[@]}" --joint_grad "$JG" --seed 1
done
# 2026-10-02 (eve): seeds 1 and 2 of the layer-L joint selector and its random control at
# L = 2, 3 (seed 0: 1.310 / 1.305 vs controls 1.333 / 1.343). Same seed-0 warm start.
for LY in 2 3; do
JF=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer "$LY"
    --head_attn oneway --pos center --epochs 20
    --init_ckpt "${CK%_L0}_L${LY}_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 1 2; do
chain "jf_warm_L${LY}_rand_seed${SD}"      2 "${JF[@]}" --seed "$SD"
chain "jf_warm_L${LY}_joint_lrL_seed${SD}" 3 "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL --seed "$SD"
done
done
# 2026-10-04: one change at a time to the layer-3 value predictor (see scripts/slurm/inject.sh
# round 33); seeds 0 and 1, vs jf_warm_L3_joint_lrL / jf_warm_L3_rand
JF=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs 20
    --init_ckpt "${CK%_L0}_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
JL=("${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "sel_gate$SX"     3 "${JL[@]}" --joint_target gate --seed "$SD"
chain "sel_drop$SX"     3 "${JL[@]}" --joint_target drop --seed "$SD"
chain "sel_enc$SX"      3 "${JL[@]}" --joint_in enc      --seed "$SD"
chain "sel_xattn$SX"    3 "${JL[@]}" --joint_arch xattn  --seed "$SD"
chain "sel_ownemb$SX"   3 "${JL[@]}" --joint_embed own   --seed "$SD"
chain "sel_jit$SX"      3 "${JL[@]}" --jitter            --seed "$SD"
chain "sel_jit_rand$SX" 2 "${JF[@]}" --jitter            --seed "$SD"
done
# 2026-10-04: the date-level, per-modality value selector at a token budget (--date_joint; see
# scripts/slurm/inject.sh round 34); seeds 0, 1
JF=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs 20 --frac_slot --joint --joint_feat lrL
    --init_ckpt "${CK%_L0}_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for DJ in only hier; do
chain "dj_${DJ}_sel$SX"  5 "${JF[@]}" --date_joint "$DJ" --eps_start 1 --eps_end 0.3 --seed "$SD"
chain "dj_${DJ}_rand$SX" 5 "${JF[@]}" --date_joint "$DJ" --eps_start 1 --eps_end 1   --seed "$SD"
done
done
# 2026-10-04: ps16-window crops (--crop_px 16; see scripts/slurm/inject.sh round 35)
chain cpx16_rand_L0 4 --selector random "${SW[@]}" --crop_px 16 --inject_layer 0
chain cpx16_rand_L3_haoneway 5 --selector random "${SW[@]}" --crop_px 16 --inject_layer 3 --head_attn oneway
JF=("${W[@]}" --selector random --select_unit slot --readout native_set --crop_px 16 --inject_layer 3
    --head_attn oneway --pos center --epochs 20
    --init_ckpt "${CK%_L0}_L3_k0.167_native_set_cpx16_poscenter_haoneway_lr0.001_ep${EP}")
B3=$LAST
AFTER=$B3 chain cpx16_warm_L3_rand      2 "${JF[@]}"
AFTER=$B3 chain cpx16_warm_L3_joint_lrL 3 "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL

# 2026-10-04: month crops pooled from the month's raw acquisitions (--acq_pool; see
# scripts/slurm/inject.sh round 34). ~7 of 24 months are present per sample, so "all" (every
# present slot, 1792 crops) is what --k_frac 1/3 already gives.
AQ=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 3 --frac_slot)
for AP in none mean attn; do
NM=$([ "$AP" = none ] && echo comp || echo "$AP")
chain "acq_L3_all_${NM}" 7 "${AQ[@]}" --k_frac 1      --acq_pool "$AP" --batch_size 16
chain "acq_L3_k17_${NM}" 6 "${AQ[@]}" --k_frac 0.1667 --acq_pool "$AP"
done

# 2026-10-05: acquisition pooling follow-ups (see scripts/slurm/inject.sh round 36): a second
# seed from scratch, and the drop-target value selector vs random picks on the 1/6 runs (warm).
# MS=base: the same on OlmoEarth base.
MS="${MS:-tiny}"; MX=$([ "$MS" = tiny ] && echo "" || echo "_$MS")
AQ=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 3 --frac_slot --model_size "$MS")
for AP in none attn; do
NM=$([ "$AP" = none ] && echo comp || echo "$AP"); AT=$([ "$AP" = none ] && echo "" || echo "_acq$AP")
for SD in ${SEEDS:-1}; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "acq_L3_all_${NM}${MX}${SX}" "${LK:-4}" "${AQ[@]}" --k_frac 1      --acq_pool "$AP" --batch_size "${BS:-16}" --seed "$SD"
chain "acq_L3_k17_${NM}${MX}${SX}" "${LK:-4}" "${AQ[@]}" --k_frac 0.1667 --acq_pool "$AP" ${BSK:+--batch_size "$BSK"} --seed "$SD"
done
WQ=("${AQ[@]}" --k_frac 0.1667 --acq_pool "$AP" --pos center --epochs 20 ${BSK:+--batch_size "$BSK"}
    --init_ckpt "inject_fft_yieldsat_raw_${MS}_random_L3_k0.167_native_set_frac${AT}_poscenter_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "acq_warm_k17_${NM}${MX}_rand${SX}" "${LW:-2}" "${WQ[@]}" --seed "$SD"
chain "acq_warm_k17_${NM}${MX}_drop${SX}" "${LW:-3}" "${WQ[@]}" --seed "$SD" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL --joint_target drop
done
done
# 2026-10-05: regular-lattice picks (--selector grid; scripts/slurm/inject.sh round 38). 1/6 of
# the 24 month slots is 57% of the ~7 present ones, so the lattice here is the space-time
# checkerboard (50%) plus a random 7%.
GQ=("${W[@]}" --selector grid --select_unit slot --readout native_set --inject_layer 3 --frac_slot)
for AP in none attn; do
NM=$([ "$AP" = none ] && echo comp || echo "$AP"); AT=$([ "$AP" = none ] && echo "" || echo "_acq$AP")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "acq_L3_k17_${NM}_grid${SX}" 4 "${GQ[@]}" --k_frac 0.1667 --acq_pool "$AP" --seed "$SD"
chain "acq_warm_k17_${NM}_grid${SX}" 2 "${GQ[@]}" --k_frac 0.1667 --acq_pool "$AP" --pos center --epochs 20 --seed "$SD" \
    --init_ckpt "inject_fft_yieldsat_raw_tiny_random_L3_k0.167_native_set_frac${AT}_poscenter_lr0.001_ep${EP}"
done
done
# 2026-10-06: lattice picks at OlmoEarth base, and the harder selector test warm from the
# lattice-pick run (scripts/slurm/inject.sh round 39). MS=base for the base chains.
MS="${MS:-tiny}"; MX=$([ "$MS" = tiny ] && echo "" || echo "_$MS")
GB=("${W[@]}" --select_unit slot --readout native_set --inject_layer 3 --frac_slot --acq_pool attn --model_size "$MS" --k_frac 0.1667 ${BSK:+--batch_size "$BSK"})
GD=""
if [ "$MS" != tiny ]; then
chain "acq_L3_k17_attn${MX}_grid"       "${LK:-6}" --selector grid "${GB[@]}"; GD=${LAST:-}
chain "acq_L3_k17_attn${MX}_grid_seed1" "${LK:-6}" --selector grid "${GB[@]}" --seed 1
fi
GW=("${GB[@]}" --pos center --epochs 20
    --init_ckpt "inject_fft_yieldsat_raw_${MS}_grid_L3_k0.167_native_set_frac_acqattn_poscenter_lr0.001_ep${EP}")
JD=(--selector random --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL --joint_target drop)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
AFTER=$GD
chain "gwarm_k17${MX}_grid${SX}"  "${LW:-2}" --selector grid "${GW[@]}" --seed "$SD"
chain "gwarm_k17${MX}_drop${SX}"  "${LW:-3}" "${JD[@]}" "${GW[@]}" --seed "$SD"
chain "gwarm_k17${MX}_dropg${SX}" "${LW:-3}" "${JD[@]}" "${GW[@]}" --seed "$SD" --joint_explore grid
unset AFTER
done
# 2026-10-05: dropout-pass target, more seeds and L2 (see scripts/slurm/inject.sh round 36)
for LY in 2 3; do
JL=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer "$LY"
    --head_attn oneway --pos center --epochs 20 --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --init_ckpt "${CK%_L0}_L${LY}_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1 2; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "sel_drop_L${LY}$SX" 3 "${JL[@]}" --joint_target drop --seed "$SD"
done
done
chain ps_rand_L0_seed1    4 --selector random "${SW[@]}" --inject_layer 0 --seed 1
chain cpx16_rand_L0_seed1 4 --selector random "${SW[@]}" --inject_layer 0 --seed 1 --crop_px 16
# 2026-10-05: attention value predictor on encoded candidates (see scripts/slurm/inject.sh round 37)
JL=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs 20 --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --joint_target drop
    --init_ckpt "${CK%_L0}_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for JA in xenc xenc_t xenc_ts; do
chain "sel2_${JA}$SX" 3 "${JL[@]}" --joint_arch "$JA" --seed "$SD"
done
done
# 2026-10-05: cheap / coarse-attention selectors and less noisy dropout labels (see
# scripts/slurm/inject.sh round 38)
JL=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs 20 --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --joint_target drop
    --init_ckpt "${CK%_L0}_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "sel3_xs$SX"   3 "${JL[@]}" --joint_arch xs  --seed "$SD"
chain "sel3_xc$SX"   3 "${JL[@]}" --joint_arch xc  --seed "$SD"
chain "sel3_df25$SX" 3 "${JL[@]}" --drop_frac 0.25 --seed "$SD"
chain "sel3_dp2$SX"  3 "${JL[@]}" --drop_passes 2  --seed "$SD"
done
# 2026-10-05: free-position crops by patch size (see scripts/slurm/inject.sh round 36)
FR=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 0 --crop_free)
for PX in 16 8 4; do
chain "free_ps$PX" 5 "${FR[@]}" --crop_px "$PX"
done
chain vary_ps16 4 "${FR[@]}" --crop_px 16 --n_crops 196  --vary_crops
chain vary_ps8  5 "${FR[@]}" --crop_px 8  --n_crops 512  --vary_crops
chain vary_ps4  5 "${FR[@]}" --crop_px 4  --n_crops 1024 --vary_crops
# 2026-10-06: vary_ps* seed 1 and higher caps (see scripts/slurm/inject.sh round 37)
FR=("${W[@]}" --selector random --select_unit slot --readout native_set --inject_layer 0 --crop_free --vary_crops)
chain vary_ps16_seed1 4 "${FR[@]}" --crop_px 16 --n_crops 196  --seed 1
chain vary_ps8_seed1  5 "${FR[@]}" --crop_px 8  --n_crops 512  --seed 1
chain vary_ps4_seed1  5 "${FR[@]}" --crop_px 4  --n_crops 1024 --seed 1
chain varyhi_ps16 5 "${FR[@]}" --crop_px 16 --n_crops 512
chain varyhi_ps8  6 "${FR[@]}" --crop_px 8  --n_crops 1024
chain varyhi_ps4  8 "${FR[@]}" --crop_px 4  --n_crops 2048
# 2026-10-06: lattice picks on the crop-count curves (see scripts/slurm/inject.sh round 40)
FR=("${W[@]}" --select_unit slot --readout native_set --inject_layer 0 --crop_free --vary_crops)
chain nsweep_ps16 1 "${FR[@]}" --selector random --crop_px 16 --n_crops 196  --sweep_only
chain nsweep_ps8  1 "${FR[@]}" --selector random --crop_px 8  --n_crops 512  --sweep_only
chain nsweep_ps4  1 "${FR[@]}" --selector random --crop_px 4  --n_crops 1024 --sweep_only
chain varyg_ps16 4 "${FR[@]}" --selector grid --crop_px 16 --n_crops 196
chain varyg_ps8  5 "${FR[@]}" --selector grid --crop_px 8  --n_crops 512
chain varyg_ps4  5 "${FR[@]}" --selector grid --crop_px 4  --n_crops 1024
FR=("${W[@]}" --select_unit slot --readout native_set --inject_layer 0 --crop_free --vary_crops --selector random --sweep_only)
chain nsweep37_ps16_seed1 1 "${FR[@]}" --crop_px 16 --n_crops 196  --seed 1
chain nsweep37_ps8_seed1  1 "${FR[@]}" --crop_px 8  --n_crops 512  --seed 1
chain nsweep37_ps4_seed1  1 "${FR[@]}" --crop_px 4  --n_crops 1024 --seed 1
chain nsweep37_hi_ps16 1 "${FR[@]}" --crop_px 16 --n_crops 512
chain nsweep37_hi_ps8  1 "${FR[@]}" --crop_px 8  --n_crops 1024
chain nsweep37_hi_ps4  1 "${FR[@]}" --crop_px 4  --n_crops 2048
# round 41 (2026-10-06): APT (arXiv:2510.18091) pixel-entropy selection (--selector apt, see
# InjectFT._apt_scores), against the plain random and staggered-lattice (grid) controls at the
# same budgets -- see scripts/slurm/inject.sh round 41 for the PASTIS / BioMassters pair.
chain apt_L0_native_set      4 "${W[@]}" --selector apt  "${NS[@]}"
chain apt_L0_k33_native_set  5 "${W[@]}" --selector apt  "${NS[@]}" --k_frac 0.3333
chain grid_L0_native_set     4 "${W[@]}" --selector grid "${NS[@]}"
chain grid_L0_k33_native_set 5 "${W[@]}" --selector grid "${NS[@]}" --k_frac 0.3333
# round 42 (2026-10-07): --apt_time none (pure per-frame spatial entropy, time-ignorant) and
# --apt_pool (space-then-time cell pick + consecutive-month merge) -- see inject.sh round 42.
chain apttime_L0_native_set      4 "${W[@]}" --selector apt "${NS[@]}" --apt_time none
chain apttime_L0_k33_native_set  5 "${W[@]}" --selector apt "${NS[@]}" --apt_time none --k_frac 0.3333
chain aptpool_L0_native_set      4 "${W[@]}" --selector apt "${NS[@]}" --apt_pool
chain aptpool_L0_k33_native_set  5 "${W[@]}" --selector apt "${NS[@]}" --apt_pool --k_frac 0.3333
# 2026-10-08: one crop-position list for every crop size (see scripts/slurm/inject.sh round 41)
FR=("${W[@]}" --select_unit slot --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --n_crops 2048)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for GM in c2f shared; do for PX in 16 8 4; do
chain "pos_${GM}_ps$PX$SX" 8 "${FR[@]}" --grid_mode "$GM" --crop_px "$PX" --seed "$SD"
done; done
done
# 2026-10-08: ps8 -> ps4 (seasons at ps8 + ps4 crops) and the fixed seasons-only endpoints
# (see scripts/slurm/inject.sh round 43)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for GM in c2f shared; do
chain "b8_${GM}_ps4$SX" 8 "${FR[@]}" --grid_mode "$GM" --crop_px 4 --lr_patch 8 --seed "$SD"
done
for PX in 16 8 4; do
chain "fix_ps$PX$SX" $([ "$PX" = 4 ] && echo 8 || echo 3) "${W[@]}" --selector none --readout native_set --lr_patch "$PX" --seed "$SD"
done
done
# 2026-10-09: count-set training and dedicated-count models (see scripts/slurm/inject.sh round 44)
FS=("${W[@]}" --select_unit slot --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --grid_mode shared)
for PAIR in "16 8" "16 4" "8 4" "16 16"; do   # 16 16: only the ded_ chains were submitted
set -- $PAIR
PA=(--lr_patch "$1" --crop_px "$2")
chain "set_$1-$2"      8  "${FS[@]}" "${PA[@]}" --crop_counts 0,64,128,256,512,1024,2048
chain "set_$1-$2_long" 14 "${FS[@]}" "${PA[@]}" --crop_counts 0,64,128,256,512,1024,2048 --epochs $((2 * EP))
for K in 64 128 256 512 1024 2048; do
chain "ded_$1-$2_n$K" $(case $K in 2048) echo 12 ;; 1024) echo 7 ;; 512) echo 5 ;; *) echo 4 ;; esac) "${FS[@]}" "${PA[@]}" --crop_counts "$K"
done
done
# 2026-10-09: dedicated-count models with crops injected at block 3 (inject.sh round 45)
F3=("${W[@]}" --select_unit slot --readout native_set --inject_layer 3 --crop_free --vary_crops --selector grid --grid_mode shared)
for PAIR in "16 8" "16 4" "8 4" "16 16"; do
set -- $PAIR
for K in 64 128 256 512 1024 2048; do
chain "ded3_$1-$2_n$K" $(case $K in 2048) echo 12 ;; 1024) echo 7 ;; 512) echo 5 ;; *) echo 4 ;; esac) "${F3[@]}" --lr_patch "$1" --crop_px "$2" --crop_counts "$K"
done
done
# 2026-10-09: is the dedicated ps8-crop model at 2048 crops undertrained? twice the epochs
chain "ded_16-8_n2048_long" 22 "${FS[@]}" --lr_patch 16 --crop_px 8 --crop_counts 2048 --epochs $((2 * EP))
# 2026-10-12: spatial resolution only, no season tokens (inject.sh round 46)
FN=("${W[@]}" --select_unit slot --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --grid_mode shared --no_seasons)
for PK in "16 64" "16 128" "16 192" "16 256" "16 512" "8 128" "8 256" "8 512" "8 768" "8 1024" "4 512" "4 1024" "4 2048" "4 3072"; do
set -- $PK
chain "nos_ps$1_n$2" $(case $2 in 3072) echo 14 ;; 2048) echo 12 ;; 1024) echo 7 ;; 512|768) echo 5 ;; *) echo 4 ;; esac) "${FN[@]}" --crop_px "$1" --crop_counts "$2"
done
# 2026-10-12: on-grid months + shifted-grid months of the same patch size (inject.sh round 47)
SG=("${W[@]}" --lr_steps all --readout native_set --batch_size 8)
for P in 16 8 4; do
chain "sg_ps${P}_base" $([ "$P" = 4 ] && echo 10 || echo 5) "${SG[@]}" --lr_patch "$P" --selector none
for S in 16 8; do   # stride 4 dropped 2026-10-12: 1400-2800 GMACs a sample, too slow
chain "sg_ps${P}_s$S" $( { [ "$S" = 4 ] || [ "$P" = 4 ]; } && echo 16 || echo 7) "${SG[@]}" --lr_patch "$P" --crop_px "$P" --shift_stride "$S" --select_unit slot --inject_layer 0 --crop_free --vary_crops --selector grid
done
done
# 2026-10-13: temporal resolution only (inject.sh round 48)
TR=("${W[@]}" --readout native_set --batch_size 8 --selector none)
for P in 16 8 4; do
chain "tr_ps${P}_year" 3 "${TR[@]}" --lr_patch "$P" --lr_steps year
done
chain tr_ps16_raw 6  "${TR[@]}" --lr_patch 16 --lr_steps raw
chain tr_ps8_raw  14 "${TR[@]}" --lr_patch 8  --lr_steps raw
# 2026-10-08 (user: "what if we use apt to query ps16 tokens?"): APT chooses patch SIZE
# (ps16 vs ps4) by entropy gain -- see scripts/slurm/inject.sh's 2026-10-08 apt_res block.
chain aptres_L0_native_set      4 "${W[@]}" --selector apt "${NS[@]}" --apt_res
chain aptres_L0_k33_native_set  5 "${W[@]}" --selector apt "${NS[@]}" --apt_res --k_frac 0.3333
# 2026-10-12 (user: "start with ps16 and t=all... split in time or space or both, depending on
# the after-split entropy -- this is almost info gain"): a true 3D (space, space, time) octree
# -- InjectFT._apt_octree_mask, no token budget, pure entropy threshold --apt_tau (calibrated
# by exp.upsamplers.calibrate_apt_octree_tau, see scripts/slurm/inject.sh's matching round and
# results/apt_octree_tau.json). YieldSAT is a special case: BOTH budgets (1/6 -> 1024, 1/3 ->
# 2048 target tokens) are UNREACHABLE -- the calibration's own near-zero-tau check tops out at
# only 339.8 tokens regardless (T=24 months is a lot of real redundancy for the gain-per-token
# test to find, apparently more than pastis/biomassters/geoidflood all have) -- queued anyway
# at the SAME small positive tau (0.05) for both budget labels, since the octree produces the
# same token count either way here (there is no second, less-aggressive tau to distinguish
# them) -- user-confirmed 2026-10-12 policy: report actual vs target honestly, never skip.
#
# 3 --apt_octree_pool arms (user: "how are the time series patches handled... input composites
# or embedding pool or something else", then: "also 1) encode and pool embedding, and 2) encode
# and cross attention pool embedding... computation wise input < embed < attn") -- see
# scripts/slurm/inject.sh's matching block for the full rationale. More links than
# pastis/biomassters' 3 (date-level-ish Python recursion over T=24 months/sample is slower than
# pastis/biomassters' T=12, matching this file's own aptres links already being higher: 4-5 vs
# inject.sh's 2).
for POOL in input embed attn; do
chain "aptoctree_${POOL}_L0_native_set"     5 "${W[@]}" --selector apt "${NS[@]}" --apt_octree --apt_octree_pool "$POOL" --apt_tau 0.05
chain "aptoctree_${POOL}_L0_k33_native_set" 5 "${W[@]}" --selector apt "${NS[@]}" --apt_octree --apt_octree_pool "$POOL" --apt_tau 0.05 --k_frac 0.3333
done
# 2026-10-15: label-error ORACLE (see scripts/slurm/inject.sh round 49). No LR-only native
# checkpoint here: the zero-shot arms start from ps16sea_ts_native_set with its learned pool
# score dropped (native_mean has none).
ZS_CKPT=inject_fft_yieldsat_raw_tiny_none_native_set_lr0.001_ep50
OR=(--inject_layer 0 --select_unit location --crop_px 8)
chain zs_none_mean 1 "${W[@]}" --selector none --readout native_mean --init_ckpt "$ZS_CKPT" --epochs 0
chain none_mean    4 "${W[@]}" --selector none --readout native_mean
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for SEL in oracle random; do
S=${SEL:0:3}
chain "zs_${S}_n${N}_mean" 1 "${W[@]}" --selector "$SEL" "${OR[@]}" --k_frac "$KF" --readout native_mean \
  --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
chain "${S}_n${N}_L0_mean" 5 "${W[@]}" --selector "$SEL" "${OR[@]}" --k_frac "$KF" --readout native_mean
chain "${S}_n${N}_L0_set"  5 "${W[@]}" --selector "$SEL" "${OR[@]}" --k_frac "$KF" --readout native_set
done
done
# 2026-10-15: the zero-shot oracle with --attn oneway (see scripts/slurm/inject.sh round 50)
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for SEL in oracle random; do
chain "zs_${SEL:0:3}_n${N}_mean_oneway" 1 "${W[@]}" --selector "$SEL" "${OR[@]}" --k_frac "$KF" \
  --readout native_mean --attn oneway --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
done
done
# 2026-10-15: fine-tuned oracle / random with --attn oneway (see scripts/slurm/inject.sh round 51)
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for SEL in oracle random; do
for R in mean set; do
chain "${SEL:0:3}_n${N}_L0_${R}_oneway" 5 "${W[@]}" --selector "$SEL" "${OR[@]}" --k_frac "$KF" \
  --readout "native_$R" --attn oneway
done
done
done
# 2026-10-15: the content-blind location lattice at the oracle's budgets (see
# scripts/slurm/inject.sh round 52)
GR=(--selector grid --inject_layer 0 --select_unit location --crop_px 8)
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for R in mean set; do
chain "grid_n${N}_L0_${R}"        5 "${W[@]}" "${GR[@]}" --k_frac "$KF" --readout "native_$R"
chain "grid_n${N}_L0_${R}_oneway" 5 "${W[@]}" "${GR[@]}" --k_frac "$KF" --readout "native_$R" --attn oneway
done
chain "zs_grid_n${N}_mean"        1 "${W[@]}" "${GR[@]}" --k_frac "$KF" --readout native_mean \
  --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
chain "zs_grid_n${N}_mean_oneway" 1 "${W[@]}" "${GR[@]}" --k_frac "$KF" --readout native_mean \
  --attn oneway --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
done
# 2026-10-15: oracle candidates on a coarser lattice (--loc_stride 6 / 8; see
# scripts/slurm/inject.sh round 53)
for S in 6 8; do
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for R in mean set; do
chain "ora_st${S}_n${N}_L0_${R}"        5 "${W[@]}" --selector oracle "${OR[@]}" --loc_stride "$S" --k_frac "$KF" --readout "native_$R"
chain "ora_st${S}_n${N}_L0_${R}_oneway" 5 "${W[@]}" --selector oracle "${OR[@]}" --loc_stride "$S" --k_frac "$KF" --readout "native_$R" --attn oneway
done
chain "zs_ora_st${S}_n${N}_mean_oneway" 1 "${W[@]}" --selector oracle "${OR[@]}" --loc_stride "$S" --k_frac "$KF" \
  --readout native_mean --attn oneway --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
done
done
