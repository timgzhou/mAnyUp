#!/bin/bash
# Submit the GEOID S1-series fine-tunes (exp.upsamplers.finetune_inject) for one task, each a
# chain of short resumable jobs, from the repo root:
#
#   [AFTER=<slurm dependency>] [ONLY=<regex>] [DRY=1] bash scripts/slurm/inject_series.sh <task>
#       task: geoidseries_flood (flood vs rest, flood IoU) | geoidseries (3 classes)
#
# All: 10 epochs of 50k train chips drawn afresh (--epoch_samples), crops enter at block 0,
# pair selection (a chosen pair of consecutive dates crops both dates).
#   ref_months     months at ps16, no crops                       (the low-res input alone)
#   ref_all16      every date at ps16, no crops                   (OlmoEarth series FT, ps16)
#   ref_all4       every date at ps4, no crops                    (the full-resolution ceiling)
#   rand           random pairs, 1/6 of date-cells; rand_k33 1/3
#   xattnT_gain    cross-attention selector + self-attention across dates, normalized
#                  scores, first-order-gain loss; xattn_gain without the temporal part;
#                  mlp_gain the per-slot MLP selector, same objective
set -euo pipefail
TASK="${1:?usage: bash scripts/slurm/inject_series.sh <geoidseries_flood|geoidseries>}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }
CPUS=9
LAST=""
T=$([ "$TASK" = geoidseries_flood ] && echo gsf || echo gs3)

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
  local prev="${AFTER:-}" id i dep
  for i in $(seq 1 "$links"); do
    dep=""
    if [ -n "$prev" ]; then dep="--dependency=$([ "$i" = 1 ] && echo "$prev" || echo "afterany:$prev")"; fi
    local cmd=(sbatch --parsable -J "inj_${T}_${name}_$i" --time=3:00:00 --cpus-per-task="$CPUS"
               --exclude=kn117,kn104,kn101,kn001 $dep
               scripts/slurm/run.sh exp.upsamplers.finetune_inject --task "$TASK" --epochs 10
               --epoch_samples 50000 --num_workers $((CPUS - 1)) --max_hours 2.5 "$@")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; id="DRY$i"; else id=$("${cmd[@]}"); fi
    prev=$id
  done
  echo "$name: chain of $links ending $prev"
  LAST=$prev                        # the chain's last job id, for a dependent chain's AFTER
}

P=(--inject_layer 0 --select_unit pair)
chain ref_months   3 --selector none
chain ref_all16    3 --selector none --lr_steps all
chain ref_all4     4 --selector none --lr_steps all --lr_patch 4 --batch_size 16
chain rand         3 --selector random "${P[@]}"
chain rand_k33     3 --selector random "${P[@]}" --k_frac 0.3333
chain xattnT_gain  3 --selector xattn  "${P[@]}" --sel_temporal --score_norm --gain_loss 1
if [ "$TASK" = geoidseries_flood ]; then
  chain xattn_gain 3 --selector xattn   "${P[@]}" --score_norm --gain_loss 1
  chain mlp_gain   3 --selector learned "${P[@]}" --score_norm --gain_loss 1
fi
# 2026-09-28: fractional crop slots (dates of one month get distinct time encodings)
chain rand_frac    3 --selector random "${P[@]}" --frac_slot
# 2026-09-30: time-continuous overlap-only readouts (manyup.inject.TimeSetReadout): each date
# reads its cell's month tokens + crops by time (Gaussian: tsplat; time-queried attention:
# tquery) -- dates of one month no longer share a token, so pair diffs are not exactly 0 off crops
for RO in tsplat tquery; do
chain "ref_months_${RO}"  3 --selector none   --readout "$RO"
chain "rand_${RO}"        3 --selector random "${P[@]}" --readout "$RO"
chain "rand_k33_${RO}"    3 --selector random "${P[@]}" --readout "$RO" --k_frac 0.3333
done
# 2026-09-30: the cheap local fixed baseline (every date at ps4, each 4 px tile its own sequence:
# ps4ts4_ts) and the 1/2 budget, to trace mixed resolution up to the every-date ps4 reference
chain ref_all4_tile4   4 --selector none --lr_steps all --lr_patch 4 --lr_tile 4 --batch_size 16
chain rand_k50         3 --selector random "${P[@]}" --k_frac 0.5
for RO in tsplat tquery; do
chain "rand_k50_${RO}" 3 --selector random "${P[@]}" --readout "$RO" --k_frac 0.5
done
# 2026-09-30: JOINT selector + backbone on the series pair heads (--joint, pair unit): warm start
# from random 1/6 pairs + time-queried attention readout; the predictor scores every (date,
# cell), a pair-cell scores the max of its two dates; eps random pair-cells, the rest the best;
# trained on each crop's in-context leave-one-out value through the pair head (pair_loo_values).
# 4 more epochs; control = the same warm start, random pairs.
WJ=(--selector random "${P[@]}" --readout tquery --epochs 4
    --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.167_pair_tquery_es50000_lr0.001_ep10")
chain warm_rand_tquery        3 "${WJ[@]}"
chain warm_joint_1-0.3_tquery 3 "${WJ[@]}" --joint --eps_start 1 --eps_end 0.3
chain warm_joint_0.5_tquery   3 "${WJ[@]}" --joint --eps_start 0.5 --eps_end 0.5
# 2026-09-30 (b): is the 3-class joint gain real? Equal DATE-crop count (--pair_match_random: the
# joint picks stop at the crop count of a random draw; random pairs share more dates, so the
# first joint runs encoded 6-12% more crops), seeds 0-2 of the control and both joint arms; and
# the joint recipe from scratch (10 epochs, no warm start)
for SD in 0 1 2; do
SA=(); [ "$SD" = 0 ] || SA=(--seed "$SD")
[ "$SD" = 0 ] || chain "warm_rand_tquery_seed${SD}"   3 "${WJ[@]}" "${SA[@]}"
chain "warm_joint_1-0.3_tquery_eq_seed${SD}" 3 "${WJ[@]}" --joint --eps_start 1 --eps_end 0.3 --pair_match_random "${SA[@]}"
chain "warm_joint_0.5_tquery_eq_seed${SD}"   3 "${WJ[@]}" --joint --eps_start 0.5 --eps_end 0.5 --pair_match_random "${SA[@]}"
done
chain scratch_joint_1-0.3_tquery_eq 4 --selector random "${P[@]}" --readout tquery --joint --eps_start 1 --eps_end 0.3 --pair_match_random
# 2026-10-01: from scratch with a random-pick phase first (--eps_hold): eps 1 for 10 epochs, then
# 1 -> 0.3 over 2 and held, 14 epochs (= the warm arms' 10 + 4)
chain scratch_hold_joint_1-0.3_tquery_eq 5 --selector random "${P[@]}" --readout tquery --epochs 14 --joint --eps_start 1 --eps_end 0.3 --eps_hold 0.714 --eps_anneal 0.143 --pair_match_random
# 2026-10-01: DENSE SELF-TEACHER, feature space (--self_teach n, tsplat): the encoder encodes n
# random dates densely at ps4 (pair head over them: task loss); the mixed-resolution splat read
# at those dates must reconstruct its features; --joint values = reconstruction gain. Warm start
# from random 1/6 pairs + tsplat, 4 epochs.
WS=(--selector random "${P[@]}" --readout tsplat --epochs 4
    --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.167_pair_tsplat_es50000_lr0.001_ep10")
chain warm_ts_rand                    3 "${WS[@]}"
chain warm_ts_teach4                  4 "${WS[@]}" --self_teach 4
chain warm_ts_teach4_joint_1-0.3_eq   4 "${WS[@]}" --self_teach 4 --joint --eps_start 1 --eps_end 0.3 --pair_match_random
chain warm_ts_teach4_joint_1-0.3_eq_seed1 4 "${WS[@]}" --self_teach 4 --joint --eps_start 1 --eps_end 0.3 --pair_match_random --seed 1
chain warm_ts_teach8_joint_1-0.3_eq   4 "${WS[@]}" --self_teach 8 --joint --eps_start 1 --eps_end 0.3 --pair_match_random
chain warm_ts_joint_1-0.3_eq          4 "${WS[@]}" --joint --eps_start 1 --eps_end 0.3 --pair_match_random
# the 8-date teacher at the default batch ran out of GPU memory on the 3-class task: batch 16
chain warm_ts_teach8_joint_1-0.3_eq_b16 4 "${WS[@]}" --self_teach 8 --joint --eps_start 1 --eps_end 0.3 --pair_match_random --batch_size 16
# 2026-10-01: --teach_event: the dense teacher's dates always include the last date before the
# event and the post-event date (2 of its 4; the rest random) -- with 4 purely random dates the
# teacher's pairs often missed the flood onset and its task loss hurt (flood-only 0.46 -> 0.31)
chain warm_ts_teach4_evt                 4 "${WS[@]}" --self_teach 4 --teach_event
chain warm_ts_teach4_evt_seed1           4 "${WS[@]}" --self_teach 4 --teach_event --seed 1
chain warm_ts_teach4_evt_joint_1-0.3_eq  4 "${WS[@]}" --self_teach 4 --teach_event --joint --eps_start 1 --eps_end 0.3 --pair_match_random
chain warm_ts_teach4_evt_joint_1-0.3_eq_seed1 4 "${WS[@]}" --self_teach 4 --teach_event --joint --eps_start 1 --eps_end 0.3 --pair_match_random --seed 1
chain warm_ts_rand_seed1                 3 "${WS[@]}" --seed 1
# 2026-10-01: --teach_detach on the event teacher: which gradient path gave the 3-class gain?
for DT in recon all; do
chain "warm_ts_teach4_evt_dt${DT}"                4 "${WS[@]}" --self_teach 4 --teach_event --teach_detach "$DT"
chain "warm_ts_teach4_evt_dt${DT}_joint_1-0.3_eq" 4 "${WS[@]}" --self_teach 4 --teach_event --teach_detach "$DT" --joint --eps_start 1 --eps_end 0.3 --pair_match_random
done
# 2026-10-02: the injection depth L with and without --head_attn oneway (crops read the cached
# lr keys / values in blocks [0, L), joint attention from L on); random 1/6 pairs, tquery
# readout, centre positions (L0 repeated). And the cross-attention pair selector reading the lr
# tokens at depth L (as xattnT_gain, which sits at L0).
SW=(--selector random --select_unit pair --readout tquery)
chain sweep_rand_L0_tquery 3 "${SW[@]}" --inject_layer 0
for LY in 2 4 6 8; do
chain "sweep_rand_L${LY}_tquery"          3 "${SW[@]}" --inject_layer "$LY"
chain "sweep_rand_L${LY}_tquery_haoneway" 4 "${SW[@]}" --inject_layer "$LY" --head_attn oneway
done
for LY in 4 8; do
chain "sweep_xattnT_gain_L${LY}_haoneway" 4 --selector xattn --select_unit pair --sel_temporal --score_norm --gain_loss 1 --inject_layer "$LY" --head_attn oneway
done
# 2026-10-02: the joint pair selector without its 12-block pre-look (--joint_feat lrL: each
# date's ps4 patch embedding + the month tokens after blocks [0, L) of the main pass; no pixel
# variance), L = 2, 3, 4, 6 with one-way head attention. Warm start (4 epochs) from the random
# one-way centre run at the same L (L3 is trained first: jf_base_rand_L3_tquery_haoneway); joint
# arms at an equal date-crop count (--pair_match_random).
chain jf_base_rand_L3_tquery_haoneway 4 "${SW[@]}" --inject_layer 3 --head_attn oneway
B3=$LAST
for LY in 2 3 4 6; do
JF=("${SW[@]}" --inject_layer "$LY" --head_attn oneway --pos center --epochs 4
    --init_ckpt "inject_fft_${TASK}_tiny_random_L${LY}_k0.167_pair_tquery_poscenter_haoneway_es50000_lr0.001_ep10")
DEP=$([ "$LY" = 3 ] && echo "afterany:${B3:-}" || echo "")
JP=(--joint --eps_start 1 --eps_end 0.3 --pair_match_random)
AFTER=$DEP chain "jf_warm_L${LY}_rand"          3 "${JF[@]}"
AFTER=$DEP chain "jf_warm_L${LY}_joint_prelook" 3 "${JF[@]}" "${JP[@]}"
AFTER=$DEP chain "jf_warm_L${LY}_joint_lrL"     3 "${JF[@]}" "${JP[@]}" --joint_feat lrL
done
# 2026-10-02 (eve): seeds 1 and 2 of the layer-L joint pair selector and its random control at
# L = 2, 3 (seed 0: 0.627 / 0.627 vs controls 0.616 / 0.608). Same seed-0 warm start.
for LY in 2 3; do
JF=("${SW[@]}" --inject_layer "$LY" --head_attn oneway --pos center --epochs 4
    --init_ckpt "inject_fft_${TASK}_tiny_random_L${LY}_k0.167_pair_tquery_poscenter_haoneway_es50000_lr0.001_ep10")
for SD in 1 2; do
chain "jf_warm_L${LY}_rand_seed${SD}"      3 "${JF[@]}" --seed "$SD"
chain "jf_warm_L${LY}_joint_lrL_seed${SD}" 3 "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --pair_match_random --joint_feat lrL --seed "$SD"
done
done
# 2026-10-05: the dropout-pass value target on the pair selector (--joint_target drop: a second
# no-grad pass of blocks [L, n) + readout + pair head without a random half of the date crops),
# L = 2, 3, seeds 0-2, vs jf_warm_L{2,3}_joint_lrL[_seedN] / _rand
for LY in 2 3; do
JF=("${SW[@]}" --inject_layer "$LY" --head_attn oneway --pos center --epochs 4
    --init_ckpt "inject_fft_${TASK}_tiny_random_L${LY}_k0.167_pair_tquery_poscenter_haoneway_es50000_lr0.001_ep10")
for SD in 0 1 2; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "sel_drop_L${LY}$SX" 3 "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --pair_match_random --joint_feat lrL --joint_target drop --seed "$SD"
done
done
