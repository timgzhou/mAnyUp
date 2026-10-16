#!/bin/bash
# Submit the injected-token fine-tuning arms (exp.upsamplers.finetune_inject) for one task, from
# the repo root, each as a chain of short resumable jobs (--max_hours; links run afterany, and a
# finished run exits at once):
#
#   [DRY=1] [ONLY=<regex>] [LINKS=n] bash scripts/slurm/inject.sh <pastis|biomassters|geoidflood>
#
#   none            S4ps16 bilinear readout, no crops
#   rand_L6         random crops at k = 1/6, injected before block 6
#   learned_L6      learned selector (joint), same k and layer
#   rand_L0/L9      where to inject
#   rand_L6_k33     a bigger budget (k = 1/3)
#   rand_L0_k33/k50, learned_L0   round 2: block 0 with bigger budgets, learned at block 0
# GEOID selects by location (both dates of a patch): its diff head needs both.
set -euo pipefail
TASK="${1:?usage: bash scripts/slurm/inject.sh <pastis|biomassters|geoidflood>}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f env_setup/env_olmo.sh ] || { echo "not the repo root: $PWD" >&2; exit 1; }

case "$TASK" in
  pastis)      EP=50 ;;
  biomassters) EP=10 ;;
  geoidflood)  EP=10 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
LINKS="${LINKS:-2}"
declare -A GDEP
UNIT=$([ "$TASK" = geoidflood ] && echo location || echo slot)
CPUS=9

chain() {   # chain <name> [finetune_inject args...]: LINKS jobs, each resuming the last
  local name=$1; shift
  if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then return; fi
  # crop positions (2026-10-01): new submissions are centre-aligned (--pos center, run names
  # gain _poscenter). Kept as they are: a chain that names --pos, a no-crop chain, and a warm
  # start (its checkpoint was trained on corner positions). CROP_POS=corner submits the old way.
  if [[ "${CROP_POS:-center}" != corner && " $* " != *" --pos "* && " $* " != *" --init_ckpt "*
        && " $* " != *" --selector none "* ]]; then
    set -- "$@" --pos "${CROP_POS:-center}"
  fi
  local prev="${AFTER:-}" id i     # AFTER=<job id>: start after another chain's last link
  for i in $(seq 1 "$LINKS"); do
    local cmd=(sbatch --parsable -J "inj_${TASK:0:3}_${name}_$i" --time=3:00:00
               --cpus-per-task="$CPUS" --exclude=kn117,kn104,kn101,kn001,kn161,kn112,kn140,kn088 ${MEM:+--mem=$MEM} ${prev:+--dependency=afterany:$prev}
               scripts/slurm/run.sh exp.upsamplers.finetune_inject --task "$TASK" --epochs "$EP"
               --num_workers $((CPUS - 1)) --max_hours 2.5 "$@")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; id="DRY$i"; else id=$("${cmd[@]}"); fi
    prev=$id
  done
  echo "$name: chain of $LINKS ending $prev"
  LAST=$prev                        # the chain's last job id, for a dependent chain's AFTER
}

chain none        --selector none
chain rand_L6     --selector random  --inject_layer 6 --select_unit "$UNIT"
chain learned_L6  --selector learned --inject_layer 6 --select_unit "$UNIT"
chain rand_L0     --selector random  --inject_layer 0 --select_unit "$UNIT"
chain rand_L9     --selector random  --inject_layer 9 --select_unit "$UNIT"
chain rand_L6_k33 --selector random  --inject_layer 6 --select_unit "$UNIT" --k_frac 0.3333
# round 2 (2026-09-27): block 0 won and more crops helped most
chain rand_L0_k33   --selector random  --inject_layer 0 --select_unit "$UNIT" --k_frac 0.3333
chain rand_L0_k50   --selector random  --inject_layer 0 --select_unit "$UNIT" --k_frac 0.5
chain learned_L0    --selector learned --inject_layer 0 --select_unit "$UNIT"
# round 3 (2026-09-27): why does the learned selector lose to random? (PASTIS / BioMassters)
#   learned_L0_pickrand  random slots gated by learned scores   -> cost of the gate
#   learned_L0_perstep   learned top-k within each month        -> spread over time
#   *_attn               temporal attention pool instead of the time mean
chain learned_L0_pickrand --selector learned --inject_layer 0 --pick random
chain learned_L0_perstep  --selector learned --inject_layer 0 --pick per_step
chain rand_L0_attn        --selector random  --inject_layer 0 --select_unit "$UNIT" --head attn
chain learned_L0_attn     --selector learned --inject_layer 0 --select_unit "$UNIT" --head attn
# round 4 (2026-09-27): the learned gate saturates at sigmoid = 1 (no gradient) -> keep it alive
chain learned_L0_snorm    --selector learned --inject_layer 0 --score_norm
chain learned_L0_l1e-2    --selector learned --inject_layer 0 --score_l1 0.01
chain learned_L0_l1e-1    --selector learned --inject_layer 0 --score_l1 0.1
# round 5 (2026-09-27): selector architecture (cross-attention) and objective (first-order gain)
chain xattn_L0_snorm           --selector xattn   --inject_layer 0 --score_norm
chain xattn_L0_snorm_gain      --selector xattn   --inject_layer 0 --score_norm --gain_loss 1
chain learned_L0_snorm_gain    --selector learned --inject_layer 0 --score_norm --gain_loss 1
chain xattnT_L0_snorm_gain     --selector xattn   --inject_layer 0 --score_norm --gain_loss 1 --sel_temporal
# round 6 (2026-09-27): readout -- Gaussian splat of the irregular tokens (random picks, block 0)
chain none_splat_t        --selector none                                        --readout splat_t
chain rand_L0_splat       --selector random --inject_layer 0 --select_unit "$UNIT" --readout splat
chain rand_L0_splat_t     --selector random --inject_layer 0 --select_unit "$UNIT" --readout splat_t
chain rand_L0_splat_tg    --selector random --inject_layer 0 --select_unit "$UNIT" --readout splat_tg
chain rand_L0_k33_splat    --selector random --inject_layer 0 --select_unit "$UNIT" --k_frac 0.3333 --readout splat
chain rand_L0_k33_splat_t  --selector random --inject_layer 0 --select_unit "$UNIT" --k_frac 0.3333 --readout splat_t
chain rand_L0_k33_splat_tg --selector random --inject_layer 0 --select_unit "$UNIT" --k_frac 0.3333 --readout splat_tg
# round 7 (2026-09-28): combine the winners -- attention-pool head + guided splat readout;
# random picks at 1/6, 1/3, 1/2 and the best learned selectors at 1/6
C=(--inject_layer 0 --select_unit "$UNIT" --head attn --readout splat_tg)
chain rand_L0_attn_splat_tg      --selector random  "${C[@]}"
chain rand_L0_k33_attn_splat_tg  --selector random  "${C[@]}" --k_frac 0.3333
chain rand_L0_k50_attn_splat_tg  --selector random  "${C[@]}" --k_frac 0.5
chain learned_L0_snorm_gain_attn_splat_tg --selector learned "${C[@]}" --score_norm --gain_loss 1
chain learned_L0_l1e-2_attn_splat_tg      --selector learned "${C[@]}" --score_l1 0.01
chain xattn_L0_snorm_gain_attn_splat_tg   --selector xattn   "${C[@]}" --score_norm --gain_loss 1
# round 8 (2026-09-28): sparse per-cell attention readout (CellAttnReadout) -- no month grid;
# each ps4 cell pools nearby season tokens + crops (the decoder a date level will need)
X=(--inject_layer 0 --select_unit "$UNIT" --readout cellattn)
chain none_cellattn          --selector none                                   --readout cellattn
chain rand_L0_cellattn       --selector random  "${X[@]}"
chain rand_L0_k33_cellattn   --selector random  "${X[@]}" --k_frac 0.3333
chain rand_L0_k50_cellattn   --selector random  "${X[@]}" --k_frac 0.5
chain learned_L0_l1e-2_cellattn --selector learned "${X[@]}" --score_l1 0.01
chain xattn_L0_snorm_cellattn   --selector xattn   "${X[@]}" --score_norm
# round 9 (2026-09-28, PASTIS only -- BioMassters has no dates): season ps16 -> month ps4 ->
# date ps4, sparse per-cell readout, fractional crop slots; random date picks (LINKS=4)
Y=(--inject_layer 0 --select_unit "$UNIT" --readout cellattn --frac_slot)
if [ "$TASK" = pastis ]; then
chain rand_L0_frac_cellattn            --selector random  "${Y[@]}"
chain rand_L0_frac_nested2             --selector random  "${Y[@]}" --date_level nested --date_per_month 2
chain rand_L0_frac_nested4             --selector random  "${Y[@]}" --date_level nested --date_per_month 4
chain rand_L0_frac_nested2_nomonth     --selector random  "${Y[@]}" --date_level nested --date_per_month 2 --no_month_crops
chain rand_L0_frac_free2               --selector random  "${Y[@]}" --date_level free   --date_per_month 2
chain rand_L0_k33_frac_nested2         --selector random  "${Y[@]}" --date_level nested --date_per_month 2 --k_frac 0.3333
chain learned_L0_l1e-2_frac_nested2    --selector learned "${Y[@]}" --date_level nested --date_per_month 2 --score_l1 0.01
fi
# round 10 (2026-09-28): (a) where-to-look ceiling -- whole cells (all months) picked by the
# model's own crop-free loss vs the TRUE label (oracle) / its entropy / at random; (b) learned
# date picks re-picking (date, cell) pairs over all dates and cells (free), PASTIS; (c) the
# learned-months-with-dates collapse with date crops NOT gated by the month score
Z=(--inject_layer 0 --readout cellattn --select_unit location)
chain loc_rand_cellattn       --selector random  "${Z[@]}"
chain loc_oracle_cellattn     --selector oracle  "${Z[@]}"
chain loc_rand_k33_cellattn   --selector random  "${Z[@]}" --k_frac 0.3333
chain loc_oracle_k33_cellattn --selector oracle  "${Z[@]}" --k_frac 0.3333
if [ "$TASK" = pastis ]; then
chain loc_entropy_cellattn    --selector entropy "${Z[@]}"
D=(--inject_layer 0 --select_unit "$UNIT" --readout cellattn --frac_slot --date_level free --date_per_month 2)
chain rand_L0_frac_free2_dxattn      --selector random  "${D[@]}" --date_selector xattn
chain rand_L0_k33_frac_free2_dxattn  --selector random  "${D[@]}" --date_selector xattn --k_frac 0.3333
chain rand_L0_k33_frac_free2         --selector random  "${D[@]}" --k_frac 0.3333
chain learned_L0_l1e-2_frac_nested2_dnogate --selector learned --inject_layer 0 --select_unit "$UNIT" \
      --readout cellattn --frac_slot --date_level nested --date_per_month 2 --score_l1 0.01 --date_gate none
fi
# round 11 (2026-09-28): complete the readout-vs-head ablation grid, and a second seed of the
# 1/6 ladder (--seed 1 -> run name suffix _seed1) to size the seed noise
#   none_attn / rand_L0_k{33,50}_attn   attention head alone (bilinear readout): no crops, 1/3, 1/2
chain none_attn          --selector none                                             --head attn
chain rand_L0_k33_attn   --selector random --inject_layer 0 --select_unit "$UNIT" --head attn --k_frac 0.3333
chain rand_L0_k50_attn   --selector random --inject_layer 0 --select_unit "$UNIT" --head attn --k_frac 0.5
S=(--seed 1)
chain none_seed1                  --selector none                                               "${S[@]}"
chain rand_L6_seed1               --selector random --inject_layer 6 --select_unit "$UNIT"       "${S[@]}"
chain rand_L0_seed1               --selector random --inject_layer 0 --select_unit "$UNIT"       "${S[@]}"
chain rand_L0_attn_seed1          --selector random --inject_layer 0 --select_unit "$UNIT" --head attn "${S[@]}"
chain rand_L0_splat_seed1         --selector random --inject_layer 0 --select_unit "$UNIT" --readout splat    "${S[@]}"
chain rand_L0_splat_t_seed1       --selector random --inject_layer 0 --select_unit "$UNIT" --readout splat_t  "${S[@]}"
chain rand_L0_splat_tg_seed1      --selector random --inject_layer 0 --select_unit "$UNIT" --readout splat_tg "${S[@]}"
chain rand_L0_attn_splat_tg_seed1 --selector random "${C[@]}"                                   "${S[@]}"
chain rand_L0_cellattn_seed1      --selector random "${X[@]}"                                   "${S[@]}"
if [ "$TASK" = pastis ]; then
chain rand_L0_frac_cellattn_seed1 --selector random "${Y[@]}"                                   "${S[@]}"
fi
# round 12 (2026-09-28): native-resolution pa2px head (--readout native) -- the plain readout
# bilinearly upsamples ps16 features before a 4x4 px head, blurring the sub-patch layout a ps16
# token encodes; native maps ps16 tokens to their 16x16 px block and crops (same head) to 16x16
# area-downsampled to 4x4. Random picks at block 0, k = 0, 1/6, 1/3, 1/2, and the 12-month ps16
# series through this same pipeline (the fixed ps16ts64 baseline, pipeline-matched)
N=(--inject_layer 0 --select_unit "$UNIT" --readout native)
chain none_native          --selector none                    --readout native
chain months_ps16_native   --selector none --lr_steps all     --readout native
chain rand_L0_native       --selector random "${N[@]}"
chain rand_L0_k33_native   --selector random "${N[@]}" --k_frac 0.3333
chain rand_L0_k50_native   --selector random "${N[@]}" --k_frac 0.5
# round 13 (2026-09-28): overlap-only readouts (no spatial-neighbour context; user: neighbours are
# extra context any method could use) and native + attention pool in logit space; random picks,
# block 0, k = 0, 1/6, 1/3, 1/2
#   native_attn        native ps16 head + temporal attention pool over months, in logit space
#   cellattn_overlap   per-cell attention pool over the covering season tokens + crops at the cell
#   splat_t_overlap    time-only Gaussian splat over the same overlapping tokens
for RO in native_attn cellattn_overlap splat_t_overlap; do
chain "none_${RO}"         --selector none                                       --readout "$RO"
chain "rand_L0_${RO}"      --selector random --inject_layer 0 --select_unit "$UNIT" --readout "$RO"
chain "rand_L0_k33_${RO}"  --selector random --inject_layer 0 --select_unit "$UNIT" --readout "$RO" --k_frac 0.3333
chain "rand_L0_k50_${RO}"  --selector random --inject_layer 0 --select_unit "$UNIT" --readout "$RO" --k_frac 0.5
done
# round 14 (2026-09-29): (a) the date level with an overlap-only readout: native_set = per-cell
# attention pool in logit space over the covering season tokens + every crop at the cell;
# months only at k = 0 .. 1 (k = 1: every month slot, the equal-budget reference for dates) and
# ps16sea_ts->ps4mon_pd->ps4date_pd with 2 free dates per month crop, random or learned dates;
# (b) selectors redone with native_attn (crops can dominate their cell now): learned, xattn,
# and whole-cell random / oracle / entropy
NS_=(--inject_layer 0 --select_unit "$UNIT" --readout native_set)
chain none_native_set           --selector none                   --readout native_set
chain rand_L0_native_set        --selector random "${NS_[@]}"
chain rand_L0_k33_native_set    --selector random "${NS_[@]}" --k_frac 0.3333
chain rand_L0_k50_native_set    --selector random "${NS_[@]}" --k_frac 0.5
if [ "$TASK" = pastis ]; then
chain rand_L0_k100_native_set   --selector random "${NS_[@]}" --k_frac 1.0
DN=("${NS_[@]}" --frac_slot --date_level free --date_per_month 2)
chain rand_L0_frac_free2_native_set            --selector random "${DN[@]}"
chain rand_L0_k33_frac_free2_native_set        --selector random "${DN[@]}" --k_frac 0.3333
chain rand_L0_frac_free2_dxattn_native_set     --selector random "${DN[@]}" --date_selector xattn
chain rand_L0_k33_frac_free2_dxattn_native_set --selector random "${DN[@]}" --date_selector xattn --k_frac 0.3333
fi
NA=(--inject_layer 0 --readout native_attn)
chain learned_L0_l1e-2_native_attn  --selector learned "${NA[@]}" --select_unit "$UNIT" --score_l1 0.01
chain xattn_L0_snorm_native_attn    --selector xattn   "${NA[@]}" --select_unit "$UNIT" --score_norm
chain loc_rand_native_attn          --selector random  "${NA[@]}" --select_unit location
chain loc_oracle_native_attn        --selector oracle  "${NA[@]}" --select_unit location
if [ "$TASK" = pastis ]; then
chain loc_entropy_native_attn       --selector entropy "${NA[@]}" --select_unit location
fi
# round 15 (2026-09-29): random MIXED-RESOLUTION crops at an equal token budget (--res_mix
# ps16,ps8,ps4 shares; a share a resolution cannot fill goes to ps4, per sample); native_set
# readout lists each crop at every ps4 cell of its footprint. ps4-only = rand_L0*_native_set.
#   res1-0-0  ps16 months first (at most all of them, ~192), the rest ps4
#   res0-1-0  ps8 only (rest ps4 if a sample runs out)
#   res0-1-1  half ps8, half ps4;   res1-1-1  a third each
R=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set)
for MIX in 1,0,0 0,1,0 0,1,1 1,1,1; do
chain "rand_L0_native_set_res${MIX//,/-}"      "${R[@]}" --res_mix "$MIX"
chain "rand_L0_k33_native_set_res${MIX//,/-}"  "${R[@]}" --res_mix "$MIX" --k_frac 0.3333
done
# round 16 (2026-09-30): JOINT selector + backbone (--joint): warm start from the random 1/6
# native_set model; a value predictor (cross-month attention over pre-look features) picks
# (1 - eps) of the crops, eps random; trained each step on its picks' in-context leave-one-out
# values; the backbone trains on the mixed picks. Test scored with eps_end picks, and also with
# selector-only (eps 0) and random picks (eps 1). Control: the same warm start, random picks.
WEP=$([ "$TASK" = pastis ] && echo 20 || echo 5)
W=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set --epochs "$WEP"
   --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.167_native_set_lr0.001_ep${EP}")
chain warm_rand                 "${W[@]}"
chain warm_joint_1-0.3          "${W[@]}" --joint --eps_start 1 --eps_end 0.3
chain warm_joint_0.5            "${W[@]}" --joint --eps_start 0.5 --eps_end 0.5
chain warm_joint_1-0            "${W[@]}" --joint --eps_start 1 --eps_end 0
# round 17 (2026-09-30): manyup.inject.TimeSetReadout on the time-mean tasks, native logits:
#   native_tsplat  overlap-only time-only Gaussian splat;  native_tquery  time-queried attention
for RO in native_tsplat native_tquery; do
chain "none_${RO}"         --selector none                                          --readout "$RO"
chain "rand_L0_${RO}"      --selector random --inject_layer 0 --select_unit "$UNIT" --readout "$RO"
chain "rand_L0_k33_${RO}"  --selector random --inject_layer 0 --select_unit "$UNIT" --readout "$RO" --k_frac 0.3333
chain "rand_L0_k50_${RO}"  --selector random --inject_layer 0 --select_unit "$UNIT" --readout "$RO" --k_frac 0.5
done
# round 18 (2026-09-30): the BioMassters joint win -- seeds 1, 2 of the control and the annealed
# joint arm, and the joint selector at 1/3 (warm start from random 1/3 native_set)
if [ "$TASK" = biomassters ]; then
for SD in 1 2; do
chain "warm_rand_seed${SD}"          "${W[@]}" --seed "$SD"
chain "warm_joint_1-0.3_seed${SD}"   "${W[@]}" --joint --eps_start 1 --eps_end 0.3 --seed "$SD"
done
W33=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set --epochs "$WEP"
     --k_frac 0.3333 --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.333_native_set_lr0.001_ep${EP}")
chain warm_k33_rand         "${W33[@]}"
chain warm_k33_joint_1-0.3  "${W33[@]}" --joint --eps_start 1 --eps_end 0.3
fi
# round 19 (2026-09-30, BioMassters): the joint selector ALSO chooses the resolution: ps4 and ps8
# (month, cell) candidates, one token each (--joint_res), warm start from the random ps8/ps4
# mix (res0-1-1); control = the same warm start, random ps8/ps4 half-half
if [ "$TASK" = biomassters ]; then
WR=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set --epochs "$WEP"
    --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.167_native_set_res0-1-1_lr0.001_ep${EP}")
chain warm_res011_rand          "${WR[@]}" --res_mix 0,1,1
chain warm_jres_1-0.3           "${WR[@]}" --joint --joint_res --eps_start 1 --eps_end 0.3
chain warm_jres_1-0             "${WR[@]}" --joint --joint_res --eps_start 1 --eps_end 0
fi
# round 20 (2026-09-30): (a) BioMassters joint at 1/2 (does it reach ps4ts64_ts?), seeds 0, 1,
# and seed 1 at 1/3; (b) the joint selector choosing ps4 / ps8 at 1/3; (c) the joint recipe FROM
# SCRATCH (no warm start: frozen-backbone phase with random picks, then eps 1 -> 0.3 over the
# first half) at 1/6 -- at the task's epochs, and on BioMassters also at 15 epochs (= the warm
# arms' 10 + 5) with its random-pick control
if [ "$TASK" = biomassters ]; then
W50=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set --epochs "$WEP"
     --k_frac 0.5 --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.5_native_set_lr0.001_ep${EP}")
chain warm_k50_rand               "${W50[@]}"
chain warm_k50_joint_1-0.3        "${W50[@]}" --joint --eps_start 1 --eps_end 0.3
chain warm_k50_rand_seed1         "${W50[@]}" --seed 1
chain warm_k50_joint_1-0.3_seed1  "${W50[@]}" --joint --eps_start 1 --eps_end 0.3 --seed 1
chain warm_k33_rand_seed1         "${W33[@]}" --seed 1
chain warm_k33_joint_1-0.3_seed1  "${W33[@]}" --joint --eps_start 1 --eps_end 0.3 --seed 1
WR33=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set --epochs "$WEP"
      --k_frac 0.3333 --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.333_native_set_res0-1-1_lr0.001_ep${EP}")
chain warm_k33_res011_rand        "${WR33[@]}" --res_mix 0,1,1
chain warm_k33_jres_1-0.3         "${WR33[@]}" --joint --joint_res --eps_start 1 --eps_end 0.3
SC=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set)
chain scratch_rand_ep15           "${SC[@]}" --epochs 15
chain scratch_joint_1-0.3_ep15    "${SC[@]}" --epochs 15 --joint --eps_start 1 --eps_end 0.3
fi
if [ "$TASK" != geoidflood ]; then
chain scratch_joint_1-0.3 --selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set --joint --eps_start 1 --eps_end 0.3
fi
# round 21 (2026-10-01): from-scratch joint with the plain anneal FAILED (the selector takes over
# while the backbone cannot use crops yet). One run that mimics the warm recipe: eps = 1 for the
# first EP epochs (--eps_hold), then 1 -> 0.3 and held, EP + WEP epochs in total
HEP=$((EP + WEP))
HOLD=$(python3 -c "print(round($EP / $HEP, 3), round($WEP / $HEP / 2, 3))")
if [ "$TASK" != geoidflood ]; then
chain scratch_hold_joint_1-0.3 --selector random --inject_layer 0 --select_unit "$UNIT" --readout native_set --epochs "$HEP" --joint --eps_start 1 --eps_end 0.3 --eps_hold "${HOLD% *}" --eps_anneal "${HOLD#* }"
fi
# round 22 (2026-10-01, BioMassters: its months ARE the raw steps): DENSE SELF-TEACHER
# (--self_teach 4): the same encoder encodes 4 random raw steps densely (ps4 over the tile, task
# loss); the mixed-resolution time-queried readout must reconstruct its per-step logits at those
# steps; with --joint the selector's values are each crop's leave-one-out reconstruction gain.
# Warm start from random + native_tquery (5 epochs), and from scratch.
if [ "$TASK" = biomassters ]; then
for K in 0.167 0.333; do
KN=$([ "$K" = 0.167 ] && echo "" || echo "_k33")
KF=$([ "$K" = 0.167 ] && echo 0.16666666666666666 || echo 0.3333)
WT=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_tquery --epochs "$WEP"
    --k_frac "$KF" --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k${K}_native_tquery_lr0.001_ep${EP}")
chain "warm_tq${KN}_rand"                   "${WT[@]}"
chain "warm_tq${KN}_teach4"                 "${WT[@]}" --self_teach 4
chain "warm_tq${KN}_teach4_joint_1-0.3"     "${WT[@]}" --self_teach 4 --joint --eps_start 1 --eps_end 0.3
done
WT1=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_tquery --epochs "$WEP"
     --init_ckpt "inject_fft_${TASK}_tiny_random_L0_k0.167_native_tquery_lr0.001_ep${EP}")
chain warm_tq_teach4_joint_1-0.3_seed1      "${WT1[@]}" --self_teach 4 --joint --eps_start 1 --eps_end 0.3 --seed 1
ST=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout native_tquery --self_teach 4)
chain scratch_tq_teach4                     "${ST[@]}"
chain scratch_tq_teach4_hold_joint_1-0.3    "${ST[@]}" --epochs 15 --joint --eps_start 1 --eps_end 0.3 --eps_hold 0.667 --eps_anneal 0.167
fi
# round 23 (2026-10-01): DENSE SELF-TEACHER, FEATURE SPACE (--self_teach 4, --readout tsplat):
# the same encoder encodes 4 random RAW timesteps densely (ps4 over the tile; task loss) and the
# mixed-resolution tokens' time-only Gaussian splat, read at those times, must reconstruct its
# dense features; --joint: the selector's values = each crop's leave-one-out reconstruction
# gain. From scratch (no feature-space tsplat checkpoints on these tasks), EP + WEP epochs.
if [ "$TASK" != geoidflood ]; then
TS=(--selector random --inject_layer 0 --select_unit "$UNIT" --readout tsplat --epochs "$HEP")
JH=(--joint --eps_start 1 --eps_end 0.3 --eps_hold "${HOLD% *}" --eps_anneal "${HOLD#* }")
chain scratch_ts_rand                         "${TS[@]}"
chain scratch_ts_teach4                       "${TS[@]}" --self_teach 4
chain scratch_ts_teach4_hold_joint_1-0.3      "${TS[@]}" --self_teach 4 "${JH[@]}"
chain scratch_ts_teach4_hold_joint_1-0.3_seed1 "${TS[@]}" --self_teach 4 "${JH[@]}" --seed 1
chain scratch_ts_k33_rand                     "${TS[@]}" --k_frac 0.3333
chain scratch_ts_k33_teach4_hold_joint_1-0.3  "${TS[@]}" --k_frac 0.3333 --self_teach 4 "${JH[@]}"
fi
# round 24 (2026-10-01): how a token's place is encoded (--pos, manyup.inject), on the random
# ps16/ps8/ps4 third-each mix at 1/6; the corner baseline = rand_L0_native_set_res1-1-1
#   poscenter  crops at their centre in the low-res frame (no scale signal)
#   posbox     corner positions + zero-init map of the bottom-right pixel on every token
if [ "$TASK" != geoidflood ]; then
for P in center box; do
chain "rand_L0_native_set_res1-1-1_pos${P}" "${R[@]}" --res_mix 1,1,1 --pos "$P"
done
fi
# round 25 (2026-10-01, PASTIS: the date level): OlmoEarth encodes slot + month, no day, so
# dates of one month are told apart only by --frac_slot's fractional slot (weak: ~0.1 rad for
# 10 days). --doy: zero-init map of the day of year (24 harmonics) on every token. Baseline =
# rand_L0_frac_free2_native_set (round 14).
#   free2_doy        day of year instead of the fractional slot
#   frac_free2_doy   both
if [ "$TASK" = pastis ]; then
DD=("${NS_[@]}" --date_level free --date_per_month 2 --doy)
chain rand_L0_free2_native_set_doy       --selector random "${DD[@]}"
chain rand_L0_frac_free2_native_set_doy  --selector random "${DD[@]}" --frac_slot
fi
# round 26 (2026-10-01): who attends to whom (--attn, manyup.inject), random ps4 month crops at
# 1/6, block 0; the joint baseline = rand_L0_native_set
#   oneway   lr alone (= the crop-free model); each crop reads its own tokens + its sample's
#            lr keys / values at every depth; crops never read each other
#   overlap  oneway, only the lr cell under the crop (every season and band set)
if [ "$TASK" != geoidflood ]; then
for A in oneway overlap; do
chain "rand_L0_native_set_${A}" "${NS_[@]}" --selector random --attn "$A"
done
fi
# round 27 (2026-10-01): dense self-teacher with a LOGIT-SPACE task readout: the task is read by
# native_set from the mixed-resolution tokens; a separate time-only Gaussian splat reconstructs
# the teacher's dense features (--self_teach 4 --readout native_set). --joint values: the
# leave-one-out reconstruction gain (default) or reconstruction + task-loss gain (vboth).
# Warm start as round 16 (W); controls there: warm_rand, warm_joint_1-0.3 (label values).
if [ "$TASK" != geoidflood ]; then
J=(--joint --eps_start 1 --eps_end 0.3)
chain warm_ns_teach4                      "${W[@]}" --self_teach 4
chain warm_ns_teach4_joint_1-0.3          "${W[@]}" --self_teach 4 "${J[@]}"
chain warm_ns_teach4_joint_1-0.3_seed1    "${W[@]}" --self_teach 4 "${J[@]}" --seed 1
chain warm_ns_teach4_vboth_joint_1-0.3    "${W[@]}" --self_teach 4 "${J[@]}" --teach_value both
fi
if [ "$TASK" = biomassters ]; then
chain warm_ns_k33_teach4_joint_1-0.3      "${W33[@]}" --self_teach 4 "${J[@]}"
fi
# round 27 (2026-10-01): what a crop reads BEFORE it joins the lr sequence (--head_attn, blocks
# [0, L); L = 0 has no such blocks), random ps4 month crops at 1/6, native_set, L = 6 and 9.
# Reference: rand_L0_native_set (joint from block 0).
#   L{6,9}            alone: a crop reads its own tokens (as every run so far)
#   L{6,9}_haoneway   + its sample's lr keys / values, cached from the lr pass
#   L{6,9}_haoverlap  + only the lr cell under it
if [ "$TASK" != geoidflood ]; then
for LY in 6 9; do
H=(--selector random --inject_layer "$LY" --select_unit "$UNIT" --readout native_set)
chain "rand_L${LY}_native_set"            "${H[@]}"
chain "rand_L${LY}_native_set_haoneway"   "${H[@]}" --head_attn oneway
chain "rand_L${LY}_native_set_haoverlap"  "${H[@]}" --head_attn overlap
done
fi
# round 28 (2026-10-01): which self-teacher gradients should reach the encoder? (--teach_detach)
#   dtrecon  the reconstruction loss trains only the splat (+ the selector's values); the
#            teacher's task loss still trains the encoder
#   dtall    neither: the teacher is a no-grad pass with no task loss (the encoder is trained by
#            the student's task loss alone; reconstruction only drives the selector)
if [ "$TASK" != geoidflood ]; then
for DT in recon all; do
chain "warm_ns_teach4_dt${DT}"                   "${W[@]}" --self_teach 4 --teach_detach "$DT"
chain "warm_ns_teach4_dt${DT}_joint_1-0.3"       "${W[@]}" --self_teach 4 --teach_detach "$DT" "${J[@]}"
chain "warm_ns_teach4_dt${DT}_joint_1-0.3_seed1" "${W[@]}" --self_teach 4 --teach_detach "$DT" "${J[@]}" --seed 1
done
fi
# round 29 (2026-10-02): the injection depth L with and without --head_attn oneway (crops read
# the cached lr keys / values in blocks [0, L), joint attention from L on), random ps4 month crops
# at 1/6, native_set, centre positions (chain's default now, so L0 / L6 are repeated here).
# Round 27 (corner): PASTIS L0 0.559, L6 alone 0.458 -> oneway 0.544, L9 0.455 -> 0.508;
# BioMassters L0 35.08, L6 35.82 -> 35.41.
#   sweep_rand_L{0,2,4,6,8}[_haoneway]      random picks
#   sweep_xattn_L{4,8}_haoneway             a selector on the lr tokens at depth L (cross-
#                                           attention, normalized scores): deeper features for free
#   sweep_warm_L6_haoneway_{rand,joint}     the joint value-predictor selector on top of the
#                                           finished L6 oneway run (corner), and its random control
if [ "$TASK" != geoidflood ]; then
LS=(--select_unit "$UNIT" --readout native_set)
chain sweep_rand_L0_native_set --selector random "${LS[@]}" --inject_layer 0
for LY in 2 3 4 6 8; do
chain "sweep_rand_L${LY}_native_set"          --selector random "${LS[@]}" --inject_layer "$LY"
chain "sweep_rand_L${LY}_native_set_haoneway" --selector random "${LS[@]}" --inject_layer "$LY" --head_attn oneway
done
for LY in 4 8; do
chain "sweep_xattn_L${LY}_native_set_haoneway" --selector xattn --score_norm "${LS[@]}" --inject_layer "$LY" --head_attn oneway
done
WH=(--selector random "${LS[@]}" --inject_layer 6 --head_attn oneway --epochs "$WEP"
    --init_ckpt "inject_fft_${TASK}_tiny_random_L6_k0.167_native_set_haoneway_lr0.001_ep${EP}")
chain sweep_warm_L6_haoneway_rand        "${WH[@]}"
chain sweep_warm_L6_haoneway_joint_1-0.3 "${WH[@]}" --joint --eps_start 1 --eps_end 0.3
fi
# round 30 (2026-10-02): the joint selector WITHOUT its 12-block pre-look (--joint_feat lrL): the
# value predictor reads each candidate's ps4 patch embedding + the season tokens after blocks
# [0, L) of the pass the model does anyway -- no extra pass, no season logits, no pixel variance.
# Warm start from round 29's random L{4,6} one-way centre runs; per L:
#   jf_warm_L{4,6}_rand            the control: same warm start, random picks
#   jf_warm_L{4,6}_joint_prelook   the joint selector as before (12-block pre-look features)
#   jf_warm_L{4,6}_joint_lrL       the same training, layer-L features only
# 2026-10-02 (later): also L 2 and 3 -- the selector reading an EARLY layer, so nearly the whole
# network still runs jointly on the crops it picked (L3's base run: sweep_rand_L3_..._haoneway).
if [ "$TASK" != geoidflood ]; then
for LY in 2 3 4 6; do
JF=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer "$LY"
    --head_attn oneway --pos center --epochs "$WEP"
    --init_ckpt "inject_fft_${TASK}_tiny_random_L${LY}_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
chain "jf_warm_L${LY}_rand"          "${JF[@]}"
chain "jf_warm_L${LY}_joint_prelook" "${JF[@]}" --joint --eps_start 1 --eps_end 0.3
chain "jf_warm_L${LY}_joint_lrL"     "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
done
fi
# round 31 (2026-10-02): train the layer-3 selector THROUGH the task loss with DPS's perturbed
# top-k (--joint_grad dps: no value regression; both: value regression + DPS), vs value-only
# (jf_warm_L3_joint_lrL = seed 0); same warm start, 2 seeds
if [ "$TASK" != geoidflood ]; then
JF=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs "$WEP" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --init_ckpt "inject_fft_${TASK}_tiny_random_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for JG in dps both; do
chain "dps_warm_L3_lrL_${JG}"       "${JF[@]}" --joint_grad "$JG"
chain "dps_warm_L3_lrL_${JG}_seed1" "${JF[@]}" --joint_grad "$JG" --seed 1
done
fi
# round 32 (2026-10-02): seeds 1 and 2 of the layer-L joint selector and its random control at
# L = 2, 3 (BioMassters only: PASTIS showed no selector gain). The warm start is the same seed-0
# random one-way run; the seed changes the warm phase (picks, selector init, data order).
if [ "$TASK" = biomassters ]; then
for LY in 2 3; do
JF=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer "$LY"
    --head_attn oneway --pos center --epochs "$WEP"
    --init_ckpt "inject_fft_${TASK}_tiny_random_L${LY}_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 1 2; do
chain "jf_warm_L${LY}_rand_seed${SD}"      "${JF[@]}" --seed "$SD"
chain "jf_warm_L${LY}_joint_lrL_seed${SD}" "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL --seed "$SD"
done
done
fi
# round 33 (2026-10-04): one change at a time to the layer-3 value predictor (jf_warm_L3_joint_lrL,
# seeds 0-2), same warm start, 1/6 budget, seeds 0 and 1:
#   sel_gate     --joint_target gate   target = -dLoss/dgate of a crop's tokens entering block L
#                                      (its value through every later layer, not the readout only)
#   sel_drop     --joint_target drop   target = loss rise in a 2nd no-grad pass of blocks [L, n) without
#                                      a random half of the crops (true removal; gate's smoke: r ~ 0
#                                      with the leave-one-out value, selector loss stuck at 1)
#   sel_enc      --joint_in enc        (e - season mean) + time encoding, season tokens; no one-hots
#   sel_xattn    --joint_arch xattn    the candidate queries the season tokens (CrossAttnSelector)
#   sel_ownemb   --joint_embed own     the selector trains its own copy of the patch embedding
#   sel_jit      --jitter              crops 0-3 px off the ps4 grid, selector unchanged
#   sel_jit_rand --jitter              ... with random picks (vs jf_warm_L3_rand)
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
JF=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs "$WEP"
    --init_ckpt "inject_fft_${TASK}_tiny_random_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
JL=("${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "sel_gate$SX"     "${JL[@]}" --joint_target gate --seed "$SD"
chain "sel_drop$SX"     "${JL[@]}" --joint_target drop --seed "$SD"
chain "sel_enc$SX"      "${JL[@]}" --joint_in enc      --seed "$SD"
chain "sel_xattn$SX"    "${JL[@]}" --joint_arch xattn  --seed "$SD"
chain "sel_ownemb$SX"   "${JL[@]}" --joint_embed own   --seed "$SD"
chain "sel_jit$SX"      "${JL[@]}" --jitter            --seed "$SD"
chain "sel_jit_rand$SX" "${JF[@]}" --jitter            --seed "$SD"
done
fi
# round 34 (2026-10-04): the value-predictor selector at the DATE level, one modality per crop,
# budget = the TOKENS 1/6 of the ps4 month crops cost (--date_joint; PASTIS has raw dates).
# Warm start from the L3 one-way run, --frac_slot (dates of one season get their own time slot);
# *_rand = the same crops picked at random (eps 1 -> 1). Seeds 0, 1.
#   dj_only_{sel,rand}   date crops anywhere, no month crops
#   dj_hier_{sel,rand}   half the tokens on month crops, half on dates inside the picked months
if [ "$TASK" = pastis ]; then
JF=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs "$WEP" --frac_slot --joint --joint_feat lrL
    --init_ckpt "inject_fft_${TASK}_tiny_random_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for DJ in only hier; do
chain "dj_${DJ}_sel$SX"  "${JF[@]}" --date_joint "$DJ" --eps_start 1 --eps_end 0.3 --seed "$SD"
chain "dj_${DJ}_rand$SX" "${JF[@]}" --date_joint "$DJ" --eps_start 1 --eps_end 1   --seed "$SD"
done
done
fi
# round 35 (2026-10-04): patch size vs token count. --crop_px 16: the same 1/6 random picks
# (month, ps4 cell), but each extra token is ONE ps16 token on the 16 px window centred on the
# cell -- single-month tokens at the low-res patch size, off the ps16 grid. Against
# sweep_rand_L0_native_set / sweep_rand_L3_native_set_haoneway (ps4 crops, same recipe), then the
# layer-3 selector and its control on top (vs jf_warm_L3_*).
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
LS=(--select_unit "$UNIT" --readout native_set --crop_px 16)
chain cpx16_rand_L0 --selector random "${LS[@]}" --inject_layer 0
chain cpx16_rand_L3_haoneway --selector random "${LS[@]}" --inject_layer 3 --head_attn oneway
JF=(--selector random "${LS[@]}" --inject_layer 3 --head_attn oneway --pos center --epochs "$WEP"
    --init_ckpt "inject_fft_${TASK}_tiny_random_L3_k0.167_native_set_cpx16_poscenter_haoneway_lr0.001_ep${EP}")
B3=${LAST:-}
AFTER=$B3 chain cpx16_warm_L3_rand      "${JF[@]}"
AFTER=$B3 chain cpx16_warm_L3_joint_lrL "${JF[@]}" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
fi

# round 34 (2026-10-04): month crops from the month's RAW ACQUISITIONS instead of its composite
# (--acq_pool): each picked (month, cell)'s acquisitions run blocks [0, 3) as one patch time
# series (temporal attention inside the cell, none across cells), are pooled per token kind
# (mean / learned softmax weight) and join the joint sequence at block 3. Random picks, from
# scratch, --frac_slot (the dates of a month need distinct time encodings; the control too).
#   acq_L3_{all,k33,k17}_{comp,mean,attn}   comp = the composite control
if [ "$TASK" = pastis ]; then
AQ=(--selector random --select_unit slot --readout native_set --inject_layer 3 --frac_slot)
for AP in none mean attn; do
NM=$([ "$AP" = none ] && echo comp || echo "$AP")
chain "acq_L3_all_${NM}" "${AQ[@]}" --k_frac 1      --acq_pool "$AP" --batch_size 8
chain "acq_L3_k33_${NM}" "${AQ[@]}" --k_frac 0.3333 --acq_pool "$AP"
chain "acq_L3_k17_${NM}" "${AQ[@]}" --k_frac 0.1667 --acq_pool "$AP"
done
fi

# round 36 (2026-10-05): acquisition pooling, follow-ups to round 34 (PASTIS all 0.651 attn vs
# 0.622 composite; 1/3 0.618 vs 0.600; ps4ts64_ts 0.614).
#   acq_L3_{all,k33,k17}_{comp,attn}_seed1              a second seed, from scratch
#   acq_warm_{k33,k17}_{comp,attn}_{rand,drop}[_seed1]  warm from round 34's seed-0 run: random
#       picks (control) vs the layer-3 value selector with the drop target (--joint_target drop)
#   MS=base: the same chains on OlmoEarth base (names gain _base; BS = batch size of the all arms)
if [ "$TASK" = pastis ]; then
MS="${MS:-tiny}"; MX=$([ "$MS" = tiny ] && echo "" || echo "_$MS")
AQ=(--selector random --select_unit slot --readout native_set --inject_layer 3 --frac_slot --model_size "$MS")
for AP in none attn; do
NM=$([ "$AP" = none ] && echo comp || echo "$AP"); AT=$([ "$AP" = none ] && echo "" || echo "_acq$AP")
for SD in ${SEEDS:-1}; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "acq_L3_all_${NM}${MX}${SX}" "${AQ[@]}" --k_frac 1      --acq_pool "$AP" --batch_size "${BS:-8}" --seed "$SD"
chain "acq_L3_k33_${NM}${MX}${SX}" "${AQ[@]}" --k_frac 0.3333 --acq_pool "$AP" ${BSK:+--batch_size "$BSK"} --seed "$SD"
chain "acq_L3_k17_${NM}${MX}${SX}" "${AQ[@]}" --k_frac 0.1667 --acq_pool "$AP" ${BSK:+--batch_size "$BSK"} --seed "$SD"
done
for KK in "k33 0.3333 0.333" "k17 0.1667 0.167"; do
set -- $KK
WQ=("${AQ[@]}" --k_frac "$2" --acq_pool "$AP" --pos center --epochs "$WEP" ${BSK:+--batch_size "$BSK"}
    --init_ckpt "inject_fft_pastis_${MS}_random_L3_k$3_native_set_frac${AT}_poscenter_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "acq_warm_$1_${NM}${MX}_rand${SX}" "${WQ[@]}" --seed "$SD"
chain "acq_warm_$1_${NM}${MX}_drop${SX}" "${WQ[@]}" --seed "$SD" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL --joint_target drop
done
done
done
fi
# round 36 (2026-10-05): the dropout-pass target (--joint_target drop) gave PASTIS its first
# selector gain (0.581 / 0.574 vs 0.557 random) -> seed 2 at L3 and seeds 0-1 at L2 (vs
# jf_warm_L{2,3}_joint_lrL / _rand); and seed 1 of the patch-size pair at L0 (ps4 vs ps16 windows).
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
for LY in 2 3; do
JL=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer "$LY"
    --head_attn oneway --pos center --epochs "$WEP" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --init_ckpt "inject_fft_${TASK}_tiny_random_L${LY}_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1 2; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "sel_drop_L${LY}$SX" "${JL[@]}" --joint_target drop --seed "$SD"
done
done
LS=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --seed 1)
chain ps_rand_L0_seed1       --selector random "${LS[@]}"
chain cpx16_rand_L0_seed1    --selector random "${LS[@]}" --crop_px 16
fi
# round 38 (2026-10-05): regular-lattice picks (--selector grid, see grid_scores) -- the
# coverage-maximising content-blind baseline next to random picks, for the acquisition-pooling
# arms of round 36. Random at 1/6 leaves 11% of the cells without any crop; the lattice none.
#   acq_L3_{k33,k17}_{comp,attn}_grid[_seed1]         from scratch (vs acq_L3_*, random picks)
#   acq_warm_{k33,k17}_{comp,attn}_grid[_seed1]       warm from the random-pick run, like _rand
#   acq_warm_{k33,k17}_{comp,attn}_gspatial[_seed1]   ... the same 1-in-m cells in every month
if [ "$TASK" = pastis ]; then
GQ=(--selector grid --select_unit slot --readout native_set --inject_layer 3 --frac_slot)
for AP in none attn; do
NM=$([ "$AP" = none ] && echo comp || echo "$AP"); AT=$([ "$AP" = none ] && echo "" || echo "_acq$AP")
for KK in "k33 0.3333 0.333" "k17 0.1667 0.167"; do
set -- $KK
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
WG=("${GQ[@]}" --k_frac "$2" --acq_pool "$AP" --pos center --epochs "$WEP" --seed "$SD"
    --init_ckpt "inject_fft_pastis_tiny_random_L3_k$3_native_set_frac${AT}_poscenter_lr0.001_ep${EP}")
chain "acq_L3_$1_${NM}_grid${SX}"        "${GQ[@]}" --k_frac "$2" --acq_pool "$AP" --seed "$SD"
chain "acq_warm_$1_${NM}_grid${SX}"      "${WG[@]}"
chain "acq_warm_$1_${NM}_gspatial${SX}"  "${WG[@]}" --grid_mode spatial
done
done
done
fi
# round 39 (2026-10-06): round 38 found the staggered lattice equals the drop selector on PASTIS
# (1/6 pooled: random 0.589, lattice 0.615, selector 0.614) -- the selector's gain was coverage.
#   acq_L3_{k33,k17}_attn_base_grid[_seed1]   OlmoEarth base with lattice picks, from scratch
#   the harder selector test, warm from the LATTICE-pick run (${MS}, env MS; names gain _base):
#   gwarm_{k33,k17}${MX}_grid[_seed1]    lattice picks, 20 more epochs (control)
#   gwarm_{k33,k17}${MX}_drop[_seed1]    drop selector, eps share = random slots
#   gwarm_{k33,k17}${MX}_dropg[_seed1]   drop selector, eps share = a coarse lattice (--joint_explore grid)
if [ "$TASK" = pastis ]; then
MS="${MS:-tiny}"; MX=$([ "$MS" = tiny ] && echo "" || echo "_$MS")
GB=(--select_unit slot --readout native_set --inject_layer 3 --frac_slot --acq_pool attn --model_size "$MS" ${BSK:+--batch_size "$BSK"})
for KK in "k33 0.3333 0.333" "k17 0.1667 0.167"; do
set -- $KK
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
if [ "$MS" != tiny ]; then
chain "acq_L3_$1_attn${MX}_grid${SX}" --selector grid "${GB[@]}" --k_frac "$2" --seed "$SD"
[ "$SD" = 0 ] && GDEP[$1]=${LAST:-}
fi
done
done
for KK in "k33 0.3333 0.333" "k17 0.1667 0.167"; do
set -- $KK
GW=("${GB[@]}" --k_frac "$2" --pos center --epochs "$WEP"
    --init_ckpt "inject_fft_pastis_${MS}_grid_L3_k$3_native_set_frac_acqattn_poscenter_lr0.001_ep${EP}")
JD=(--selector random --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL --joint_target drop)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
[ "$MS" != tiny ] && AFTER=${GDEP[$1]:-}
LINKS="${LW:-$LINKS}" chain "gwarm_$1${MX}_grid${SX}"  --selector grid "${GW[@]}" --seed "$SD"
LINKS="${LW:-$LINKS}" chain "gwarm_$1${MX}_drop${SX}"  "${JD[@]}" "${GW[@]}" --seed "$SD"
LINKS="${LW:-$LINKS}" chain "gwarm_$1${MX}_dropg${SX}" "${JD[@]}" "${GW[@]}" --seed "$SD" --joint_explore grid
unset AFTER
done
done
fi
# round 37 (2026-10-05): the value predictor as attention on ENCODED candidates (--joint_arch
# xenc*: EncCrossSelector -- the candidate carries OlmoEarth's own crop encodings, no one-hots;
# cross-attention to the tile's season tokens instead of concatenating them), all with the
# dropout-pass target, L3, vs sel_drop (the concat MLP, same target). Seeds 0, 1.
#   sel2_xenc      cross-attention only          sel2_xenc_t   + attention over a cell's months
#   sel2_xenc_ts   + attention over a month's cells (information flows in space too)
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
JL=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs "$WEP" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --joint_target drop
    --init_ckpt "inject_fft_${TASK}_tiny_random_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for JA in xenc xenc_t xenc_ts; do
chain "sel2_${JA}$SX" "${JL[@]}" --joint_arch "$JA" --seed "$SD"
done
done
fi
# round 38 (2026-10-05): after round 37 (EncCrossSelector ties the concat MLP at MORE compute:
# overhead 2.6 / 3.5 / 4.7 GMAC for xenc / _t / _ts vs 2.4; selector loss ~0.97 everywhere).
# L3, dropout-pass target, seeds 0, 1, vs sel_drop:
#   sel3_xs    --joint_arch xs      xenc_ts at width 128, one-layer feed-forward (cheap)
#   sel3_xc    --joint_arch xc      width 128; candidates mix through 2 full self-attention
#                                   layers over their (month, ps16 cell) means instead
#   sel3_df25  --drop_frac 0.25     concat MLP, less label noise: drop a quarter, not half
#   sel3_dp2   --drop_passes 2      concat MLP, two dropout passes averaged
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
JL=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer 3
    --head_attn oneway --pos center --epochs "$WEP" --joint --eps_start 1 --eps_end 0.3 --joint_feat lrL
    --joint_target drop
    --init_ckpt "inject_fft_${TASK}_tiny_random_L3_k0.167_native_set_poscenter_haoneway_lr0.001_ep${EP}")
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
chain "sel3_xs$SX"   "${JL[@]}" --joint_arch xs  --seed "$SD"
chain "sel3_xc$SX"   "${JL[@]}" --joint_arch xc  --seed "$SD"
chain "sel3_df25$SX" "${JL[@]}" --drop_frac 0.25 --seed "$SD"
chain "sel3_dp2$SX"  "${JL[@]}" --drop_passes 2  --seed "$SD"
done
fi
# round 36 (2026-10-05): patch size of the extra tokens, random picks only, crops TRULY off every
# grid (--crop_free: a crop_px window of one random month at a uniformly random pixel position).
#   free_ps{16,8,4}   the 1/6 budget (512 crops), L0, from scratch
#   vary_ps{16,8,4}   ONE model per patch size trained with 0..max crops per sample (max 196 /
#                     512 / 1024), tested at 0, 16, 32, ... max -> results/<task>/inject_ncrops.csv
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free)
for PX in 16 8 4; do
chain "free_ps$PX" "${FR[@]}" --crop_px "$PX"
done
chain vary_ps16 "${FR[@]}" --crop_px 16 --n_crops 196  --vary_crops
chain vary_ps8  "${FR[@]}" --crop_px 8  --n_crops 512  --vary_crops
chain vary_ps4  "${FR[@]}" --crop_px 4  --n_crops 1024 --vary_crops
fi
# round 37 (2026-10-06): the vary_ps* curves again with seed 1, and with higher caps (512 / 1024 /
# 2048 crops for ps16 / ps8 / ps4) -- no curve had flattened at 196 / 512 / 1024
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--selector random --select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops)
chain vary_ps16_seed1 "${FR[@]}" --crop_px 16 --n_crops 196  --seed 1
chain vary_ps8_seed1  "${FR[@]}" --crop_px 8  --n_crops 512  --seed 1
chain vary_ps4_seed1  "${FR[@]}" --crop_px 4  --n_crops 1024 --seed 1
chain varyhi_ps16 "${FR[@]}" --crop_px 16 --n_crops 512
chain varyhi_ps8  "${FR[@]}" --crop_px 8  --n_crops 1024
chain varyhi_ps4  "${FR[@]}" --crop_px 4  --n_crops 2048
fi
# round 40 (2026-10-06): the crop-count curves with staggered-lattice picks (crops on their own
# patch grid, every cell cropped every m-th month; grid_scores) next to the random ones. Every
# --vary_crops test sweep now scores both picks at 0, 16, 32, 64, 128, 196, 256, ... cap.
#   nsweep_ps{16,8,4}  the finished random-trained vary_ps* models, test sweep only (LINKS=1)
#   varyg_ps{16,8,4}   trained with lattice picks (random lattice offset per sample), same caps
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops)
chain nsweep_ps16 "${FR[@]}" --selector random --crop_px 16 --n_crops 196  --sweep_only
chain nsweep_ps8  "${FR[@]}" --selector random --crop_px 8  --n_crops 512  --sweep_only
chain nsweep_ps4  "${FR[@]}" --selector random --crop_px 4  --n_crops 1024 --sweep_only
chain varyg_ps16 "${FR[@]}" --selector grid --crop_px 16 --n_crops 196
chain varyg_ps8  "${FR[@]}" --selector grid --crop_px 8  --n_crops 512
chain varyg_ps4  "${FR[@]}" --selector grid --crop_px 4  --n_crops 1024
fi
# round 40b: the round-37 runs finished under the old sweep (random picks, no 196) -> sweep again
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops --selector random --sweep_only)
chain nsweep37_ps16_seed1 "${FR[@]}" --crop_px 16 --n_crops 196  --seed 1
chain nsweep37_ps8_seed1  "${FR[@]}" --crop_px 8  --n_crops 512  --seed 1
chain nsweep37_ps4_seed1  "${FR[@]}" --crop_px 4  --n_crops 1024 --seed 1
chain nsweep37_hi_ps16 "${FR[@]}" --crop_px 16 --n_crops 512
chain nsweep37_hi_ps8  "${FR[@]}" --crop_px 8  --n_crops 1024
chain nsweep37_hi_ps4  "${FR[@]}" --crop_px 4  --n_crops 2048
fi
# round 41 (2026-10-06): APT (Choudhury et al. 2025, arXiv:2510.18091) as a content-aware,
# label-free, MODEL-FREE patch-selection baseline (--selector apt): pixel-intensity entropy
# H(P) per ps4 cell per month, plus our own H_time term over the present months (see
# InjectFT._apt_scores). Needs --select_unit slot, so PASTIS / BioMassters only (GEOID uses
# location). Same budgets and readout as the plain rand_L0_native_set control (line ~212) and
# the staggered-lattice control (--selector grid), so all three are directly comparable at
# matched k_frac: random (content-blind, i.i.d.), grid (content-blind, coverage-maximising),
# apt (content-aware, no model pass needed to score).
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
AP=(--inject_layer 0 --select_unit "$UNIT" --readout native_set)
chain apt_L0_native_set        --selector apt "${AP[@]}"
chain apt_L0_k33_native_set    --selector apt "${AP[@]}" --k_frac 0.3333
chain grid_L0_native_set       --selector grid "${AP[@]}"
chain grid_L0_k33_native_set   --selector grid "${AP[@]}" --k_frac 0.3333
fi
# round 42 (2026-10-07), two user-requested extensions of round 41's --selector apt:
#   apt_time none   drop the H_time term -- the MORE LITERAL APT: pure per-frame spatial
#                   entropy H_space alone, time-ignorant, flat top-k over every (month,
#                   cell) slot like every other selector (same budgets as round 41)
#   apt_pool        space-then-time: cells picked first by score summed over their present
#                   months, consecutive similar months at a picked cell merged into one
#                   interpolated-slot crop (--apt_merge_mse), freed budget spent globally on
#                   more cells (see InjectFT._apt_pool_mask / _apt_pool_crops)
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
AP=(--inject_layer 0 --select_unit "$UNIT" --readout native_set --selector apt)
chain apttime_L0_native_set       "${AP[@]}" --apt_time none
chain apttime_L0_k33_native_set   "${AP[@]}" --apt_time none --k_frac 0.3333
chain aptpool_L0_native_set       "${AP[@]}" --apt_pool
chain aptpool_L0_k33_native_set   "${AP[@]}" --apt_pool --k_frac 0.3333
fi
# round 41 (2026-10-08): ONE crop-position list shared by every crop size (same count -> same
# places and months; only the window differs), cap 2048 for all, lattice picks in training:
#   pos_c2f_ps{16,8,4}     --grid_mode c2f     ps16 grid corners, then ps8, then ps4 (coarse to fine)
#   pos_shared_ps{16,8,4}  --grid_mode shared  the staggered ps4 lattice, windows centred on it
# tested at 0, 16, 32, 64, 96, 128, 192, ... 2048 with measured coverage -> inject_ncrops.csv
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --n_crops 2048)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for GM in c2f shared; do for PX in 16 8 4; do
chain "pos_${GM}_ps$PX$SX" "${FR[@]}" --grid_mode "$GM" --crop_px "$PX" --seed "$SD"
done; done
done
fi
# round 43 (2026-10-08): the other two resolution pairs, for the accuracy / compute picture
# (round 41's pos_*_ps8 is ps16 -> ps8, pos_*_ps4 is ps16 -> ps4):
#   b8_{c2f,shared}_ps4   ps8 -> ps4: seasons at ps8 (--lr_patch 8) + ps4 month crops, 0..2048
#   fix_ps{16,8,4}        fixed endpoints: seasons only at one patch size, same readout, no crops
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --n_crops 2048)
for SD in 0 1; do
SX=$([ "$SD" = 0 ] && echo "" || echo "_seed$SD")
for GM in c2f shared; do
LINKS="${LB:-8}" chain "b8_${GM}_ps4$SX" "${FR[@]}" --grid_mode "$GM" --crop_px 4 --lr_patch 8 --seed "$SD"
done
for PX in 16 8 4; do
LINKS=$([ "$PX" = 4 ] && echo "${LB:-8}" || echo 3) chain "fix_ps$PX$SX" --selector none --readout native_set --lr_patch "$PX" --seed "$SD"
done
done
fi
# round 44 (2026-10-09): the low-compute end of the mixed curves (shared position list, seed 0)
#   set_<pair>[_long]   counts drawn from {0,64,...,2048} in training (0 crops 1 time in 7);
#                       _long: twice the epochs
#   ded_<pair>_n<K>     a model DEDICATED to K crops (--crop_counts K), still tested at <= K
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --grid_mode shared)
for PAIR in "16 8" "16 4" "8 4" "16 16"; do   # 16 16: only the ded_ chains were submitted
set -- $PAIR
PA=(--lr_patch "$1" --crop_px "$2")
LINKS=8  chain "set_$1-$2"      "${FR[@]}" "${PA[@]}" --crop_counts 0,64,128,256,512,1024,2048
LINKS=14 chain "set_$1-$2_long" "${FR[@]}" "${PA[@]}" --crop_counts 0,64,128,256,512,1024,2048 --epochs $((2 * EP))
for K in 64 128 256 512 1024 2048; do
LINKS=$(case $K in 2048) echo 12 ;; 1024) echo 7 ;; 512) echo 5 ;; *) echo 4 ;; esac) \
  chain "ded_$1-$2_n$K" "${FR[@]}" "${PA[@]}" --crop_counts "$K"
done
done
fi
# round 45 (2026-10-09): round 44's dedicated-count models with the crops injected at block 3
# instead of 0 (crops run blocks 0-2 alone, then 9 joint blocks):  ded3_<pair>_n<K>
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 3 --crop_free --vary_crops --selector grid --grid_mode shared)
for PAIR in "16 8" "16 4" "8 4" "16 16"; do
set -- $PAIR
for K in 64 128 256 512 1024 2048; do
LINKS=$(case $K in 2048) echo 12 ;; 1024) echo 7 ;; 512) echo 5 ;; *) echo 4 ;; esac) \
  chain "ded3_$1-$2_n$K" "${FR[@]}" --lr_patch "$1" --crop_px "$2" --crop_counts "$K"
done
done
fi
# round 46 (2026-10-12): spatial resolution ONLY -- no season tokens (--no_seasons): the month
# crops are the whole input. Dedicated models on the shared lattice, to set against the fixed
# monthly on-grid ps16 / ps8 / ps4 models (matched token count at 16 / 64 / 256 crops a month:
# 192 / 768 / 3072 for 12 months).   nos_ps<P>_n<K>
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
FR=(--select_unit "$UNIT" --readout native_set --inject_layer 0 --crop_free --vary_crops --selector grid --grid_mode shared --no_seasons)
for PK in "16 64" "16 128" "16 192" "16 256" "16 512" "8 128" "8 256" "8 512" "8 768" "8 1024" "4 512" "4 1024" "4 2048" "4 3072"; do
set -- $PK
LINKS=$(case $2 in 3072) echo 14 ;; 2048) echo 12 ;; 1024) echo 7 ;; 512|768) echo 5 ;; *) echo 4 ;; esac) \
  chain "nos_ps$1_n$2" "${FR[@]}" --crop_px "$1" --crop_counts "$2"
done
fi
# round 47 (2026-10-12): spatial resolution only, NO seasonal composites anywhere. Base = the
# monthly on-grid model at patch size P; the test adds EVERY P px window of a grid shifted off
# the patch grid, in every month (--shift_stride S; zero-padded past the tile edge):
#   sg_ps<P>_base        monthly on-grid only (same readout, same batch size)
#   sg_ps<P>_s<S>        + the shifted grid of stride S = 16 / 8 / 4 (5/10/20, 4/9/18, 4/8/17
#                        windows a side for P = 16, 8, 4)
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
SG=(--lr_steps all --readout native_set --batch_size 8)
for P in 16 8 4; do
LINKS=$([ "$P" = 4 ] && echo 10 || echo 5) chain "sg_ps${P}_base" "${SG[@]}" --lr_patch "$P" --selector none
for S in 16 8; do   # stride 4 dropped 2026-10-12: 1400-2800 GMACs a sample, too slow
LINKS=$( { [ "$S" = 4 ] || [ "$P" = 4 ]; } && echo 16 || echo 7) chain "sg_ps${P}_s$S" "${SG[@]}" --lr_patch "$P" \
  --crop_px "$P" --shift_stride "$S" --select_unit "$UNIT" --inject_layer 0 --crop_free --vary_crops --selector grid
done
done
fi
# round 48 (2026-10-13): TEMPORAL resolution only -- regular grid at ps16 / ps8 / ps4, no crops,
# the input steps coarsened or refined (--lr_steps), same readout:
#   tr_ps<P>_year    ONE composite of all months        tr_ps<P>_raw   every raw acquisition
#   seasonal = fix_ps<P> (round 43), monthly = sg_ps<P>_base (round 47)
# raw: PASTIS only here (BioMassters' months ARE its raw steps); no ps4 (~50k tokens a sample)
SG=(--readout native_set --batch_size 8 --selector none)
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
for P in 16 8 4; do
LINKS=3 chain "tr_ps${P}_year" "${SG[@]}" --lr_patch "$P" --lr_steps year
done
fi
if [ "$TASK" = pastis ]; then
LINKS=6  chain tr_ps16_raw "${SG[@]}" --lr_patch 16 --lr_steps raw
LINKS=14 chain tr_ps8_raw  "${SG[@]}" --lr_patch 8  --lr_steps raw
fi
# 2026-10-08 (user: "what if we use apt to query ps16 tokens?"): APT chooses patch SIZE
# (ps16 vs ps4) by its own entropy gain, the part of the original paper's coarse/fine
# quadtree this codebase's plain --selector apt never implemented (it only ranks a grid
# fixed at ps4). Every present (month, ps16 cell) starts as one ps16 crop (1 token);
# splitting into its 16 ps4 children costs 15 more and is spent on the highest entropy-GAIN
# cells first, cumulative-cost cutoff (see InjectFT._apt_res_mask). Same budgets as apt /
# apttime / aptpool above, so all 5 arms compare at matched k_frac.
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
AP=(--inject_layer 0 --select_unit "$UNIT" --readout native_set --selector apt)
chain aptres_L0_native_set       "${AP[@]}" --apt_res
chain aptres_L0_k33_native_set   "${AP[@]}" --apt_res --k_frac 0.3333
fi
# round 46 (2026-10-12, user: "start with ps16 and t=all... split in time or space or both,
# depending on the after-split entropy -- this is almost info gain"): a true 3D (space, space,
# time) octree -- InjectFT._apt_octree_mask, no token budget (pure entropy threshold --apt_tau,
# calibrated per (task, k_frac) by exp.upsamplers.calibrate_apt_octree_tau to match apt/aptres'
# own token counts at k_frac 1/6 and 1/3; see results/apt_octree_tau.json). 1/3 is UNREACHABLE
# on pastis (tops out ~720 of 1024 target tokens -- the gain-per-token test itself stops
# splitting further once no split beats staying coarse, independent of tau) -- queued anyway,
# user-confirmed 2026-10-12: report actual vs target honestly rather than skip it.
#
# 3 --apt_octree_pool arms (user: "how are the time series patches handled... input composites
# or embedding pool or something else", then: "also 1) encode and pool embedding, and 2) encode
# and cross attention pool embedding... computation wise input < embed < attn"):
#   input  pixel+slot average every leaf's member months BEFORE embedding (cheapest, default)
#   embed  embed every member month separately, blocks [0,L), MEAN-pool the embeddings
#   attn   same as embed but a LEARNED per-member softmax weight (reuses --acq_pool's mechanism)
#
# tau values from the FULL calibration sweep (64-sample, train-split, results/apt_octree_tau.json,
# 2026-10-12). 1/3 is UNREACHABLE on pastis/biomassters (gain-per-token test itself stops
# splitting further once no split beats staying coarse, independent of tau -- tops out at
# ~720/658 of the 1024/1024 target tokens respectively) -- queued anyway at a small positive tau
# (0.05, confirmed to already hit the same ceiling as tau<=0) per user-confirmed 2026-10-12
# policy: report actual vs target honestly rather than skip an unreachable pair.
# geoidflood NOT included here (unlike apt_res/apt_pool above, which already gate on
# pastis/biomassters only): --apt_octree REQUIRES --select_unit slot (see InjectFT's own
# validation), but geoidflood's UNIT is "location" (its pair_* head needs both dates of a
# patch) -- the same constraint apt/apt_res/apt_pool already work around by simply never
# running on geoidflood; apt_octree is no different, just made explicit here since this
# block would otherwise silently pass --select_unit location and hit a ValueError at
# construction time on the cluster instead of failing fast locally.
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
AO=(--inject_layer 0 --select_unit "$UNIT" --readout native_set --selector apt --apt_octree)
case "$TASK" in
  pastis)      TAU16=10.8693 ;;
  biomassters) TAU16=8.6529  ;;
esac
for POOL in input embed attn; do
LINKS=3 chain "aptoctree_${POOL}_L0_native_set"     "${AO[@]}" --apt_octree_pool "$POOL" --apt_tau "$TAU16"
LINKS=3 chain "aptoctree_${POOL}_L0_k33_native_set" "${AO[@]}" --apt_octree_pool "$POOL" --apt_tau 0.05 --k_frac 0.3333
done
fi
# round 49 (2026-10-15): label-error ORACLE -- how much is there to gain from choosing WHERE to
# re-encode? ps16 seasons predict alone; the n locations whose 8 px window has the highest loss
# vs the TRUE label (train AND test) get a ps8 crop at every present month (--k_frac n/256 = n
# locations). Random locations at the same n are the control.
#   zs_*    no training: the LR-only native checkpoint, --readout native_mean (parameter-free
#           softmax mean over the overlapping tokens), --epochs 0
#   *_mean  fine-tuned, native_mean          *_set   fine-tuned, native_set (attention pool;
#           no-crop baseline = none_native_set)
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
ZS_CKPT=$([ "$TASK" = pastis ] && echo inject_fft_pastis_tiny_none_native_lr0.001_ep50 \
          || echo inject_fft_biomassters_tiny_none_native_lr0.001_ep10)
OR=(--inject_layer 0 --select_unit location --crop_px 8)
LINKS=1 chain zs_none_mean --selector none --readout native_mean --init_ckpt "$ZS_CKPT" --epochs 0
LINKS=3 chain none_mean    --selector none --readout native_mean
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for SEL in oracle random; do
S=${SEL:0:3}
LINKS=1 chain "zs_${S}_n${N}_mean" --selector "$SEL" "${OR[@]}" --k_frac "$KF" --readout native_mean \
  --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
LINKS=3 chain "${S}_n${N}_L0_mean" --selector "$SEL" "${OR[@]}" --k_frac "$KF" --readout native_mean
LINKS=3 chain "${S}_n${N}_L0_set"  --selector "$SEL" "${OR[@]}" --k_frac "$KF" --readout native_set
done
done
fi
# round 50 (2026-10-15): round 49's zero-shot oracle with --attn oneway -- the lr tokens never
# read the crops (the lr sequence IS the crop-free model), each crop reads the lr keys / values.
# Outside the crop windows the prediction is exactly the LR model's; only the windows change.
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for SEL in oracle random; do
LINKS=1 chain "zs_${SEL:0:3}_n${N}_mean_oneway" --selector "$SEL" --inject_layer 0 --select_unit location \
  --crop_px 8 --k_frac "$KF" --readout native_mean --attn oneway --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
done
done
fi
# round 51 (2026-10-15): round 49's fine-tuned oracle / random arms with --attn oneway -- the lr
# tokens never read the crops in ANY block (blocks [0, L) are separate by construction; --attn
# covers [L, 12)). No-crop baselines unchanged: none_mean, none_native_set.
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for SEL in oracle random; do
for R in mean set; do
LINKS=3 chain "${SEL:0:3}_n${N}_L0_${R}_oneway" --selector "$SEL" --inject_layer 0 --select_unit location \
  --crop_px 8 --k_frac "$KF" --readout "native_$R" --attn oneway
done
done
done
fi
# round 52 (2026-10-15, user: "how does it compare to grid?"): the content-blind lattice at the
# oracle's budgets -- n ps8 windows on a regular lattice of the tile's 8 x 8 window grid
# (location_grid_scores: checkerboard / staggered 1-in-4 / one per row and column for n = 32 /
# 16 / 8), every present month; random phase in training, fixed at test. Same arms as rounds
# 49 - 51 (fine-tuned mean / set, joint and oneway; zero-shot mean, joint and oneway).
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
GR=(--selector grid --inject_layer 0 --select_unit location --crop_px 8)
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for R in mean set; do
LINKS=3 chain "grid_n${N}_L0_${R}"        "${GR[@]}" --k_frac "$KF" --readout "native_$R"
LINKS=3 chain "grid_n${N}_L0_${R}_oneway" "${GR[@]}" --k_frac "$KF" --readout "native_$R" --attn oneway
done
LINKS=1 chain "zs_grid_n${N}_mean"        "${GR[@]}" --k_frac "$KF" --readout native_mean \
  --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
LINKS=1 chain "zs_grid_n${N}_mean_oneway" "${GR[@]}" --k_frac "$KF" --readout native_mean \
  --attn oneway --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
done
fi
# round 53 (2026-10-15, user: "restrict the amount of overlap"): the oracle's candidate ps8
# windows on a coarser lattice (--loc_stride): 8 = the 8 x 8 non-overlapping windows, 6 = 10 x 10
# windows spread evenly (gaps 6-7 px, <= 2 px overlap). Round 49 is stride 4 (a window centred
# on every ps4 cell: neighbours overlap by half). Fine-tuned mean / set x joint / oneway, and
# zero-shot oneway. Content-blind control at stride 8: grid (round 52).
if [ "$TASK" = pastis ] || [ "$TASK" = biomassters ]; then
OS=(--selector oracle --inject_layer 0 --select_unit location --crop_px 8)
for S in 6 8; do
for N in 8 16 32; do
KF=$(python3 -c "print($N / 256)")
for R in mean set; do
LINKS=3 chain "ora_st${S}_n${N}_L0_${R}"        "${OS[@]}" --loc_stride "$S" --k_frac "$KF" --readout "native_$R"
LINKS=3 chain "ora_st${S}_n${N}_L0_${R}_oneway" "${OS[@]}" --loc_stride "$S" --k_frac "$KF" --readout "native_$R" --attn oneway
done
LINKS=1 chain "zs_ora_st${S}_n${N}_mean_oneway" "${OS[@]}" --loc_stride "$S" --k_frac "$KF" --readout native_mean \
  --attn oneway --init_ckpt "$ZS_CKPT" --epochs 0 --pos center
done
done
fi
