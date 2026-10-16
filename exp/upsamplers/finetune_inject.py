"""Fine-tune OlmoEarth with high-res tokens INJECTED mid-network (manyup.inject), end to end.

    seasons   the season composites (GroupSteps(task.season), mid-season timestamps) at ps16
              through blocks [0, L)                                              -> lr_L
    select    k (month, ps4 patch) slots by --selector, from lr_L and each month's own ps4
              patch embedding
    crops     each selected slot: its 4 x 4 px crop of that month at ps4 through blocks
              [0, L), at its true pixel position, its month, its season's time slot
    tail      blocks [L, 12) on each sample's seasons + its crops together, then the norm
    readout   every month = its season's final tokens, bilinear to the ps4 grid; a selected
              slot is replaced by its crop's final token
    head      the task's pa2px head (time mean over present months, or post - pre)

No upsampler, no guidance network: the only guidance is the month's own input, through the
selector and the crops. ONE set of FlexiViT weights runs both patch sizes and is fine-tuned
end to end with the head, the type embedding and (learned) the selector.

--selector (arm names, see arm_label):
    none     no crops (k = 0)                                  "S4ps16 bilinear"
    random   k random present slots                            "S4ps16 + 512xps4-1mo inject@L6 @ random"
    learned  per-slot MLP on [e_m, e_m - season mean e, lr_L at the slot's season]
             (e_m = month m's ps4 patch embedding), trained jointly on the task loss:
             score gate at injection (crop token -> its season token as sigmoid(s) -> 0) and at
             readout, Gumbel-top-k while training                     "... @ learned"
    xattn    manyup.inject.CrossAttnSelector: each (month, cell) queries the lr_L sequence;
             --sel_temporal adds self-attention across the months of a cell
             Both learned selectors also take --gain_loss lambda: an auxiliary rank-BCE teaching
             the scores the FIRST-ORDER GAIN of each picked crop, -<dL/d out, crop - base> at
             its slot (out = the readout the head sees, base = the season's bilinear token
             there), ranked within each sample; dL/d out comes from the task loss for free.
--select_unit location: k // T patches, each at every present month (GEOID: both dates).
--select_unit pair (series tasks with a pair_* head): candidates are (consecutive date pair,
    cell); a chosen pair crops BOTH its dates, so a pair's difference is either
    high-res vs high-res or season token vs season token -- never a crop against a season
    token, which would read as a change. Budget per sample: k_frac x its dates x cells / 2
    pairs; a pair scores the max of its two dates.
--lr_steps all / --lr_patch 4 (references): the low-res sequence is every step itself (no
    seasons) and/or at ps4 -- with --selector none, the plain OlmoEarth series fine-tune at ps16
    or ps4, through the same head.
--pick (which slots a selector's scores choose; diagnostics of the learned selector):
    topk      the k best slots over all months (default)
    random    k random slots; a learned selector still GATES them -- isolates the gate's cost
    per_step  the best k / (present months) slots within EACH month -- forces spread over time
--score_norm / --score_l1 (learned): a trained selector's scores run off until the gate is
    sigmoid = 1 everywhere, which passes no gradient (seen: mean gate 1.000 on PASTIS and
    BioMassters). score_norm gates by sigmoid of the scores standardized over each sample's
    present slots (top-k is unchanged: standardizing is monotonic per sample); score_l1 adds
    lambda * mean |score| to the loss.
--readout (final tokens -> the (month, ps4 cell) grid the head reads):
    bilinear   season tokens bilinear, crops overwrite their cell (default)
    splat      manyup.inject.GaussianSplatReadout: every token predicts an anisotropic Gaussian
               and a weight; a month reads its season's tokens and its own crops
    splat_t    + a temporal kernel: every month reads every token (crops inform nearby months)
    splat_tg   + the bilateral guidance term (the month's patch embedding vs the token's)
    tsplat / tquery   manyup.inject.TimeSetReadout: each cell reads its overlapping tokens
                      (covering ps16 token of every slot + crops at the cell), weighted per
                      QUERY TIME by a Gaussian in time (tsplat) or by attention from a time
                      encoding (tquery) -> a feature at every step (any head, incl. pair diff:
                      dates of one month no longer share a token)
    native_tsplat / native_tquery   the same weights, averaged over the months, applied to
                      the tokens' native logits (native_time_logits; time_mean tasks)
    native_set        native_attn as a per-cell SET (NativeSetReadout): the covering season
                      tokens + every crop at the cell (months AND dates), no month grid --
                      the date level's overlap-only readout
    native_mean       native_set with NO parameters: weights = the priors alone (season token:
                      its present months, crop: 1), the mean taken over each candidate's
                      SOFTMAX (classification); loads a --readout native checkpoint as is
    native_attn       native + a temporal attention pool over months IN LOGIT SPACE
                      (native_attn_logits) instead of the plain mean
    cellattn_overlap  cellattn restricted to overlapping tokens (the covering ps16 token of
                      every season + the crops at the cell), no spatial Gaussian
    splat_t_overlap   splat_t restricted to overlapping tokens, no spatial Gaussian: a
                      time-only splat (each month reads the covering tokens by time distance)
    native     no feature upsampling: the pa2px head maps each ps16 season token to its own
               16 x 16 px block (sub-patch detail intact) and each crop, through the SAME head,
               to a 16 x 16 map area-downsampled to its 4 x 4 px block; per month the crops
               overwrite their blocks, then the mean over months (in logit space; the head is
               linear). No crops = the fixed ps16 fine-tune's head. See native_logits.
    cellattn   manyup.inject.CellAttnReadout (time_mean tasks; replaces --head): no grid of
               months -- each ps4 cell attention-pools the season tokens around it and the
               crops within 1 cell, weights = content score + each token's spatial Gaussian
--res_mix a,b,c (--readout native_set, random): the k-token budget is split a:b:c over random
    single-month crops at ps16 (one token = one month over a 16 x 16 px ps16 cell), ps8
    (8 x 8 px) and ps4 (4 x 4 px) -- each one token, at its true position and its month's
    slot; the readout lists each crop at every ps4 cell of its footprint.
--frac_slot: a crop's time slot = its season + its position inside the season (in
    [-0.5, 0.5)) instead of the bare season index, so months of one season -- and dates of one
    month -- get distinct time encodings (OlmoEarth encodes slot + month, no day).
--date_level nested|free (PASTIS; needs --readout cellattn): a third level of single-date ps4
    crops, each modality on its own dates (S2 and S1A dates differ). nested: only inside the
    month-cells picked at level 1 (season -> month -> date), free: any cell. Random picks,
    --date_per_month per month pick; a learned month selector gates them by their (month,
    cell) score. --no_month_crops: the month picks only route the dates.
--selector oracle | entropy (with --select_unit location --readout cellattn): the cells to
    crop (all their months) scored by a crop-free pass of the model itself -- oracle = its
    loss against the TRUE label (a ceiling on choosing where to look; labels at test too),
    entropy = its predictive entropy (label-free). Compare with random at the same unit.
    Also --readout native_set | native_mean with --crop_px 8 | 16: a pick is the crop_px window
    centred on the cell, scored by the mean loss over that window; --k_frac n / 256 = n
    locations, each at every present month.
--selector apt (--select_unit slot, any readout): Choudhury et al. 2025's Adaptive Patch
    Transformers (arXiv:2510.18091) pixel-intensity entropy, H(P) = -sum p_i log2 p_i over a
    binned histogram of a cell's own pixels (their Eq. 1; no model pass, no labels -- unlike
    --selector entropy above, which is PREDICTIVE entropy of the model's own logits). We add
    the time axis APT has none of: score(month, cell) = H_space(cell, month) + H_time(cell),
    H_time = H(P) of that cell's per-month means across the present months (see _apt_scores).
    Luma = task.rgb's bands, mean over channels (APT is single-channel).
--date_selector xattn: learned date picks (a CrossAttnSelector per modality over its dates),
    each date crop gated by sigmoid of its own standardized score. --date_gate none: random
    date crops NOT gated by their month pick's learned score (the learned-months-with-dates
    collapse diagnostic).
--head (time_mean tasks):
    mean      the task's head: mean over present months, then pa2px
    attn      temporal attention pool: per location, a softmax over present months of a
              learned score (LayerNorm + Linear on the token) weights the months, then pa2px;
              lets the head lean on the months that got crops, whatever their spread

A month counts (is selected, read out, averaged by the head) only if EVERY input modality has
a frame there and its season has data. OlmoEarth's fine-tuning protocol as
exp.upsamplers.finetune_seasonal; resumable per epoch, --max_hours for chains of short jobs.
One row in results/<task>/inject_fft.csv.

    python -u -m exp.upsamplers.finetune_inject --task pastis --epochs 50 --selector random
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from exp.common import olmo_bootstrap  # noqa: E402,F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import argparse  # noqa: E402
import copy  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402
from olmoearth_pretrain.datatypes import TokensAndMasks  # noqa: E402
from olmoearth_pretrain.nn.encodings import get_1d_sincos_pos_encoding  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402
from dataclasses import replace  # noqa: E402

from torch.utils.data import DataLoader, Dataset, RandomSampler  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.heads import Pa2PxHead, consecutive_pairs  # noqa: E402
from exp.common.tasks import TASKS, Task  # noqa: E402
from exp.olmoearth.train import (FREEZE_EPOCH_FRACTION, SCHEDULER_COOLDOWN,  # noqa: E402
                                 SCHEDULER_FACTOR, SCHEDULER_PATIENCE, UNFREEZE_LR_FACTOR,
                                 _append, _brief)
from exp.upsamplers.finetune_seasonal import Paired  # noqa: E402
from exp.upsamplers.task_data import OUT_GRID  # noqa: E402
from manyup.data import All, GroupSteps, mtmm_collate  # noqa: E402
from manyup.data.temporal import TemporalPolicy  # noqa: E402
from manyup.encoders.olmoearth import (OlmoEarthEncoder, OlmoEarthInput,  # noqa: E402
                                       _map, _to_device, crop_patches, pool_timestep)
from manyup.inject import (CellAttnReadout, CrossAttnSelector, EncCrossSelector, GaussianSplatReadout,  # noqa: E402
                           ATTN, BASE_GSD, HEAD_ATTN, POS, MixedResOE, Packed, TimeSetReadout,
                           sincos_2d)
from manyup.seasonal import season_index, season_mean, seasons_used, select  # noqa: E402
from manyup.timAnyUp import topk_mask  # noqa: E402

LR_PS, LOOKUP_PS, TILE = 16, 4, 64
SELECTORS = ("none", "random", "grid", "learned", "xattn", "oracle", "entropy", "apt")
PRIOR = ("oracle", "entropy")     # scored by a crop-free pass of the model itself
PIXEL_PRIOR = ("apt",)            # scored from raw pixels alone, no model pass at all
APT_BINS = 32                     # histogram bins for H(P) = -sum p_i log2 p_i (paper: Eq. 1)
LEARNED = ("learned", "xattn")
DATE_LEVELS = ("none", "nested", "free")

READOUTS = ("bilinear", "splat", "splat_t", "splat_tg", "cellattn", "native",
            "native_attn", "cellattn_overlap", "splat_t_overlap", "native_set", "native_mean",
            "tsplat", "tquery", "native_tsplat", "native_tquery")
SET = ("native_set", "native_mean")                 # NativeSetReadout
NATIVE = ("native", "native_attn", "native_set", "native_mean",   # ps16 tokens -> 16 x 16 px
          "native_tsplat", "native_tquery")
TIME = ("tsplat", "tquery", "native_tsplat", "native_tquery")   # manyup.inject.TimeSetReadout


@torch.no_grad()
def pair_loo_values(head, wgt, feat, crop_of, pairs, target, NS, chunk=2048):
    """In-context value of each crop on a series pair head: remove it from its cell's time
    weights at EVERY date, (out_t - w_tj f_j) / (1 - w_tj); recompute the cell's pair logits
    (later - earlier [, mean]) and their logsumexp over pairs; value = loss over the cell's
    4 x 4 px without it - with it. -> (N,) in crop order."""
    B, T, Q, K = wgt.shape
    ok = crop_of >= 0
    bq = ok.nonzero(as_tuple=True)
    crop = crop_of[ok]
    m = bq[2]
    N = int(crop.max()) + 1 if len(crop) else 0
    v = torch.zeros(N, device=wgt.device)
    if not len(crop):
        return v
    C, p = head.num_outputs, head.patch_size
    pairs = pairs.to(wgt.device)
    tg = target.to(wgt.device)
    tg = tg[:, 0] if tg.ndim == 4 else tg
    lab = rearrange(tg, "b (y i) (x j) -> b (y x) (i j)", i=p, j=p)          # (B, Q, 16)
    for s0 in range(0, len(crop), chunk):
        sl = slice(s0, s0 + chunk)
        b_, q_, j_ = bq[0][sl], bq[1][sl], NS + m[sl]
        w = wgt[b_, :, q_, j_].clamp(max=1 - 1e-4)                          # (E, T)
        f = feat[b_, q_, j_]                                                # (E, D)
        o = torch.einsum("etk,ekd->etd", wgt[b_, :, q_], feat[b_, q_])      # (E, T, D)
        o_wo = (o - w[..., None] * f[:, None]) / (1 - w)[..., None]
        pr = pairs[b_]                                                      # (E, P, 2)
        okp = (pr >= 0).all(-1)
        ar = torch.arange(len(b_), device=o.device)[:, None]
        L = []
        for z in (o, o_wo):
            ea, la = z[ar, pr[..., 0].clamp(min=0)], z[ar, pr[..., 1].clamp(min=0)]
            xx = la - ea
            if head.reduction == "pair_diffmean":
                xx = torch.cat([xx, (la + ea) / 2], -1)
            lg = head.linear(xx).float().view(len(b_), -1, C, p * p)          # (E, P, C, 16)
            lg = lg.masked_fill(~okp[:, :, None, None], float("-inf"))
            L.append(torch.logsumexp(lg, 1).nan_to_num(0.0, neginf=0.0))    # (E, C, 16)
        yl = lab[b_, q_]                                                    # (E, 16)
        mm = (yl >= 0).float()
        ce = [F.cross_entropy(z, yl.long().clamp(min=0), reduction="none") for z in L]
        val = ((ce[1] - ce[0]) * mm).sum(1) / mm.sum(1).clamp(min=1)
        v.index_add_(0, crop[sl], val.float())
    return v


def tile_sample(sample, size: int):
    """Split each sample's 64 px frame into (64 / size)^2 independent size x size px tiles
    (every step kept): (B, 64, 64, T, C) -> (B * g * g, size, size, T, C), tile-major within a
    sample (row gy, then col gx). For --lr_tile: psXtsY with Y < 64 (each tile its own
    sequence, no attention between tiles)."""
    g = TILE // size

    def cut(k, v):
        if k == "timestamps":                                                   # (B, T, 3)
            return v.repeat_interleave(g * g, 0)
        if torch.is_tensor(v) and v.ndim == 5:
            return rearrange(v, "b (gy y) (gx x) t c -> (b gy gx) y x t c", gy=g, gx=g)
        return v

    return _map(sample, cut)


def native_time_logits(linear, ro, lr_F, lr_valid, season, present, frac, hr, hr_idx,
                       hr_logw=None, lr_px: int = LR_PS, crop_px: int = LOOKUP_PS,
                       parts: bool = False):
    """--readout native_tsplat / native_tquery (time_mean tasks): TimeSetReadout's per-step
    weights over each cell's overlapping tokens, averaged over the present steps, applied to
    the tokens' NATIVE logits (a ps16 token its 16 x 16 block's 4 x 4 px in the cell, a crop
    its 16 x 16 map area-downsampled to 4 x 4): mean_t sum_i w_i(t) logits_i. -> (B, C, P, P)"""
    B, NS, h, w, D = lr_F.shape
    C = linear.out_features // lr_px ** 2
    r, c = lr_px // crop_px, crop_px
    wgt, _, crop_of = ro.weights(lr_F, lr_valid, season, present, frac, hr, hr_idx, hr_logw)
    real = (present & (season >= 0)).float()                                    # (B, T)
    Wm = (wgt * real[:, :, None, None]).sum(1) / real.sum(1).clamp(min=1)[:, None, None]
    s_log = rearrange(linear(lr_F.float()), "b s gh gw (k i j) -> b s k (gh i) (gw j)",
                      k=C, i=lr_px, j=lr_px)                                    # (B, NS, C, P, P)
    cand = [rearrange(s_log, "b s k (y u) (x v) -> b (y x) s k (u v)", u=c, v=c)]
    if crop_of is not None:
        N = len(hr)
        crop = F.avg_pool2d(linear(hr.float()).view(N, C, lr_px, lr_px), r).reshape(N, C, c * c)
        cand.append(crop[crop_of.clamp(min=0)])                                 # (B, Q, M, C, 16)
    cand = torch.cat(cand, 2)                                                   # (B, Q, K, C, 16)
    out = torch.einsum("bqk,bqkcp->bqcp", Wm, cand)
    out = rearrange(out, "b (y x) k (i j) -> b k (y i) (x j)", y=h * r, x=w * r, i=c, j=c)
    # parts (--self_teach): the per-step weights (B, T, Q, K), every candidate's native logits
    # (B, Q, K, C, 16; the first NS are the season tokens) and the crop index of each crop
    # slot (B, Q, M) -- what a per-date prediction and its leave-one-out need
    return (out, (wgt, cand, crop_of)) if parts else out
JOINT_ARCH = ("mlp", "xattn", "xenc", "xenc_t", "xenc_ts", "xs", "xc")
NEW_PARAMS = ("acq_score.", "joint.", "recon_splat.", "joint_embed.", "date_pred.", "date_type")     # modules a warm start (--init_ckpt) may add
SPARSE = ("cellattn", "cellattn_overlap") + SET   # no month grid: take date crops


class NativeSetReadout(nn.Module):
    """--readout native_set: native_attn_logits as a per-cell SET, so any number of crops of
    any kind (month composites, single dates, several per cell) fit without a month grid --
    the date level's readout with overlapping tokens only.

    Cell p's candidates (padded per cell):
        season tokens   the ps16 token covering p, every used season   (NS)
        crops           every crop whose footprint covers p, any step or date, any level,
                        any patch size (ps4: 1 cell, ps8: 2 x 2 cells, ps16: 4 x 4)  (0 .. M)
    Each candidate i gives
        logits_i(px in p)   the native head: a season token its 16 x 16 block's 4 x 4 px in p,
                            a crop its 16 x 16 map area-downsampled to its own footprint
                            (ps4: 4 x 4 px, ps8: 8 x 8, ps16: kept at 16 x 16), then the
                            4 x 4 px of that footprint lying in p
        a_i                 score(f_i) + prior_i + gate_i
                            score = Linear(LayerNorm(f)) (scalar, zero init)
                            prior = log(present months of the season) for a season token (it
                                    stands for that many months, as in the month grid), 0 for
                                    a crop
                            gate  = log sigmoid(selector score) for a gated crop, else 0
    and out(px in p) = sum_i softmax_i(a_i) logits_i(px). No spatial term: the pool is over
    what overlaps p. At init (score 0) a cell with no crops is the native time mean exactly.

    Shape letters: B samples, NS season slots, h = w = 4 ps16 cells, H = W = 16 ps4 cells,
    Q = H * W, N crops in the batch, M the most crops at one cell, D features, C outputs,
    P = 64 px. forward(...) -> (B, C, P, P) logits.
    """

    def __init__(self, dim: int, lr_px: int = LR_PS, crop_px: int = LOOKUP_PS,
                 base_px: int | None = None, mean: bool = False):
        """mean (--readout native_mean): no score -- NO parameters. Weights = the priors alone
        (a season token its present months, a crop 1: per month the mean of the tokens
        covering the pixel, then the mean over months), pooled in PROBABILITY space
        (classification: the weighted mean of each candidate's softmax; returns its log)."""
        super().__init__()
        self.lr_px, self.crop_px, self.mean = lr_px, crop_px, mean
        # base_px: px side of a SEASON token (--lr_patch 8: ps8 seasons). The head still writes
        # a lr_px x lr_px map; a smaller token's map is average-pooled to its footprint, the
        # same rule the crops follow.
        self.base_px = base_px or lr_px
        self.no_base = False            # --no_seasons: the season tokens are not candidates
        self.score = None if mean else nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        if self.score is not None:
            nn.init.zeros_(self.score[1].weight)
            nn.init.zeros_(self.score[1].bias)

    def _score(self, f):
        return f.new_zeros(f.shape[:-1]) if self.score is None else self.score(f.float())[..., 0]

    def forward(self, linear: nn.Linear, lr_F, season, present, hr, hr_idx, hr_logw=None,
                hr_size=None, parts: bool = False, hr_px=None, hr_fp: int | None = None):
        """
        linear    the head, D -> C * lr_px^2
        lr_F      (B, NS, h, w, D)  final season tokens
        season    (B, T) long       season slot of each month (-1: none)
        present   (B, T) bool       month is real
        hr        (N, D)            final crop tokens (all levels)
        hr_idx    4 x (N,) long     (sample b, step t -- unused, row y, col x) on the ps4 grid:
                                    the top-left ps4 cell of the crop's footprint
        hr_logw   (N,) or None      crop gate log-weight
        hr_size   (N,) long or None footprint side in ps4 cells (1 ps4, 2 ps8, 4 ps16); None: 1
        hr_px     2 x (N,) long or None   top-left PIXEL of each crop when crops lie off the ps4
                                    grid (--jitter, --crop_px): a crop then covers part of
                                    several cells, and the pool runs per PIXEL over what covers
                                    it (wgt (B, Q, K, 16) in parts)
        hr_fp     the off-grid crops' footprint side in px (None: crop_px = 4; --crop_px 16:
                  one ps16 token read at its native 16 x 16 px)
        parts     also return (wgt (B, Q, K), candidate logits (B, Q, K, C, 16), the pooled
                  per-cell logits (B, Q, C, 16), the crop index of each crop slot (B, Q, M),
                  -1 empty (a ps8 crop fills 4 cells' slots), or None; the
                  first NS of K are the season tokens) -- for --joint's leave-one-out values
        """
        B, NS, h, w, D = lr_F.shape
        c, r = self.crop_px, self.lr_px // self.crop_px
        bp = self.base_px
        rs = bp // c                                         # ps4 cells per season-token side
        H, W = h * rs, w * rs
        Q, P = H * W, h * bp
        C = linear.out_features // self.lr_px ** 2
        dev = lr_F.device
        # -- season candidates: (B, NS, Q, C, c*c) logits of each season token over cell q
        s_log = linear(lr_F.float())
        if bp < self.lr_px:
            s_log = F.avg_pool2d(s_log.reshape(-1, C, self.lr_px, self.lr_px),
                                 self.lr_px // bp).reshape(B, NS, h, w, C * bp * bp)
        s_log = rearrange(s_log, "b s gh gw (k i j) -> b s k (gh i) (gw j)",
                          k=C, i=bp, j=bp)                                     # (B, NS, C, P, P)
        s_log = rearrange(s_log, "b s k (y i) (x j) -> b (y x) s k (i j)", i=c, j=c)  # (B,Q,NS,C,c*c)
        n_s = (F.one_hot(season.clamp(min=0), NS).float()
               * (present & (season >= 0)).float()[..., None]).sum(1)          # (B, NS) months
        s_a = self._score(lr_F.float()) + n_s.clamp(min=1).log()[:, :, None, None]
        s_a = s_a.masked_fill((n_s == 0)[:, :, None, None], float("-inf"))     # unused seasons
        if self.no_base:                                                       # crops only
            s_a = torch.full_like(s_a, float("-inf"))
        s_a = (s_a.repeat_interleave(rs, 2).repeat_interleave(rs, 3)          # (B, NS, H, W)
               .flatten(2).transpose(1, 2))                                    # (B, Q, NS)
        logs, acts = [s_log], [s_a]
        # -- crop candidates: one ENTRY per (crop, ps4 cell of its footprint), padded per
        #    cell to M
        b, _, y, x = hr_idx
        N = len(b)
        cover = None
        if N and hr_px is not None:        # off-grid ps4 crops: entries per overlapped cell
            fp = hr_fp or c                                     # footprint side, px
            maps = linear(hr.float()).view(N, C, self.lr_px, self.lr_px)
            maps = F.avg_pool2d(maps, self.lr_px // fp) if fp < self.lr_px else maps   # (N,C,fp,fp)
            c_a_tok = self._score(hr.float())
            if hr_logw is not None:
                c_a_tok = c_a_tok + hr_logw.float()
            py, px = hr_px
            ar_c, ar_n = torch.arange(c, device=dev), torch.arange(N, device=dev)
            e_log, e_cov, e_cell, e_crop = [], [], [], []
            for da in range(fp // c + 1):
                for db in range(fp // c + 1):
                    cy, cx = py // c + da, px // c + db
                    ui = cy[:, None] * c + ar_c[None] - py[:, None]            # (N, c) crop row
                    uj = cx[:, None] * c + ar_c[None] - px[:, None]
                    cov = (((ui >= 0) & (ui < fp))[:, :, None] & ((uj >= 0) & (uj < fp))[:, None, :]
                           & ((cy < H) & (cx < W) & (cy >= 0) & (cx >= 0))[:, None, None])
                    keep = cov.flatten(1).any(1)
                    g = maps[ar_n[:, None, None], :, ui.clamp(0, fp - 1)[:, :, None],
                             uj.clamp(0, fp - 1)[:, None, :]]                    # (N, c, c, C)
                    e_log.append(g.permute(0, 3, 1, 2).flatten(2)[keep])        # (n, C, c*c)
                    e_cov.append(cov.flatten(1)[keep])
                    e_cell.append((b * Q + cy * W + cx)[keep])       # (kept: inside the tile)
                    e_crop.append(ar_n[keep])
            crop, cov_e = torch.cat(e_log), torch.cat(e_cov)
            cell, ecrop = torch.cat(e_cell), torch.cat(e_crop)
            c_a = c_a_tok[ecrop]
            N = len(cell)
            order = torch.argsort(cell, stable=True)
            cell_s = cell[order]
            count = torch.bincount(cell_s, minlength=B * Q)
            M = int(count.max())
            slot = torch.arange(N, device=dev) - (torch.cumsum(count, 0) - count)[cell_s]
            idx = torch.full((B * Q, M), -1, dtype=torch.long, device=dev)
            idx[cell_s, slot] = order
            idx = idx.view(B, Q, M)
            ok = idx >= 0
            safe = idx.clamp(min=0)
            logs.append(crop[safe] * ok[..., None, None])
            acts.append(c_a[safe].masked_fill(~ok, float("-inf")))
            cover = torch.cat([torch.ones(B, Q, NS, c * c, dtype=torch.bool, device=dev),
                               cov_e[safe] & ok[..., None]], 2)                  # (B, Q, K, c*c)
        elif N:
            size = hr_size if hr_size is not None else torch.ones_like(b)
            maps = linear(hr.float()).view(N, C, self.lr_px, self.lr_px)       # (N, C, 16, 16)
            c_a_tok = self._score(hr.float())                                   # (N,)
            if hr_logw is not None:
                c_a_tok = c_a_tok + hr_logw.float()
            e_log, e_a, e_b, e_y, e_x, e_crop = [], [], [], [], [], []
            for sz in size.unique().tolist():                  # 1, 2, 4 cells per side
                sel = (size == sz).nonzero(as_tuple=True)[0]
                m = F.avg_pool2d(maps[sel], r // sz) if sz < r else maps[sel]   # (n, C, c*sz, c*sz)
                m = rearrange(m, "n k (i u) (j v) -> n (i j) k (u v)", u=c, v=c)  # (n, sz*sz, C, c*c)
                ii, jj = torch.meshgrid(torch.arange(sz, device=dev), torch.arange(sz, device=dev),
                                        indexing="ij")
                e_log.append(m.flatten(0, 1))
                e_a.append(c_a_tok[sel].repeat_interleave(sz * sz))
                e_b.append(b[sel].repeat_interleave(sz * sz))
                e_crop.append(sel.repeat_interleave(sz * sz))                   # entry -> crop
                e_y.append((y[sel][:, None] + ii.flatten()[None]).flatten())
                e_x.append((x[sel][:, None] + jj.flatten()[None]).flatten())
            crop, c_a = torch.cat(e_log), torch.cat(e_a)                       # (E, C, c*c), (E,)
            eb, ey, ex, ecrop = torch.cat(e_b), torch.cat(e_y), torch.cat(e_x), torch.cat(e_crop)
            N = len(eb)                                                         # entries
            cell = eb * Q + ey * W + ex                                         # (E,)
            order = torch.argsort(cell, stable=True)
            cell_s = cell[order]
            count = torch.bincount(cell_s, minlength=B * Q)
            M = int(count.max())
            slot = torch.arange(N, device=dev) - (torch.cumsum(count, 0) - count)[cell_s]
            idx = torch.full((B * Q, M), -1, dtype=torch.long, device=dev)
            idx[cell_s, slot] = order
            idx = idx.view(B, Q, M)
            ok = idx >= 0
            safe = idx.clamp(min=0)
            logs.append(crop[safe])                                             # (B, Q, M, C, c*c)
            acts.append(c_a[safe].masked_fill(~ok, float("-inf")))              # (B, Q, M)
        a = torch.cat(acts, 2)                                                  # (B, Q, K)
        cand = torch.cat(logs, 2)                                               # (B, Q, K, C, c*c)
        prob = self.mean and C > 1
        if prob:
            cand = cand.softmax(3)
        if cover is not None:               # per pixel, over the candidates covering it
            wgt = torch.softmax(a[..., None].masked_fill(~cover, float("-inf")), 2).nan_to_num(0.0)
            out = torch.einsum("bqkp,bqkcp->bqcp", wgt, cand)
        else:
            wgt = torch.softmax(a, 2).nan_to_num(0.0)
            out = torch.einsum("bqk,bqkcp->bqcp", wgt, cand)                    # (B, Q, C, c*c)
        if prob:                            # log of a distribution: cross_entropy reads it exactly
            out = out.clamp(min=1e-8).log()
        img = rearrange(out, "b (y x) k (i j) -> b k (y i) (x j)", y=H, x=W, i=c, j=c)
        if parts:                                   # crop index (not entry) per crop slot
            cid = torch.where(idx >= 0, ecrop[idx.clamp(min=0)], -1) if N else None
            return img, (wgt, cand, out, cid)
        return img


def native_attn_logits(linear: nn.Linear, score: nn.Module, lr_F: torch.Tensor,
                       season: torch.Tensor, present: torch.Tensor,
                       crop_tok: torch.Tensor | None = None, crop_idx: tuple | None = None,
                       crop_gate: torch.Tensor | None = None, lr_px: int = LR_PS,
                       crop_px: int = LOOKUP_PS) -> torch.Tensor:
    """--readout native_attn: native_logits with the plain time mean replaced by a temporal
    attention pool IN LOGIT SPACE -- fixes both the ps16 blur (native) and the dilution of a
    crop among ~12 months (attention), with overlapping tokens only.

    Per month t and ps4 cell p the one token there is the crop at (t, p) if any, else the ps16
    season token of t's season covering p. It gives
        logits(t, px in p)   its native head map (season: its 16 x 16 block's pixels in p;
                             crop: its 16 x 16 map area-downsampled to 4 x 4)
        s(t, p)              score(token) = Linear(LayerNorm(token)), a scalar
    and out(px in p) = sum_t softmax_t(s(t, p)) logits(t, px), over present months.
    A learned selector's gate blends a crop's logits toward the season's (as native_logits),
    not its score.

    Inputs as native_logits, plus score: nn.Module (D -> 1). Dense over months:
    (B, T, C, P, P) logits, P = h * lr_px = 64. -> (B, C, P, P)
    """
    B, NS, h, w, D = lr_F.shape
    T = season.shape[1]
    C = linear.out_features // lr_px ** 2
    r = lr_px // crop_px                                      # 4 ps4 cells per ps16 side
    dev = lr_F.device
    bidx = torch.arange(B, device=dev)[:, None]
    s_idx = season.clamp(min=0)
    # (B, NS, h, w, D) -> season logit maps (B, NS, C, P, P) and scores (B, NS, h, w)
    s_log = rearrange(linear(lr_F.float()), "b s gh gw (c i j) -> b s c (gh i) (gw j)",
                      c=C, i=lr_px, j=lr_px)
    month_log = s_log[bidx, s_idx].clone()                                    # (B, T, C, P, P)
    s_score = score(lr_F.float())[..., 0]                                     # (B, NS, h, w)
    month_score = (s_score[bidx, s_idx].repeat_interleave(r, 2)
                   .repeat_interleave(r, 3).clone())                          # (B, T, H, W)
    if crop_tok is not None and len(crop_tok):
        b, t, y, x = crop_idx
        N = len(b)
        crop = F.avg_pool2d(linear(crop_tok.float()).view(N, C, lr_px, lr_px), r)  # (N,C,4,4)
        ar = torch.arange(crop_px, device=dev)
        py = (y * crop_px)[:, None] + ar                                      # (N, 4) px rows
        px = (x * crop_px)[:, None] + ar
        cc = torch.arange(C, device=dev)
        e = (N, C, crop_px, crop_px)
        idx = (b[:, None, None, None].expand(e), t[:, None, None, None].expand(e),
               cc[None, :, None, None].expand(e), py[:, None, :, None].expand(e),
               px[:, None, None, :].expand(e))
        if crop_gate is not None:
            base = month_log[idx]                                             # (N, C, 4, 4)
            crop = base + crop_gate.float()[:, None, None, None] * (crop - base)
        month_log = month_log.index_put(idx, crop)
        month_score = month_score.index_put((b, t, y, x), score(crop_tok.float())[:, 0])
    month_score = month_score.masked_fill(~present[:, :, None, None], float("-inf"))
    wgt = torch.softmax(month_score, 1).nan_to_num(0.0)                       # (B, T, H, W)
    wgt = wgt.repeat_interleave(crop_px, 2).repeat_interleave(crop_px, 3)     # (B, T, P, P)
    return (wgt[:, :, None] * month_log).sum(1)                              # (B, C, P, P)



def native_logits(linear: nn.Linear, lr_F: torch.Tensor, season: torch.Tensor,
                  present: torch.Tensor, crop_tok: torch.Tensor | None = None,
                  crop_idx: tuple | None = None, crop_gate: torch.Tensor | None = None,
                  lr_px: int = LR_PS, crop_px: int = LOOKUP_PS) -> torch.Tensor:
    """--readout native: the time-mean pa2px head applied to every token at its OWN patch
    size, so a ps16 token's sub-patch layout ("water left, land right") reaches the pixels
    instead of being blurred by a bilinear upsample of the features.

    One linear head `linear`: D -> C * lr_px * lr_px (a ps16 token -> its 16 x 16 px block).
      season token   its 16 x 16 px logit block, as in the fixed ps16 fine-tune
      crop token     the SAME head gives a 16 x 16 map; area-downsampled (mean of each 4 x 4
                     px group) to the crop's own 4 x 4 px block. No new parameters.
    Per month, the logit map is its season's map with each of that month's crops overwriting
    its 4 x 4 px block; the output is the mean over present months. The head is linear, so this
    is the time mean done in logit space, and with no crops it equals the head on the
    time-mean features (= the fixed ps16 series fine-tune on the season copies).

    Computed sparsely (never builds the (B, T, C, P, P) month stack):
        mean_t logit_t = sum_s n_s / n * season_map_s  +  sum_crops (crop - base) / n
    n_s = present months in season slot s, n = present months, base = the season map's
    4 x 4 px block under the crop.

    Inputs (shape letters: B samples, NS season slots, h = w = 4 ps16 cells, T months,
    D features, C outputs, N crops, P = h * lr_px px output side):
        lr_F       (B, NS, h, w, D)  final season tokens
        season     (B, T) long       season slot of each month (clamped at 0, as the bilinear
                                     readout does)
        present    (B, T) bool       month is real (not batch padding)
        crop_tok   (N, D)            final crop tokens
        crop_idx   4 x (N,) long     (sample b, month t, ps4 row y, ps4 col x)
        crop_gate  (N,) or None      sigmoid(selector score): a crop moves its block from the
                                     season's by this fraction (as the bilinear readout gates)
    -> (B, C, P, P) logits
    """
    B, NS, h, w, D = lr_F.shape
    C = linear.out_features // lr_px ** 2
    per_px = present.float()                                                  # (B, T)
    n = per_px.sum(1).clamp(min=1)                                            # (B,)
    n_s = (F.one_hot(season.clamp(min=0), NS).float() * per_px[..., None]).sum(1)   # (B, NS)
    feat = (n_s[:, :, None, None, None] * lr_F.float()).sum(1) / n[:, None, None, None]
    # (B, h, w, D) -> (B, h, w, C * lr_px^2) -> (B, C, h * lr_px, w * lr_px)
    logits = rearrange(linear(feat), "b gh gw (c i j) -> b c (gh i) (gw j)",
                       c=C, i=lr_px, j=lr_px).contiguous()
    if crop_tok is None or not len(crop_tok):
        return logits
    b, t, y, x = crop_idx
    N, r = len(b), lr_px // crop_px                           # r = 4 ps4 cells per ps16 side
    crop = linear(crop_tok.float()).view(N, C, lr_px, lr_px)
    crop = F.avg_pool2d(crop, r)                                              # (N, C, 4, 4)
    tok = lr_F.float()[b, season[b, t].clamp(min=0), y // r, x // r]          # (N, D)
    base = linear(tok).view(N, C, lr_px, lr_px)
    iy, ix = (y % r) * crop_px, (x % r) * crop_px             # the crop's block inside it
    ar = torch.arange(crop_px, device=lr_F.device)
    rows = (iy[:, None] + ar)[:, :, None].expand(N, crop_px, crop_px)         # (N, 4, 4)
    cols = (ix[:, None] + ar)[:, None, :].expand(N, crop_px, crop_px)
    base = base[torch.arange(N, device=lr_F.device)[:, None, None], :, rows, cols]  # (N,4,4,C)
    delta = crop - base.permute(0, 3, 1, 2)                                   # (N, C, 4, 4)
    if crop_gate is not None:
        delta = crop_gate.float()[:, None, None, None] * delta
    delta = delta / n[b][:, None, None, None]
    # scatter-add each crop's change into its 4 x 4 px block of its sample's map
    py = (y * crop_px)[:, None] + ar                                          # (N, 4)
    px = (x * crop_px)[:, None] + ar
    cc = torch.arange(C, device=lr_F.device)
    idx = (b[:, None, None, None].expand(N, C, crop_px, crop_px),
           cc[None, :, None, None].expand(N, C, crop_px, crop_px),
           py[:, None, :, None].expand(N, C, crop_px, crop_px),
           px[:, None, None, :].expand(N, C, crop_px, crop_px))
    return logits.index_put(idx, delta, accumulate=True)


def arm_label(selector: str, k: int, L: int, G: int, T: int, unit: str = "slot",
              pick: str = "topk", head: str = "mean", reg: str = "", crop_px: int = 4,
              trained: bool = True) -> str:
    """E.g. "S4ps16 + 512xps4-1mo inject@L6 @ random, full FT" (GEOID "S1ps16 + 42locx2D-ps4
    ..."): G seasons at ps16, k single-date ps<crop_px> crops injected before block L.
    trained=False (--epochs 0): "no training (zero-shot)" instead of "full FT"."""
    base = f"S{G}ps16"
    ft = "full FT" if trained else "no training (zero-shot)"
    tail = ", attn-pool head" if head == "attn" else ""
    if selector == "none":
        return f"{base}, no crops, {ft}{tail}"
    d = "M" if T > 2 else "D"
    c = f"ps{crop_px}"
    look = f"{k}x{c}-1{d.lower()}o" if unit == "slot" else f"{k // T}locx{T}{d}-{c}"
    selector += f" ({reg})" if reg else ""
    how = {"topk": selector, "random": f"random pick, {selector} gate",
           "per_step": f"{selector}, top-k per {'month' if T > 2 else 'date'}"}[pick]
    return f"{base} + {look} inject@L{L} @ {how}, {ft}{tail}"


def location_grid_scores(B: int, H: int, n: int, crop_px: int, random_phase: bool,
                         device) -> torch.Tensor:
    """(B, H, H) location scores whose top n are a regular lattice of crop windows
    (--selector grid --select_unit location): the tile's S x S non-overlapping crop_px windows
    (S = TILE / crop_px), point (i, j) kept when (i + c j + phase) mod m == 0, m = S^2 / n,
    c = round(sqrt(m)) -- n = S^2 / 2: checkerboard; / 4: staggered; / 8 (S = 8): one per row
    and column. A point marks the ps4 cell under its window's centre (a crop is cut centred on
    the picked cell, so a ps8 window lands 2 px off the 8 px lattice). Ties broken at random."""
    S = TILE // crop_px
    m = max(1, round(S * S / max(n, 1)))
    c = max(1, round(m ** 0.5))
    i, j = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device),
                          indexing="ij")
    phase = (torch.randint(0, m, (B, 1, 1), device=device) if random_phase
             else torch.zeros(B, 1, 1, dtype=torch.long, device=device))
    on = ((i + c * j)[None] + phase) % m == 0                                  # (B, S, S)
    scores = torch.rand(B, H, H, device=device) * 1e-3
    cy = (torch.arange(S, device=device) * crop_px + crop_px // 2) // (TILE // H)
    scores[:, cy[:, None], cy[None, :]] += on.float()
    return scores


def grid_scores(present: torch.Tensor, H: int, k: int, mode: str,
                random_phase: bool) -> torch.Tensor:
    """Scores whose top k present slots form a regular lattice -- the coverage-maximising,
    content-blind baseline next to uniform random picks (--selector grid).

    present (B, T) bool: months that exist for the sample.  H: ps4 cells per tile side.
    k: crops per sample.  Returns (B, T, H, H) float; feed it to select(scores, present, k).

    Per sample the lattice modulus is m = round(1 / share), share = k / (present months * H * H)
    (1/6 -> 6, 1/3 -> 3; a share above 1/2 gives the m = 2 checkerboard plus random fill).
    Slot (j, y, x), j = the month's rank among the present months, sits in coset

        stagger:  (j + y + c x + phase) mod m     c = round(sqrt(m))
        spatial:  (    y + c x + phase) mod m

    and coset 0 is picked first; any remainder (m does not divide the slot count, or the share
    is not 1 / integer) is filled at random from coset 1, then 2, ...
      stagger  every cell is cropped in every m-th present month and every month has the same
               spatial lattice, shifted by one coset per month: no cell is left without a crop.
      spatial  the same 1-in-m cells in EVERY month ("a crop every i pixels"); the other cells
               never get one.
    random_phase: a fresh lattice offset per sample (training), else offset 0 (evaluation).
    """
    B, T = present.shape
    device = present.device
    n_present = present.sum(1).clamp(min=1)                                   # (B,)
    share = k / (n_present * H * H).float()                                   # (B,)
    modulus = (1.0 / share.clamp(min=1e-6)).round().long().clamp(min=2)       # (B,)
    x_step = modulus.float().sqrt().round().long().clamp(min=1)               # (B,)
    month_rank = (present.long().cumsum(1) - 1).clamp(min=0)                  # (B, T)
    yy = torch.arange(H, device=device)[None, None, :, None]                  # (1, 1, H, 1)
    xx = torch.arange(H, device=device)[None, None, None, :]                  # (1, 1, 1, H)
    phase = (torch.randint(0, 1 << 16, (B,), device=device) if random_phase
             else torch.zeros(B, dtype=torch.long, device=device))            # (B,)
    index = yy + x_step[:, None, None, None] * xx + phase[:, None, None, None]   # (B, 1, H, H)
    if mode == "stagger":
        index = index + month_rank[:, :, None, None]                          # (B, T, H, H)
    coset = (index % modulus[:, None, None, None]).expand(B, T, H, H)         # (B, T, H, H)
    # coset 0 scores in (-0.5, 0], coset 1 in (-1.5, -1], ...: random order inside a coset
    return -coset.float() - 0.5 * torch.rand(B, T, H, H, device=device)


def shifted_positions(P: int, S: int) -> list[int]:
    """Top-left pixel (one axis) of the P px windows centred on every cell of the HALF-SHIFTED
    grid of stride S: centres 0, S, ... 64 -> top-left S k - P / 2. The first and last hang
    out of the 64 px tile by half and are zero-padded.  P = S = 16: -8, 8, 24, 40, 56.
      S >= P  no window lies on the P px patch grid (P / 2 mod P).
      S < P   every (P / S)-th value per axis IS on the patch grid; a window on it in BOTH axes
              repeats an on-grid token and is left out by the caller (shifted_count).
    Layouts before 2026-10-13 (run tag `_shift<S>` with S != P): S > P used P / 2 + S k (no
    window at the tile edge), S < P used S / 2 + S k (one more window a side, none on the
    grid). The current layout is tagged `_shift<S>c` when S != P; S = P never changed."""
    return [S * k - P // 2 for k in range(TILE // S + 1)]


def shifted_count(P: int, S: int) -> int:
    """Shifted-grid windows per month: all of shifted_positions squared, less those on the
    patch grid in both axes."""
    pos = shifted_positions(P, S)
    return len(pos) ** 2 - sum(p % P == 0 for p in pos) ** 2


def pad_sample(sample, pad: int):
    """The sample with `pad` px around every frame: pixel values 0 (the normalized mean), mask
    fields replicated from the edge (a missing month stays missing in its padding)."""
    def one(k, v):
        if not torch.is_tensor(v) or v.ndim != 5:
            return v
        if "mask" in k:
            idx = torch.arange(-pad, v.shape[1] + pad, device=v.device).clamp(0, v.shape[1] - 1)
            return v[:, idx][:, :, idx]
        return F.pad(v, (0, 0, 0, 0, pad, pad, pad, pad))

    return _map(sample, one)


def shared_grid_order(present: torch.Tensor, cnt: torch.Tensor, mode: str,
                      random_phase: bool) -> torch.Tensor:
    """One ordered list of crop anchors per sample, the SAME for every crop size (--crop_free
    --selector grid --grid_mode c2f | shared): a model with n crops takes the first n.

    Anchors are (month t, ps4 cell y, x) flattened as t * 256 + y * 16 + x; absent months sort
    last.  present (B, T) bool, cnt (B,) crops per sample (sets the stagger modulus).
      shared  the staggered ps4 lattice (grid_scores): spread over the tile from the first crop
      c2f     coarse to fine: the ps16 grid corners (16 a month), then the other ps8 grid
              corners (48), then the other ps4 cells (192) -- each stage staggered over the
              months on its own grid and filled before the next starts
    """
    B, T = present.shape
    dev = present.device
    H = TILE // LOOKUP_PS
    absent = ~present[:, :, None, None]
    if mode == "shared":
        sc = grid_scores(present, H, cnt, "stagger", random_phase)
        return sc.masked_fill(absent, float("-inf")).flatten(1).argsort(1, descending=True)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(H, device=dev), indexing="ij")
    stage = 2 - ((yy % 2 == 0) & (xx % 2 == 0)).long() - ((yy % 4 == 0) & (xx % 4 == 0)).long()
    N = T * H * H
    key = torch.full((B, N), 4 * N, dtype=torch.long, device=dev)
    used = torch.zeros(B, dtype=torch.long, device=dev)
    for s_, (Hs, n_new) in enumerate(((H // 4, 16), (H // 2, 48), (H, 192))):
        sc = grid_scores(present, Hs, (cnt - used).clamp(min=0), "stagger", random_phase)
        full = sc.repeat_interleave(H // Hs, 2).repeat_interleave(H // Hs, 3)
        member = (stage == s_)[None, None] & ~absent                          # (B, T, H, H)
        rank = full.masked_fill(~member, float("-inf")).flatten(1).argsort(
            1, descending=True).argsort(1)
        key = torch.where(member.flatten(1), s_ * N + rank, key)
        used = used + present.sum(1) * n_new
    return key.argsort(1)


def within(t_start, t_end, s_start, s_end, idx):
    """Fractional position of each step's midpoint inside the step it belongs to (idx, -1 =
    none), in [-0.5, 0.5): e.g. a month inside its season, a date inside its month.
    t_* (B, T), s_* (B, S), idx (B, T) -> (B, T) float (0 where idx < 0)."""
    i = idx.clamp(min=0)
    lo, hi = s_start.gather(1, i).double(), s_end.gather(1, i).double()
    mid = (t_start + t_end).double() / 2
    f = ((mid - lo) / (hi - lo + 1).clamp(min=1) - 0.5).float()
    return torch.where(idx >= 0, f, torch.zeros_like(f))


class YearGroup(TemporalPolicy):
    """--lr_steps year: ALL steps of the sample reduced into one (GroupSteps over the whole
    axis: PASTIS / BioMassters 12 months, YieldSAT 24)."""
    key = "groupall-mean"

    def _apply(self, md, spec):
        return GroupSteps(md.num_frames)._apply(md, spec)


class Collate:
    """In the workers: months and seasons packed for OlmoEarth, each month's season, which
    months count, and the task target. extra["frac"] (B, T): each step's position inside its
    season (--frac_slot). With `dates`: extra["dates"][m] = (m's raw dates packed for OlmoEarth
    (T_m steps), date valid (B, T_m), the month each date falls in (B, T_m; -1 = outside the
    month grid), its position inside that month's season (B, T_m))."""

    def __init__(self, task: Task, dates: bool = False, teach_n: int = 0,
                 mid_months: bool = False, any_modality: bool = False):
        m, imp = task.olmoearth.modalities, task.olmoearth.impute
        self.any_modality = any_modality    # --lr_steps raw: a date holds ONE modality's frame
        if teach_n:                 # WithDates(teach_n) kept only the teacher's modality
            m = {k: v for k, v in m.items() if k == teach_modality(task)}
        m_all = task.olmoearth.modalities
        self.task, self.mods, self.ref = task, list(m_all), next(iter(m_all))
        # mid_months (--doy): a month step is stamped with its middle day, not its first
        self.month_in = OlmoEarthInput(m_all, imp, timestamp="mid" if mid_months else "start")
        self.season_in = OlmoEarthInput(m_all, imp, timestamp="mid")
        self.date_in = ({k: OlmoEarthInput({k: v}, {k: imp[k]} if k in imp else {},
                                           timestamp="mid") for k, v in m.items()}
                        if dates else None)

    def __call__(self, items):
        months = mtmm_collate([it[0] for it in items])
        seasons = mtmm_collate([it[1] for it in items])
        ms, _ = self.month_in.to_olmoearth(months)
        ss, s_valid = self.season_in.to_olmoearth(seasons)
        ref, sref = months.modalities[self.ref], seasons.modalities[self.ref]
        season = season_index(sref.t_start, sref.t_end, ref.t_start, ref.t_end)  # (B, T)
        every = torch.stack([months.modalities[m].frame_valid for m in self.mods])
        every = every.any(0) if self.any_modality else every.all(0)
        present = every & (season >= 0) & s_valid.gather(1, season.clamp(min=0))
        pairs = None
        if self.task.reduction.startswith("pair_"):
            pairs = consecutive_pairs(present, None)      # consecutive dates, any orbit
        target = self.task.target(months)
        extra = {"frac": within(ref.t_start, ref.t_end, sref.t_start, sref.t_end, season),
                 "target": target}                        # read only by --selector oracle
        # change detection (series with a known event): the steps the dense self-teacher must
        # always include (--teach_event) -- the last present date BEFORE the event and the
        # present date nearest the labelled post-event scene (at or after the event);
        # extra["event_steps"] (B, 2) long, -1 where there is none
        if all("event_day" in mt for mt in months.meta):
            ev = torch.full((len(items), 2), -1, dtype=torch.long)
            for i, mt in enumerate(months.meta):
                day = ref.t_start[i]
                ok = present[i]
                before = (ok & (day < mt["event_day"])).nonzero()[:, 0]
                after = (ok & (day >= mt["event_day"])).nonzero()[:, 0]
                if len(before):
                    ev[i, 0] = before[-1]
                if len(after):
                    ev[i, 1] = after[(day[after] - mt["post_day"]).abs().argmin()]
            extra["event_steps"] = ev
        if self.date_in is not None:
            raw = mtmm_collate([it[2] for it in items])
            extra["dates"] = {}
            for m, inp in self.date_in.items():
                dm = raw.modalities[m]
                d_sample, d_valid = inp.to_olmoearth(raw)
                month = season_index(ref.t_start, ref.t_end, dm.t_start, dm.t_end)
                d_season = torch.where(month >= 0, season.gather(1, month.clamp(min=0)),
                                       torch.full_like(month, -1))
                frac = within(dm.t_start, dm.t_end, sref.t_start, sref.t_end, d_season)
                extra["dates"][m] = (d_sample, d_valid & (month >= 0), month, frac)
        return ms, ss, season, present, pairs, extra, target


def _mlp(d_in, hidden):
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))


def gumbel(x):
    return x - torch.log(-torch.log(torch.rand_like(x).clamp(1e-6, 1 - 1e-6)))


class DatePredictor(nn.Module):
    """--date_joint: the value predictor of ONE modality's date crops. Candidate (date d, ps4
    cell):  q = Linear([e_d, e_d - mean e over the dates of its season, time encoding of d]),
    e_d = that modality's ps4 patch embedding of the date;  c = Linear(the season tokens after
    blocks [0, L) at the covering ps16 cell, all seasons);  h = GELU(q + c);  the dates of one
    cell see each other (self-attention);  score = MLP(h)."""

    def __init__(self, dim: int, seasons: int, hidden: int = 128):
        super().__init__()
        self.q = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, hidden))
        self.ctx = nn.Sequential(nn.LayerNorm(seasons * dim), nn.Linear(seasons * dim, hidden))
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.n1 = nn.LayerNorm(hidden)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, e, e_bar, tenc, F0, valid):
        """e, e_bar (B, Td, H, W, D); tenc (B, Td, D); F0 (B, NS, h, w, D); valid (B, Td)
        -> scores (B, Td, H, W)."""
        B, Td, H, W, D = e.shape
        q = self.q(torch.cat([e, e - e_bar, tenc[:, :, None, None].expand(B, Td, H, W, D)], -1))
        f = H // F0.shape[2]
        c = self.ctx(F0.permute(0, 2, 3, 1, 4).flatten(3))                      # (B, h, w, Hd)
        c = c.repeat_interleave(f, 1).repeat_interleave(f, 2)
        hc = F.gelu(q + c[:, None]).permute(0, 2, 3, 1, 4).reshape(B * H * W, Td, -1)
        pad = ~valid[:, None].expand(B, H * W, Td).reshape(B * H * W, Td)
        pad = pad & ~pad.all(1, keepdim=True)        # a sample with no date: attend to all (unused)
        n = self.n1(hc)
        a, _ = self.attn(n, n, n, key_padding_mask=pad, need_weights=False)
        h = (hc + a).reshape(B, H, W, Td, -1).permute(0, 3, 1, 2, 4)
        return self.out(h)[..., 0]


def zscore_by(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """v (N,) standardized within each group b (N,) long."""
    n = int(b.max()) + 1
    cnt = torch.bincount(b, minlength=n).clamp(min=1).float()
    mu = torch.zeros(n, device=v.device).index_add_(0, b, v) / cnt
    var = torch.zeros(n, device=v.device).index_add_(0, b, (v - mu[b]) ** 2) / cnt
    return (v - mu[b]) * torch.rsqrt(var[b] + 1e-6)


class TemporalAttnPool(nn.Module):
    """(B, T, H, W, D), present (B, T) -> (B, H, W, D): per location, softmax over present
    steps of a learned per-token score."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x, present):
        s = self.score(x)[..., 0].float()                                     # (B,T,H,W)
        s = s.masked_fill(~present[:, :, None, None], float("-inf"))
        w = torch.softmax(s, 1).nan_to_num(0.0)
        return (w[..., None] * x.float()).sum(1)


class PerturbedTopK(torch.autograd.Function):
    """DPS's differentiable top-k (Cordonnier et al. 2021, after Berthet et al. 2020): each
    candidate's probability of being in the top k when the scores are perturbed by Gaussian
    noise, with the perturbed-optimizer gradient.

        forward   s (B, N) scores, -inf = not a candidate; k top slots per sample
                  -> p (B, N) = mean over `draws` noise draws Z of 1[i in top-k(s + sigma Z)]
        backward  dp_i / ds_j = E[1[i in top-k] Z_j] / sigma, so for an upstream gradient g
                  ds = E[(g . y - baseline) Z] / sigma, y the draw's 0/1 top-k indicator; the
                  baseline (the mean of g . y over draws) lowers the variance (E[Z] = 0).
    The gradient reaches EVERY candidate's score, also the ones not picked."""

    @staticmethod
    def forward(ctx, s, k: int, sigma: float, draws: int):
        cand = torch.isfinite(s)                                                # (B, N)
        z = torch.randn(draws, *s.shape, device=s.device)                       # (S, B, N)
        noisy = (s[None] + sigma * z).masked_fill(~cand[None], float("-inf"))
        y = torch.zeros_like(noisy, dtype=torch.bool)
        y.scatter_(2, noisy.topk(min(k, s.shape[1]), 2).indices, True)
        y &= cand[None]                                                         # (S, B, N)
        ctx.save_for_backward(y, z, cand)
        ctx.sigma = sigma
        return y.float().mean(0)

    @staticmethod
    def backward(ctx, g):
        y, z, cand = ctx.saved_tensors
        gy = (g[None].float() * y).sum(2, keepdim=True)                         # (S, B, 1)
        gy = gy - gy.mean(0, keepdim=True)
        ds = (gy * z).mean(0) / ctx.sigma                                       # (B, N)
        return ds * cand, None, None, None


def per_step_topk(scores: torch.Tensor, present: torch.Tensor, k: int) -> torch.Tensor:
    """(B, T, H, W) -> bool mask: the best k // (present steps) slots of every present step."""
    B, T = present.shape
    k_step = (k // present.sum(1).clamp(min=1))[:, None, None]                # (B,1,1)
    flat = scores.flatten(2)                                                  # (B,T,HW)
    rank = flat.argsort(2, descending=True).argsort(2)
    return ((rank < k_step) & present[:, :, None]).view_as(scores)


class InjectFT(nn.Module):
    def __init__(self, task: Task, model_size: str, inject_layer: int, selector: str,
                 k_frac: float, select_unit: str = "slot", selector_hidden: int = 64,
                 pick: str = "topk", head: str = "mean", score_norm: bool = False,
                 sel_temporal: bool = False, readout: str = "bilinear", lr_patch: int = 16,
                 frac_slot: bool = False, date_level: str = "none", date_per_month: float = 2.0,
                 month_crops: bool = True, date_selector: str = "random", date_gate: str = "month",
                 res_mix: tuple | None = None, joint: bool = False, lr_tile: int = TILE,
                 joint_res: bool = False, pair_match_random: bool = False,
                 self_teach: int = 0, pos: str = "corner", doy: bool = False,
                 attn: str = "joint", head_attn: str = "alone", joint_feat: str = "prelook",
                 joint_target: str = "loo", joint_in: str = "onehot", joint_arch: str = "mlp",
                 joint_embed: str = "shared", jitter: bool = False, date_joint: str = "none",
                 crop_px: int = LOOKUP_PS, pretrained: bool = True, apt_pool: bool = False,
                 apt_merge_mse: float = 0.01, apt_time: str = "sum", apt_res: bool = False,
                 apt_octree: bool = False, apt_tau: float = 0.0, apt_octree_pool: str = "input",
                 apt_octree_space_only: bool = False):
        super().__init__()
        # --joint_feat: what the --joint value predictor reads of the low-res sequence.
        #   prelook  a separate crop-free pass of the seasons through all 12 blocks: their
        #            tokens, their native logits, + each candidate's pixel variance
        #   lrL      the season tokens after blocks [0, L) of THIS pass (already computed: no
        #            extra pass), no logits, no pixel variance
        if joint_feat not in ("prelook", "lrL"):
            raise ValueError(f"joint_feat {joint_feat!r} not in ('prelook', 'lrL')")
        if joint_feat == "lrL" and joint_res:
            raise ValueError("--joint_feat lrL: no --joint_res")
        self.joint_feat = joint_feat
        # Round 33 (2026-10-04), each a single change to the layer-L value predictor:
        #   joint_target gate   the target is the FULL-PATH first-order value of a crop: its
        #                       tokens enter blocks [L, n) times a gate fixed at 1, and
        #                       -dLoss/dgate ~ loss(without) - loss(with) through every later
        #                       layer (the leave-one-out value only removes it from the readout)
        #   joint_in enc        candidate = (e - season mean e) + its step's time encoding, and
        #                       the season tokens; no raw e, no one-hots
        #   joint_arch xattn    CrossAttnSelector: the candidate queries the season tokens
        #        xenc[_t[s]]    EncCrossSelector: the same, on candidates carrying OlmoEarth's own
        #                       crop encodings instead of one-hots; _t + attention over the
        #                       months of a cell, _ts + attention over the cells of a month
        #   joint_embed own     e from the selector's OWN copy of the patch embedding, trained
        #                       by the selector's loss
        #   joint_target drop   the target is a TRUE removal through every layer: a second,
        #                       no-grad pass of blocks [L, n) + readout without a random half of
        #                       the crops; a dropped crop's value = the loss rise over its own
        #                       cell (shared equally by the dropped crops of that cell)
        if joint_target not in ("loo", "gate", "drop") or joint_in not in ("onehot", "enc") \
                or joint_arch not in JOINT_ARCH or joint_embed not in ("shared", "own"):
            raise ValueError("joint_target loo|gate|drop, joint_in onehot|enc, joint_arch mlp|xattn, "
                             "joint_embed shared|own")
        changed = (joint_target != "loo" or joint_in != "onehot" or joint_arch != "mlp"
                   or joint_embed != "shared")
        pair_drop = (joint and select_unit == "pair" and joint_target == "drop"
                     and readout in ("tsplat", "tquery") and lr_tile == TILE and not self_teach
                     and joint_in == "onehot" and joint_arch == "mlp" and joint_embed == "shared")
        if changed and not pair_drop and not (
                joint and joint_feat == "lrL" and select_unit == "slot"
                and not joint_res and not self_teach):
            raise ValueError("--joint_target / _in / _arch / _embed: the layer-L slot selector "
                             "(--joint --joint_feat lrL), no --joint_res / --self_teach")
        if joint_arch != "mlp" and joint_in != "onehot":
            raise ValueError("--joint_arch xattn has its own inputs: no --joint_in")
        self.joint_target, self.joint_in, self.joint_arch = joint_target, joint_in, joint_arch
        self._gate = self._gate_idx = None
        # --jitter: every ps4 month crop is cut 0..3 px off the ps4 grid (uniform, -2..+1 px
        # per axis, kept inside the tile), train and test
        # --crop_px 16 (8): the SAME picks (month, ps4 cell), but each crop is the 16 x 16 px
        # window centred on the cell (kept inside the tile), embedded as ONE ps16 token and read
        # out over its own 16 x 16 px -- extra single-month tokens at the low-res patch size,
        # off the ps16 grid: separates "more tokens" from "finer patches"
        if crop_px not in (4, 8, 16):
            raise ValueError("crop_px 4 | 8 | 16")
        self.crop_px = crop_px
        if crop_px != LOOKUP_PS and (jitter or date_joint != "none"):
            raise ValueError("--crop_px: no --jitter / --date_joint")
        # Free crops (2026-10-05; set by main: crop_free, n_crops, vary_crops, n_eval): every
        # crop is a crop_px window of ONE month at a uniformly random PIXEL position (top-left
        # anywhere in the tile -- no grid of any size) and a uniformly random present month.
        #   n_crops     crops per sample (default: k = k_frac T 256)
        #   vary_crops  training draws each sample's count uniformly from {0 .. n_crops}; the
        #               one model is then tested at every count (n_eval); validation draws the
        #               counts too, under a fixed seed
        #   crop_counts with vary_crops: the counts are drawn uniformly from this SET instead
        #               (e.g. 0,64,128,...: zero crops as often as any other; one value: a
        #               model dedicated to that count, still tested at every count below it)
        self.crop_free, self.n_crops, self.vary_crops, self.n_eval = False, 0, False, None
        self.crop_counts = None
        self._cov = [0.0, 0.0, 0]       # eval: sums of space-time / spatial crop coverage, samples
        self.free_pick = None           # crop_count_sweep: "random" / "grid" whatever was trained
        jitter_arg, jitter = jitter, jitter or crop_px != LOOKUP_PS      # same constraints
        if jitter and joint_target == "drop":
            raise ValueError("--joint_target drop credits a crop's own ps4 cell: no --jitter")
        if jitter and not (readout in SET and select_unit in ("slot", "location")
                           and date_level == "none" and res_mix is None and not joint_res
                           and not self_teach):
            raise ValueError("--jitter / --crop_px: --readout native_set | native_mean, slot or "
                             "location unit, ps4 month crops only")
        self.jitter = jitter_arg
        # --date_joint (2026-10-04): the value-predictor selector at the DATE level, one
        # modality per crop. Candidates (modality, raw date, ps4 cell); a crop costs its
        # modality's tokens (bandsets), and the budget is the TOKENS that k ps4 month crops (all
        # modalities) cost. eps of the budget is spent at random, the rest on the best
        # predicted value PER TOKEN (DatePredictor; target = leave-one-out value / tokens).
        #   only   date crops anywhere, no month crops
        #   hier   half the budget on month crops picked by the month predictor (as --joint),
        #          the other half on date crops INSIDE the picked (month, cell)s
        if date_joint not in ("none", "only", "hier"):
            raise ValueError(f"date_joint {date_joint!r} not in none|only|hier")
        if date_joint != "none" and not (
                joint and joint_feat == "lrL" and readout == "native_set" and select_unit == "slot"
                and date_level == "none" and not changed and not jitter and not joint_res
                and not self_teach and res_mix is None):
            raise ValueError("--date_joint: the plain layer-L slot selector (--joint --joint_feat "
                             "lrL --readout native_set), no other date level / selector variant")
        self.date_joint = date_joint
        self._date_sets = None
        # --self_teach n: see _self_teach. teach_loss = {'task': ..., 'recon': ...} after a
        # training forward; task_loss_fn is set by the training script
        self.self_teach, self.teach_loss, self.task_loss_fn = self_teach, None, None
        self.teach_event = False    # --teach_event: the teacher always sees pre + post event
        if self_teach and readout not in ("tsplat", "tquery", "native_set"):
            raise ValueError("--self_teach: --readout tsplat / tquery (the splat is also the "
                             "task readout) or native_set (logit-space task readout + a "
                             "separate reconstruction splat)")
        # --self_teach + native_set: the TASK is read in logit space (native_set) from the same
        # mixed-resolution tokens; this time-only Gaussian splat exists only to reconstruct the
        # teacher's dense features (and to value the crops for --joint)
        self.recon_splat = None             # built below, once the feature size is known
        # --teach_value (set by the training script): what the --joint selector regresses --
        # "recon" the leave-one-out reconstruction gain, "label" the leave-one-out task-loss
        # gain (native_set only), "both" their mean (each z-scored per sample)
        self.teach_value = "recon"
        # --teach_detach (set by the training script): which self-teacher gradients reach the
        # ENCODER. "none": both (reconstruction through the student tokens, and the teacher's
        # task loss through its dense pass). "recon": the reconstruction loss trains only the
        # splat (and gives the selector its values); the teacher's task loss still trains the
        # encoder. "all": neither -- the teacher is a no-grad pass with no task loss.
        self.teach_detach = "none"
        # --pair_match_random: joint pair picks encode as many date crops as a random draw
        self.pair_match_random = pair_match_random
        # --lr_tile: the low-res sequence per lr_tile x lr_tile px tile (psXtsY, Y < 64); a
        # reference arm only (no crops, bilinear readout)
        if lr_tile != TILE and (selector != "none" or readout != "bilinear"):
            raise ValueError("--lr_tile: a no-crop reference (--selector none, bilinear readout)")
        self.lr_tile = lr_tile
        # --res_mix: shares of the token budget spent at ps16 / ps8 / ps4 (random crops of
        # single months at each patch size; one token each, so the budget is in tokens)
        if res_mix is not None:
            if len(res_mix) != 3 or min(res_mix) < 0 or sum(res_mix) <= 0:
                raise ValueError(f"res_mix {res_mix}: three non-negative shares (ps16, ps8, ps4)")
            if (readout != "native_set" or selector != "random" or date_level != "none"
                    or select_unit != "slot"):
                raise ValueError("--res_mix: --readout native_set, --selector random, slot unit, "
                                 "no date level")
            tot = float(sum(res_mix))
            res_mix = {16: res_mix[0] / tot, 8: res_mix[1] / tot, 4: res_mix[2] / tot}
        self.res_mix = res_mix
        # --joint: a learned value predictor picks (1 - eps) of the crops, eps stay random
        self.joint_pairs = joint and select_unit == "pair"
        if joint and not ((readout == "native_set" and select_unit == "slot") or self_teach
                          or (readout in ("tsplat", "tquery") and select_unit == "pair")):
            raise ValueError("--joint: --readout native_set + slot unit (time-mean tasks), or "
                             "--readout tsplat / tquery + pair unit (series pair heads)")
        if joint and (selector != "random" or date_level != "none" or res_mix is not None):
            raise ValueError("--joint: --selector random, no date level / res_mix")
        self.joint_on, self.joint_eps, self.joint_loss = joint, 1.0, None
        # --joint_grad: how the predictor learns. value = MSE to each pick's leave-one-out
        # value (_joint_values); dps = the task loss through a perturbed top-k (PerturbedTopK):
        # each selector pick's crop gets log-weight p - stopgrad(p) in the readout (forward
        # unchanged, weight 1), p its top-k probability; both = the two summed. Set by main.
        self.joint_grad, self.dps_sigma, self.dps_draws = "value", 0.5, 64
        self.drop_frac, self.drop_passes = 0.5, 1      # --joint_target drop; set by main
        self._dps_P = None          # (B, T, H, W) top-k probability at the selector's picks
        if joint_res and not joint:
            raise ValueError("--joint_res extends --joint")
        self.joint_res = joint_res          # the joint selector also chooses ps8 crops
        self._res_masks = None
        self.last_ps8_share = None
        self._apt_groups = None      # --selector apt --apt_pool: {(b,t,y,x): member months}
        if date_level not in DATE_LEVELS:
            raise ValueError(f"date_level {date_level!r} not in {DATE_LEVELS}")
        if date_level != "none" and readout not in SPARSE:
            raise ValueError("the date level needs --readout cellattn (no month grid for dates)")
        if not month_crops and date_level != "nested":
            raise ValueError("--no_month_crops: the month picks only route nested date crops")
        self.frac_slot, self.date_level = frac_slot, date_level
        self.date_per_month, self.month_crops = date_per_month, month_crops

        m, imp = task.olmoearth.modalities, task.olmoearth.impute
        # pretrained=False (--no_pretrain): the same network from its random initialization
        oe = OlmoEarthEncoder(MODEL_SIZE_TO_ID[model_size], OlmoEarthInput(m, imp),
                              LR_PS, TILE, name=model_size, pretrained=pretrained)
        self.mix = MixedResOE(oe.encoder, inject_layer, pos, lr_patch, doy, attn, head_attn)
        self.lr_ps = lr_patch
        if lr_patch not in (4, 8, 16):
            raise ValueError("lr_patch must be 16, 8 or 4")
        D = oe.encoder.embedding_size
        # date crops are single acquisitions, month crops composites: tell them apart
        self.date_type = (nn.Parameter(torch.zeros(D))
                          if date_level != "none" or date_joint != "none" else None)
        if date_level != "none" and not (frac_slot or doy):
            raise ValueError("the date level needs --frac_slot or --doy: OlmoEarth encodes slot "
                             "and month only, so the dates of one month would be identical")
        if date_level != "none" and selector == "none":
            raise ValueError("the date level refines month picks: it needs a selector")
        if date_selector not in ("random", "xattn") or date_gate not in ("month", "none"):
            raise ValueError(f"date_selector {date_selector!r} / date_gate {date_gate!r}")
        if selector in PRIOR and (readout not in SPARSE + NATIVE or select_unit != "location"):
            raise ValueError(f"--selector {selector} scores cells from a crop-free cellattn pass: "
                             "--readout cellattn --select_unit location")
        if selector == "entropy" and task.kind != "classification":
            raise ValueError("--selector entropy needs class probabilities")
        if selector in PIXEL_PRIOR and select_unit != "slot":
            raise ValueError(f"--selector {selector} scores every (month, cell) slot directly: "
                             "--select_unit slot")
        if apt_pool and (selector != "apt" or crop_px != LOOKUP_PS or jitter):
            raise ValueError("--apt_pool: --selector apt, plain ps4-on-grid crops "
                             "(no --crop_px / --jitter -- _apt_pool_crops cannot recover a "
                             "merged group's cell from an already-adjusted crop window)")
        if apt_time not in ("sum", "none"):
            raise ValueError(f"apt_time {apt_time!r} not in ('sum', 'none')")
        if apt_time == "none" and selector != "apt":
            raise ValueError("--apt_time is a --selector apt option")
        if apt_res and (selector != "apt" or apt_pool or res_mix is not None
                        or readout != "native_set" or select_unit != "slot"):
            raise ValueError("--apt_res: --selector apt, --readout native_set, slot unit, "
                             "no --apt_pool / --res_mix (apt_res picks its own ps16-vs-ps4 "
                             "split and reuses --res_mix's footprint machinery directly)")
        if apt_octree and (selector != "apt" or apt_pool or apt_res or res_mix is not None
                           or readout != "native_set" or select_unit != "slot"):
            raise ValueError("--apt_octree: --selector apt, --readout native_set, slot unit, "
                             "no --apt_pool / --apt_res / --res_mix (apt_octree picks its own "
                             "space AND time shape per node, incompatible with every other "
                             "mask-building path -- see _apt_octree_mask)")
        if apt_octree and apt_tau <= 0:
            raise ValueError("--apt_octree needs --apt_tau > 0 (the entropy threshold driving "
                             "its recursion -- see exp.upsamplers.calibrate_apt_octree_tau for "
                             "how a dataset/budget's tau was picked)")
        if apt_octree_pool not in ("input", "embed", "attn"):
            raise ValueError(f"apt_octree_pool {apt_octree_pool!r} not in "
                             "('input', 'embed', 'attn')")
        if apt_octree_pool != "input" and not apt_octree:
            raise ValueError("--apt_octree_pool embed/attn is an --apt_octree option")
        if apt_octree_space_only and not apt_octree:
            raise ValueError("--apt_octree_space_only is an --apt_octree option")
        self.apt_pool, self.apt_merge_mse, self.apt_time = apt_pool, apt_merge_mse, apt_time
        self.apt_res, self.apt_octree, self.apt_tau = apt_res, apt_octree, apt_tau
        self.apt_octree_pool = apt_octree_pool
        self.apt_octree_space_only = apt_octree_space_only
        if apt_octree_pool == "attn":            # user's "encode + cross attention pool
            # embedding" option -- a learned per-member-month weight, same mechanism/shape as
            # --acq_pool's self.acq_score (zero-init Linear -> exp-clamp softmax-style weight),
            # but its OWN module: apt_octree_pool pools LEAF MEMBER MONTHS, acq_pool pools
            # RAW ACQUISITIONS WITHIN a month -- different arms, never both at once (see the
            # apt_octree validation above: no --apt_pool, and --acq_pool is set post-hoc via
            # set_acq_pool() so there's no risk of the two colliding, but keeping the module
            # separate avoids any future coupling between two independently-toggled options).
            D = self.mix.encoder.embedding_size
            self.apt_octree_score = nn.Linear(D, 1)
            nn.init.zeros_(self.apt_octree_score.weight)
            nn.init.zeros_(self.apt_octree_score.bias)
        self.task = task      # --selector apt: task.olmoearth.modalities (every band, raw pixels)
        # date level: learned scores of (modality, date, cell), one cross-attention selector
        # per modality (its dates' patch embeddings as queries, the season tokens as keys);
        # a date crop is gated by its own score. Random date picks are gated by their month
        # pick's score (date_gate month, a learned month selector only) or not at all.
        self.date_gate = date_gate
        self.date_selector = (nn.ModuleDict({m: CrossAttnSelector(D, selector_hidden)
                                             for m in task.olmoearth.modalities})
                              if date_level != "none" and date_selector == "xattn" else None)
        # native: the head maps a ps16 token to its 16 x 16 px block (see native_logits)
        self.native = readout in NATIVE
        self.task_kind = task.kind
        self.head = Pa2PxHead(task.reduction, D, task.num_outputs,
                              LR_PS if self.native else TILE // OUT_GRID)
        # the joint selector: exp.upsamplers.fit_innovation.Predictor on innovation_diag's
        # pre-look features: [e, e - season mean e] (2D) + all seasons' crop-free tokens (NS D)
        # + their native logits (NS C) + onehots (NS season, T month, 16 place) + pixel var
        self.joint = None
        if joint:
            from exp.upsamplers.fit_innovation import Predictor
            # month slots T and seasons NS of the task (YieldSAT: 24 months = 8 seasons)
            T_ = 24 if task.name.startswith("yieldsat") else 12
            NS_, C_ = T_ // task.season if task.season else 4, task.num_outputs
            if self.joint_pairs:            # series: see _pair_prelook
                self.joint = Predictor(5 * D + 2 * len(TimeSetReadout.PERIODS) + 16
                                       + int(joint_feat == "prelook"), temporal=True)
            elif joint_arch == "xattn":
                self.joint = CrossAttnSelector(D, 256, temporal=True)
            elif joint_arch == "xs":    # xenc_ts at half the width, one-layer feed-forward
                self.joint = EncCrossSelector(D, 128, temporal=True, spatial=True, ff=1)
            elif joint_arch == "xc":    # the same width; candidates mix through 2 full
                # self-attention layers over the (month, ps16 cell) means of their tokens
                self.joint = EncCrossSelector(D, 128, ff=1, coarse=2)
            elif joint_arch.startswith("xenc"):
                self.joint = EncCrossSelector(D, 256, temporal=joint_arch in ("xenc_t", "xenc_ts"),
                                              spatial=joint_arch == "xenc_ts")
            elif joint_in == "enc":
                self.joint = Predictor(D + NS_ * D, temporal=True)
            else:
                extra_f = NS_ * C_ + 1 if joint_feat == "prelook" else 0   # logits + pixel var
                self.joint = Predictor(2 * D + NS_ * D + extra_f + NS_ + T_ + 16
                                       + int(joint_res), temporal=True)
        self.date_pred = (nn.ModuleDict({m: DatePredictor(D, NS_) for m in task.olmoearth.modalities})
                          if date_joint != "none" else None)
        self.joint_embed = (copy.deepcopy(self.mix.encoder.patch_embeddings)
                            if joint_embed == "own" else None)
        if head not in ("mean", "attn"):
            raise ValueError(f"head {head!r} not in ('mean', 'attn')")
        if head == "attn" and task.reduction != "time_mean":
            raise ValueError(f"--head attn replaces a time mean; {task.name} uses {task.reduction}")
        self.pool = TemporalAttnPool(D) if head == "attn" else None
        if pick not in ("topk", "random", "per_step"):
            raise ValueError(f"pick {pick!r} not in ('topk', 'random', 'per_step')")
        if pick != "topk" and (selector not in LEARNED or select_unit != "slot"):
            raise ValueError("--pick random / per_step diagnose a learned slot selector")
        self.pick, self.score_norm = pick, score_norm
        self.aux_scores = None          # the scores, for a regularizer in the training loop
        self.fixed_scores = None        # (B, T, H, W): test-time picks from an outside scorer
        self.grid_mode = "stagger"      # --selector grid: lattice layout (see grid_scores)
        self.grid_phase = "random"      # ... a fresh lattice offset per training sample, or fixed
        self.force_grid = False         # score any arm's weights with lattice picks (eval only)
        self.loc_px = None              # --loc_stride: (H,) window top-left px of an anchor cell, -1 else
        # --no_seasons (set by main): no season tokens in the sequence or the readout -- only
        # the crops; a pixel no crop covers in any month reads 0
        self.no_seasons = False
        # --shift_stride S (set by main): the crops are EVERY window of the shifted grid of
        # stride S in every present month (shifted_positions), zero-padded past the tile edge
        self.shift_stride = 0
        if selector not in SELECTORS:
            raise ValueError(f"selector {selector!r} not in {SELECTORS}")
        if select_unit not in ("slot", "location", "pair"):
            raise ValueError(f"select_unit {select_unit!r} not in ('slot', 'location', 'pair')")
        if (select_unit == "pair") != task.reduction.startswith("pair_") and selector != "none":
            raise ValueError("select_unit pair goes with (and only with) a pair_* head")
        self.sel_name, self.unit, self.k_frac = selector, select_unit, k_frac
        self.selector = (_mlp(3 * D, selector_hidden) if selector == "learned" else
                         CrossAttnSelector(D, selector_hidden, temporal=sel_temporal)
                         if selector == "xattn" else None)
        if sel_temporal and selector != "xattn":
            raise ValueError("--sel_temporal is a CrossAttnSelector option")
        self.gain_ctx = None            # for --gain_loss: readout, picks, crop - base
        if readout not in READOUTS:
            raise ValueError(f"readout {readout!r} not in {READOUTS}")
        if readout.startswith("cellattn") and (task.reduction != "time_mean" or head != "mean"):
            raise ValueError("--readout cellattn pools over time itself: time_mean tasks, "
                             "no --head attn")
        if self.native and (task.reduction != "time_mean" or head != "mean"
                            or lr_patch != 16 and readout != "native_set"):
            raise ValueError("--readout native: time_mean tasks, --head mean, ps16 seasons "
                             "(native_set also ps8 / ps4)")
        # native_attn: the temporal attention score (token -> scalar), in logit space
        self.native_score = (nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
                             if readout == "native_attn" else None)
        if self_teach and readout == "native_set":
            self.recon_splat = TimeSetReadout(D, "gauss")
        self.readout = (TimeSetReadout(D, "gauss" if "splat" in readout else "query")
                        if readout in TIME else
                        NativeSetReadout(D, base_px=lr_patch, mean=readout == "native_mean")
                        if readout in SET else
                        None if readout in ("bilinear",) + NATIVE else
                        CellAttnReadout(D, overlap=readout == "cellattn_overlap")
                        if readout.startswith("cellattn") else
                        GaussianSplatReadout(D, temporal=readout != "splat",
                                             overlap=readout == "splat_t_overlap",
                                             guided=readout == "splat_tg"))
        self.k, self.GT = 0, (0, 0)

    def backbone(self):
        return self.mix.encoder

    def _pick_pairs(self, pick, present, pairs):
        """(B, T, H, W) date scores -> crop mask: the best (pair, cell) candidates, both
        dates of each; budget k_frac x present dates x cells / 2 pairs per sample."""
        B, T, H, W = pick.shape
        pairs = pairs.to(pick.device)
        ok = (pairs >= 0).all(-1)                                              # (B, P)
        bi = torch.arange(B, device=pick.device)[:, None]
        e, l_ = pairs[..., 0].clamp(min=0), pairs[..., 1].clamp(min=0)
        ps = torch.maximum(pick[bi, e], pick[bi, l_])                          # (B,P,H,W)
        ps = ps.masked_fill(~ok[:, :, None, None], float("-inf")).flatten(1)
        budget = (self.k_frac * present.sum(1).float() * H * W / 2).round().long()
        budget = torch.minimum(budget, (ok.sum(1) * H * W))
        rank = ps.argsort(1, descending=True).argsort(1)
        chosen = (rank < budget[:, None]) & torch.isfinite(ps)
        b, q = chosen.nonzero(as_tuple=True)
        p_, yx = q // (H * W), q % (H * W)
        y, x = yx // W, yx % W
        mask = torch.zeros(B, T, H, W, dtype=torch.bool, device=pick.device)
        mask[b, pairs[b, p_, 0], y, x] = True
        mask[b, pairs[b, p_, 1], y, x] = True
        return mask

    def _guide(self, ms) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.month_embedding(ms).float()

    def _tokens_per_crop(self, sample) -> int:
        """Tokens one single-patch crop of `sample` costs: its modalities' bandsets."""
        one = torch.zeros(1, dtype=torch.long, device=sample.timestamps.device)
        return int(self.mix.embed(crop_patches(sample, one, one, one, one, LOOKUP_PS), LOOKUP_PS,
                                  slot=one, py=one, px=one).keep.shape[1])

    def month_embedding(self, ms, embed=None, slot=None) -> torch.Tensor:
        """(B, T, 16, 16, D): each month's ps4 patch embedding (no encodings), pooled like
        OlmoEarth's output (bandsets, then present modalities). embed: another patch
        embedding module than the encoder's (--joint_embed own).
        slot (B, T): WITH OlmoEarth's encodings, as a ps4 crop of that (month, cell) gets them
        in MixedResOE.embed -- channel and calendar month as is, time slot `slot` (its season)
        instead of the month index, position = the crop's (centre under --pos center)."""
        enc = self.mix.encoder
        x = (embed or enc.patch_embeddings).forward(ms, LOOKUP_PS)
        if slot is not None:
            tokens, masks, _ = enc.split_tokens_masks_and_dims(x)
            tokens = enc.composite_encodings.forward(tokens, ms.timestamps, LOOKUP_PS, BASE_GSD)
            n = enc.composite_encodings.embedding_dim_per_embedding_type
            dev = slot.device
            B, T = slot.shape
            d_time = (get_1d_sincos_pos_encoding(slot.flatten().float(), n)
                      - get_1d_sincos_pos_encoding(torch.arange(T, device=dev).repeat(B).float(), n)
                      ).view(B, T, n)
            gy, gx = torch.meshgrid(torch.arange(OUT_GRID, device=dev) * LOOKUP_PS,
                                    torch.arange(OUT_GRID, device=dev) * LOOKUP_PS, indexing="ij")
            off = (LOOKUP_PS - self.mix.lr_ps) / 2 if self.mix.pos == "center" else 0.0
            d_pos = (sincos_2d(gy.flatten().float() + off, gx.flatten().float() + off, n)
                     - sincos_2d(gy.flatten(), gx.flatten(), n)).view(OUT_GRID, OUT_GRID, n)
            for m, t in tokens.items():             # (B, H, W, T, bandsets, D)
                delta = torch.zeros(B, OUT_GRID, OUT_GRID, T, 1, t.shape[-1], device=dev)
                delta[..., n:2 * n] = d_time[:, None, None, :, None]
                delta[..., 3 * n:4 * n] = d_pos[None, :, :, None, None]
                tokens[m] = t + delta.to(t.dtype)
            x = {**tokens, **masks}
        tam = TokensAndMasks(**x)
        T = ms.timestamps.shape[1]
        return torch.stack([pool_timestep(tam, t) for t in range(T)], 1)

    def set_loc_stride(self, stride: int):
        """--loc_stride: candidate crop_px windows, (TILE - P) // stride + 1 per side spread
        evenly over the tile (first at 0, last at the edge: stride 8 -> every 8 px; 6 -> gaps of
        6-7 px, <= 2 px overlap), each represented by the ps4 cell under its top-left corner
        (unique for stride >= 6). loc_px[cell] = that px, -1 for a non-candidate cell."""
        P_ = self.crop_px
        n = (TILE - P_) // stride + 1
        tl = sorted({round(i * (TILE - P_) / (n - 1)) for i in range(n)})
        cells = [v // LOOKUP_PS for v in tl]
        if len(set(cells)) != len(cells):
            raise ValueError(f"--loc_stride {stride}: two windows share an anchor cell")
        self.loc_px = torch.full((TILE // LOOKUP_PS,), -1, dtype=torch.long)
        self.loc_px[cells] = torch.tensor(tl)

    @torch.no_grad()
    def _prior_scores(self, lr, season, present, G, extra, B, H):
        """(B, H, W) per ps4 cell, from the model's own crop-free prediction (the lr tokens
        through the remaining blocks, the cellattn readout, the head):
            oracle   the loss against the TRUE label (a ceiling for choosing where to look;
                     uses labels at test time too -- never a real method)
            entropy  the predictive entropy (label-free: where the model is unsure)"""
        was = self.training
        self.eval()
        try:
            lr_x, _ = self.mix.tail(lr, None, None)
            lr_F = lr.pooled(lr_x).float()
            empty = torch.zeros(0, dtype=torch.long, device=lr_F.device)
            none = (lr_F.new_zeros(0, lr_F.shape[-1]), (empty,) * 4)
            if isinstance(self.readout, NativeSetReadout):
                logits = self.readout(self.head.linear, lr_F, season, present, *none)
            elif self.native_score is not None:
                logits = native_attn_logits(self.head.linear, self.native_score, lr_F, season,
                                            present)
            elif self.native:
                logits = native_logits(self.head.linear, lr_F, season, present)
            else:
                pooled = self.readout(lr_F, seasons_used(season, present, G), *none)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.head(pooled[:, None], torch.ones_like(present[:, :1])).float()
        finally:
            self.train(was)
        if self.sel_name == "entropy":
            p = logits.softmax(1)
            px = -(p * p.clamp(min=1e-8).log()).sum(1)                              # (B, 64, 64)
        else:
            target = extra["target"].to(logits.device)
            if logits.shape[1] > 1:
                px = F.cross_entropy(logits, target.long(), ignore_index=-1, reduction="none")
            else:
                y = target[:, 0] if target.ndim == 4 else target   # YieldSAT: (value, field id)
                px = ((logits[:, 0] - y.float()) ** 2).nan_to_num(0.0)        # NaN: unlabelled
        P_ = self.crop_px
        if P_ == LOOKUP_PS:
            return F.avg_pool2d(px[:, None], TILE // H)[:, 0]                      # (B, H, W)
        # a pick = the P_ window centred on the cell, clamped into the tile (as the crops are
        # cut): score the cell by the mean over THAT window
        win = F.avg_pool2d(px[:, None], P_, stride=1)[:, 0]          # (B, TILE-P_+1, TILE-P_+1)
        if self.loc_px is not None:     # --loc_stride: only the anchor cells are candidates
            lut = self.loc_px.to(px.device)
            c = (lut >= 0).nonzero(as_tuple=True)[0]
            out = px.new_full((B, H, H), float("-inf"))
            out[:, c[:, None], c[None, :]] = win[:, lut[c]][:, :, lut[c]]
            return out
        tl = (torch.arange(H, device=px.device) * LOOKUP_PS + LOOKUP_PS // 2
              - P_ // 2).clamp(0, TILE - P_)
        return win[:, tl][:, :, tl]                                                # (B, H, W)

    def _apt_pixels(self, ms):
        """(B, T, Hpx, Wpx, C) -- EVERY band of EVERY modality OlmoEarth has for this task,
        concatenated on the channel axis (not just task.rgb's 1-3 proxy bands). Reads
        task.olmoearth.modalities directly (the dataset -> OlmoEarth attribute map already
        used to build `ms`), so there is no band-name lookup / imputation to get wrong -- just
        every channel of every distinct attribute `ms` actually carries."""
        task = self.task
        oe_names = sorted(set(task.olmoearth.modalities.values()))
        return torch.cat([getattr(ms, name).float() for name in oe_names], -1)

    @torch.no_grad()
    def _apt_scores(self, ms, present, B, H, normalize=False):
        """(B, T, H, W): Choudhury et al. 2025 (arXiv:2510.18091, "Adaptive Patch
        Transformers") adapted to a time series. No model pass, no labels -- raw pixels only.

        APT scores a STATIC image's candidate patches by H(P) = -sum_i p_i log2 p_i, p_i the
        histogram of pixel intensities inside the patch (their Eq. 1; they bin because patch
        pixels are discrete; the paper is single-channel). Here every band of every guidance
        modality is scored SEPARATELY (own per-cell normalization, own histogram, own entropy)
        and summed -- using all available bands rather than collapsing to one luma channel.
        We add the time axis the paper has none of:
            H_space[b,t,y,x]  sum over bands/modalities of APT's H(P) at cell (y,x), month t
            H_time[b,y,x]     H(P) of the SAME cell's pixels pooled across every present month,
                              summed over bands the same way -- entropy over which months this
                              location's intensity takes, independent of any one month's content
            score = H_space + H_time (broadcast over T), per the user's "time them up"
        Cell = the TILE // H px block (no multi-scale quadtree: the grid is already fixed at
        H x H, by design -- see the GEOID arms).

        --apt_time none: skip H_time entirely (score = H_space alone, not even computed) --
        the user's more literal APT: ranks (month, cell) slots purely by per-frame spatial
        entropy, time-ignorant, same as running APT separately on every frame of the series.

        normalize=True: divide by each cell's OWN achievable entropy ceiling before summing
        over bands -- min(log2(APT_BINS), log2(n_pixels_in_cell)) per band (a histogram over
        APT_BINS bins built from fewer than APT_BINS samples cannot reach log2(APT_BINS) bits
        no matter how diverse its content is -- a ps2 cell has only 4 pixels to spread across
        32 bins, capping it at log2(4)=2 bits, vs. a ps16 cell's 256 pixels reaching the full
        log2(32)=5 bit ceiling). Without this, raw entropy is NOT comparable across different
        grid sizes H -- most of the cross-level drop measured empirically (e.g. PASTIS ps16
        vs ps2 medians ~4.5 vs ~2.0 mean-per-band bits) is this ceiling artifact, not a real
        difference in content diversity (confirmed: normalized values for the SAME cells sit
        at a near-constant 0.75-0.98 fraction of their own ceiling across all 4 levels, vs.
        the raw values spanning a 2.5x range -- see [[apt-entropy-selector]]). Needed for
        _apt_quadtree_mask's multi-level threshold comparison to mean the same thing at every
        level; NOT used by any other _apt_scores caller (apt/apttime/aptpool/aptres all score
        a single fixed grid size, so the cross-level comparability problem never arises for
        them -- changing their behaviour by flipping this default was never asked for, so the
        default stays False and every existing call site is unaffected)."""
        x = self._apt_pixels(ms)                                 # (B, Hpx, Wpx, T, C)
        x = x.permute(0, 3, 1, 2, 4)                              # (B, T, Hpx, Wpx, C)
        B_, T, Hpx, Wpx, C = x.shape
        cell = Hpx // H
        # per-cell, per-band min-max to [0, 1): APT normalizes nothing explicit, but bands here
        # are OlmoEarth's per-dataset standardized units (unbounded) rather than raw DN /
        # reflectance, so a per-cell rescale is needed before binning -- otherwise two cells at
        # the same relative contrast but different absolute level would bin differently.
        cells = x.unfold(2, cell, cell).unfold(3, cell, cell)    # (B,T,H,W,C,cell,cell)
        flat = cells.reshape(B_, T, H, H, C, cell * cell)
        lo, hi = flat.amin(-1, keepdim=True), flat.amax(-1, keepdim=True)
        binned = ((flat - lo) / (hi - lo + 1e-6) * (APT_BINS - 1)).round().long()
        onehot = F.one_hot(binned.clamp(0, APT_BINS - 1), APT_BINS).sum(-2).float()  # (...,bins)
        p = onehot / onehot.sum(-1, keepdim=True).clamp(min=1e-8)
        h_space = -(p * (p + 1e-8).log2()).sum(-1)                # (B, T, H, W, C): per band
        if normalize:
            import math
            ceil_per_band = min(math.log2(APT_BINS), math.log2(cell * cell))
            h_space = h_space / ceil_per_band                     # each band now in [0, 1]
        h_space = h_space.sum(-1)                                 # (B, T, H, W): summed over bands
        if self.apt_time == "none":
            return h_space

        # H_time: the same per-band histogram machinery, over the T present months' per-cell
        # means (one scalar per month per cell per band -- APT has no notion of "pixel" here,
        # so each month's cell mean stands in for a "pixel" of the time axis), summed over bands.
        cell_mean = flat.mean(-1)                                # (B, T, H, W, C)
        pres = present[:, :, None, None, None].expand(B_, T, H, H, C)
        lo_t = cell_mean.masked_fill(~pres, float("inf")).amin(1, keepdim=True)
        hi_t = cell_mean.masked_fill(~pres, float("-inf")).amax(1, keepdim=True)
        binned_t = ((cell_mean - lo_t) / (hi_t - lo_t + 1e-6) * (APT_BINS - 1)).round().long()
        binned_t = binned_t.clamp(0, APT_BINS - 1).masked_fill(~pres, APT_BINS)   # extra bin: absent
        onehot_t = F.one_hot(binned_t, APT_BINS + 1)[..., :APT_BINS].sum(1).float()  # (B,H,W,C,bins)
        p_t = onehot_t / onehot_t.sum(-1, keepdim=True).clamp(min=1e-8)
        h_time = -(p_t * (p_t + 1e-8).log2()).sum(-1)             # (B, H, W, C): per band
        if normalize:
            import math
            ceil_per_band_t = min(math.log2(APT_BINS), math.log2(T))  # "pixel" = one month's mean
            h_time = h_time / ceil_per_band_t
        h_time = h_time.sum(-1)                                   # (B, H, W): summed over bands
        return h_space + h_time[:, None]                         # (B, T, H, W)

    @torch.no_grad()
    def _apt_pool_mask(self, ms, present, B, H, k):
        """User's space-then-time extension (2026-10-07), on top of --selector apt:

        1. CELLS are picked first, by their score summed over present months (not per-slot
           top-k): rank every (b, y, x) by sum_t score[b,t,y,x], admit cells in that order.
        2. At each admitted cell, its present months (in time order) are scanned for RUNS of
           consecutive months whose pixels barely differ (MSE < --apt_merge_mse, measured on
           the same per-cell-per-band-normalized representation _apt_scores bins): a run
           collapses into ONE crop at the run's MEAN slot (season + frac, the user's "just
           simply interpolate"; OlmoEarth's sincos time encoding is continuous, so a
           fractional / averaged slot is natively supported -- no new embedding needed) and
           the run's MEAN pixels (averaged before patch embedding). A run of length n frees
           n - 1 crops.
        3. Freed crops are spent GLOBALLY on the next-best-scoring un-admitted cells (not kept
           local to the cell that freed them) -- more spatial coverage at the same total k.
        Admission: cells are ranked by score and admitted by a CUMULATIVE-COST CUTOFF (every
        cell up to the first one whose running total exceeds k is admitted; user confirmed
        2026-10-08 after an exact "skip a cell that doesn't fit, keep trying smaller ones
        further down the ranking" greedy pass measured at 300-800ms/batch -- a ~3-8x slowdown
        of the whole forward pass -- vs. this version's single vectorized pass, no Python loop
        over cells at all). The two rules differ only when costs vary AND an oversized cell
        sits ahead of the cutoff -- exact greedy would skip it and admit 1-2 more cells past
        where cumulative cost crosses k; the cutoff rule stops there instead.

        CAVEAT (found while visualizing picks, 2026-10-08): a merged run's representative
        month is its FIRST member (rep_t = members[0]), so a run spanning a sample's entire
        present-month range collapses the whole sample onto one timestep. On a 2-STEP
        pair/change-detection task (GEOID: pre vs post event) a loose --apt_merge_mse can
        merge the pre/post pair itself into one crop, which defeats the task -- confirmed via
        exp.upsamplers.viz_apt_picks on geoidflood (demo threshold 1.5 merged all 85 budgeted
        crops into month 0 alone, leaving month 1 -- the POST image -- with zero crops). The
        production default (0.01) does not trigger this in practice (GEOID's own pre/post
        pair differs far more than 0.01 MSE), but a user raising --apt_merge_mse on a pair
        task should be aware the mechanism has no notion of "never merge across a required
        comparison" -- it only compares consecutive PRESENT months, blind to task semantics.

        Returns (mask (B, T, H, W) bool, groups: dict[(b, t, y, x)] -> sorted list of month
        indices the mask's True slot there stands in for -- len 1 for an unmerged pick). The
        mask has exactly one True per group (at the group's FIRST month, the representative
        (t, y, x) coordinate for the generic crop_patches / embed path); forward() substitutes
        that crop's pixels and slot for the group's mean when len(group) > 1."""
        scores = self._apt_scores(ms, present, B, H)                      # (B, T, H, H)
        x = self._apt_pixels(ms).permute(0, 3, 1, 2, 4)                   # (B, T, Hpx, Wpx, C)
        B_, T, Hpx, Wpx, C = x.shape
        cell = Hpx // H
        cells = x.unfold(2, cell, cell).unfold(3, cell, cell).reshape(B_, T, H, H, C, cell * cell)
        lo, hi = cells.amin(-1, keepdim=True), cells.amax(-1, keepdim=True)
        norm = (cells - lo) / (hi - lo + 1e-6)                            # (B,T,H,H,C,cell*cell)
        dev = scores.device

        # Consecutive-PRESENT-month MSE, fully vectorized (no per-cell / per-sample Python loop,
        # no .item() host sync -- the earlier version did this with a triple-nested Python loop
        # over (sample, cell, month pair), each step a GPU->CPU sync: ~200x slower in practice,
        # see [[apt-entropy-selector]]). present varies per sample, so each sample's own present
        # months are gathered into a left-packed (B, T) index (pad = last present month, masked
        # off below) before taking consecutive differences -- "consecutive" means consecutive
        # PRESENT months of that sample, never consecutive raw month indices.
        order_t = torch.argsort(present.float(), dim=1, descending=True, stable=True)  # present
        # months first; ties (present/absent) broken by original index -- present months stay
        # in time order
        n_present = present.sum(1)                                       # (B,)
        valid_pair = (torch.arange(T - 1, device=dev)[None] < (n_present - 1)[:, None])  # (B, T-1)
        t0 = order_t[:, :-1]                                             # (B, T-1): present month i
        t1 = order_t[:, 1:]                                              # (B, T-1): present month i+1
        gather_idx = lambda t: t[:, :, None, None, None, None].expand(   # noqa: E731
            B_, T - 1, H, H, C, cell * cell)
        a = torch.gather(norm, 1, gather_idx(t0))
        b_ = torch.gather(norm, 1, gather_idx(t1))
        mse = (a - b_).pow(2).mean(-1).sum(-1)                            # (B, T-1, H, H): sum bands
        # a run ends at pair i wherever the pixels differ too much -- NOT at padding pairs
        # (i >= n_present - 1): those carry no real break information and must never seed a
        # (spurious) run-start at a padding rank (fixed 2026-10-08: the previous version used
        # `| ~valid_pair`, forcing breaks=True in the padding region, which made run_starts
        # True there too -- the mask then had entries at ABSENT months with no groups entry,
        # a KeyError in forward() the moment any sample in a batch had a missing month; see
        # [[apt-entropy-selector]]).
        breaks = (mse >= self.apt_merge_mse) & valid_pair[:, :, None, None]

        cell_score = scores.masked_fill(
            ~present[:, :, None, None], 0).sum(1)                        # (B, H, H): sum over t
        # run_starts[b, i, y, x]: present-rank i is the FIRST month of a run at this cell --
        # i == 0, or the previous rank ended a run (breaks[i - 1]) -- AND i is itself a real
        # present-rank (i < n_present): ranks beyond a sample's present-month count are padding
        # and must never start a (fictitious) run. This alone gives the mask (True at every
        # run-start, in RAW month coordinates after scatter below) with no grouping pass -- no
        # Python loop over cells or months anywhere in this function.
        run_starts = torch.cat([torch.ones(B_, 1, H, H, dtype=torch.bool, device=dev),
                                breaks], dim=1)                          # (B, T, H, H)
        rank_valid = (torch.arange(T, device=dev)[None] < n_present[:, None])  # (B, T)
        run_starts &= rank_valid[:, :, None, None]
        cost = run_starts.sum(1)                                        # (B, H, H): runs per cell

        # Cumulative-cutoff admission, fully vectorized: sort cells by score, take a running
        # sum of their cost in that order, admit while the running sum stays within budget.
        HH = H * H
        flat_score = cell_score.flatten(1)                               # (B, HH)
        flat_cost = cost.flatten(1)                                      # (B, HH)
        order = flat_score.argsort(1, descending=True)                   # (B, HH)
        cost_sorted = torch.gather(flat_cost, 1, order)                  # (B, HH)
        cum = cost_sorted.cumsum(1)                                      # (B, HH)
        budget = k if torch.is_tensor(k) else torch.full((B_,), k, device=dev)
        admit_sorted = cum <= budget[:, None]                            # (B, HH)
        admit_flat = torch.zeros(B_, HH, dtype=torch.bool, device=dev).scatter_(1, order, admit_sorted)
        admit = admit_flat.view(B_, H, H)                                 # (B, H, H)

        # The mask: True at every run-start month of every ADMITTED cell. run_starts is in
        # PRESENT-RANK order (axis 1 indexes rank, not raw month) -- scatter it back to raw
        # month indices via order_t (the same permutation _apt_scores' caller never sees, kept
        # local to this function) before applying the cell admission mask.
        mask = torch.zeros(B_, T, H, H, dtype=torch.bool, device=dev)
        mask.scatter_(1, order_t[:, :, None, None].expand(B_, T, H, H), run_starts)
        mask &= admit[:, None]

        # groups: only admitted cells need their member-month lists (for pixel averaging /
        # slot interpolation in _apt_pool_crops), so this loop runs over ADMITTED (b, y, x)
        # triples only -- at most k of them, not every scanned cell -- and reads data already
        # moved to CPU/NumPy in one shot (as before), not inside a per-cell GPU round-trip.
        import numpy as np
        bb, yy, xx = admit.nonzero(as_tuple=True)
        groups: dict[tuple[int, int, int, int], list[int]] = {}
        if len(bb):
            order_t_np = order_t.cpu().numpy()                           # (B, T)
            run_id_np = run_starts.cumsum(1).cpu().numpy()                # (B, T, H, H): 1-indexed
            n_present_np = n_present.cpu().numpy()
            bb_np, yy_np, xx_np = bb.cpu().numpy(), yy.cpu().numpy(), xx.cpu().numpy()
            for bi, yi, xi in zip(bb_np.tolist(), yy_np.tolist(), xx_np.tolist()):
                n_pres = int(n_present_np[bi])
                months = order_t_np[bi, :n_pres]                         # raw month, present-rank order
                ids = run_id_np[bi, :n_pres, yi, xi]
                bounds = np.flatnonzero(np.diff(ids)) + 1
                for members in np.split(months, bounds):
                    rep_t = int(members[0])
                    groups[(bi, rep_t, yi, xi)] = members.tolist()
        return mask, groups

    @torch.no_grad()
    def _apt_res_mask(self, ms, present, B, k):
        """User's 2026-10-08 extension: let APT's own entropy choose patch SIZE (ps16 vs ps4),
        not just which ps4 slots to fill -- the part of the original paper's coarse/fine
        quadtree idea this codebase's plain --selector apt never implemented (it only ranks a
        grid already fixed at ps4).

        Every present (month, ps16 cell) starts covered by ONE ps16 crop (1 token). A ps16
        crop's footprint is a 4x4 block of ps4 CELLS (16px / 4px = 4 per side -- same
        convention _res_crops already uses, f = ps // LOOKUP_PS), so splitting it into its
        16 ps4 children costs 15 MORE tokens (16 total) and gains value = sum(score_ps4 over
        the 16 children) - score_ps16 (entropy of the fine crops over the coarse one).
        Candidates are ranked by value and SPLIT by a vectorized cumulative-cost cutoff (same
        mechanism as _apt_pool_mask, user-approved 2026-10-08): every (month, ps16 cell) up to
        the first whose running EXTRA cost exceeds the remaining budget (k -
        n_present_ps16_cells, the base ps16-everywhere cost) is split; the rest stay at ps16.
        Total tokens spent = n_present_ps16_cells + 15 x n_split, <= k by construction (never
        exceeds it, may slightly underspend when no exact split count lands on k -- same
        "stops rather than overshoots" rule as apt_pool).

        Returns (mask4, res_masks): mask4 (B, T, 16, 16) bool is the ps4-grid mask of SPLIT
        cells' children, in the SAME convention as every other selector's `mask` (one True per
        ps4 crop) -- forward() treats it exactly like a plain top-k pick, no new code needed
        on that path. res_masks = {16: mask16 (B, T, 4, 4) bool} is the UNSPLIT ps16 cells, in
        _res_picks' own dict format (ps8 key never used) -- handed to the EXISTING
        _res_crops / NativeSetReadout footprint machinery unchanged."""
        H4, H16 = 16, 4
        score_ps4 = self._apt_scores(ms, present, B, H4)           # (B, T, 16, 16)
        score_ps16 = self._apt_scores(ms, present, B, H16)         # (B, T, 4, 4)
        children_sum = F.avg_pool2d(score_ps4, 4) * 16             # (B, T, 4, 4): sum of 4x4 kids
        value = (children_sum - score_ps16).masked_fill(~present[:, :, None, None], 0)

        dev = value.device
        n_present = present.sum(1)                                  # (B,)
        base_cost = n_present * (H16 * H16)                         # (B,): one ps16 crop each
        budget = torch.as_tensor(k, device=dev).expand(B) if not torch.is_tensor(k) else k
        if bool((budget < base_cost).any()):
            # a ps16 crop at every present (month, cell) is the FLOOR cost -- this function has
            # no tier below ps16 (no "neither" option), so a budget this small cannot even
            # cover the coarsest crop everywhere, let alone decide where to split finer.
            min_k_frac = (H16 * H16) / (16 * 16)       # 16 / 256 of the ps4 grid
            raise ValueError(f"--apt_res: k={int(budget.amin())} is below the ps16-everywhere "
                             f"floor ({int(base_cost.amax())} for the sample with the most "
                             f"present months) -- need --k_frac >= {min_k_frac:.4g} ({H16}x{H16} "
                             f"/ {16}x{16} grid cells) at this T")
        extra_budget = (budget - base_cost).clamp(min=0)            # (B,): tokens left for splits

        flat_val = value.flatten(1)                                 # (B, T*4*4)
        order = flat_val.argsort(1, descending=True)
        present_flat = present[:, :, None, None].expand(
            B, value.shape[1], H16, H16).reshape(B, -1)
        valid_sorted = torch.gather(present_flat, 1, order)
        cost_sorted = torch.where(valid_sorted, torch.full_like(order, (16 * 16) // (H16 * H16) - 1),
                                  torch.zeros_like(order))
        cum = cost_sorted.cumsum(1)
        split_sorted = valid_sorted & (cum <= extra_budget[:, None])
        split_flat = torch.zeros_like(split_sorted).scatter_(1, order, split_sorted)
        split = split_flat.view_as(value)                          # (B, T, 4, 4) bool

        mask16 = present[:, :, None, None].expand(B, present.shape[1], H16, H16) & ~split
        mask4 = split.repeat_interleave(4, -1).repeat_interleave(4, -2)   # (B, T, 16, 16)
        return mask4, {16: mask16}

    @torch.no_grad()
    def _apt_quadtree_mask(self, ms, present, B, grid_sizes, taus, normalize=True):
        """The ACTUAL APT algorithm (Choudhury et al. 2025, Section 3.1), not this codebase's
        token-budget adaptations (_apt_pool_mask / _apt_res_mask both target a fixed k; the
        paper never does -- it has no budget concept at all, only per-level entropy
        thresholds). No time axis either: the paper is single still-image, full stop (checked
        against the paper text directly, 2026-10-08) -- every month here is scored and
        recursed on INDEPENDENTLY, with --apt_time forced to "none" by the caller (plain
        per-frame H_space, no H_time term: this codebase's time extensions are additions
        beyond the paper, not part of it).

        grid_sizes: coarsest-to-finest grid side lengths, e.g. (4, 8, 16, 32) for patch sizes
        (16, 8, 4, 2) px on a 64px tile -- a strict quadtree, each next grid exactly 2x the
        side (4x the cells) of the one before, matching the paper's "2^i * p" hierarchy
        exactly (Section 3.1: "we impose the constraint that all patches follow a quadtree-
        like structure"). taus: len(grid_sizes) - 1 thresholds, taus[i] gating the decision
        at grid_sizes[i] (coarser) vs recursing into grid_sizes[i+1] (finer) -- one threshold
        per non-finest level, per the paper ("a tunable hyperparameter for each level"); the
        finest level never needs one (nothing left to split into).

        normalize=True (default HERE, unlike _apt_scores' own default False): entropy at
        different grid sizes is NOT comparable raw -- a finer cell has fewer pixels, so its
        histogram cannot reach the same entropy ceiling no matter how diverse its content is
        (ps2's 4 pixels cap at log2(4)=2 bits; ps16's 256 pixels reach the full log2(32)=5 bit
        ceiling). Since taus here are compared ACROSS levels in the same recursion, using raw
        (un-normalized) entropy would make taus[0] and taus[-1] mean completely different
        things -- normalize=False is for completeness/debugging only, not a real use case for
        this specific method (see _apt_scores' own docstring for the full derivation).

        Algorithm (paper's own prose IS its algorithm, no separate pseudocode in the source):
        start at the coarsest grid, compute entropy per cell; KEEP a cell (assign it that
        patch size) if its entropy < tau for that level; else SPLIT it into its 4 quadtree
        children at the next-finer grid and repeat the test on each child; cells still
        unresolved at the finest grid are simply assigned that smallest size (no test there).

        Returns a dict {grid_size: mask (B, T, grid_size, grid_size) bool} -- mask[b,t,y,x]
        True means "this cell, at THIS grid's resolution, is a kept (not-further-split) patch
        for this sample/month". Every present cell appears exactly once across the whole dict
        (at exactly one grid_size) -- same "partition of the tile" invariant _apt_res_mask's
        mask4/mask16 pair already keeps, just generalized to N levels instead of 2."""
        assert len(taus) == len(grid_sizes) - 1
        old_apt_time = self.apt_time
        self.apt_time = "none"                                       # paper has no time axis
        try:
            scores = {g: self._apt_scores(ms, present, B, g, normalize=normalize)
                     for g in grid_sizes}
        finally:
            self.apt_time = old_apt_time

        out: dict[int, torch.Tensor] = {}
        # "live" = still unresolved at the current level (every present cell, initially).
        live = present[:, :, None, None].expand_as(scores[grid_sizes[0]]).clone()
        for i, g in enumerate(grid_sizes):
            if i == len(grid_sizes) - 1:
                out[g] = live                                        # finest: no test, assign all
                break
            keep = live & (scores[g] < taus[i])
            out[g] = keep
            split = live & ~keep
            live = split.repeat_interleave(2, -1).repeat_interleave(2, -2)  # -> next-finer grid
        return out

    def _pool_entropy(self, block) -> float:
        """Single-block wrapper around _pool_entropy_batch (kept for the unit tests and any
        one-off caller -- see that function's docstring for the real, vectorized primitive
        _apt_octree_mask's recursion actually uses). Accepts either a torch tensor or a numpy
        array (old tests/callers may still pass numpy; _pool_entropy_batch itself is torch-
        only, see its docstring)."""
        if not torch.is_tensor(block):
            block = torch.as_tensor(block)
        return float(self._pool_entropy_batch(block[None])[0])

    def _pool_entropy_batch(self, blocks: torch.Tensor) -> torch.Tensor:
        """H(P) for N blocks AT ONCE, blocks (N, ps, ps, n_t, C) all sharing one (ps, n_t, C)
        shape -- returns (N,) normalized entropy values, SAME device as `blocks`. POOLS every
        (band, timestep, pixel) within each block into ONE histogram, same semantics
        _pool_entropy always had (see history below); batched so _apt_octree_mask's recursion
        pays ONE call's overhead for an entire tree LEVEL (every live node across the whole
        batch, grouped by shared block shape) instead of one call per node per sample.

        TORCH, on whatever device `blocks` already lives on (2026-10-12 perf fix #4, GPU not
        CPU) -- NOT the numpy version two earlier attempts went through: fix #1 (per-sample
        .cpu().numpy(), removing per-NODE GPU round-trips) and fix #2 (batching nodes within a
        level into one call per shared shape, removing the B=32 Python-loop multiplier)
        together only got this to ~9.8-14.7s/batch on PASTIS-scale data (still ~20-30x too
        slow for a 3h Slurm limit to cover meaningful training) -- profiling fix #2's output
        showed the real cost is DATA VOLUME, not dispatch overhead: a single level's largest
        group can be ~22M scalars (e.g. 512 live ps16 x 12-month x 14-band nodes), and a tree
        level produces several such groups. That volume is exactly where a GPU's throughput
        (not latency) matters, and fix #2's batching already made every call large enough
        (thousands-to-millions of elements) to amortize kernel-launch cost -- the numpy
        versions were leaving the GPU idle AFTER already paying to batch for it. This version
        keeps binning/histogramming/entropy as torch ops on `blocks`' own device, using
        scatter_add_ in place of np.bincount (no CPU histogram primitive to reach for on GPU).

        Original (per-block) semantics, preserved exactly:
        a block that looks IDENTICAL at every timestep scores LOW (little joint space-time
        diversity) even if each individual frame is itself detailed -- user-confirmed
        2026-10-09: this is what makes "stay coarse in time" a meaningful, information-driven
        choice rather than a free action with no entropy cost.

        Normalized the same way as _apt_scores(normalize=True): divided by this block's own
        achievable ceiling min(log2(APT_BINS), log2(n_pixel_samples)) -- ESSENTIAL here, not
        just nice-to-have, since _apt_octree_mask compares ONE tau across nodes of wildly
        different (ps, n_timesteps) shapes; without normalization a tau could never mean the
        same thing for a ps16 x 20-timestep node as for a ps2 x 1-timestep one.

        n_t == 0 (every month in this range is absent for this cell) returns 0.0 for that
        block -- an all-absent block has nothing to pool; the caller treats this as a trivial
        always-stop leaf costing 0 tokens, same as every other selector in this file simply
        skipping absent slots rather than assigning them a crop.

        CRITICAL: the ceiling must divide EACH BAND's entropy before summing over bands, not
        the summed total -- otherwise the normalization under-corrects by a factor of C (first
        draft got this backwards: divided the ALREADY-summed-over-bands entropy by one scalar
        ceiling, so a 14-band block's normalized score came out ~12 instead of landing in
        [0, C] like _apt_scores(normalize=True) does -- caught by a direct smoke test on real
        PASTIS data showing h_parent=12.4, nowhere near a sane tau like 0.85; matches
        _apt_scores' OWN ceil_per_band convention exactly, see its docstring)."""
        import math
        N = blocks.shape[0]
        device = blocks.device
        if blocks.shape[3] == 0:                                        # n_t == 0 for the group
            return torch.zeros(N, device=device)
        flat = blocks.reshape(N, -1, blocks.shape[-1]).float()              # (N, ps*ps*n_t, C)
        lo = flat.amin(1, keepdim=True)
        hi = flat.amax(1, keepdim=True)
        norm = (flat - lo) / (hi - lo + 1e-6)
        binned = (norm * (APT_BINS - 1)).round().clamp(0, APT_BINS - 1).long()
        n_samples, C = flat.shape[1], flat.shape[2]
        # Vectorized per-(N,C) histogram via scatter_add_ (torch's GPU-native alternative to
        # np.bincount): offset each (n, c) slice's bin codes into its own disjoint range so one
        # scatter over the WHOLE group fills every (n, c)'s histogram at once.
        c_idx = (torch.arange(N * C, device=device) * APT_BINS).view(N, 1, C)
        flat_idx = (c_idx + binned).reshape(-1)                          # (N*n_samples*C,)
        counts = torch.zeros(N * C * APT_BINS, device=device)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        onehot = counts.view(N, C, APT_BINS)
        p = onehot / onehot.sum(-1, keepdim=True).clamp(min=1e-8)
        h_per_band = -(p * (p + 1e-8).log2()).sum(-1)                    # (N, C)
        ceil_per_band = min(math.log2(APT_BINS), math.log2(max(n_samples, 2)))
        return (h_per_band / ceil_per_band).sum(-1)                      # (N,), summed AFTER normalizing

    @torch.no_grad()
    def _apt_octree_mask(self, ms, present, B, tau, ps_min=2, ps_max=16, space_only=False):
        """User's 2026-10-09 idea: "start with ps16 and t=all, calculate entropy, and if above
        threshold then split in either time dimension or space dimension, or both, depending
        on the after-split entropy -- this is almost info gain." A true 3D (y, x, t) octree,
        generalizing _apt_quadtree_mask's space-only quadtree to let the SAME recursion also
        choose to coarsen/split in TIME -- the first variant in this file where staying coarse
        in time is an entropy-driven choice at every tree node, not a fixed param (--apt_time)
        or a separate post-hoc merge pass (--apt_pool).

        space_only=True (2026-10-13, isolating experiment: does apt_octree's PASTIS loss to
        --selector grid at a truly matched token budget, k=1/6, come from its spatial pick
        placement or from the ~92% of its leaves that pixel+slot-AVERAGE several months into
        one blurred token, confirmed by direct measurement on 20 real test samples -- vs
        grid's one sharp, single-month token per picked cell) starts every ps_max node at a
        SINGLE present month instead of the full present-rank range, so rank_lens is always 1
        and can_time is always False for every node at every level -- the time candidate is
        never offered, no Python branch needed (the existing can_space/can_time masking in the
        recursion below already handles this for free). Produces a per-month space-only
        quadtree, structurally identical to grid's one-token-per-(month,cell) layout, so any
        remaining accuracy gap against grid isolates the SPATIAL placement choice alone.

        No token budget (user-confirmed 2026-10-09: "I wanna see the tokenization that would
        come out of it first, without training at all") -- pure entropy THRESHOLD, same
        philosophy as _apt_quadtree_mask, ONE shared tau across every node regardless of its
        (ps, n_timesteps) shape (see _pool_entropy's own docstring for why normalization makes
        this comparison meaningful).

        Recursion per node (ps, present-rank range [r0,r1)), starting at every present ps_max
        cell x its FULL present-rank range:
          1. H = _pool_entropy(node). If H < tau (or node is ALREADY ps_min with a single
             present month, nothing left to split into): STOP, LEAF (1 token -- its time range
             would be pixel+slot averaged into one crop by the same mechanism _apt_pool_crops
             already uses, user-confirmed 2026-10-09 this is what makes "stay coarse in time"
             a real token saving, not free; this function itself only decides the TREE shape,
             building actual crops from it is a separate, not-yet-wired step, see below).
          2. Else, evaluate whichever of the 3 candidate splits are available (space needs
             ps > ps_min, time needs more than 1 present month in range):
               space:  ps -> ps/2, 4 children (same range)              -- cost 4
               time:   range -> 2 halves of present RANKS (never raw month indices -- an
                       absent month can never anchor a half on its own: ranks are built from
                       an ALREADY present-only month list, so a halved range is automatically
                       present-only too), 2 children (same ps)           -- cost 2
               both:   both at once, 8 children                          -- cost 8
             gain(split) = H(parent) - sum(children's H); GAIN PER TOKEN = gain / (cost - 1)
             (cost - 1 = the EXTRA tokens spent over the 1-token leaf alternative). Pick
             whichever available split has the highest gain-per-token; if none is POSITIVE
             (splitting would cost tokens for no entropy reduction), STOP here too -- this is
             the actual "info gain" structure requested: compare NOT splitting against every
             way OF splitting, take the best.
          3. Recurse into the winning split's children.

        Returns a list (len B) of per-sample leaf lists, each leaf
        (y0, x0, ps, present_months: list[int]) -- present_months (not a [r0,r1) rank range)
        so any caller can build a crop without needing this function's internal present-rank
        bookkeeping. Intentionally NOT a dict-of-masks like _apt_quadtree_mask's return (that
        shape assumes a small fixed set of GLOBAL grid levels every cell aligns to; here every
        leaf can have its own independent (ps, time-range) shape) -- viz-only for now, not yet
        wired to any training-time crop-building path."""
        pixels = self._apt_pixels(ms)                              # (B, Hpx, Wpx, T, C)
        Hpx, T = pixels.shape[1], pixels.shape[3]
        device = pixels.device
        months_b = [present[b].nonzero(as_tuple=True)[0].tolist() for b in range(B)]
        # months_t[b, r] = months_b[b][r] for r < len(months_b[b]), else 0 (padding -- never
        # read: every (r0, r1) a node can carry satisfies r1 <= len(months_b[b]), see
        # _apt_octree_entropy_group's docstring). Built once per batch for that function's
        # vectorized gather.
        months_t = torch.zeros(B, T, dtype=torch.long, device=device)
        for b in range(B):
            months_t[b, :len(months_b[b])] = torch.tensor(months_b[b], device=device)

        # 2026-10-12 perf fix #2 (grouped batching) + fix #4 (GPU, not CPU numpy -- see
        # _pool_entropy_batch's docstring for the full history/measurements): breadth-first by
        # LEVEL instead of one Python-recursive call per node per sample -- a "node" is
        # (b, y0, x0, ps, r0, r1); [r0, r1) indexes INTO months_b[b] (a present-RANK range,
        # never raw month indices, same convention the old per-sample recursion used). Every
        # live node at a level is evaluated for its h_parent AND every one of its live
        # candidate children in ONE _pool_entropy_batch call per shared (ps, n_months) GROUP --
        # grouping is required because _pool_entropy_batch needs one dense (N, ps, ps, n_t, C)
        # array per call, and different nodes can have different ps/n_months even within one
        # level (space-only vs time-only vs both splits fan out to different child shapes).
        # `pixels` stays on `device` throughout (no .cpu().numpy(), unlike fix #1 -- the
        # groups built here are large enough that GPU throughput beats CPU for them, see
        # _pool_entropy_batch's docstring).
        live = torch.tensor(
            ([(b, gy * ps_max, gx * ps_max, ps_max, r, r + 1)
              for b in range(B)
              for r in range(len(months_b[b]))
              for gy in range(Hpx // ps_max) for gx in range(Hpx // ps_max)] if space_only else
             [(b, gy * ps_max, gx * ps_max, ps_max, 0, len(months_b[b]))
              for b in range(B)
              for gy in range(Hpx // ps_max) for gx in range(Hpx // ps_max)]),
            dtype=torch.long, device=device)
        all_leaves: list[list[tuple]] = [[] for _ in range(B)]

        # 2026-10-12 perf fix #6 (vectorize the LEVEL LOOP itself, not just the gather inside
        # _apt_octree_entropy_group): fix #5 made that function fast (0.081s of a 4.49s batch,
        # 2%) but left the Python loops THIS function builds around it (constructing each
        # live node's candidates, flattening children, the final per-node decision) untouched
        # -- those are 98% of the remaining cost, iterating thousands of live nodes per level
        # in pure Python. Rewritten below as pure tensor ops: every node's 3 candidates (space
        # cost4, time cost2, both cost8 -- same menu as the original per-node recursion) are
        # built via fixed-shape broadcasts over the WHOLE live set at once, with can_space/
        # can_time masking (not Python if/else) controlling which actually count toward the
        # argmax -- an "unavailable" candidate's children get r1<=r0 (empty), which
        # _apt_octree_entropy_group already short-circuits to entropy 0, but gain_per_token is
        # additionally -inf'd for masked candidates below so a free "0 entropy" child could
        # never look like a winning split.
        hp_half = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1)], device=device)  # 4 space offsets
        while live.numel():
            h_parent = self._apt_octree_entropy_group(pixels, months_t, live)
            b, y0, x0, ps, r0, r1 = live.unbind(1)
            rank_lens = r1 - r0
            can_space = ps > ps_min
            can_time = rank_lens > 1
            hp, rm = ps // 2, (r0 + r1) // 2
            N = live.shape[0]

            # Candidate "space": 4 children per node, same (r0, r1), half ps, offset by hp.
            sp_y = y0[:, None] + hp_half[None, :, 0] * hp[:, None]            # (N, 4)
            sp_x = x0[:, None] + hp_half[None, :, 1] * hp[:, None]            # (N, 4)
            space_children = torch.stack([
                b[:, None].expand(-1, 4), sp_y, sp_x, hp[:, None].expand(-1, 4),
                r0[:, None].expand(-1, 4), r1[:, None].expand(-1, 4)], -1)    # (N, 4, 6)
            space_children = torch.where(can_space[:, None, None], space_children,
                                          torch.zeros_like(space_children))   # r1=r0=0 if disabled

            # Candidate "time": 2 children, same (y0, x0, ps), halved rank range.
            time_children = torch.stack([
                torch.stack([b, y0, x0, ps, r0, rm], -1),
                torch.stack([b, y0, x0, ps, rm, r1], -1)], 1)                # (N, 2, 6)
            time_children = torch.where(can_time[:, None, None], time_children,
                                         torch.zeros_like(time_children))

            # Candidate "both": 8 children, half ps x halved rank range (4 space offsets x 2
            # time halves).
            both_mask = can_space & can_time
            bo_y = y0[:, None] + hp_half[None, :, 0] * hp[:, None]            # (N, 4)
            bo_x = x0[:, None] + hp_half[None, :, 1] * hp[:, None]            # (N, 4)
            both_r0 = torch.stack([r0, rm], -1)                              # (N, 2)
            both_r1 = torch.stack([rm, r1], -1)                              # (N, 2)
            both_children = torch.stack([
                b[:, None, None].expand(-1, 4, 2),
                bo_y[:, :, None].expand(-1, 4, 2), bo_x[:, :, None].expand(-1, 4, 2),
                hp[:, None, None].expand(-1, 4, 2),
                both_r0[:, None, :].expand(-1, 4, 2), both_r1[:, None, :].expand(-1, 4, 2),
            ], -1).reshape(N, 8, 6)
            both_children = torch.where(both_mask[:, None, None], both_children,
                                         torch.zeros_like(both_children))

            flat_children = torch.cat(
                [space_children.reshape(-1, 6), time_children.reshape(-1, 6),
                 both_children.reshape(-1, 6)], 0)                           # (N*14, 6)
            h_children_flat = self._apt_octree_entropy_group(pixels, months_t, flat_children)
            h_space = h_children_flat[:N * 4].view(N, 4).mean(1)
            h_time = h_children_flat[N * 4:N * 6].view(N, 2).mean(1)
            h_both = h_children_flat[N * 6:].view(N, 8).mean(1)

            neg_inf = torch.full((N,), float("-inf"), device=device)
            gain_space = torch.where(can_space, (h_parent - h_space) / 3, neg_inf)
            gain_time = torch.where(can_time, (h_parent - h_time) / 1, neg_inf)
            gain_both = torch.where(both_mask, (h_parent - h_both) / 7, neg_inf)
            gains = torch.stack([gain_space, gain_time, gain_both], -1)       # (N, 3)
            best_gain, best_cand = gains.max(-1)                              # (N,)

            rank_months_len = rank_lens                                       # alias for clarity
            is_leaf = (rank_months_len == 0) | (h_parent < tau) | (best_gain <= 0)
            leaf_idx = is_leaf.nonzero(as_tuple=True)[0].tolist()
            for i in leaf_idx:
                bi = int(b[i])
                r0i, r1i = int(r0[i]), int(r1[i])
                all_leaves[bi].append((int(y0[i]), int(x0[i]), int(ps[i]), months_b[bi][r0i:r1i]))

            split_idx = (~is_leaf).nonzero(as_tuple=True)[0]
            chosen_children = []
            for cand_i, children in ((0, space_children), (1, time_children), (2, both_children)):
                sel = split_idx[best_cand[split_idx] == cand_i]
                if sel.numel():
                    chosen_children.append(children[sel].reshape(-1, 6))
            live = (torch.cat(chosen_children, 0) if chosen_children
                    else torch.zeros(0, 6, dtype=torch.long, device=device))
        return all_leaves

    def _apt_octree_entropy_group(self, pixels, months_t, nodes_t):
        """_apt_octree_mask's per-level batching primitive: `nodes_t` is a (Nn, 6) LONG tensor
        of (b, y0, x0, ps, r0, r1) rows, possibly spanning several distinct (ps, r1-r0) shapes
        (e.g. one level's parents are all ps_max-sized but their candidate children fan out to
        ps_max/2 and/or halved time ranges) -- groups them by shared (ps, r1-r0) so each group
        can be gathered into one dense tensor for a SINGLE _pool_entropy_batch call, then
        scatters the per-group results back into one (Nn,) tensor in the caller's original
        order. `months_t` is (B, T) LONG, months_t[b, r] = that sample's r'th PRESENT raw
        month (see _apt_octree_mask for how it's built; padded entries beyond a sample's own
        present-month count are never read, since no valid r0:r1 range reaches them).

        2026-10-12 perf fix #5 (vectorized gather, no Python node loop) -- fix #4 (batching +
        GPU) alone barely moved the needle (8.7-16.1s/batch, same ballpark as the CPU numpy
        version) because THIS function's per-group gather was still a Python list
        comprehension slicing one node at a time before stacking: profiling isolated this
        exact loop at 0.324s for a single 5312-node group alone (of ~32 such groups across a
        batch's ~7 tree levels) -- the real cost was never the entropy math, it was building
        `blocks` one slow Python-level tensor slice per node. Replaced with torch advanced
        indexing (one gather over y/x/t offset grids for the WHOLE group at once) -- same
        5312-node case: 0.054s, 6x faster, verified bit-identical to the old per-node slicing.

        r1 == r0 (an empty rank range -- can only arise from an odd present-month count's time
        split) is its own degenerate group, short-circuited to entropy 0.0 without touching
        _pool_entropy_batch at all (an empty block has nothing to pool, same convention
        _apt_octree_recurse always used for "not rank_months")."""
        device = pixels.device
        Nn = nodes_t.shape[0]
        out = torch.zeros(Nn, device=device)
        b, y0, x0, ps, r0, r1 = nodes_t.unbind(1)
        n_t = r1 - r0
        valid = n_t > 0                                     # out[i] stays 0.0 where False
        shape_key = ps * 10_000 + n_t                        # unique per (ps, n_t) combo
        for key in shape_key[valid].unique().tolist():
            idx = ((shape_key == key) & valid).nonzero(as_tuple=True)[0]
            ps_g, n_t_g = int(ps[idx[0]]), int(n_t[idx[0]])
            bg, y0g, x0g, r0g = b[idx], y0[idx], x0[idx], r0[idx]
            dy = torch.arange(ps_g, device=device)
            dx = torch.arange(ps_g, device=device)
            dt = torch.arange(n_t_g, device=device)
            yy = y0g[:, None] + dy[None, :]                           # (G, ps)
            xx = x0g[:, None] + dx[None, :]                           # (G, ps)
            rr = r0g[:, None] + dt[None, :]                           # (G, n_t) -- PRESENT-RANK offsets
            tt = months_t[bg[:, None], rr]                            # (G, n_t) -- raw month indices
            blocks = pixels[bg[:, None, None, None], yy[:, :, None, None],
                             xx[:, None, :, None], tt[:, None, None, :]]  # (G, ps, ps, n_t, C)
            out[idx] = self._pool_entropy_batch(blocks)
        return out

    def _apt_octree_crops(self, ms, all_leaves, season, frac):
        """Build real crops from _apt_octree_mask's leaves ((y0, x0, ps, months) per sample),
        -> the SAME dict-list format _res_crops returns ({"hr", "idx", "logw", "size"}), so
        forward()'s existing `dates = dates + self._res_crops(...)` / NativeSetReadout path
        consumes octree crops identically to ps16/ps8 --res_mix crops, no new readout code.

        A multi-month leaf's member months are combined into ONE token by
        self.apt_octree_pool (user's 2026-10-12 question: "how are the time series patches
        handled... input composites or embedding pool or something else" -- all 3 are real,
        selectable options now, compute cost input < embed < attn as the user expects, with
        "input" the original/default):
          "input": pixel+slot AVERAGING before embedding (_apt_pool_crops' "just simply
            interpolate" mechanism, generalized here to any ps -- OlmoEarth's sincos time
            encoding is continuous, so a fractional/averaged slot needs no new embedding).
            Cheapest: one embed() call per leaf regardless of its member count.
          "embed" / "attn": embed EVERY member month SEPARATELY (own true date/slot each, same
            convention _pool_acquisitions already uses for raw acquisitions-within-a-month),
            run blocks [0, self.mix.L) on them, then pool the OUTPUT embeddings into one token
            per leaf -- mean ("embed") or a learned per-token softmax weight ("attn", reusing
            --acq_pool's existing self.acq_score Linear + exp-weighted pool exactly). Costs
            one embed() + one blocks[0,L) pass per MEMBER MONTH, not per leaf -- the "it's
            clear input < embed < attn computation-wise" the user named (attn adds one more
            Linear + softmax-like normalization over embed's extra per-member blocks pass).

        A leaf's span is drawn from PRESENT months only (months list). A leaf's footprint for
        NativeSetReadout is f = ps // LOOKUP_PS (identical convention to _res_crops; ps_min=4
        is REQUIRED for training -- see _apt_octree_mask's ps_min docstring -- so f is always
        a whole number >= 1, never the ps2 case that would give f=0 and has no representation
        in this codebase's footprint-pooling scheme)."""
        by_ps: dict[int, list[tuple]] = {}
        for b, leaves in enumerate(all_leaves):
            for y0, x0, ps, months in leaves:
                if not months:                       # an all-absent node: no crop to build
                    continue
                by_ps.setdefault(ps, []).append((b, y0, x0, months))

        out = []
        for ps, entries in by_ps.items():
            b = torch.tensor([e[0] for e in entries], device=season.device)
            # _apt_octree_mask's y0/x0 are already TOP-LEFT PIXEL coords (gy*ps_max, not a
            # block-grid index) -- unlike _res_crops' y/x (ps4-CELL indices, hence its own
            # `py, px = y * ps` conversion). Using that same conversion here was a real bug
            # caught by the first real-model smoke test (an out-of-bounds CUDA assert in
            # patch_embeddings.forward -- e.g. a ps16 leaf at pixel y0=48 got re-multiplied to
            # py=768, nowhere near the 64px tile) -- py/px are y0/x0 directly, no rescale.
            py = torch.tensor([e[1] for e in entries], device=season.device)
            px = torch.tensor([e[2] for e in entries], device=season.device)
            t_rep = torch.tensor([e[3][0] for e in entries], device=season.device)  # 1st month
            sizes = torch.tensor([len(e[3]) for e in entries], device=season.device)
            f = ps // LOOKUP_PS
            idx = (b, t_rep, py // LOOKUP_PS, px // LOOKUP_PS)

            if self.apt_octree_pool == "input" or bool((sizes == 1).all()):
                hr = self._apt_octree_crops_input(ms, b, py, px, t_rep, ps, sizes, entries,
                                                   season, frac)
            else:
                hr = self._apt_octree_crops_embed(ms, b, py, px, t_rep, ps, sizes, entries,
                                                   season, frac)
            out.append({"hr": self.mix.head_blocks(hr, self._lr, b), "idx": idx,
                        "logw": None, "size": f})
        return out

    def _apt_octree_crops_input(self, ms, b, py, px, t_rep, ps, sizes, entries, season, frac):
        """--apt_octree_pool input (default): pixel+slot average every leaf's member months
        BEFORE embedding, same mechanism _apt_pool_crops uses (generalized to any ps -- its
        own pooling math was never actually ps4-specific, only its caller was). One embed()
        call total, covering every leaf in this ps group at once."""
        slot = self._crop_slot(season[b, t_rep], frac[b, t_rep])
        if bool((sizes == 1).all()):
            crops = crop_patches(ms, b, t_rep, py, px, ps, pixels=True)
            return self.mix.embed(crops, ps, slot=slot, py=py, px=px)
        from manyup.encoders.olmoearth import _map
        rep = torch.repeat_interleave(torch.arange(len(entries), device=season.device), sizes)
        b_e, py_e, px_e = b[rep], py[rep], px[rep]
        t_e = torch.tensor([m for e in entries for m in e[3]], device=season.device)
        members = crop_patches(ms, b_e, t_e, py_e, px_e, ps, pixels=True)

        def pool(k, v):
            if k == "timestamps":
                return v[:, :1]
            if v.ndim != 5:
                raise ValueError(f"_apt_octree_crops_input: {k} is not (N, H, W, T, C)")
            acc = torch.zeros(len(entries), *v.shape[1:], dtype=torch.float32, device=v.device)
            return acc.index_add_(0, rep, v.float()).div_(
                sizes[:, None, None, None, None].float()).to(v.dtype)

        crops = _map(members, pool)
        crops = crops._replace(timestamps=members.timestamps[
            torch.cumsum(sizes, 0) - sizes])          # first member's timestamp per leaf
        slot_e = slot[rep].float()
        slot = torch.zeros(len(entries), device=season.device, dtype=torch.float32).index_add_(
            0, rep, slot_e).div_(sizes.float())
        return self.mix.embed(crops, ps, slot=slot, py=py, px=px)

    def _apt_octree_crops_embed(self, ms, b, py, px, t_rep, ps, sizes, entries, season, frac):
        """--apt_octree_pool embed / attn: embed every member month SEPARATELY (own true date,
        same py/px as its leaf), run blocks [0, self.mix.L), pool the resulting EMBEDDINGS
        (not pixels) into one token per leaf -- mean ("embed") or a learned per-token softmax
        weight ("attn", reusing --acq_pool's self.acq_score exactly: same Linear, same
        exp-clamp-weighted pool). Direct generalization of _pool_acquisitions' pattern (there:
        pool raw acquisitions INTO a month composite's token slots; here: pool member MONTHS
        into one leaf token) -- one embed() + blocks[0,L) pass per MEMBER MONTH, the
        "more compute than input-pool" cost the user named.

        Every member of one leaf shares identical dims/token-layout (same ps, same modality
        set -- only the pixels/date differ), and EVERY leaf in this call shares the SAME ps
        too (by construction: _apt_octree_crops groups leaves by ps before calling this), so
        n_slots (tokens per member) is one shared constant, not something that needs tracking
        per leaf or per member."""
        n_leaves = len(entries)
        rep = torch.repeat_interleave(torch.arange(n_leaves, device=season.device), sizes)
        b_e, py_e, px_e = b[rep], py[rep], px[rep]
        t_e = torch.tensor([m for e in entries for m in e[3]], device=season.device)
        slot_e = self._crop_slot(season[b_e, t_e], frac[b_e, t_e])
        members = crop_patches(ms, b_e, t_e, py_e, px_e, ps, pixels=True)
        n_members = len(b_e)                                      # = int(sizes.sum())
        emb = self.mix.embed(members, ps, slot=slot_e, py=py_e, px=px_e)
        n_slots = int(emb.seqlens[0])                              # tokens per member (shared)
        if bool((emb.seqlens != n_slots).any()):
            # keep (modality/bandset PRESENCE) is a per-TASK property (which modalities
            # task.olmoearth.modalities declares), not per-date -- per-pixel validity within a
            # date is a separate VALUE mask, not a token-count mask, so every member crop of
            # one task/ps should embed to the SAME token count regardless of which month it's
            # from. Verified directly on pastis/biomassters/yieldsat_raw (uniform seqlens on
            # real present-month sequences, incl. gaps) before relying on this in the flat
            # row // n_slots indexing below -- if a task ever violates it, fail loudly here
            # rather than silently mispool tokens of different modalities/bandsets together.
            raise NotImplementedError(
                "apt_octree_pool embed/attn assumes every member month of a leaf embeds to "
                f"the SAME token count (verified on pastis/biomassters/yieldsat_raw) -- this "
                f"task/ps violates it (seqlens={emb.seqlens.tolist()}); the flat row//n_slots "
                "pooling index below would silently mispool mismatched token layouts")
        x = self.mix.blocks(emb.x, emb.seqlens, 0, self.mix.L).float()
        w = torch.ones(len(x), device=x.device)
        if self.apt_octree_pool == "attn":
            w = torch.exp(self.apt_octree_score(x)[:, 0].clamp(-20, 20))
        # emb.x is one row per (member, token-slot), member-major, slot-minor (seqlens are all
        # n_slots, so member_of / tok_slot are both a simple div/mod of the flat row index --
        # no need for emb.seqlens' cumsum the way a ragged-seqlen case like _pool_acquisitions
        # needs, since every member here has the IDENTICAL token count by construction).
        row = torch.arange(len(x), device=x.device)
        member_of, tok_slot = row // n_slots, row % n_slots
        leaf_of = rep[member_of]                                   # (N_tok,): which LEAF
        key = leaf_of * n_slots + tok_slot
        total = x.new_zeros(n_leaves * n_slots, x.shape[-1]).index_add_(0, key, x * w[:, None])
        weight = x.new_zeros(n_leaves * n_slots).index_add_(0, key, w)
        pooled = (total / weight.clamp(min=1e-6)[:, None]).to(emb.x.dtype)
        # keep/dims/masks describe the (identical, shared) per-member token layout -- take
        # them from the FIRST member of each leaf (any member works, they're all identical);
        # emb.yx is each member's own (py,px) centre -- every member of a leaf shares py,px by
        # construction too, so the first member's yx is also the leaf's.
        first = torch.cumsum(sizes, 0) - sizes                      # each leaf's first member row
        ref = emb.subset(torch.zeros(n_members, dtype=torch.bool, device=x.device).scatter_(
            0, first, True))
        return Packed(pooled, torch.full((n_leaves,), n_slots, device=x.device, dtype=torch.long),
                     ref.keep, ref.dims, ref.masks, ref.yx)

    def _date_scores(self, views, cands, present, lr_L, G, ss_month):
        """Learned date scores (B, T_m, H, W) per modality, standardized per sample over all
        its date candidates (so sigmoid gates stay away from 0 / 1)."""
        out = []
        for (m, d_sample, d_month, d_frac, d_season, ok), c in zip(views, cands):
            e = self.month_embedding(_to_device(d_sample, present.device, torch.bfloat16)).float()
            e_bar = season_mean(e, d_season, ok, G)
            valid = seasons_used(d_season, ok, G)
            out.append(self.date_selector[m](e, e_bar, lr_L, d_season, ok, valid,
                                             d_sample.timestamps[:, :, 1].to(present.device),
                                             ss_month))
        flat = torch.cat([o.flatten(1) for o in out], 1)
        w = torch.cat([c.flatten(1) for c in cands], 1).float()
        n = w.sum(1, keepdim=True).clamp(min=2)
        mu = (flat * w).sum(1, keepdim=True) / n
        var = (((flat - mu) * w) ** 2).sum(1, keepdim=True) / (n - 1)
        z = (flat - mu) * torch.rsqrt(var + 1e-6)
        return list(torch.split(z, [o[0].numel() for o in out], 1)), [o.shape for o in out]

    def _date_crops(self, extra, month_mask, present, season, lr_m, scores, r, B, H,
                    lr_L=None, G=0, ss_month=None):
        """Level 2: date crops. Candidates (modality, date, cell) of the dates inside the month
        grid whose month is present; `nested`: only cells whose MONTH was picked at level 1
        (a picked month-cell is refined into its dates), `free`: any cell. Random picks,
        date_per_month x (month picks) per sample over all modalities together. A learned
        month selector gates each date crop by its (month, cell) score, as a month crop.
        -> per modality with picks: {"hr": Packed after blocks [0, L), "idx": (b, month, y, x),
        "logw": gate log-weight or None}."""
        if month_mask is None:
            return []
        dev = present.device
        bidx = torch.arange(B, device=dev)[:, None]
        cands, views = [], []
        for m, (d_sample, d_valid, d_month, d_frac) in extra["dates"].items():
            d_valid, d_month, d_frac = d_valid.to(dev), d_month.to(dev), d_frac.to(dev)
            ok = d_valid & (d_month >= 0) & present.gather(1, d_month.clamp(min=0))   # (B, Td)
            if self.date_level == "nested":
                c = ok[:, :, None, None] & month_mask[bidx, d_month.clamp(min=0)]      # (B,Td,H,W)
            else:
                c = ok[:, :, None, None].expand(-1, -1, H, H)
            cands.append(c)
            d_season = torch.where(d_month >= 0, season.gather(1, d_month.clamp(min=0)),
                                   torch.full_like(d_month, -1))
            views.append((m, d_sample, d_month, d_frac, d_season, ok))
        C = torch.cat(cands, 1)                                                  # (B, sum Td, H, W)
        budget = (self.date_per_month * month_mask.flatten(1).sum(1).float()).round().long()
        date_z = None
        if self.date_selector is not None:
            zs, shapes = self._date_scores(views, cands, present, lr_L, G, ss_month)
            date_z = [z.view(sh) for z, sh in zip(zs, shapes)]
            self.aux_date_scores = torch.cat([z[c] for z, c in zip(date_z, cands)])
            pick = torch.cat(date_z, 1).detach()
            pick = gumbel(pick) if self.training else pick
        else:
            pick = torch.rand(C.shape, device=dev)
        sc = pick.masked_fill(~C, float("-inf")).flatten(1)
        rank = sc.argsort(1, descending=True).argsort(1)
        chosen = ((rank < budget[:, None]) & torch.isfinite(sc)).view_as(C)
        out = []
        for i, ((m, d_sample, d_month, d_frac, _, _), ch) in enumerate(
                zip(views, torch.split(chosen, [c.shape[1] for c in cands], 1))):
            bb, dd, yy, xx = ch.nonzero(as_tuple=True)
            if not len(bb):
                continue
            d_sample = _to_device(d_sample, dev, torch.bfloat16)
            mo = d_month[bb, dd]
            hr = self.mix.embed(crop_patches(d_sample, bb, dd, yy, xx, LOOKUP_PS), LOOKUP_PS,
                                slot=self._crop_slot(season[bb, mo], d_frac[bb, dd]),
                                py=yy * LOOKUP_PS, px=xx * LOOKUP_PS)
            hr.x = (hr.x + self.date_type).to(hr.x.dtype)
            hr = self.mix.head_blocks(hr, self._lr, bb)
            logw = None
            s = None
            if date_z is not None:                         # own learned score
                s = date_z[i][bb, dd, yy, xx]
            elif self.selector is not None and self.date_gate == "month":
                s = scores[bb, mo, yy, xx]
            if s is not None:
                tok = torch.repeat_interleave(torch.arange(len(bb), device=dev), hr.seqlens)
                g = torch.sigmoid(s)[tok, None]
                ref = lr_m[bb, mo, yy // r, xx // r][tok]
                hr.x = (ref + g * (hr.x.float() - ref)).to(hr.x.dtype)
                logw = F.logsigmoid(s.float())
            out.append({"hr": hr, "idx": (bb, mo, yy, xx), "logw": logw})
        return out

    def _joint_date_crops(self, extra, month_mask, present, season, budget, B, H):
        """--date_joint: pick + encode the date crops. month_mask (B, T, H, W) or None: only
        dates inside a picked (month, cell) are candidates (hier). budget (B,) tokens.
        -> crop sets as _date_crops; self._date_sets = [(scores at the picks (n,) with grad,
        sample b (n,), tokens per crop)] in the same order, for _joint_values."""
        dev = present.device
        bidx = torch.arange(B, device=dev)[:, None]
        with torch.no_grad():
            F0 = self._lr.pooled().float()                                     # (B, NS, h, w, D)
        G, D = F0.shape[1], F0.shape[-1]
        views, cands, ps, cost = [], [], [], []
        for m, (d_sample, d_valid, d_month, d_frac) in extra["dates"].items():
            d_valid, d_month, d_frac = d_valid.to(dev), d_month.to(dev), d_frac.to(dev)
            ok = d_valid & (d_month >= 0) & present.gather(1, d_month.clamp(min=0))   # (B, Td)
            c = (ok[:, :, None, None] & month_mask[bidx, d_month.clamp(min=0)]
                 if month_mask is not None else ok[:, :, None, None].expand(-1, -1, H, H))
            d_season = torch.where(d_month >= 0, season.gather(1, d_month.clamp(min=0)),
                                   torch.full_like(d_month, -1))
            d_sample = _to_device(d_sample, dev, torch.bfloat16)
            with torch.no_grad():
                e = self.month_embedding(d_sample).float()                      # (B, Td, H, W, D)
                e_bar = season_mean(e, d_season, ok, G)
                slot = d_season.clamp(min=0).float() + d_frac.float()
                tenc = get_1d_sincos_pos_encoding(slot.flatten(), D).view(*slot.shape, D).float()
            ps.append(self.date_pred[m](e, e_bar, tenc, F0, ok).float())
            views.append((m, d_sample, d_month, d_frac))
            cands.append(c)
            with torch.no_grad():
                views[-1] += (self._tokens_per_crop(d_sample),)
            cost.append(torch.full((c[0].numel(),), float(views[-1][-1]), device=dev))
        C = torch.cat([c.flatten(1) for c in cands], 1)                         # (B, N)
        P = torch.cat([p.flatten(1) for p in ps], 1)
        cost = torch.cat(cost)                                                  # (N,) tokens

        def take(key, tokens):       # best keys first while their tokens fit the budget
            order = key.argsort(1, descending=True)
            fit = (cost[order].cumsum(1) <= tokens[:, None]) & torch.isfinite(key.gather(1, order))
            return torch.zeros_like(C).scatter_(1, order, fit)

        n_rand = (self.joint_eps * budget.float()).round()
        rmask = take(torch.rand(C.shape, device=dev).masked_fill(~C, float("-inf")), n_rand)
        tmask = take(P.detach().masked_fill(~C | rmask, float("-inf")), budget.float() - n_rand)
        chosen = rmask | tmask
        out, self._date_sets = [], []
        self.date_crop_tokens = float(sum(v_[-1] for v_ in views))   # one month crop: all modalities
        for (m, d_sample, d_month, d_frac, n_tok), ch, p in zip(
                views, torch.split(chosen, [c[0].numel() for c in cands], 1), ps):
            bb, dd, yy, xx = ch.view_as(p).nonzero(as_tuple=True)
            if not len(bb):
                continue
            mo = d_month[bb, dd]
            hr = self.mix.embed(crop_patches(d_sample, bb, dd, yy, xx, LOOKUP_PS), LOOKUP_PS,
                                slot=self._crop_slot(season[bb, mo], d_frac[bb, dd]),
                                py=yy * LOOKUP_PS, px=xx * LOOKUP_PS)
            hr.x = (hr.x + self.date_type).to(hr.x.dtype)
            hr = self.mix.head_blocks(hr, self._lr, bb)
            out.append({"hr": hr, "idx": (bb, mo, yy, xx), "logw": None})
            self._date_sets.append((p[bb, dd, yy, xx], bb, float(n_tok)))
        self.last_date_tokens = float((chosen.float() * cost).sum(1).mean())
        return out

    def _joint_picks(self, ms, ss, season, present, k, H):
        """--joint picks -> (B, T, H, W) scores whose top k are the picks: round(eps k) uniform
        random slots, the rest the predictor's best other slots. Keeps the predictor's scores
        (with grad) in self._joint_p for its loss (see _joint_values)."""
        from exp.upsamplers.fit_innovation import features, pool_prelook
        from exp.upsamplers.innovation_diag import prelook
        B, T = present.shape
        dev = present.device
        ar = torch.arange(B)
        if self.joint_feat == "lrL":     # the season tokens at depth L of this pass, detached
            with torch.no_grad():
                pre = {"F0": self._lr.pooled().float(), "season": season, "present": present}
                if self.joint_embed is None:
                    pre["e"] = self.month_embedding(ms).float()
            if self.joint_embed is not None:     # the selector's own embedding: with grad
                pre["e"] = self.month_embedding(ms, self.joint_embed).float()
        else:
            _, pre = prelook(self, ms, ss, season, present)      # no grad: the pre-look pass
        if self.joint_arch.startswith("xenc") or self.joint_arch in ("xs", "xc"):
            G = pre["F0"].shape[1]
            frac = self._frac if self._frac is not None else torch.zeros_like(season, dtype=torch.float)
            with torch.no_grad():
                e = self.month_embedding(ms, slot=self._crop_slot(season.clamp(min=0), frac)).float()
            p = self.joint(e, season_mean(e, season, present, G), pre["F0"], present,
                           seasons_used(season, present, G))
        elif self.joint_arch == "xattn":
            G = pre["F0"].shape[1]
            p = self.joint(pre["e"], season_mean(pre["e"], season, present, G), pre["F0"],
                           season, present, seasons_used(season, present, G),
                           ms.timestamps[:, :, 1].to(dev), ss.timestamps[:, :, 1].to(dev))
        elif self.joint_in == "enc":
            e, F0 = pre["e"], pre["F0"]
            G, D = F0.shape[1], e.shape[-1]
            n = D // 4                           # OlmoEarth's time-slot block of a token
            tenc = e.new_zeros(T, D)
            tenc[:, n:2 * n] = get_1d_sincos_pos_encoding(torch.arange(T, device=dev).float(), n)
            f = H // F0.shape[2]
            F0 = F0.repeat_interleave(f, 2).repeat_interleave(f, 3)
            F0 = F0.permute(0, 2, 3, 1, 4).reshape(B, 1, H, H, G * D).expand(B, T, H, H, G * D)
            x = torch.cat([e - season_mean(e, season, present, G) + tenc[None, :, None, None],
                           F0], -1).reshape(B, T, H * H, -1)
            p = self.joint(x.float(), present[:, :, None].expand(B, T, H * H)).view(B, T, H, H)
        else:
            x, _, valid = features(pre, ar, dev)
            if self.joint_res:                                    # resolution flag: 0 = ps4
                x = torch.cat([x, x.new_zeros(*x.shape[:-1], 1)], -1)
            p = self.joint(x.float(), valid).view(B, T, H, H)
        self._joint_p = p
        where = present[:, :, None, None].expand(B, T, H, H)
        n_rand = round(self.joint_eps * k)
        if not self.joint_res:
            # the eps share: uniform random slots, or (--joint_explore grid) a coarse regular
            # lattice of the same size -- a coverage floor the predictor's picks are added to
            explore = (grid_scores(present, H, n_rand, self.grid_mode,
                                   self.training and self.grid_phase == "random")
                       if getattr(self, "joint_explore", "random") == "grid" and n_rand
                       else torch.rand(B, T, H, H, device=dev))               # (B, T, H, H)
            rmask = select(explore, present, n_rand) if n_rand else torch.zeros_like(where)
            tmask = torch.zeros_like(where)
            if k - n_rand > 0:
                ps = p.detach().float().masked_fill(~where | rmask, float("-inf"))
                tmask = select(ps, present, k - n_rand) & ~rmask
                if self.training and self.joint_grad in ("dps", "both"):
                    # scores z-scored per sample over its candidates (not random-picked), so
                    # sigma is in units of the score spread; with grad into the predictor
                    cand = where & ~rmask
                    cf = cand.float()
                    n = cf.flatten(1).sum(1).clamp(min=2)[:, None, None, None]
                    mu = (p.float() * cf).flatten(1).sum(1)[:, None, None, None] / n
                    var = (((p.float() - mu) * cf) ** 2).flatten(1).sum(1)[:, None, None, None] / (n - 1)
                    pz = ((p.float() - mu) * torch.rsqrt(var + 1e-6)).masked_fill(~cand, float("-inf"))
                    P = PerturbedTopK.apply(pz.flatten(1), k - n_rand, self.dps_sigma,
                                            self.dps_draws).view(B, T, H, H)
                    self._dps_P = P * tmask             # 0 at random picks: no gradient there
            return (rmask | tmask).float() + 1e-3 * torch.rand(B, T, H, H, device=dev)
        # ps4 AND ps8 candidates, one token each: the random share split evenly between the two
        # grids; the selector's share = the top of both grids' scores together
        H8 = H // 2
        x8, _, valid8 = features(pool_prelook(pre, 2), ar, dev)
        x8 = torch.cat([x8, x8.new_ones(*x8.shape[:-1], 1)], -1)
        p8 = self.joint(x8.float(), valid8).view(B, T, H8, H8)
        self._joint_p8 = p8
        where8 = present[:, :, None, None].expand(B, T, H8, H8)
        n8r = n_rand // 2
        n4r = n_rand - n8r
        r4 = (select(torch.rand(B, T, H, H, device=dev), present, n4r) if n4r
              else torch.zeros_like(where))
        r8 = (select(torch.rand(B, T, H8, H8, device=dev), present, n8r) if n8r
              else torch.zeros_like(where8))
        t4, t8 = torch.zeros_like(where), torch.zeros_like(where8)
        if k - n_rand > 0:
            s4 = p.detach().float().masked_fill(~where | r4, float("-inf")).flatten(1)
            s8 = p8.detach().float().masked_fill(~where8 | r8, float("-inf")).flatten(1)
            both = torch.cat([s4, s8], 1)                                    # (B, T*H*H + T*H8*H8)
            top = torch.zeros_like(both, dtype=torch.bool)
            top.scatter_(1, both.topk(k - n_rand, 1).indices, True)
            top &= torch.isfinite(both)
            t4, t8 = top[:, :s4.shape[1]].view_as(where), top[:, s4.shape[1]:].view_as(where8)
        self._res_masks = {8: r8 | t8}
        self.last_ps8_share = (float(t8.sum()) / max(1, k - n_rand) / B, float((r8 | t8).sum()) / k / B)
        return (r4 | t4).float() + 1e-3 * torch.rand(B, T, H, H, device=dev)

    @torch.no_grad()
    def _pair_prelook(self, ms, ss, season, present, frac):
        """Series (GEOID) pre-look features per (date t, ps4 cell) -> (B, T, 256, F):
            e_t, e_t - its month's mean e       raw ps4 patch embedding of the date (2 D)
            F0 of its month, the month before, the month after at its ps16 cell (3 D)
                                                crop-free contextual month tokens
            phi(slot + frac)                    the date's time (TimeSetReadout's sin/cos)
            place of the cell in its ps16 cell (16), log pixel variance (1)
        --joint_feat lrL: F0 = the month tokens after blocks [0, L) of the main pass (no
        crop-free pass through all blocks), and no pixel variance."""
        B, T = present.shape
        lrL = self.joint_feat == "lrL"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if not lrL:
                lr = self.mix.head_blocks(self.mix.embed(ss, self.lr_ps))
                lr_x0, _ = self.mix.tail(lr, None, None)
            e = self.month_embedding(ms).float()                                # (B,T,16,16,D)
        F0 = (self._lr.pooled() if lrL else lr.pooled(lr_x0)).float()           # (B,NS,4,4,D)
        NS = F0.shape[1]
        e_bar = season_mean(e, season, present, NS)
        F0 = F0.repeat_interleave(4, 2).repeat_interleave(4, 3)                 # (B,NS,16,16,D)
        bi = torch.arange(B, device=e.device)[:, None]
        s = season.clamp(min=0)
        own = F0[bi, s]                                                         # (B,T,16,16,D)
        prv = F0[bi, (s - 1).clamp(min=0)]
        nxt = F0[bi, (s + 1).clamp(max=NS - 1)]
        phi = self.readout._phi(s.float() + frac.float())                       # (B,T,14)
        phi = phi[:, :, None, None].expand(B, T, 16, 16, phi.shape[-1])
        sub = torch.arange(16, device=e.device).view(4, 4).repeat(4, 4)
        oh = F.one_hot(sub, 16).float()[None, None].expand(B, T, 16, 16, 16)
        pix = 0
        for kk, val in ms.as_dict().items():
            if torch.is_tensor(val) and val.ndim == 5 and not kk.endswith("_mask"):
                u = rearrange(val.float(), "B (Y u) (X w) T c -> B T Y X (u w) c", u=4, w=4)
                pix = pix + u.var(4).mean(-1)
        pv = torch.as_tensor(pix, device=e.device).float().clamp(min=0).log1p()
        pv = pv.expand(B, T, 16, 16)[..., None] if pv.ndim == 4 else torch.zeros(B, T, 16, 16, 1, device=e.device)
        x = torch.cat([e, e - e_bar, own, prv, nxt, phi, oh] + ([] if lrL else [pv]), -1)
        return x.reshape(B, T, 256, -1)

    def _joint_pair_picks(self, ms, ss, season, present, extra, pairs, H):
        """--joint on a series pair head: every (date, cell) scored by the predictor; a
        (pair, cell) candidate scores the max of its two dates (as _pick_pairs); round(eps x
        budget) pair-cells uniform at random, the rest the best other pair-cells. Sets
        self._joint_pair_mask (B, T, H, W); returns the date scores (with grad in _joint_p)."""
        B, T = present.shape
        dev = present.device
        frac = extra["frac"].to(dev) if extra is not None else torch.zeros_like(present, dtype=torch.float)
        x = self._pair_prelook(ms, ss, season, present, frac)
        valid = present[:, :, None].expand(B, T, 256)
        p = self.joint(x, valid).view(B, T, H, H)
        self._joint_p = p
        pairs = pairs.to(dev)
        ok = (pairs >= 0).all(-1)                                               # (B, P)
        bi = torch.arange(B, device=dev)[:, None]
        e_, l_ = pairs[..., 0].clamp(min=0), pairs[..., 1].clamp(min=0)
        ps = torch.maximum(p.detach()[bi, e_], p.detach()[bi, l_]).float()     # (B, P, H, W)
        ps = ps.masked_fill(~ok[:, :, None, None], float("-inf")).flatten(1)
        budget = (self.k_frac * present.sum(1).float() * H * H / 2).round().long()
        budget = torch.minimum(budget, ok.sum(1) * H * H)
        n_rand = (self.joint_eps * budget.float()).round().long()
        r = torch.rand_like(ps).masked_fill(~torch.isfinite(ps), -1.0)
        r_rank = r.argsort(1, descending=True).argsort(1)
        rsel = (r_rank < n_rand[:, None]) & torch.isfinite(ps)
        ps2 = ps.masked_fill(rsel, float("-inf"))
        t_rank = ps2.argsort(1, descending=True).argsort(1)
        tsel = (t_rank < (budget - n_rand)[:, None]) & torch.isfinite(ps2)

        def date_mask(chosen):
            """chosen (B, P*H*H) bool pair-cells -> (B, T, H, H) bool: both dates of each."""
            bb, q = chosen.nonzero(as_tuple=True)
            p_, yx = q // (H * H), q % (H * H)
            y, x_ = yx // H, yx % H
            m = torch.zeros(B, T, H, H, dtype=torch.bool, device=dev)
            m[bb, pairs[bb, p_, 0], y, x_] = True
            m[bb, pairs[bb, p_, 1], y, x_] = True
            return m

        if self.pair_match_random:
            # Equal DATE-crop count: pair-cells that share a date in a cell cost one crop, and
            # uniform random pairs share more dates than the selector's, so at an equal number
            # of pair-cells the selector would encode more crops. Per sample, the target is
            # the date-crop count of a fresh all-random draw at the same pair budget; the
            # selector's pair-cells are added best-first until that count is reached.
            r2 = torch.rand_like(ps).masked_fill(~torch.isfinite(ps), -1.0)
            rand_all = (r2.argsort(1, descending=True).argsort(1) < budget[:, None]) & torch.isfinite(ps)
            target = date_mask(rand_all).flatten(1).sum(1)                      # (B,) date crops
            lo, hi = torch.zeros_like(budget), budget - n_rand                  # selector pair-cells
            for _ in range(int(hi.max().clamp(min=1)).bit_length() + 1):       # binary search
                mid = (lo + hi + 1) // 2
                n_mid = date_mask(rsel | ((t_rank < mid[:, None]) & torch.isfinite(ps2))).flatten(1).sum(1)
                fits = n_mid <= target
                lo = torch.where(fits, mid, lo)
                hi = torch.where(fits, hi, mid - 1)
            tsel = (t_rank < lo[:, None]) & torch.isfinite(ps2)
        self._joint_pair_mask = date_mask(rsel | tsel)
        return p

    def _joint_values(self, parts, b, t, y, x, target, NS):
        """In-context value of each picked crop: the loss over its cell's 4 x 4 px WITHOUT it
        (its softmax weight removed from the cell's pool) minus WITH it -- no re-encoding.
        -> the predictor's loss: MSE of its scores at the picks to the values z-scored per
        sample (their ranking is what top-k needs). Stored in self.joint_loss."""
        wgt, cand, out, idx = parts
        if idx is None or target is None:
            self.joint_loss = None
            return
        ok = idx >= 0                                                           # (B, Q, M)
        bq = ok.nonzero(as_tuple=True)
        crop = idx[ok]                                                          # (E,) crop ids
        w = wgt[:, :, NS:][ok].clamp(max=1 - 1e-4)                             # (E,) | (E, 16)
        w = w[:, None, None] if w.ndim == 1 else w[:, None, :]
        L = cand[:, :, NS:][ok]                                                 # (E, C, 16)
        o = out[bq[0], bq[1]]                                                   # (E, C, 16)
        o_wo = (o - w * L) / (1 - w)
        tg = target.to(o.device)
        tg = tg[:, 0] if tg.ndim == 4 else tg                                    # (B, 64, 64)
        lab = rearrange(tg, "b (y i) (x j) -> b (y x) (i j)", i=4, j=4)[bq[0], bq[1]]  # (E, 16)
        if self.task_kind == "classification":
            m = (lab >= 0).float()
            f = lambda z: F.cross_entropy(z, lab.long().clamp(min=0), reduction="none")  # noqa: E731
        else:
            m = torch.isfinite(lab.float()).float()
            f = lambda z: (z[:, 0] - lab.float().nan_to_num(0.0)) ** 2           # noqa: E731
        with torch.no_grad():
            val = ((f(o_wo) - f(o)) * m).sum(1) / m.sum(1).clamp(min=1)           # (E,) >0 helps
        n4 = len(b) if b is not None else 0
        n8 = 0
        ds = self._date_sets or []
        nd = sum(len(pp) for pp, _, _ in ds)
        if self.joint_res and self._res_masks is not None:
            b8, t8, y8, x8 = self._res_masks[8].nonzero(as_tuple=True)       # _res_crops order
            n8 = len(b8)
        v = torch.zeros(n4 + n8 + nd, device=o.device).index_add_(0, crop, val.float())  # sum over cells
        if ds:          # --date_joint: month crops (if any) as ever, date crops per token
            losses = []
            if n4:
                losses.append(F.mse_loss(self._joint_p[b, t, y, x].float(), zscore_by(v[:n4], b)))
            if nd:
                per_tok = v[n4:] / torch.cat([torch.full((len(pp),), c_, device=v.device)
                                              for pp, _, c_ in ds])
                bd = torch.cat([bb_ for _, bb_, _ in ds])
                losses.append(F.mse_loss(torch.cat([pp for pp, _, _ in ds]).float(),
                                         zscore_by(per_tok, bd)))
            self.joint_loss = sum(losses) / len(losses) if losses else None
            return
        self._loo_v = v
        if self.joint_target != "loo":      # trained on another target (gate_value_loss,
            self.joint_loss = None          # _drop_values): leave-one-out is only logged
            return
        # z-score per sample
        pred = self._joint_p[b, t, y, x].float()
        bb = b
        if n8:
            pred = torch.cat([pred, self._joint_p8[b8, t8, y8, x8].float()])
            bb = torch.cat([b, b8])
        B = int(bb.max()) + 1
        cnt = torch.bincount(bb, minlength=B).clamp(min=1).float()
        mu = torch.zeros(B, device=v.device).index_add_(0, bb, v) / cnt
        var = torch.zeros(B, device=v.device).index_add_(0, bb, (v - mu[bb]) ** 2) / cnt
        z = (v - mu[bb]) * torch.rsqrt(var[bb] + 1e-6)
        self.joint_loss = F.mse_loss(pred, z)

    @torch.no_grad()
    def _drop_values(self, lr, hr, b, t, y, x, logits, season, present, target, read=None):
        """--joint_target drop -> (dropped (N,) bool, value (N,)): blocks [L, n) and the readout
        again WITHOUT a random share self.drop_frac of the crops (their tokens after blocks
        [0, L) do not depend on the other crops). Value of a dropped crop = mean loss over its
        cell's 4 x 4 px without them minus with all, divided by the dropped crops at that cell.
        self.drop_passes > 1: that many independent passes; a crop's value is its mean over the
        passes that dropped it, `dropped` = in any pass.
        read(lr_F, crop tokens, crop idx) -> logits: another readout than native_set."""
        dev = b.device
        tg = target.to(dev)
        tg = tg[:, 0] if tg.ndim == 4 else tg                                    # (B, 64, 64)
        if self.task_kind == "classification":
            m = (tg >= 0).float()
            f = lambda z: F.cross_entropy(z.float(), tg.long().clamp(min=0), reduction="none")  # noqa: E731
        else:
            m = torch.isfinite(tg.float()).float()
            f = lambda z: (z[:, 0].float() - tg.float().nan_to_num(0.0)) ** 2    # noqa: E731
        base = f(logits)
        tot = torch.zeros(len(b), device=dev)
        cnt = torch.zeros(len(b), device=dev)
        for _ in range(self.drop_passes):
            drop = torch.rand(len(b), device=dev) < self.drop_frac
            keep = ~drop
            hr_sub = hr.subset(keep)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lr_x, hr_x = self.mix.tail(lr, hr_sub, b[keep])
            v = (hr_sub.pooled(hr_x).float()[:, 0, 0, 0] if keep.any()
                 else lr_x.new_zeros(0, lr_x.shape[-1]).float())
            kidx = (b[keep], t[keep], y[keep], x[keep])
            out = (read(lr.pooled(lr_x).float(), v, kidx) if read is not None else
                   self.readout(self.head.linear, lr.pooled(lr_x).float(), season, present, v, kidx))
            d = (f(out) - base) * m                                              # (B, 64, 64)
            cell = (F.avg_pool2d(d[:, None], LOOKUP_PS)
                    / F.avg_pool2d(m[:, None], LOOKUP_PS).clamp(min=1e-6))[:, 0]  # (B, 16, 16)
            n = torch.zeros_like(cell).index_put_(
                (b[drop], y[drop], x[drop]), torch.ones(int(drop.sum()), device=dev), accumulate=True)
            tot += (cell[b, y, x] / n[b, y, x].clamp(min=1)) * drop
            cnt += drop
        return cnt > 0, tot / cnt.clamp(min=1)

    def _drop_loss(self, lr, hr, b, t, y, x, logits, season, present, target, read=None):
        """--joint_target drop: self.joint_loss = the predictor's MSE to the dropped crops'
        values (_drop_values), z-scored per sample."""
        drop, val = self._drop_values(lr, hr, b, t, y, x, logits, season, present, target, read)
        loo = getattr(self, "_loo_v", None)
        if loo is not None and len(loo) == len(val) and drop.sum() > 2:
            self.gate_loo_corr = float(torch.corrcoef(torch.stack([val[drop], loo[drop]]))[0, 1])
        bd = b[drop]
        self.joint_loss = F.mse_loss(self._joint_p[bd, t[drop], y[drop], x[drop]].float(),
                                     zscore_by(val[drop], bd))

    def _self_teach(self, ms, season, present, frac, extra, lr_F, v, picks, logw, pairs_all):
        """--self_teach n: the mixed-resolution encoding as a FEATURE UPSAMPLER whose target is
        the same encoder's own dense encoding (training only).

        Teacher   n random real RAW timesteps of the sample, drawn here independently of the
                  selector's picks (which never see them): the raw acquisitions where the
                  loader carries them (extra["dates"]: PASTIS, YieldSAT), else the step series
                  `ms` itself (BioMassters months, GEOID dates: already raw). The SAME encoder
                  encodes them densely as one ps4 sequence over the whole tile (ps4ts64 with n
                  steps) -> f_teach (B, n, 16, 16, D); the task head on them takes the TASK
                  loss (time mean, or the pair head over the n dates' consecutive pairs).
        Student   the mixed-resolution tokens (season / month ps16 tokens + the picked ps4
                  crops) splatted by the time-continuous readout AT THE TEACHER'S n TIMES ->
                  (B, n, 16, 16, D); loss = MSE to f_teach (stop-gradient on the teacher).
        Selector  (--joint) each picked crop's in-context value = the reconstruction error of
                  its cell at the n teacher times WITHOUT it (its weight removed) minus WITH
                  it; the predictor regresses the per-sample z-scored values.
        Sets self.teach_loss = {"task", "recon"} and self.joint_loss.

        ms (B, 64, 64, T, c) step series; season, present (B, T); frac (B, T) step position
        in its low-res slot; lr_F (B, NS, 4, 4, D) final low-res tokens; v (N, D) final crop
        tokens and picks = (b, t, y, x) (N,) each, or None; logw (N,) crop gate or None."""
        from manyup.encoders.olmoearth import _map
        B, T = present.shape
        dev, D = present.device, lr_F.shape[-1]
        NS = lr_F.shape[1]
        target = extra["target"].to(dev)
        # -- the raw step pool: sample (B, 64, 64, Tr, c), real (B, Tr), time (B, Tr) in slots
        views = extra.get("dates") or {}
        if views:
            mod = "S2" if "S2" in views else sorted(views)[0]
            raw, d_valid, d_month, d_frac = views[mod]
            raw = _to_device(raw, dev, torch.bfloat16)
            d_valid, d_month, d_frac = d_valid.to(dev), d_month.to(dev), d_frac.to(dev)
            d_season = season.gather(1, d_month.clamp(min=0))
            real = d_valid & (d_month >= 0) & (d_season >= 0)
            time = d_season.clamp(min=0).float() + d_frac.float()
        else:
            raw, real = ms, present & (season >= 0)
            time = season.clamp(min=0).float() + frac.float()
        n = min(self.self_teach, real.shape[1])
        draw = torch.rand(real.shape, device=dev).masked_fill(~real, -1.0)
        if self.teach_event and not views and extra.get("event_steps") is not None:
            ev = extra["event_steps"].to(dev)              # (B, 2): pre- and post-event steps
            forced = torch.zeros_like(real).scatter_(1, ev.clamp(min=0), ev >= 0) & real
            draw = draw.masked_fill(forced, 2.0)           # always among the n drawn
        steps = draw.topk(n, 1).indices.sort(1).values                          # (B, n) in time order
        teach_valid = real.gather(1, steps)                                     # (B, n) bool
        t_query = time.gather(1, steps)                                         # (B, n)
        bi = torch.arange(B, device=dev)[:, None]

        def take(k, val):               # (B, H, W, Tr, c) -> (B, H, W, n, c); timestamps (B, Tr, 3)
            if k == "timestamps":
                return val[bi, steps]
            idx = steps[:, None, None, :, None].expand(B, val.shape[1], val.shape[2], n, val.shape[4])
            return val.gather(3, idx)

        # -- teacher: dense ps4 over the tile, n steps, one sequence; task loss
        frozen_teacher = self.teach_detach == "all"
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(not frozen_teacher):
            dense = self.mix.head_blocks(self.mix.embed(_map(raw, take), LOOKUP_PS))
            dense_x, _ = self.mix.tail(dense, None, None)
            f_teach = dense.pooled(dense_x)                                     # (B, n, 16, 16, D)
            f_teach = f_teach * teach_valid[:, :, None, None, None]
            if self.native:   # native head: a ps4 token's 16 x 16 map -> its 4 x 4 px, time mean
                lin = self.head.linear
                C, cells = lin.out_features // LR_PS ** 2, LR_PS // LOOKUP_PS
                tl = F.avg_pool2d(lin(f_teach).float().view(-1, C, LR_PS, LR_PS), cells)
                tl = tl.view(B, n, f_teach.shape[2], f_teach.shape[3], C, cells, cells)
                tl = (tl * teach_valid[:, :, None, None, None, None, None]).sum(1) / \
                    teach_valid.float().sum(1).clamp(min=1)[:, None, None, None, None, None]
                t_logits = rearrange(tl, "b y x k i j -> b k (y i) (x j)")
            elif self.pool is None:
                t_pairs = (consecutive_pairs(teach_valid.cpu(), None).to(dev)
                           if self.head.reduction.startswith("pair_") else None)
                t_logits = self.head(f_teach, teach_valid, t_pairs)
            else:
                t_logits = self.head(self.pool(f_teach, teach_valid)[:, None],
                                     torch.ones_like(teach_valid[:, :1]))
        task = (t_logits.float().sum() * 0 if frozen_teacher
                else self.task_loss_fn(t_logits.float(), target))
        if frozen_teacher:              # the no-grad pass above cached detached weight casts
            torch.clear_autocast_cache()
        if self.teach_detach != "none":   # the reconstruction trains the splat, not the encoder
            lr_F = lr_F.detach()
            v = v.detach() if v is not None else None
        # -- student: the splat read at the teacher's times
        empty = torch.zeros(0, dtype=torch.long, device=dev)
        splat = self.recon_splat if self.recon_splat is not None else self.readout
        wgt, feat, crop_of = splat.weights(
            lr_F, seasons_used(season, present, NS), season, present, frac,
            v if picks is not None else lr_F.new_zeros(0, D),
            picks if picks is not None else (empty,) * 4, logw,
            t_query=t_query, q_valid=teach_valid)                               # (B, n, Q, K)
        student = torch.einsum("bnqk,bqkd->bnqd", wgt, feat)                    # (B, n, Q, D)
        goal = f_teach.float().detach().flatten(2, 3)                           # (B, n, Q, D)
        tv = teach_valid.float()
        norm = tv.sum(1).clamp(min=1)                                           # (B,)
        err = ((student - goal) ** 2).mean((2, 3))                              # (B, n)
        # scale-free: / the teacher features' variance over the batch's real cells (so the
        # loss is the unexplained share of the dense feature map, ~1 for a constant guess)
        scale = goal[teach_valid].var(0).mean().clamp(min=1e-6) if teach_valid.any() else 1.0
        recon = ((err * tv).sum(1) / norm).mean() / scale
        self.teach_loss = {"task": task, "recon": recon}
        label_loss = self.joint_loss if self.teach_value in ("label", "both") else None
        self.joint_loss = label_loss
        if (self.joint is None or picks is None or crop_of is None
                or self.teach_value == "label"):
            return
        b, t, y, x = picks
        ok = crop_of >= 0                                                       # (B, Q, M)
        eb, eq, em = ok.nonzero(as_tuple=True)                                  # E crop entries
        crop = crop_of[ok]                                                      # (E,) crop index
        with torch.no_grad():
            val = torch.zeros(len(b), device=dev)
            for s0 in range(0, len(crop), 8192):                               # (E, n, D) chunks
                sl = slice(s0, s0 + 8192)
                cb, cq, cm = eb[sl], eq[sl], NS + em[sl]
                w = wgt[cb, :, cq, cm].clamp(max=1 - 1e-4)                      # (e, n)
                own = feat[cb, cq, cm]                                          # (e, D)
                o = student[cb, :, cq]                                          # (e, n, D)
                o_wo = (o - w[..., None] * own[:, None]) / (1 - w)[..., None]
                g = goal[cb, :, cq]
                gain = ((o_wo - g) ** 2 - (o - g) ** 2).mean(-1)                # (e, n) > 0: helps
                val.index_add_(0, crop[sl], ((gain * tv[cb]).sum(1) / norm[cb]).float())
            cnt = torch.bincount(b, minlength=B).clamp(min=1).float()
            mu = torch.zeros(B, device=dev).index_add_(0, b, val) / cnt
            var = torch.zeros(B, device=dev).index_add_(0, b, (val - mu[b]) ** 2) / cnt
            z = (val - mu[b]) * torch.rsqrt(var[b] + 1e-6)
        self.joint_loss = F.mse_loss(self._joint_p[b, t, y, x].float(), z)
        if label_loss is not None:          # "both": the two regressions, equal weight
            self.joint_loss = 0.5 * (self.joint_loss + label_loss)

    def _res_picks(self, present, k):
        """--res_mix: random (month, cell) picks at ps16 and ps8, round(share x k) tokens each
        -> {ps: (B, T, g, g) bool mask}. A sample with fewer candidates than asked (few
        present months) gets all of them; the ps4 picks absorb the shortfall, so every sample
        spends exactly k tokens."""
        B, T = present.shape
        out = {}
        for ps in (16, 8):
            n = round(k * self.res_mix[ps])
            if n:
                g = TILE // ps                                                 # 4 or 8 cells
                out[ps] = select(torch.rand(B, T, g, g, device=present.device), present, n)
        return out

    def _res_crops(self, ms, season, frac, masks):
        """Encode the _res_picks crops like the ps4 crops (true position, their month's slot)
        -> crop sets as _date_crops returns them, plus "size" = footprint side in ps4 cells;
        "idx" is the footprint's top-left ps4 cell."""
        out = []
        for ps, mask in masks.items():
            b, t, y, x = mask.nonzero(as_tuple=True)
            if not len(b):
                continue
            crops = crop_patches(ms, b, t, y, x, ps)
            hr = self.mix.embed(crops, ps, slot=self._crop_slot(season[b, t], frac[b, t]),
                                py=y * ps, px=x * ps)
            f = ps // LOOKUP_PS                                                # 4 or 2
            out.append({"hr": self.mix.head_blocks(hr, self._lr, b), "idx": (b, t, y * f, x * f),
                        "logw": None, "size": f})
        return out

    def set_acq_pool(self, mode: str):
        """--acq_pool (see _pool_acquisitions); called by main after construction."""
        self.acq_pool = mode
        if mode == "attn":          # a learned weight per acquisition token; zero init = mean
            D = self.mix.encoder.embedding_size
            self.acq_score = nn.Linear(D, 1)
            nn.init.zeros_(self.acq_score.weight)
            nn.init.zeros_(self.acq_score.bias)

    def _pool_acquisitions(self, hr, extra, picks, season, present):
        """--acq_pool: a month crop's tokens are not those of the month COMPOSITE but a pool
        over the month's raw acquisitions, each encoded on its own date.

        Every picked (month, ps4 cell) becomes ONE sequence = the ps4 tokens of that cell on
        every valid acquisition inside the month, all modalities together (a patch time
        series: attention across the dates of one cell, never across cells), run through blocks
        [0, L). Then, per token kind (modality, bandset), the acquisitions are pooled (mean, or
        softmax of a learned score) into the one token a composite crop has of that kind, and
        the crop joins the joint sequence from block L exactly as a composite crop does.
        A token kind with no valid acquisition in the month keeps the composite's token.

        hr      Packed of the N composite month crops after blocks [0, L)
        picks   (b, t, y, x), each (N,): sample, month, ps4 cell of every crop
        extra["dates"][m] = (raw dates of modality m packed (T_m steps), valid (B, T_m),
                             month of each date (B, T_m), its position in the season (B, T_m))
        -> hr with its tokens replaced. Sets self.last_acq = (acquisition tokens per sample,
        share of crop tokens that kept the composite)."""
        b, t, y, x = picks
        dev = hr.x.device
        B, T = present.shape
        H = OUT_GRID
        n_crops, n_slots = hr.keep.shape                       # token slots of a composite crop
        crop_id = torch.full((B, T, H, H), -1, dtype=torch.long, device=dev)
        crop_id[b, t, y, x] = torch.arange(n_crops, device=dev)
        slot_offset, i = {}, 0      # where each modality's bandsets sit among a crop's slots
        for m, shape in hr.dims.items():
            n = 1
            for d in shape[1:-1]:
                n *= d
            slot_offset[m] = i
            i += n
        bidx = torch.arange(B, device=dev)[:, None]
        tok_x, tok_crop, tok_slot = [], [], []
        for m, (d_sample, d_valid, d_month, d_frac) in extra["dates"].items():
            d_valid, d_month, d_frac = d_valid.to(dev), d_month.to(dev), d_frac.to(dev)
            ok = d_valid & (d_month >= 0) & present.gather(1, d_month.clamp(min=0))   # (B, T_m)
            crop_of = crop_id[bidx, d_month.clamp(min=0)]                    # (B, T_m, H, W)
            bb, dd, yy, xx = (ok[:, :, None, None] & (crop_of >= 0)).nonzero(as_tuple=True)
            if not len(bb):
                continue
            d_sample = _to_device(d_sample, dev, torch.bfloat16)
            acq = self.mix.embed(crop_patches(d_sample, bb, dd, yy, xx, LOOKUP_PS), LOOKUP_PS,
                                 slot=self._crop_slot(season[bb, d_month[bb, dd]], d_frac[bb, dd]),
                                 py=yy * LOOKUP_PS, px=xx * LOOKUP_PS)
            which, bandset = acq.keep.nonzero(as_tuple=True)    # acq.x's order (row-major)
            tok_x.append(acq.x)                                              # (n_tok_m, D)
            tok_crop.append(crop_of[bb, dd, yy, xx][which])                  # its month crop
            (m_oe,) = acq.dims                  # the encoder's name of this one modality
            tok_slot.append(slot_offset[m_oe] + bandset)                     # its token kind
        if not tok_x:
            self.last_acq = (0.0, 1.0)
            return hr
        tok_x, tok_crop, tok_slot = torch.cat(tok_x), torch.cat(tok_crop), torch.cat(tok_slot)
        order = torch.argsort(tok_crop, stable=True)            # one sequence per month crop
        tok_x, tok_crop, tok_slot = tok_x[order], tok_crop[order], tok_slot[order]
        seqlens = torch.bincount(tok_crop, minlength=n_crops)                # (N,)
        if hr.typed:                # the composite crops got hr_type before their blocks
            tok_x = (tok_x + self.mix.hr_type).to(tok_x.dtype)
        out = self.mix.blocks(tok_x, seqlens[seqlens > 0], 0, self.mix.L).float()
        w = torch.ones(len(out), device=dev)
        if self.acq_pool == "attn":
            w = torch.exp(self.acq_score(out)[:, 0].clamp(-20, 20))
        key = tok_crop * n_slots + tok_slot                     # (crop, token kind) of a token
        total = out.new_zeros(n_crops * n_slots, out.shape[-1]).index_add_(0, key, out * w[:, None])
        weight = out.new_zeros(n_crops * n_slots).index_add_(0, key, w)
        kept = hr.keep.flatten()                                # hr.x's order (row-major)
        pooled = (total / weight.clamp(min=1e-6)[:, None])[kept]
        have = (weight > 0)[kept]
        hr.x = torch.where(have[:, None], pooled, hr.x.float()).to(hr.x.dtype)
        self.last_acq = (len(out) / B, 1.0 - float(have.float().mean()))
        return hr

    def _crop_slot(self, season_of, frac):
        """Time slot of crops: their season, + their position inside it (--frac_slot) so
        crops of one season (and dates of one month) get distinct time encodings."""
        return season_of.float() + frac if self.frac_slot else season_of

    def _apt_pool_crops(self, ms, b, t, y, x, py, px, P_, slot, groups):
        """--apt_pool: replaces each merged group's single-month crop (and its slot) with the
        MEAN pixels and MEAN slot of every month the group's run covers (user: "just simply
        interpolate" -- OlmoEarth's sincos time encoding is continuous, so an averaged /
        fractional slot needs no new embedding). b/t/y/x are the MASK coordinates (y, x the
        ps4 cell _apt_pool_mask ranked, not py/px -- those may already be a --crop_px /
        --jitter / --crop_free adjusted window and cannot be reversed back to a cell); py/px
        are that crop's actual pixel window, reused unchanged for every group member since
        --apt_pool requires the plain ps4-on-grid path (checked in __init__). slot is the
        representative (first-month) value _apt_pool_mask already picked; groups[(b,t,y,x)]
        lists every member month -- len 1 (no merge) crops are left untouched, byte for byte."""
        from manyup.encoders.olmoearth import _map
        if not any(len(g) > 1 for g in groups.values()):
            return crop_patches(ms, b, t, py, px, P_, pixels=True), slot
        reps = list(zip(b.tolist(), t.tolist(), y.tolist(), x.tolist()))
        members = [groups[k] for k in reps]
        sizes = torch.tensor([len(m) for m in members], device=b.device)
        if (sizes == 1).all():
            return crop_patches(ms, b, t, py, px, P_, pixels=True), slot
        # expand to one (b, t_member, y, x) row per member, same py/px as its group's crop
        rep = torch.repeat_interleave(torch.arange(len(reps), device=b.device), sizes)
        b_e, py_e, px_e = b[rep], py[rep], px[rep]
        t_e = torch.tensor([m for ms_ in members for m in ms_], device=b.device)
        crops = crop_patches(ms, b_e, t_e, py_e, px_e, P_, pixels=True)

        def pool(k, v):
            if k == "timestamps":
                return v[:, :1]              # kept per-member by crop_patches; take the first
            if v.ndim != 5:
                raise ValueError(f"_apt_pool_crops: {k} is not (N, H, W, T, C)")
            out = torch.zeros(len(reps), *v.shape[1:], dtype=torch.float32, device=v.device)
            return out.index_add_(0, rep, v.float()).div_(
                sizes[:, None, None, None, None].float()).to(v.dtype)

        pooled = _map(crops, pool)
        pooled = pooled._replace(timestamps=crops.timestamps[
            torch.cumsum(sizes, 0) - sizes])            # first member's timestamp per group
        # the mean slot is an INTERPOLATED (fractional) value even when --frac_slot is off and
        # slot is otherwise a bare integer season index: pooling forces it to float regardless.
        slot_e = slot[rep].float()
        mean_slot = torch.zeros(len(reps), device=slot.device, dtype=torch.float32).index_add_(
            0, rep, slot_e).div_(sizes.float())
        return pooled, mean_slot

    def forward(self, ms, ss, season, present, pairs=None, extra=None):
        """-> (logits, present (B, T), crops spent). pairs (B, P, 2): a pair_* head's date
        pairs (consecutive present steps). extra: Collate's (step fractions, date views)."""
        device = self.head.linear.weight.device
        ms, ss = _to_device(ms, device, torch.bfloat16), _to_device(ss, device, torch.bfloat16)
        season, present = season.to(device), present.to(device)
        B, T = present.shape
        G, H = ss.timestamps.shape[1], OUT_GRID
        self.GT = (G, T)
        self._dps_P = None
        bidx = torch.arange(B, device=device)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr = self.mix.head_blocks(self.mix.embed(
                tile_sample(ss, self.lr_tile) if self.lr_tile != TILE else ss, self.lr_ps))
            if self.no_seasons:                 # crops only: an EMPTY low-res sequence
                from manyup.inject import Packed
                lr = Packed(lr.x[:0], torch.zeros_like(lr.seqlens), torch.zeros_like(lr.keep),
                            lr.dims, lr.masks, None)
            self._lr = lr                       # crops read it in blocks [0, L) (--head_attn)
            if (self.lr_tile == TILE and lr.keep.shape[1]
                    and lr.dims[next(iter(lr.dims))][1] * self.lr_ps != TILE):
                raise ValueError("InjectFT expects one 64 px tile per sample")
            k = round(self.k_frac * T * H * H) if self.sel_name != "none" else 0
            if self.crop_free and self.n_crops:
                k = self.n_crops
            self.k = k
            self._frac = extra["frac"].to(device) if extra is not None else None
            hr = b = scores = None
            n_crops = 0
            self._res_masks = None
            self._apt_groups = None
            self._octree_leaves = None
            if k:
                where = present[:, :, None, None].expand(B, T, H, H)
                lr_L = lr.pooled().float()                                     # (B,G,h,w,D)
                r = H // lr_L.shape[2]
                lr_m = lr_L[bidx, season.clamp(min=0)]                         # (B,T,h,w,D)
                if self.sel_name == "grid" and self.unit == "location":   # lattice of windows
                    scores = location_grid_scores(
                        B, H, max(1, k // T), self.crop_px,
                        self.training and self.grid_phase == "random", device)
                    scores = scores[:, None].expand(B, T, H, H).contiguous()
                elif self.sel_name == "grid" or self.force_grid:   # regular lattice, content-blind
                    scores = grid_scores(present, H, k, self.grid_mode,
                                         self.training and self.grid_phase == "random")
                elif self.sel_name in PRIOR:                   # crop-free pass, per cell
                    cell = self._prior_scores(lr, season, present, G, extra, B, H)   # (B,H,W)
                    scores = cell[:, None].expand(B, T, H, H).contiguous()
                elif self.sel_name in PIXEL_PRIOR:             # raw pixels, no model pass (APT)
                    scores = self._apt_scores(ms, present, B, H)
                elif self.joint is not None and self.joint_pairs:   # series: joint pair picks
                    scores = self._joint_pair_picks(ms, ss, season, present, extra, pairs, H)
                elif self.joint is not None:     # eps random + (1 - eps) predictor picks
                    # --date_joint hier: half the budget on month crops; only: none at all
                    k_m = k // 2 if self.date_joint == "hier" else k
                    scores = (self._joint_picks(ms, ss, season, present, k_m, H)
                              if self.date_joint != "only" else torch.rand(B, T, H, H, device=device))
                elif self.selector is None:     # random, or picks set from outside
                    scores = (self.fixed_scores if self.fixed_scores is not None
                              else torch.rand(B, T, H, H, device=device))
                else:
                    e = self.month_embedding(ms).float()                        # (B,T,H,W,D)
                    e_bar = season_mean(e, season, present, G)
                    if self.sel_name == "xattn":
                        lr_valid = seasons_used(season, present, G)
                        scores = self.selector(e, e_bar, lr_L, season, present, lr_valid,
                                               ms.timestamps[:, :, 1].to(device),
                                               ss.timestamps[:, :, 1].to(device))
                    else:
                        up = lr_m.repeat_interleave(r, 2).repeat_interleave(r, 3)
                        scores = self.selector(torch.cat([e, e - e_bar, up], -1).detach())[..., 0]
                    self.aux_scores = scores.float()[where]
                    raw_scores = scores.float()
                    if self.score_norm:            # per sample, over its present slots
                        wf = where.float()
                        n = wf.flatten(1).sum(1).clamp(min=2)[:, None, None, None]
                        mu = (scores.float() * wf).flatten(1).sum(1)[:, None, None, None] / n
                        var = (((scores.float() - mu) * wf) ** 2).flatten(1).sum(1)[
                            :, None, None, None] / (n - 1)
                        scores = (scores.float() - mu) * torch.rsqrt(var + 1e-6)   # finite grad at var 0
                if self.loc_px is not None:     # --loc_stride: anchor cells only (random too)
                    a1 = self.loc_px.to(device) >= 0
                    scores = scores.masked_fill(~(a1[:, None] & a1[None, :]), float("-inf"))
                pick = scores.detach().float()
                explore = self.selector is not None and self.training
                k_ps4 = (k_m if self.joint is not None and not self.joint_pairs
                         and not self.force_grid else k)
                if self.res_mix is not None:    # ps16 / ps8 first; ps4 takes the rest per sample
                    self._res_masks = self._res_picks(present, k)
                    used = sum(m.flatten(1).sum(1) for m in self._res_masks.values())   # (B,)
                    k_ps4 = k - used                                            # (B,) long
                if self.joint_res:              # the joint selector's ps8 picks (set in _joint_picks)
                    used = sum(m.flatten(1).sum(1) for m in self._res_masks.values())
                    k_ps4 = k - used
                if self.sel_name == "apt" and self.apt_octree:
                    # user's 2026-10-09 space-time octree: APT chooses patch SIZE AND time
                    # SPAN by entropy-gain-per-token, not just which ps4 slots to fill (see
                    # _apt_octree_mask). EVERY crop comes from the octree -- no ps4 "remainder"
                    # the way --res_mix/--apt_res leave one -- so mask is all-False (n_crops=0,
                    # the normal per-cell crop path below never fires) and self._octree_leaves
                    # is read back by _apt_octree_crops at crop construction, same place
                    # _res_crops reads self._res_masks.
                    mask = torch.zeros(B, T, H, H, dtype=torch.bool, device=device)
                    self._octree_leaves = self._apt_octree_mask(
                        ms, present, B, self.apt_tau, ps_min=4,
                        space_only=self.apt_octree_space_only)
                elif self.sel_name == "apt" and self.apt_pool:
                    # user's space-then-time extension: cells first, consecutive-month MSE
                    # merge within a cell, freed budget spent globally on more cells (see
                    # _apt_pool_mask). self._apt_groups read back at crop construction below.
                    mask, self._apt_groups = self._apt_pool_mask(ms, present, B, H, k_ps4)
                elif self.sel_name == "apt" and self.apt_res:
                    # user's 2026-10-08 extension: APT chooses patch SIZE (ps16 vs ps4) by its
                    # own entropy gain, not just which ps4 slots to fill (see _apt_res_mask).
                    # The ps4-grid mask (split cells' children) flows through the ordinary
                    # mask.nonzero() -> crop_patches(...) path below exactly like every other
                    # selector; the UNSPLIT ps16 cells go through the existing --res_mix
                    # footprint machinery (_res_crops / NativeSetReadout) unchanged.
                    mask, self._res_masks = self._apt_res_mask(ms, present, B, k)
                elif self._res_masks is not None:
                    rank = pick.masked_fill(~where, float("-inf")).flatten(1).argsort(
                        1, descending=True).argsort(1).view_as(pick)
                    mask = (rank < k_ps4[:, None, None, None]) & where
                elif self.pick == "random":                       # the scores only gate
                    mask = select(torch.rand_like(pick), present, k)
                elif self.pick == "per_step":
                    mask = per_step_topk(gumbel(pick) if explore else pick, present, k)
                elif self.unit == "pair" and self.joint is not None:
                    mask = self._joint_pair_mask
                elif self.unit == "pair":
                    mask = self._pick_pairs(gumbel(pick) if explore else pick, present, pairs)
                elif self.unit == "slot":
                    mask = select(gumbel(pick) if explore else pick, present, k_ps4)
                else:
                    loc = pick.masked_fill(~where, float("-inf")).amax(1)      # (B, H, W)
                    mask = topk_mask((gumbel(loc) if explore else loc)[:, None],
                                     max(1, k // T)) & where
                b, t, y, x = mask.nonzero(as_tuple=True)
                n_crops = len(b)
                if self.crop_free:              # uniform pixel positions, uniform present months
                    P_ = self.crop_px
                    if self.n_eval is not None and not self.training:
                        cnt = torch.full((B,), min(self.n_eval, k), device=device)
                    elif self.vary_crops and self.crop_counts:
                        cc = torch.tensor(self.crop_counts, device=device)
                        cnt = cc[torch.randint(0, len(cc), (B,), device=device)]
                    elif self.vary_crops:
                        cnt = torch.randint(0, k + 1, (B,), device=device)
                    else:
                        cnt = torch.full((B,), k, device=device)
                    grid_pick = (self.free_pick == "grid" if self.free_pick
                                 else self.sel_name == "grid" or self.force_grid)
                    shift = self.shift_stride if grid_pick else 0
                    if shift:                   # every window of the shifted grid, every month
                        pos = shifted_positions(P_, shift)
                        ns = len(pos)
                        cnt = present.sum(1) * ns * ns
                    M = max(int(cnt.max()), 1)
                    keep = torch.arange(M, device=device)[None] < cnt[:, None]          # (B, M)
                    tt = torch.multinomial(present.float(), M, replacement=True)
                    fy, fx = (torch.randint(0, TILE - P_ + 1, (B, M), device=device) for _ in "yx")
                    if shift:                   # crop i: present month i // ns^2, window i % ns^2
                        i = torch.arange(M, device=device)
                        pos = torch.tensor(pos, device=device)
                        mo = torch.argsort((~present).long(), dim=1, stable=True)       # (B, T)
                        tt = mo[:, (i // (ns * ns)).clamp(max=T - 1)]
                        fy = pos[i % (ns * ns) // ns][None].expand(B, M)
                        fx = pos[i % ns][None].expand(B, M)
                        # a window on the patch grid in both axes repeats an on-grid token
                        keep = keep & ~((fy % P_ == 0) & (fx % P_ == 0))
                    elif grid_pick:
                        # staggered lattice on the crops' OWN patch grid (TILE / P_ cells a side):
                        # per sample its first cnt lattice slots (grid_scores, modulus from the
                        # sample's own count); counts beyond every slot keep the random crops above
                        # --grid_mode c2f / shared: ONE anchor list for every crop size
                        # (shared_grid_order, ps4 cells); the window sits with its top-left on
                        # the anchor (c2f) or centred on the anchor's ps4 cell (shared), clamped
                        # into the tile
                        rp = self.training and self.grid_phase == "random"
                        if self.grid_mode in ("c2f", "shared"):
                            HP, st = TILE // LOOKUP_PS, LOOKUP_PS
                            off = 0 if self.grid_mode == "c2f" else LOOKUP_PS // 2 - P_ // 2
                            order = shared_grid_order(present, cnt, self.grid_mode, rp)
                        else:
                            HP, st, off = TILE // P_, P_, 0
                            sc = grid_scores(present, HP, cnt, self.grid_mode, rp)
                            order = sc.masked_fill(~present[:, :, None, None],
                                                   float("-inf")).flatten(1).argsort(1, descending=True)
                        Mg = min(M, order.shape[1])
                        order = order[:, :Mg]                                           # (B, Mg)
                        on = (torch.arange(Mg, device=device)[None]
                              < (present.sum(1) * HP * HP)[:, None])                    # (B, Mg)
                        gy, gx = ((v * st + off).clamp(0, TILE - P_)
                                  for v in (order // HP % HP, order % HP))
                        tt[:, :Mg] = torch.where(on, order // (HP * HP), tt[:, :Mg])
                        fy[:, :Mg] = torch.where(on, gy, fy[:, :Mg])
                        fx[:, :Mg] = torch.where(on, gx, fx[:, :Mg])
                    b = keep.nonzero(as_tuple=True)[0]
                    t, fy, fx = tt[keep], fy[keep], fx[keep]
                    if not self.training:       # share of (present month, pixel) under a crop,
                        d = torch.zeros(B, T, TILE + 1, TILE + 1, device=device)   # and of pixels
                        for sy, sx, sg in ((0, 0, 1.), (P_, 0, -1.), (0, P_, -1.), (P_, P_, 1.)):
                            d.index_put_((b, t, (fy + sy).clamp(0, TILE), (fx + sx).clamp(0, TILE)),
                                         torch.full((len(b),), sg, device=device), accumulate=True)
                        cov = (d.cumsum(2).cumsum(3)[:, :, :TILE, :TILE] > 0.5) & present[:, :, None, None]
                        self._cov[0] += float((cov.flatten(1).sum(1)
                                               / (present.sum(1).clamp(min=1) * TILE * TILE)).sum())
                        self._cov[1] += float(cov.any(1).flatten(1).float().mean(1).sum())
                        self._cov[2] += B
                    y, x = ((v // LOOKUP_PS).clamp(0, H - 1) for v in (fy, fx))   # ps4 cell of the top-left
                    self._free_px = (fy, fx)
                    n_crops = len(b)
                if self.date_joint == "only":
                    n_crops = 0
                self.last_pick = (mask, scores.detach(), present)   # for analyze_inject_picks
            # A no-grad pass above (the joint selector's pre-look, the oracle / entropy prior)
            # ran INSIDE this autocast context: autocast caches the bfloat16 copies of the
            # weights it casts, and copies made under no_grad are detached -- reused below, the
            # blocks' Linear weights would get NO gradient (only the float32 LayerNorms would).
            torch.clear_autocast_cache()
            frac = extra["frac"].to(device) if extra is not None else torch.zeros_like(present, dtype=torch.float)
            month_idx = (b, t, y, x) if n_crops else None      # the month picks (routing)
            dates = self._date_crops(extra, mask if n_crops else None, present, season, lr_m if k else None,
                                     scores, r if k else 4, B, H,
                                     lr_L if k else None, G, ss.timestamps[:, :, 1].to(device)
                                     ) if self.date_level != "none" else []
            self._date_sets = None
            if self.date_joint != "none" and k:
                with torch.no_grad():                               # tokens / month crop
                    c_full = self._tokens_per_crop(ms)
                dates = self._joint_date_crops(
                    extra, mask if self.date_joint == "hier" else None, present, season,
                    torch.full((B,), (k - (k_m if self.date_joint == "hier" else 0)) * c_full,
                               device=device), B, H)
            if n_crops and not self.month_crops:
                n_crops = 0                                     # months only route the dates
            if self._res_masks is not None and k:               # ps16 / ps8 month crops
                dates = dates + self._res_crops(ms, season, frac, self._res_masks)
            if self._octree_leaves is not None:                 # --apt_octree: every crop
                dates = dates + self._apt_octree_crops(ms, self._octree_leaves, season, frac)
            hr_px = None
            self._gate = self._gate_idx = None
            if n_crops:
                py, px = y * LOOKUP_PS, x * LOOKUP_PS
                P_ = self.crop_px
                if self.loc_px is not None:     # --loc_stride: the anchor's own window
                    lut = self.loc_px.to(device)
                    py, px = lut[y], lut[x]
                    hr_px = (py, px)
                elif P_ != LOOKUP_PS:           # the window centred on the picked cell
                    py, px = ((v + LOOKUP_PS // 2 - P_ // 2).clamp(0, TILE - P_) for v in (py, px))
                    hr_px = (py, px)
                if self.crop_free:
                    py, px = hr_px = self._free_px
                if self.jitter:
                    py, px = ((v + torch.randint(0, LOOKUP_PS, v.shape, device=device)
                               - LOOKUP_PS // 2).clamp(0, TILE - LOOKUP_PS) for v in (py, px))
                    hr_px = (py, px)
                crop_slot = self._crop_slot(season[b, t], frac[b, t])
                if self._apt_groups is not None:
                    crops, crop_slot = self._apt_pool_crops(ms, b, t, y, x, py, px, P_,
                                                            crop_slot, self._apt_groups)
                elif self.crop_free and self.shift_stride:   # windows may leave the tile
                    crops = crop_patches(pad_sample(ms, P_), b, t, py + P_, px + P_, P_,
                                         pixels=True)
                else:
                    crops = crop_patches(ms, b, t, py, px, P_, pixels=True)
                hr = self.mix.embed(crops, P_, slot=crop_slot, py=py, px=px)
                hr = self.mix.head_blocks(hr, lr, b)
                if getattr(self, "acq_pool", "none") != "none":     # pooled raw acquisitions
                    hr = self._pool_acquisitions(hr, extra, (b, t, y, x), season, present)
                if self.joint_target == "gate" and self.training:
                    self._gate = torch.ones(n_crops, device=device, requires_grad=True)
                    self._gate_idx = (b, t, y, x)
                    tok = torch.repeat_interleave(torch.arange(n_crops, device=device), hr.seqlens)
                    hr.x = (hr.x.float() * self._gate[tok, None]).to(hr.x.dtype)
                if self.selector is not None:  # sigmoid(s) -> 0: the crop IS its season token
                    tok = torch.repeat_interleave(torch.arange(n_crops, device=device),
                                                  hr.seqlens)
                    g = torch.sigmoid(scores[b, t, y, x])[tok, None]
                    ref = lr_m[b, t, y // r, x // r][tok]
                    hr.x = (ref + g * (hr.x.float() - ref)).to(hr.x.dtype)
            if dates:
                sets = ([(hr, b)] if n_crops else []) + [(d["hr"], d["idx"][0]) for d in dates]
                lr_x, outs = self.mix.tail_multi(lr, sets)
                hr_x = outs[0] if n_crops else None
                date_x = outs[1:] if n_crops else outs
            else:
                lr_x, hr_x = self.mix.tail(lr, hr, b if n_crops else None)
        lr_F = lr.pooled(lr_x).float()                                         # (B,G,h,w,D)
        if self.lr_tile != TILE:          # tiles back into place: (B g g, G, t, t, D) -> (B, G, h, w, D)
            g = TILE // self.lr_tile
            lr_F = rearrange(lr_F, "(b gy gx) s h w d -> b s (gy h) (gx w) d", gy=g, gx=g)
        v = hr.pooled(hr_x).float()[:, 0, 0, 0] if n_crops else None           # (N, D)
        if isinstance(self.readout, (CellAttnReadout, NativeSetReadout)):   # sparse, per cell
            # every crop set with its gate log-weight, or None (ungated: log-weight 0)
            vs, idx, logw = [], [], []
            if n_crops:
                vs.append(v)
                idx.append(torch.stack([b, t, y, x]))
                if self.selector is not None:
                    logw.append(F.logsigmoid(scores[b, t, y, x].float()))
                elif self._dps_P is not None:   # weight exp(p - sg p) = 1, gradient dp/ds
                    pv = self._dps_P[b, t, y, x]
                    logw.append(pv - pv.detach())
                else:
                    logw.append(None)
            sizes = [torch.ones_like(b)] if n_crops else []
            for d, dx_ in zip(dates, date_x if dates else []):
                vs.append(d["hr"].pooled(dx_).float()[:, 0, 0, 0])
                idx.append(torch.stack(d["idx"]))
                logw.append(d["logw"])
                sizes.append(torch.full_like(d["idx"][0], d.get("size", 1)))
            if all(w is None for w in logw):
                logw = []
            else:
                logw = [w if w is not None else u.new_zeros(len(u)) for w, u in zip(logw, vs)]
            n_all = sum(len(u) for u in vs)
            empty = torch.zeros(0, dtype=torch.long, device=device)
            if isinstance(self.readout, NativeSetReadout):
                self.gain_ctx = None
                teach = (self.self_teach > 0 and self.training and extra is not None
                         and extra.get("target") is not None and self.task_loss_fn is not None)
                values = (self.joint is not None and self.training
                          and (n_crops > 0 or bool(self._date_sets))
                          and not (teach and self.teach_value == "recon")
                          and self.joint_grad != "dps")
                out = self.readout(
                    self.head.linear, lr_F, season, present,
                    torch.cat(vs) if vs else lr_F.new_zeros(0, lr_F.shape[-1]),
                    tuple(torch.cat(idx, 1)) if idx else (empty,) * 4,
                    torch.cat(logw) if logw else None,
                    torch.cat(sizes) if sizes else None, parts=values, hr_px=hr_px,
                    hr_fp=self.crop_px)
                self.joint_loss = self.teach_loss = None
                if values:
                    logits, parts = out
                    self._joint_values(parts, *((b, t, y, x) if n_crops else (None,) * 4),
                                       extra.get("target") if extra is not None else None, G)
                else:
                    logits = out
                if (self.joint_target == "drop" and self.training and n_crops
                        and extra is not None and extra.get("target") is not None):
                    self._drop_loss(lr, hr, b, t, y, x, logits.detach(), season, present,
                                    extra["target"])
                if teach:       # dense self-teacher on the ps4 month crops (v, picks)
                    self._self_teach(ms, season, present, frac, extra, lr_F,
                                     v if n_crops else None, (b, t, y, x) if n_crops else None,
                                     None, pairs)
                return logits, present, n_all
            pooled = self.readout(
                lr_F, seasons_used(season, present, G),
                torch.cat(vs) if vs else lr_F.new_zeros(0, lr_F.shape[-1]),
                tuple(torch.cat(idx, 1)) if idx else (empty,) * 4,
                torch.cat(logw) if logw else None)
            self.gain_ctx = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.head(pooled[:, None], torch.ones_like(present[:, :1]))
            return logits.float(), present, n_all
        drop_pair = False
        if self.native:                   # every token through the head at its own patch size
            self.gain_ctx = None
            gate = (torch.sigmoid(scores[b, t, y, x].float())
                    if n_crops and self.selector is not None else None)
            if isinstance(self.readout, TimeSetReadout):
                empty = torch.zeros(0, dtype=torch.long, device=device)
                logits = native_time_logits(
                    self.head.linear, self.readout, lr_F, seasons_used(season, present, G),
                    season, present, frac, v if n_crops else lr_F.new_zeros(0, lr_F.shape[-1]),
                    (b, t, y, x) if n_crops else (empty,) * 4,
                    F.logsigmoid(scores[b, t, y, x].float())
                    if n_crops and self.selector is not None else None)
                return logits, present, n_crops
            crop_args = (v if n_crops else None, (b, t, y, x) if n_crops else None, gate)
            logits = (native_logits(self.head.linear, lr_F, season, present, *crop_args)
                      if self.native_score is None else
                      native_attn_logits(self.head.linear, self.native_score, lr_F, season,
                                         present, *crop_args))
            return logits, present, n_crops
        base = lr_F[bidx, season.clamp(min=0)]                                 # (B,T,h,w,D)
        h, w, D = base.shape[2:]
        base = F.interpolate(base.flatten(0, 1).permute(0, 3, 1, 2), size=(H, H),
                             mode="bilinear", align_corners=False)
        out = base.permute(0, 2, 3, 1).reshape(B, T, H, H, D)
        if n_crops:
            o = out[b, t, y, x]
            delta = (v - o).detach()      # what the crop changes at its slot (vs bilinear)
        if isinstance(self.readout, TimeSetReadout):          # per-step time slices
            empty = torch.zeros(0, dtype=torch.long, device=device)
            wgt, feat, crop_of = self.readout.weights(
                lr_F, seasons_used(season, present, G), season, present, frac,
                v if n_crops else lr_F.new_zeros(0, D), (b, t, y, x) if n_crops else (empty,) * 4,
                F.logsigmoid(scores[b, t, y, x].float())
                if n_crops and self.selector is not None else None)
            out = torch.einsum("btqk,bqkd->btqd", wgt, feat).reshape(B, T, H, H, D)
            self.joint_loss = self.teach_loss = None
            if (self.self_teach and self.training and extra is not None
                    and extra.get("target") is not None and self.task_loss_fn is not None):
                # dense self-teacher: reconstruction + teacher task loss (+ the selector's values)
                self._self_teach(ms, season, present, frac, extra, lr_F, v if n_crops else None,
                                 (b, t, y, x) if n_crops else None,
                                 F.logsigmoid(scores[b, t, y, x].float())
                                 if n_crops and self.selector is not None else None, pairs)
            elif (self.joint is not None and self.training and n_crops and crop_of is not None
                    and extra is not None and extra.get("target") is not None):
                val = pair_loo_values(self.head, wgt.detach(), feat.detach(), crop_of, pairs,
                                      extra["target"], lr_F.shape[1])
                cnt = torch.bincount(b, minlength=B).clamp(min=1).float()
                mu = torch.zeros(B, device=device).index_add_(0, b, val) / cnt
                var = torch.zeros(B, device=device).index_add_(0, b, (val - mu[b]) ** 2) / cnt
                z = (val - mu[b]) * torch.rsqrt(var[b] + 1e-6)
                self.joint_loss = F.mse_loss(self._joint_p[b, t, y, x].float(), z)
                self._loo_v = val
                if self.joint_target == "drop":     # trained on the dropout-pass values below
                    self.joint_loss, drop_pair = None, True
        elif self.readout is not None:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            lr_valid = seasons_used(season, present, G)
            out = self.readout(
                lr_F, lr_valid, season, present,
                v if n_crops else lr_F.new_zeros(0, D),
                (b, t, y, x) if n_crops else (empty,) * 4,
                F.logsigmoid(scores[b, t, y, x].float())
                if n_crops and self.selector is not None else None,
                self._guide(ms) if self.readout.guided else None)
        elif n_crops:
            if self.selector is not None:
                v = o + torch.sigmoid(scores[b, t, y, x])[:, None] * (v - o)
            out = out.index_put((b, t, y, x), v)
        out = out * present[:, :, None, None, None]
        self.gain_ctx = ((out, (b, t, y, x), delta, raw_scores)
                         if n_crops and self.selector is not None and self.training else None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.pool is None:
                logits = self.head(out, present, pairs)
            else:
                pooled = self.pool(out, present)[:, None]
                logits = self.head(pooled, torch.ones_like(present[:, :1]))
        if drop_pair:

            def read(lr_F2, v2, kidx):          # the series readout + pair head, kept crops only
                w2, f2, _ = self.readout.weights(lr_F2, seasons_used(season, present, G), season,
                                                 present, frac, v2, kidx, None)
                o2 = torch.einsum("btqk,bqkd->btqd", w2, f2).reshape(B, T, H, H, D)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.head(o2 * present[:, :, None, None, None], present, pairs).float()

            self._drop_loss(lr, hr, b, t, y, x, logits.detach().float(), season, present,
                            extra["target"], read)
        return logits.float(), present, n_crops


def gate_value_loss(model) -> torch.Tensor | None:
    """--joint_target gate, AFTER the task loss's backward: the predictor's MSE to each picked
    crop's -dLoss/dgate, z-scored per sample (as the leave-one-out values are)."""
    if model._gate is None or model._gate.grad is None:
        return None
    b, t, y, x = model._gate_idx
    v = -model._gate.grad.float()
    loo = getattr(model, "_loo_v", None)
    if loo is not None and len(loo) == len(v):      # how far the two targets agree
        model.gate_loo_corr = float(torch.corrcoef(torch.stack([v, loo]))[0, 1])
    B = int(b.max()) + 1
    cnt = torch.bincount(b, minlength=B).clamp(min=1).float()
    mu = torch.zeros(B, device=v.device).index_add_(0, b, v) / cnt
    var = torch.zeros(B, device=v.device).index_add_(0, b, (v - mu[b]) ** 2) / cnt
    z = (v - mu[b]) * torch.rsqrt(var[b] + 1e-12)
    return F.mse_loss(model._joint_p[b, t, y, x].float(), z)


def rank_within(values: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Rank of each value within its group, scaled to [0, 1] (0 = lowest)."""
    o1 = torch.argsort(values)
    order = o1[torch.argsort(group[o1], stable=True)]
    counts = torch.bincount(group)
    starts = torch.cumsum(counts, 0) - counts
    g = group[order]
    out = torch.empty_like(values)
    out[order] = ((torch.arange(len(values), device=values.device) - starts[g]).float()
                  / (counts[g] - 1).clamp(min=1).float())
    return out


def gain_loss(model, task_loss) -> torch.Tensor:
    """Rank-BCE of the selector's raw scores on the first-order gain of its picked crops."""
    out, (b, t, y, x), delta, raw = model.gain_ctx
    grad, = torch.autograd.grad(task_loss, out, retain_graph=True)
    gain = -(grad[b, t, y, x].float() * delta).sum(-1)       # loss drop the crop buys, 1st order
    target = rank_within(gain, b)
    return F.binary_cross_entropy_with_logits(raw[b, t, y, x], target)


def teach_modality(task: Task) -> str:
    """The modality whose raw acquisitions the dense self-teacher reads (--self_teach)."""
    mods = list(task.olmoearth.modalities)
    return "S2" if "S2" in mods else mods[0]


class WithDates(Dataset):
    """(months, seasons, raw): the task's sample, its seasons, and every raw frame of each
    modality on its own dates (the date level; PASTIS: S2 and S1A dates differ).

    teach_n > 0 (--self_teach without a date level): raw keeps only the teacher's modality and
    only teach_n of its frames, drawn HERE, uniformly at random among the valid frames with at
    least half their pixels valid (clear), else among all valid frames -- independent of the
    model's picks. Packing ~100 acquisitions per tile for a teacher that reads 4 was most of
    the loading time."""

    def __init__(self, task: Task, split: str, lr, teach_n: int = 0):
        self.teach_n, self.teach_mod = teach_n, teach_modality(task)
        self.raw = task.open(split, {m: All() for m in task.olmoearth.modalities})
        self.policies, self.lr = task.policies(), lr
        self.spec = self.raw.SPEC.modalities

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        raw = self.raw[i]
        months = replace(raw, modalities={m: self.policies[m](md, self.spec[m])
                                          for m, md in raw.modalities.items()})
        seasons = replace(months, modalities={m: self.lr(md, self.spec[m])
                                              for m, md in months.modalities.items()})
        if self.teach_n:
            md = raw.modalities[self.teach_mod]
            ok = md.frame_valid.clone()
            if md.pixel_valid is not None:
                clear = ok & (md.pixel_valid.flatten(1).float().mean(1) >= 0.5)
                ok = clear if int(clear.sum()) >= self.teach_n else ok
            cand = ok.nonzero()[:, 0]
            pick = cand[torch.randperm(len(cand))[: self.teach_n]].sort().values
            if len(pick):
                raw = replace(raw, modalities={self.teach_mod: md.select_frames(pick)})
        return months, seasons, raw


class UnionDates(Dataset):
    """--lr_steps raw: every raw acquisition, each modality moved onto the UNION of the sample's
    acquisition dates (OlmoEarth wants one time axis; PASTIS S2 and S1A dates differ). A
    modality without a frame on a date is an invalid (MISSING) frame there: no tokens."""

    def __init__(self, task: Task, split: str):
        self.ds = task.open(split, {m: All() for m in task.olmoearth.modalities})
        self.SPEC = self.ds.SPEC

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        from manyup.data.time import TimeAxis
        s = self.ds[i]
        days = torch.unique(torch.cat([md.time.start for md in s.modalities.values()]))
        out = {}
        for m, md in s.modalities.items():
            at = torch.searchsorted(days, md.time.start)
            T = len(days)

            def put(v, fill=0):
                full = torch.full((T,) + tuple(v.shape[1:]), fill, dtype=v.dtype)
                full[at] = v
                return full
            out[m] = replace(
                md, data=put(md.data), time=TimeAxis(days, days.clone(), md.time.precision),
                frame_valid=put(md.frame_valid, False),
                pixel_valid=None if md.pixel_valid is None else put(md.pixel_valid, False),
                aux={k: put(v) for k, v in md.aux.items()})
        return replace(s, modalities=out)


def loader(task, split, batch_size, num_workers, shuffle, lr_steps: str = "season",
           epoch_samples: int = 0, dates: bool = False, teach_n: int = 0,
           mid_months: bool = False):
    """epoch_samples > 0 (train): each epoch is that many chips drawn afresh at random.
    dates: also every raw frame (WithDates) for the date level."""
    # season: the task's seasons; all: its steps (months); year: ONE composite of all of them;
    # raw: every acquisition (UnionDates) -- for none of the last three is there a second level
    lr = (All() if lr_steps in ("all", "raw") else YearGroup() if lr_steps == "year"
          else task.lr_policy())
    lr = GroupSteps(lr) if isinstance(lr, int) else lr
    ds = (WithDates(task, split, lr, teach_n) if dates
          else Paired(UnionDates(task, split) if lr_steps == "raw" else task.dataset(split), lr))
    sampler = (RandomSampler(ds, num_samples=min(epoch_samples, len(ds)))
               if shuffle and epoch_samples else None)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle and sampler is None, num_workers=num_workers,
                      collate_fn=Collate(task, dates, teach_n if dates else 0, mid_months,
                                         any_modality=lr_steps == "raw"),
                      # dates: a batch is ~1.4 GB of raw frames whose length changes from batch
                      # to batch -- pinned, the CUDA host allocator kept ~29 GB of blocks in the
                      # main process (2026-10-04), and idle val / test workers held their own
                      persistent_workers=num_workers > 0 and (shuffle or not dates),
                      prefetch_factor=(1 if dates else 2) if num_workers else None,
                      drop_last=shuffle, pin_memory=not dates)


@torch.no_grad()
def evaluate(model, task, dl, limit):
    model.eval()
    if getattr(model, "crop_free", False):      # the same crops (and counts) at every call
        torch.manual_seed(1234)
    meter, crops, n = task.meter(), 0, 0
    for i, (*x, target) in enumerate(dl):
        if limit and i >= limit:
            break
        target = target.cuda()
        logits, _, nc = model(*x)
        meter.update(logits, target)
        crops, n = crops + nc, n + target.shape[0]
    evaluate.last_meter = meter     # the raw counts of this call (the folds' summed confusion)
    return meter.compute(), crops / max(n, 1)


SWEEP_COUNTS = (0, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def crop_count_sweep(model, task, dl, a, name):
    """--vary_crops: the one model tested at every crop count in SWEEP_COUNTS up to its cap,
    with uniformly random crops and with the staggered lattice (free_pick), plus the measured
    compute of each count -> results/<task>/inject_ncrops.csv (column `pick`; `train_pick` is in
    the run name: _random_ / _grid_)."""
    from exp.bench.flop_trace import append_row, count_macs, flop_row
    top = a.n_crops or model.k
    model.eval()
    shift = getattr(model, "shift_stride", 0)       # the whole shifted grid: one count, grid only
    for n_ in sorted({n for n in SWEEP_COUNTS
                      if n < top and (n or not model.no_seasons) and not shift} | {top}):
        model.n_eval = n_
        torch.manual_seed(1234)
        n_s = 0
        with torch.no_grad(), count_macs() as count:
            for i, (*x, target) in enumerate(dl["test"]):
                if i >= 2:
                    break
                model(*x)
                n_s += target.shape[0]
        fr = flop_row(name, a.task, a.model_size, count, n_s, len(model.backbone().blocks), n_)
        for pick in (("grid",) if shift else ("random", "grid")):
            model.free_pick = pick
            model._cov = [0.0, 0.0, 0]
            r, c_ = evaluate(model, task, dl["test"], a.limit_batches)
            model.free_pick = None
            cov = [round(v / max(model._cov[2], 1), 4) for v in model._cov[:2]]
            if not a.limit_batches:
                append_row(Path(f"results/{a.task}/inject_ncrops.csv"), {
                    "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "task": a.task,
                    "px": a.crop_px, "inject_layer": a.inject_layer, "seed": a.seed,
                    "n_max": top, "n_test": n_, "test_crops_per_sample": round(c_, 1),
                    "gmacs": round(fr["gmacs"], 3), "tokens_per_block": fr["tokens_per_block"],
                    "test_primary": round(r["primary"], 5), "pick": pick,
                    "grid_mode": a.grid_mode if pick == "grid" else "",
                    "coverage": cov[0], "coverage_space": cov[1]})
            print(f"TEST n_crops {n_} {pick}: {fr['gmacs']:.2f} GMACs | coverage {cov[0]:.3f} "
                  f"(space {cov[1]:.3f}) | {_brief(r)}", flush=True)
    model.n_eval = top


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--selector", required=True, choices=SELECTORS)
    ap.add_argument("--inject_layer", type=int, default=6)
    ap.add_argument("--k_frac", type=float, default=1 / 6,
                    help="crops per sample, as a fraction of months x 16 x 16")
    ap.add_argument("--select_unit", default="slot", choices=("slot", "location", "pair"))
    ap.add_argument("--lr_steps", default="season", choices=("season", "all", "year", "raw"),
                    help="the base sequence's steps: the task's seasons, all its steps (months), "
                         "one composite of every step (year), or every raw acquisition (raw); "
                         "year / raw: --selector none only")
    ap.add_argument("--epoch_samples", type=int, default=0,
                    help="train chips per epoch, drawn afresh (0 = the whole split)")
    ap.add_argument("--lr_patch", type=int, default=16, choices=(16, 8, 4))
    ap.add_argument("--lr_tile", type=int, default=64, choices=(64, 4),
                    help="reference only: encode each lr_tile px tile as its own sequence")
    ap.add_argument("--selector_hidden", type=int, default=64)
    ap.add_argument("--pick", default="topk", choices=("topk", "random", "per_step"))
    ap.add_argument("--head", default="mean", choices=("mean", "attn"))
    ap.add_argument("--score_norm", action="store_true")
    ap.add_argument("--score_l1", type=float, default=0.0)
    ap.add_argument("--sel_temporal", action="store_true")
    ap.add_argument("--gain_loss", type=float, default=0.0)
    ap.add_argument("--readout", default="bilinear", choices=READOUTS)
    ap.add_argument("--frac_slot", action="store_true",
                    help="crops' time slot = season + position inside it (distinct encodings "
                         "for crops of one season, dates of one month)")
    ap.add_argument("--date_level", default="none", choices=DATE_LEVELS,
                    help="3rd level: single-date ps4 crops of each modality (PASTIS); nested = "
                         "only in picked month-cells, free = anywhere; needs --readout cellattn")
    ap.add_argument("--date_per_month", type=float, default=2.0,
                    help="date crops per month crop (per sample)")
    ap.add_argument("--date_selector", default="random", choices=("random", "xattn"),
                    help="date picks: random, or a learned cross-attention selector per modality")
    ap.add_argument("--date_gate", default="month", choices=("month", "none"),
                    help="random date crops gated by their month pick's learned score, or not")
    ap.add_argument("--res_mix", default=None,
                    help="token-budget shares at ps16,ps8,ps4, e.g. 1,1,1 (random single-month "
                         "crops at each patch size; needs --readout native_set)")
    ap.add_argument("--pos", default="corner", choices=POS,
                    help="crop / token position encoding (manyup.inject): corner = top-left "
                         "pixel; center = crops at their centre in the low-res frame; box = "
                         "corner + a zero-init map of the bottom-right pixel on every token")
    ap.add_argument("--attn", default="joint", choices=ATTN,
                    help="blocks [L, n) (manyup.inject): joint = one sequence of lr tokens + "
                         "crops; oneway = lr alone, each crop reads its own tokens + its "
                         "sample's lr keys / values; overlap = oneway, only the lr cell under it")
    ap.add_argument("--head_attn", default="alone", choices=HEAD_ATTN,
                    help="blocks [0, L), --inject_layer > 0 (manyup.inject): alone = a crop "
                         "reads its own tokens; oneway = + its sample's cached lr keys / "
                         "values; overlap = + only the lr cell under it")
    ap.add_argument("--doy", action="store_true",
                    help="every token also gets a zero-init map of its day of year "
                         "(manyup.inject); month steps are then stamped mid-month")
    ap.add_argument("--joint", action="store_true",
                    help="joint selector: a value predictor picks (1 - eps) of the crops, trained "
                         "on each pick's in-context leave-one-out value (needs native_set)")
    ap.add_argument("--joint_feat", default="prelook", choices=("prelook", "lrL"),
                    help="--joint predictor inputs: a crop-free 12-block pre-look (tokens, logits, "
                         "pixel variance), or the season tokens after blocks [0, L) of this pass only")
    ap.add_argument("--joint_target", default="loo", choices=("loo", "gate", "drop"),
                    help="--joint value target: readout leave-one-out | full-path gate gradient "
                         "| loss rise in a second pass without a random half of the crops")
    ap.add_argument("--joint_in", default="onehot", choices=("onehot", "enc"),
                    help="--joint inputs: e, e - season mean, one-hots | (e - season mean) + time encoding")
    ap.add_argument("--joint_arch", default="mlp", choices=JOINT_ARCH,
                    help="--joint predictor: per-candidate MLP | candidate queries the season tokens")
    ap.add_argument("--joint_embed", default="shared", choices=("shared", "own"),
                    help="--joint: the encoder's patch embedding | the selector's own trained copy")
    ap.add_argument("--jitter", action="store_true", help="ps4 month crops 0-3 px off the grid")
    ap.add_argument("--loc_stride", type=int, default=4, choices=(4, 6, 8),
                    help="--select_unit location, --crop_px 8, oracle / entropy / random: the "
                         "candidate windows' top-left corners every loc_stride px (6: <= 2 px "
                         "overlap, 8: none); 4 = a window centred on every ps4 cell (default)")
    ap.add_argument("--crop_px", type=int, default=4, choices=(4, 8, 16),
                    help="crop window side: 4 = ps4 crops; 16 = one ps16 token on the 16 px "
                         "window centred on each picked ps4 cell")
    ap.add_argument("--apt_pool", action="store_true",
                    help="--selector apt: pick ps4 CELLS first (summed over their present "
                         "months' score), merge consecutive similar months at each picked "
                         "cell (--apt_merge_mse) into one interpolated-slot crop, and spend "
                         "the freed budget on more cells (see InjectFT._apt_pool_mask)")
    ap.add_argument("--apt_merge_mse", type=float, default=0.01,
                    help="--apt_pool: consecutive months at a cell merge into one crop when "
                         "their per-band-normalized pixel MSE is below this")
    ap.add_argument("--apt_time", default="sum", choices=("sum", "none"),
                    help="--selector apt: sum H_space + H_time (default), or 'none' for the "
                         "more literal APT -- pure per-frame spatial entropy, time-ignorant, "
                         "flat top-k over every (month, cell) slot same as every other selector")
    ap.add_argument("--apt_res", action="store_true",
                    help="--selector apt: let entropy choose patch SIZE (ps16 vs ps4) instead "
                         "of just which ps4 slots to fill -- every present (month, ps16 cell) "
                         "starts as one ps16 crop, splits into 4 ps4 children when the entropy "
                         "GAIN of doing so is highest-ranked and the 3 extra tokens fit the "
                         "budget (see InjectFT._apt_res_mask). No --apt_pool / --res_mix.")
    ap.add_argument("--apt_octree", action="store_true",
                    help="--selector apt: a 3D (space, space, time) octree -- every present "
                         "ps16 cell starts spanning its FULL present time range as one crop; "
                         "entropy ABOVE --apt_tau splits it in space, time, or both (whichever "
                         "gives the best entropy-reduction per extra token), recursed down to "
                         "ps4 (the finest size this codebase's footprint machinery supports). "
                         "NO TOKEN BUDGET -- token count is whatever the recursion produces at "
                         "the given tau (see InjectFT._apt_octree_mask / _apt_octree_crops, "
                         "and exp.upsamplers.calibrate_apt_octree_tau to pick a tau that lands "
                         "near a target average count). No --apt_pool / --apt_res / --res_mix.")
    ap.add_argument("--apt_tau", type=float, default=0.0,
                    help="--apt_octree: the entropy threshold driving its recursion (required, "
                         "> 0, when --apt_octree is set) -- dataset- and band-count-specific, "
                         "see calibrate_apt_octree_tau.py")
    ap.add_argument("--apt_octree_pool", default="input", choices=("input", "embed", "attn"),
                    help="--apt_octree: how a multi-month leaf's member months combine into "
                         "one token -- 'input' (default): pixel+slot average BEFORE embedding "
                         "(cheapest, one embed() per leaf); 'embed': embed every member month "
                         "SEPARATELY, run blocks [0,L), MEAN-pool the resulting embeddings; "
                         "'attn': same but a LEARNED per-member softmax weight instead of a "
                         "plain mean (reuses --acq_pool's attn-weight mechanism). Compute cost "
                         "input < embed < attn (embed/attn cost one embed()+blocks[0,L) pass "
                         "PER MEMBER MONTH, not per leaf). embed/attn need --apt_octree.")
    ap.add_argument("--apt_octree_space_only", action="store_true",
                    help="--apt_octree: start every ps16 cell at a SINGLE present month "
                         "(rank_lens always 1) instead of its full present-rank range, so the "
                         "time candidate is never offered -- a per-month space-only quadtree, "
                         "structurally identical to --selector grid's one-token-per-(month,"
                         "cell) layout (2026-10-13 isolating experiment: does apt_octree's "
                         "PASTIS loss to grid at a MATCHED token budget come from where it "
                         "places picks, or from the ~92% of its leaves that pixel+slot-AVERAGE "
                         "several months into one blurred token where grid gives a sharp "
                         "single-month token per cell -- see [[apt-spacetime-octree]]).")
    ap.add_argument("--crop_free", action="store_true",
                    help="crops at uniformly random pixel positions and months (off every grid)")
    ap.add_argument("--n_crops", type=int, default=0, help="--crop_free: crops per sample (0: k)")
    ap.add_argument("--sweep_only", action="store_true",
                    help="--vary_crops: load the finished run's checkpoint and redo only the "
                         "crop-count test sweep (crop_count_sweep)")
    ap.add_argument("--shift_stride", type=int, default=0, choices=(0, 4, 8, 16),
                    help="--crop_free --selector grid --vary_crops: the crops are every crop_px "
                         "window of the stride-S grid shifted off the patch grid, in every "
                         "month, zero-padded past the tile edge (with --lr_steps all: on-grid "
                         "months + off-grid months of the same patch size)")
    ap.add_argument("--no_seasons", action="store_true",
                    help="--crop_free --inject_layer 0: no season tokens at all, the month crops "
                         "are the whole input (spatial-resolution-only comparison)")
    ap.add_argument("--crop_counts", default="",
                    help="--vary_crops: draw each training sample's count from this set, e.g. "
                         "0,64,128,256,512,1024,2048 (sets --n_crops to its largest)")
    ap.add_argument("--vary_crops", action="store_true",
                    help="--crop_free: train with 0..n_crops crops per sample (uniform); test "
                         "the one model at 0, 16, 32, ... n_crops -> results/<task>/inject_ncrops.csv")
    ap.add_argument("--date_joint", default="none", choices=("none", "only", "hier"),
                    help="--joint at the date level, one modality per crop, token budget: date "
                         "crops only | month crops (half) + dates inside the picked months")
    ap.add_argument("--joint_res", action="store_true",
                    help="--joint also chooses the resolution: ps4 and ps8 (month, cell) "
                         "candidates, one token each")
    ap.add_argument("--self_teach", type=int, default=0,
                    help="n > 0: the encoder is its own dense teacher on n random raw steps "
                         "(ps4 over the tile, task loss); the mixed-resolution readout must "
                         "reconstruct its dense FEATURES at those times (--readout tsplat / tquery); "
                         "--joint values = reconstruction gain")
    ap.add_argument("--teach_event", action="store_true",
                    help="--self_teach on an event series (GEOID): the teacher's steps always "
                         "include the last date before the event and the post-event date; the "
                         "other n - 2 stay random")
    ap.add_argument("--teach_value", default="recon", choices=("recon", "label", "both"),
                    help="--self_teach + --joint: the selector regresses each crop's "
                         "leave-one-out reconstruction gain, task-loss gain (native_set), or both")
    ap.add_argument("--teach_detach", default="none", choices=("none", "recon", "all"),
                    help="--self_teach gradients kept from the encoder: recon = the "
                         "reconstruction loss trains only the splat (+ the selector's values); "
                         "all = also no teacher task loss (the teacher is a no-grad pass)")
    ap.add_argument("--teach_weight", type=float, default=1.0, help="--self_teach: weight of "
                    "the reconstruction loss (the teacher's task loss has weight 1)")
    ap.add_argument("--pair_match_random", action="store_true",
                    help="--joint, pair unit: cap the picks at the date-crop count of a random "
                         "draw with the same pair budget (pairs sharing a date cost one crop)")
    ap.add_argument("--eps_start", type=float, default=1.0, help="--joint: random share, start")
    ap.add_argument("--eps_end", type=float, default=0.3, help="--joint: random share, end")
    ap.add_argument("--eps_hold", type=float, default=0.0,
                    help="--joint: fraction of the epochs held at eps_start before the anneal "
                         "(from scratch: the backbone first learns to use random crops)")
    ap.add_argument("--eps_anneal", type=float, default=0.5,
                    help="--joint: fraction of the epochs over which eps goes start -> end")
    ap.add_argument("--joint_weight", type=float, default=1.0, help="--joint: predictor loss weight")
    ap.add_argument("--init_ckpt", default="",
                    help="warm start from checkpoints/<run>.pt (no frozen phase; the backbone "
                         "at lr x UNFREEZE_LR_FACTOR from the start, a new selector at lr)")
    ap.add_argument("--no_month_crops", action="store_true",
                    help="the month picks only route nested date crops; no month crops")
    ap.add_argument("--joint_explore", default="random", choices=("random", "grid"),
                    help="--joint: the eps share of the picks is uniform random slots, or a "
                         "coarse staggered lattice of that many slots (grid_scores) -- an even "
                         "coverage floor under the predictor's picks")
    ap.add_argument("--joint_grad", default="value", choices=("value", "dps", "both"),
                    help="--joint: train the predictor on leave-one-out values, through the task "
                    "loss with DPS's perturbed top-k, or both (see InjectFT.__init__)")
    ap.add_argument("--dps_sigma", type=float, default=0.5,
                    help="--joint_grad dps/both: noise scale, in per-sample score std units")
    ap.add_argument("--dps_draws", type=int, default=64, help="--joint_grad dps/both: noise draws")
    ap.add_argument("--drop_frac", type=float, default=0.5,
                    help="--joint_target drop: share of the crops removed in a dropout pass")
    ap.add_argument("--drop_passes", type=int, default=1,
                    help="--joint_target drop: dropout passes averaged per step")
    ap.add_argument("--grid_mode", default="stagger", choices=("stagger", "spatial", "c2f", "shared"),
                    help="--selector grid: stagger = every cell cropped in every m-th month, the "
                         "lattice shifted month to month; spatial = the same 1-in-m cells in "
                         "every month (see grid_scores)")
    ap.add_argument("--grid_phase", default="random", choices=("random", "fixed"),
                    help="--selector grid: a fresh lattice offset per training sample, or the "
                         "evaluation offset (0) in training too")
    ap.add_argument("--acq_pool", default="none", choices=("none", "mean", "attn"),
                    help="month crops from the month's raw acquisitions, not its composite: each "
                    "picked (month, cell)'s acquisitions run blocks [0, L) as one patch time "
                    "series and are pooled per token kind (InjectFT._pool_acquisitions); needs "
                    "--inject_layer > 0 and --frac_slot or --doy")
    ap.add_argument("--flops", type=int, default=0, help="measure inference MACs per sample "
                    "over this many val batches (exp.bench.flop_trace), write one row to "
                    "results/<task>/inject_flops.csv and exit; nothing is trained")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model_size", default="tiny", choices=list(MODEL_SIZE_TO_ID))
    ap.add_argument("--batch_size", type=int, default=None, help="default: the task's ps16")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrain", action="store_true",
                    help="OlmoEarth from its random initialization: no frozen phase, the whole "
                         "model at --lr from the first step")
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="--no_pretrain: linear learning-rate warmup over the first epochs")
    ap.add_argument("--fold", type=int, default=1, choices=(1, 2, 3, 4, 5),
                    help="PASTIS benchmark protocol fold (1 = the split of every earlier run: "
                         "train folds 1-3, val 4, test 5; f rotates all three by f - 1)")
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=None)
    a = ap.parse_args()
    if a.lr_steps in ("year", "raw") and a.selector != "none":
        raise SystemExit("--lr_steps year / raw: --selector none (temporal-resolution references)")
    if a.shift_stride:
        if not (a.crop_free and a.vary_crops and a.selector == "grid") or a.crop_counts:
            raise SystemExit("--shift_stride needs --crop_free --vary_crops --selector grid, "
                             "no --crop_counts")
        a.n_crops = ((24 if a.task.startswith("yieldsat") else 12)
                     * len(shifted_positions(a.crop_px, a.shift_stride)) ** 2)   # upper bound
    if a.crop_counts:
        if not (a.crop_free and a.vary_crops):
            raise SystemExit("--crop_counts needs --crop_free --vary_crops")
        a.n_crops = max(int(v) for v in a.crop_counts.split(","))
    res_mix = tuple(float(v) for v in a.res_mix.split(",")) if a.res_mix else None

    task = TASKS[a.task]
    if a.fold != 1:                 # the benchmark's rotated train / val / test folds
        if not hasattr(task, "fold_groups"):
            raise SystemExit(f"--fold: {a.task} has no benchmark folds")
        task.fold = a.fold
        print(f"Protocol fold {a.fold}: dataset folds {task.fold_groups()}")
    if a.no_pretrain and a.init_ckpt:
        raise SystemExit("--no_pretrain with --init_ckpt: the checkpoint replaces the init anyway")
    if a.warmup_epochs and not a.no_pretrain:
        raise SystemExit("--warmup_epochs is for --no_pretrain (the fine-tune starts frozen)")
    torch.manual_seed(a.seed)
    bs = a.batch_size or task.batch_size[16]
    k_tag = "" if a.selector == "none" else (
        f"_L{a.inject_layer}_k{a.k_frac:.3g}"
        + {"slot": "", "location": "_loc", "pair": "_pair"}[a.select_unit])
    name = (f"inject_fft_{a.task}_{a.model_size}_{a.selector}{k_tag}"
            + ("" if a.pick == "topk" else f"_pick{a.pick}")
            + ("" if a.head == "mean" else f"_head{a.head}")
            + ("_snorm" if a.score_norm else "") + (f"_sl1{a.score_l1:g}" if a.score_l1 else "")
            + ("_seltemp" if a.sel_temporal else "") + (f"_gain{a.gain_loss:g}" if a.gain_loss else "")
            + ("" if a.readout == "bilinear" else f"_{a.readout}")
            + ("_frac" if a.frac_slot else "")
            + ("" if a.date_level == "none" else
               f"_date{a.date_level}{a.date_per_month:g}" + ("_nomonth" if a.no_month_crops else "")
               + ("_dxattn" if a.date_selector == "xattn" else "")
               + ("_dnogate" if a.date_gate == "none" else ""))
            + ("_res" + "-".join(f"{v:g}" for v in res_mix) if res_mix else "")
            + (f"_joint{a.eps_start:g}-{a.eps_end:g}a{a.eps_anneal:g}"
               + (f"h{a.eps_hold:g}" if a.eps_hold else "") if a.joint else "")
            + ("_jfL" if a.joint and a.joint_feat == "lrL" else "")
            + ("" if a.acq_pool == "none" else f"_acq{a.acq_pool}")
            + (f"_apool{a.apt_merge_mse:g}" if a.selector == "apt" and a.apt_pool else "")
            + ("_atnone" if a.selector == "apt" and a.apt_time == "none" else "")
            + ("_ares" if a.selector == "apt" and a.apt_res else "")
            + (f"_aoct{a.apt_tau:g}" + ("" if a.apt_octree_pool == "input"
                                        else f"p{a.apt_octree_pool}")
                                     + ("_sonly" if a.apt_octree_space_only else "")
               if a.selector == "apt" and a.apt_octree else "")
            + ("_gspatial" if a.selector == "grid" and a.grid_mode == "spatial" else "")
            + (f"_g{a.grid_mode}" if a.selector == "grid" and a.grid_mode in ("c2f", "shared") else "")
            + ("_gfixed" if a.selector == "grid" and a.grid_phase == "fixed" else "")
            + ("_jxgrid" if a.joint and a.joint_explore == "grid" else "")
            + (f"_jg{a.joint_grad}" + ("" if a.dps_sigma == 0.5 else f"s{a.dps_sigma:g}")
               if a.joint and a.joint_grad != "value" else "")
            + ("" if a.joint_target == "loo" else f"_jt{a.joint_target}"
               + ("" if a.drop_frac == 0.5 else f"f{a.drop_frac:g}")
               + ("" if a.drop_passes == 1 else f"x{a.drop_passes}")) + ("_jienc" if a.joint_in == "enc" else "")
            + ("" if a.joint_arch == "mlp" else f"_ja{a.joint_arch}")
            + ("_jeown" if a.joint_embed == "own" else "") + ("_jit" if a.jitter else "")
            + ("" if a.date_joint == "none" else f"_dj{a.date_joint}")
            + ("" if a.crop_px == 4 else f"_cpx{a.crop_px}")
            + ("" if a.loc_stride == 4 else f"_st{a.loc_stride}")
            + ("_free" if a.crop_free else "") + (f"_n{a.n_crops}" if a.n_crops else "")
            + ("_vary" if a.vary_crops else "")
            + (f"_cnt{a.crop_counts.replace(',', '-')}" if a.crop_counts else "")
            + ("_noseason" if a.no_seasons else "")
            + (f"_shift{a.shift_stride}" + ("c" if a.shift_stride != a.crop_px else "")
               if a.shift_stride else "")
            + ("_gfix" if a.joint else "")   # 2026-10-01: runs before had gradient-free blocks
            + ("_jres8" if a.joint_res else "")
            + ("" if a.pos == "corner" else f"_pos{a.pos}") + ("_doy" if a.doy else "")
            + ("" if a.attn == "joint" else f"_{a.attn}")
            + ("" if a.head_attn == "alone" else f"_ha{a.head_attn}")
            + ("_eqcrops" if a.pair_match_random else "")
            + (f"_teach{a.self_teach}w{a.teach_weight:g}" if a.self_teach else "")
            + ("_evt" if a.teach_event else "")
            + (f"_v{a.teach_value}" if a.self_teach and a.teach_value != "recon" else "")
            + (f"_dt{a.teach_detach}" if a.self_teach and a.teach_detach != "none" else "")
            # _warmg: warm from a lattice-pick (--selector grid) checkpoint, not a random-pick one
            + (("_warmg" if "_grid_" in a.init_ckpt else "_warm")
               if a.init_ckpt else "")
            + (f"_lr{a.lr_steps}" if a.lr_steps != "season" else "") + (f"_lrps{a.lr_patch}" if a.lr_patch != 16 else "")
            + (f"_lrtile{a.lr_tile}" if a.lr_tile != 64 else "")
            + (f"_es{a.epoch_samples}" if a.epoch_samples else "")
            # _nopre: OlmoEarth from random init; _wuN: N warmup epochs; _foldF: protocol fold
            + ("_nopre" if a.no_pretrain else "")
            + (f"_wu{a.warmup_epochs}" if a.warmup_epochs else "")
            + (f"_fold{a.fold}" if a.fold != 1 else "")
            + f"_lr{a.lr:g}_ep{a.epochs}"
            + (f"_seed{a.seed}" if a.seed else ""))     # seed 0 keeps the original run names
    smoke = "_smoke" if a.limit_batches else ""
    state_path = Path(f"checkpoints/{name}{smoke}_state.pt")
    if not a.flops and not a.sweep_only and state_path.exists() and torch.load(state_path, weights_only=False).get("done"):
        print(f"{state_path} says this run is finished; nothing to do")
        return
    model = InjectFT(task, a.model_size, a.inject_layer, a.selector, a.k_frac,
                     a.select_unit, a.selector_hidden, a.pick, a.head, a.score_norm,
                     a.sel_temporal, a.readout, a.lr_patch, a.frac_slot, a.date_level,
                     a.date_per_month, not a.no_month_crops, a.date_selector, a.date_gate,
                     res_mix, a.joint, a.lr_tile, a.joint_res,
                     a.pair_match_random, a.self_teach, a.pos, a.doy, a.attn, a.head_attn,
                     a.joint_feat, a.joint_target, a.joint_in, a.joint_arch, a.joint_embed,
                     a.jitter, a.date_joint, a.crop_px, not a.no_pretrain,
                     a.apt_pool, a.apt_merge_mse, a.apt_time, a.apt_res,
                     a.apt_octree, a.apt_tau, a.apt_octree_pool,
                     a.apt_octree_space_only).cuda()
    model.grid_mode, model.grid_phase = a.grid_mode, a.grid_phase
    if a.loc_stride != 4:
        if (a.select_unit != "location" or a.crop_px != 8
                or a.selector not in ("oracle", "entropy", "random")):
            raise SystemExit("--loc_stride: --select_unit location --crop_px 8, "
                             "--selector oracle | entropy | random")
        model.set_loc_stride(a.loc_stride)
    if a.selector == "grid" and (a.joint or a.select_unit not in ("slot", "location")
                                 or a.res_mix or a.date_level != "none"):
        raise SystemExit("--selector grid is a content-blind baseline: no --joint, "
                         "--res_mix, --date_level, and --select_unit slot | location")
    if a.crop_free:
        if (a.selector not in ("random", "grid") or a.readout != "native_set" or a.select_unit != "slot"
                or a.joint or a.jitter or a.res_mix or a.date_level != "none"
                or a.acq_pool != "none" or a.self_teach):
            raise SystemExit("--crop_free: --selector random --readout native_set, slot unit, "
                             "no selector / date / res / jitter / acq_pool options")
        model.crop_free, model.n_crops, model.vary_crops = True, a.n_crops, a.vary_crops
        model.crop_counts = [int(v) for v in a.crop_counts.split(",")] if a.crop_counts else None
        model.shift_stride = a.shift_stride
        if a.no_seasons:
            if a.inject_layer != 0 or not model.crop_counts or min(model.crop_counts) == 0:
                raise SystemExit("--no_seasons: --inject_layer 0 and --crop_counts without 0")
            model.no_seasons = model.readout.no_base = True
    elif a.n_crops or a.vary_crops or a.no_seasons:
        raise SystemExit("--n_crops / --vary_crops / --no_seasons need --crop_free")
    # (ps8 months through the bilinear head, no crops: the plain series fine-tune, a reference)
    if (a.lr_patch == 8 or a.lr_patch == 4 and a.readout == "native_set") and not (
            a.readout == "native_set"
            and (a.crop_free or a.selector == "none")) and not (
            a.lr_patch == 8 and a.readout == "bilinear" and a.selector == "none"):
        raise SystemExit("--lr_patch 8, or 4 with native_set (ps8 / ps4 seasons): --readout "
                         "native_set with --crop_free crops or --selector none")
    if a.acq_pool != "none":
        if a.inject_layer == 0 or not (a.frac_slot or a.doy) or a.date_level != "none":
            raise SystemExit("--acq_pool: --inject_layer > 0 (the acquisitions are pooled after "
                             "blocks [0, L)), --frac_slot or --doy, no --date_level")
        model.set_acq_pool(a.acq_pool)
        model.cuda()
    if a.init_ckpt and not a.flops:         # warm start: every weight but a new joint selector
        sd = torch.load(f"checkpoints/{a.init_ckpt}.pt", map_location="cpu",
                        weights_only=False)["model"]
        # the training loss module (class weights) rides along in newer checkpoints
        sd = {k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")}
        if a.readout == "native_mean":      # a native_set checkpoint: drop its learned pool score
            sd = {k: v for k, v in sd.items() if not k.startswith("readout.score.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected or any(not k.startswith(NEW_PARAMS) for k in missing):
            raise SystemExit(f"--init_ckpt mismatch: missing {missing[:5]} unexpected "
                             f"{unexpected[:5]}")
        if model.joint_embed is not None:   # the selector's copy starts from the WARM embedding
            model.joint_embed.load_state_dict(model.mix.encoder.patch_embeddings.state_dict())
    if a.gain_loss and a.readout in SPARSE + NATIVE:
        raise SystemExit("--gain_loss reads the dense month feature grid; "
                         "cellattn / native never build it")
    if (a.score_norm or a.score_l1 or a.gain_loss) and a.selector not in LEARNED:
        raise SystemExit("--score_norm / --score_l1 regularize a learned selector")
    # --self_teach: the teacher reads the raw acquisitions where the step series is composites
    teach_raw = bool(a.self_teach) and task.name in ("pastis", "pastis_s2", "yieldsat_raw")
    dl = {s: loader(task, s, bs, a.num_workers, s == "train", a.lr_steps, a.epoch_samples,
                    dates=a.date_level != "none" or a.date_joint != "none" or a.acq_pool != "none" or (teach_raw and s == "train"),
                    teach_n=a.self_teach if teach_raw and s == "train" and a.date_level == "none" else 0, mid_months=a.doy)
          for s in ("train", "val", "test")}
    if a.sweep_only:
        model.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location="cpu",
                                         weights_only=False)["model"])
        crop_count_sweep(model, task, dl, a, name)
        return
    if a.flops:     # inference cost of this arm: pre-look + selector + crops + readout
        from exp.bench.flop_trace import append_row, count_macs, flop_row
        model.eval()
        model.joint_eps = a.eps_end         # the random share the arm is scored with
        n_samples, n_crops = 0, 0
        with torch.no_grad(), count_macs() as count:
            for i, (*x, target) in enumerate(dl["val"]):
                if i >= a.flops:
                    break
                n_crops += model(*x)[2]
                n_samples += target.shape[0]
        row = flop_row(name, a.task, a.model_size, count, n_samples,
                       len(model.backbone().blocks), round(n_crops / n_samples, 1))
        append_row(Path(f"results/{a.task}/inject_flops.csv"), row)
        print(f"FLOPS {name}: {row['gmacs']:.2f} GMACs/sample (attention "
              f"{row['gmacs_attention']:.2f}, other {row['gmacs_other']:.2f}) | "
              f"{row['tokens_per_block']:.0f} tokens/block, longest sequence "
              f"{row['longest_sequence']} | {row['crops_per_sample']} crops/sample")
        return
    better = task.meter().higher_is_better
    # one optimizer over everything; the backbone has no gradients until it is unfrozen
    if a.init_ckpt:     # warm start: fine-tune rate for the trained weights, full for the new
        new = [p for n_, p in model.named_parameters() if n_.startswith(NEW_PARAMS)]
        old = [p for n_, p in model.named_parameters() if not n_.startswith(NEW_PARAMS)]
        opt = torch.optim.AdamW([{"params": old, "lr": a.lr * UNFREEZE_LR_FACTOR}]
                                + ([{"params": new, "lr": a.lr}] if new else []))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = ReduceLROnPlateau(opt, mode="max" if better else "min", factor=SCHEDULER_FACTOR,
                              patience=SCHEDULER_PATIENCE, cooldown=SCHEDULER_COOLDOWN)
    # --no_pretrain: nothing to protect, so no frozen phase and one rate (a.lr) throughout
    freeze_epochs = (0 if a.init_ckpt or a.no_pretrain
                     else math.ceil(FREEZE_EPOCH_FRACTION * a.epochs))
    loss_fn = task.loss().cuda()
    model.task_loss_fn = loss_fn
    model.teach_event, model.teach_value = a.teach_event, a.teach_value
    model.teach_detach = a.teach_detach
    model.joint_grad, model.dps_sigma, model.dps_draws = a.joint_grad, a.dps_sigma, a.dps_draws
    model.joint_explore = a.joint_explore
    model.drop_frac, model.drop_passes = a.drop_frac, a.drop_passes
    if a.joint_grad != "value" and not (a.joint and a.readout == "native_set"
                                        and a.select_unit == "slot" and not a.joint_res):
        raise SystemExit("--joint_grad dps/both: --joint, --readout native_set, slot unit, no --joint_res")
    start, hours_before, epoch_s = 0, 0.0, None
    best, best_epoch, best_val, best_state = None, -1, None, None
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, hours_before, epoch_s = st["epoch"], st["hours"], st["epoch_s"]
        best, best_epoch, best_val, best_state = (st["best"], st["best_epoch"], st["best_val"],
                                                  st["best_state"])
        if tuple(st.get("val_excluded", ())) != task.val_excluded_events:
            raise SystemExit(f"{state_path} was scored on another val definition")
        print(f"Resumed from {state_path} after epoch {start}")
    print(f"Run: {name} | batch {bs} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"params (backbone {sum(p.numel() for p in model.backbone().parameters()) / 1e6:.2f}M)")
    t_start = time.time()

    for epoch in range(start, a.epochs):
        if a.max_hours and epoch_s and (time.time() - t_start + 1.1 * epoch_s) / 3600 > a.max_hours:
            print(f"STOPPED before epoch {epoch + 1} (--max_hours); rerun to resume")
            return
        e0 = time.time()
        torch.manual_seed(a.seed + epoch)
        tuning = epoch >= freeze_epochs
        model.backbone().requires_grad_(tuning)
        if a.joint:                         # random share: eps_start -> eps_end, then held
            span = max(1, round(a.eps_anneal * a.epochs))
            hold = round(a.eps_hold * a.epochs)     # epochs of eps_start before the anneal
            model.joint_eps = a.eps_start + (a.eps_end - a.eps_start) * min(
                1.0, max(0, epoch - hold) / span)
        if epoch == freeze_epochs and not a.init_ckpt and not a.no_pretrain:
            for g in opt.param_groups:
                g["lr"] = a.lr * UNFREEZE_LR_FACTOR
            print(f"epoch {epoch + 1}: backbone unfrozen, lr -> {a.lr * UNFREEZE_LR_FACTOR:g}")
        model.train()
        model.backbone().train(tuning)
        loss_sum, n, crops, samples = 0.0, 0, 0, 0
        for i, (*x, target) in enumerate(dl["train"]):
            if a.limit_batches and i >= a.limit_batches:
                break
            if epoch < a.warmup_epochs:     # linear warmup, per step: 0 -> a.lr
                steps = len(dl["train"])
                for g in opt.param_groups:
                    g["lr"] = a.lr * (epoch * steps + i + 1) / (a.warmup_epochs * steps)
            target = target.cuda()
            logits, _, nc = model(*x)
            task_loss = loss_fn(logits, target)
            loss = task_loss
            if a.score_l1 and model.aux_scores is not None:
                loss = loss + a.score_l1 * model.aux_scores.abs().mean()
            if a.gain_loss and model.gain_ctx is not None:
                loss = loss + a.gain_loss * gain_loss(model, task_loss)
            if model.joint is not None and model.joint_loss is not None:
                loss = loss + a.joint_weight * model.joint_loss
            if model.teach_loss is not None:    # dense self-teacher: its task loss + reconstruction
                loss = loss + model.teach_loss["task"] + a.teach_weight * model.teach_loss["recon"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.joint_target == "gate":    # the target is a gradient of the loss above
                model.joint_loss = gate_value_loss(model)
                if model.joint_loss is not None:
                    (a.joint_weight * model.joint_loss).backward()
            opt.step()
            loss_sum, n = loss_sum + loss.item(), n + 1
            crops, samples = crops + nc, samples + target.shape[0]
            if i % 100 == 0:
                extra_ = ""
                if getattr(model, "last_ps8_share", None) is not None:
                    extra_ = (f" | ps8 share: selector {model.last_ps8_share[0]:.2f}, "
                              f"all {model.last_ps8_share[1]:.2f}")
                if model.teach_loss is not None:
                    extra_ += (f" | student task {task_loss.item():.4f} teacher task "
                               f"{model.teach_loss['task'].item():.4f} recon "
                               f"{model.teach_loss['recon'].item():.4f}")
                if model.joint is not None and model.joint_loss is not None:
                    extra_ += f" | selector loss {model.joint_loss.item():.3f} eps {model.joint_eps:.2f}"
                if a.acq_pool != "none" and getattr(model, "last_acq", None):
                    extra_ += (f" | {model.last_acq[0]:.0f} acquisition tokens/sample, "
                               f"{100 * model.last_acq[1]:.1f}% composite fallback")
                    if a.date_joint != "none":
                        extra_ += f" | date-crop tokens/sample {model.last_date_tokens:.0f}"
                    if a.joint_target != "loo" and getattr(model, "gate_loo_corr", None) is not None:
                        extra_ += f" | target vs leave-one-out value r {model.gate_loo_corr:.2f}"
                print(f"  step {i}/{len(dl['train'])} loss {loss.item():.4f}{extra_}", flush=True)
        train_s = time.time() - e0
        val, _ = evaluate(model, task, dl["val"], a.limit_batches)
        sched.step(val["primary"])
        print(f"epoch {epoch + 1}/{a.epochs} | {samples / train_s:.1f} samples/s "
              f"({train_s / 60:.1f} min) | {crops / max(samples, 1):.0f} crops/sample | "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB\n"
              f"   train_loss {loss_sum / max(n, 1):.4f} | val {_brief(val)}", flush=True)
        if best is None or (val["primary"] > best if better else val["primary"] < best):
            best, best_epoch, best_val = val["primary"], epoch + 1, val
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        epoch_s = time.time() - e0
        tmp = state_path.with_suffix(".tmp")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch + 1,
                    "hours": hours_before + (time.time() - t_start) / 3600, "epoch_s": epoch_s,
                    "best": best, "best_epoch": best_epoch, "best_val": best_val,
                    "best_state": best_state,
                    "val_excluded": list(task.val_excluded_events)}, tmp)
        tmp.replace(state_path)

    if best_state is None:          # no epoch was ever scored (e.g. resumed with nothing left)
        best_val, _ = evaluate(model, task, dl["val"], a.limit_batches)
        best, best_epoch = best_val["primary"], a.epochs
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    if a.joint:
        model.joint_eps = a.eps_end         # scored as trained at the end (model selection too)
    if a.vary_crops:                        # the main row: the full budget
        model.n_eval = a.n_crops or model.k
    test, test_crops = evaluate(model, task, dl["test"], a.limit_batches)
    # test confusion matrix (classes x classes, rows = truth): the benchmark's 5-fold number is
    # the mIoU of the SUM of the folds' matrices, not the mean of their mIoUs
    test_conf = getattr(evaluate.last_meter, "conf", None)
    if test_conf is not None and not a.limit_batches:
        os.makedirs(f"results/{a.task}/conf", exist_ok=True)
        torch.save(test_conf.cpu(), f"results/{a.task}/conf/{name}.pt")
    if a.vary_crops:        # one model, every crop count: accuracy + measured compute
        crop_count_sweep(model, task, dl, a, name)
    joint_extra = {}
    if a.joint:                             # the same weights with pure-selector / random picks
        for tag, eps in (("selector_only", 0.0), ("random_picks", 1.0)):
            model.joint_eps = eps
            r, _ = evaluate(model, task, dl["test"], a.limit_batches)
            joint_extra[f"test_primary_{tag}"] = round(r["primary"], 5)
            print(f"TEST picks {tag} (eps {eps:g}): {_brief(r)}")
        model.joint_eps = a.eps_end
        if model.unit == "slot" and not model.joint_pairs and not model.joint_res \
                and model.res_mix is None and a.date_joint == "none":
            model.force_grid = True         # ... and with the regular-lattice picks
            r, _ = evaluate(model, task, dl["test"], a.limit_batches)
            joint_extra["test_primary_grid_picks"] = round(r["primary"], 5)
            print(f"TEST picks grid_picks ({model.grid_mode} lattice): {_brief(r)}")
            model.force_grid = False
    hours = hours_before + (time.time() - t_start) / 3600
    reg = ", ".join((["temporal self-attn"] if a.sel_temporal else [])
                    + (["normalized scores"] if a.score_norm else [])
                    + ([f"L1 {a.score_l1:g} on scores"] if a.score_l1 else [])
                    + ([f"gain loss {a.gain_loss:g}"] if a.gain_loss else []))
    readout = {"bilinear": "", "splat": ", splat readout", "splat_t": ", space-time splat readout",
               "splat_tg": ", guided space-time splat readout",
               "cellattn": ", sparse per-cell attention readout",
               "native": ", native-res pa2px head (ps16 16x16 px, crops area-down to 4x4)",
               "native_attn": ", native-res pa2px head + temporal attn pool in logit space",
               "cellattn_overlap": ", overlap-only per-cell attention readout",
               "splat_t_overlap": ", overlap-only time-only splat readout",
               "native_set": ", native-res pa2px head + per-cell attn pool over overlapping tokens",
               "native_mean": ", native-res pa2px head + parameter-free softmax mean over "
                              "overlapping tokens",
               "tsplat": ", overlap-only time-continuous Gaussian splat readout (per step)",
               "tquery": ", overlap-only time-queried attention readout (per step)",
               "native_tsplat": ", native-res head + overlap-only time Gaussian splat, time mean",
               "native_tquery": ", native-res head + overlap-only time-queried attention, time mean",
               }[a.readout]
    readout += ", fractional crop slots" if a.frac_slot else ""
    if a.joint:
        readout += (f", joint value-predictor picks (random share {a.eps_start:g} -> "
                    f"{a.eps_end:g})" + (", choosing ps4 or ps8" if a.joint_res else ""))
    if a.pos != "corner":
        readout += {"center": ", crops positioned at their centre",
                    "box": ", + bottom-right corner encoding (zero-init)"}[a.pos]
    if a.doy:
        readout += ", + day-of-year encoding (zero-init)"
    if a.head_attn != "alone":
        readout += {"oneway": ", crops read the lr tokens in blocks [0, L)",
                    "overlap": ", crops read their lr cell in blocks [0, L)"}[a.head_attn]
    if a.attn != "joint":
        readout += {"oneway": ", one-way attention (crops read lr, lr never reads crops)",
                    "overlap": ", one-way attention to the overlapping lr cell only"}[a.attn]
    if a.init_ckpt:
        readout += ", warm start"
    if res_mix:
        readout += (", token budget at ps16/ps8/ps4 = " + "/".join(f"{v:g}" for v in res_mix)
                    + " (random single-month crops)")
    if a.date_level != "none":
        readout += (f", + {a.date_per_month:g} single-date ps4 crops per month pick "
                    + ("inside picked month-cells" if a.date_level == "nested" else "anywhere")
                    + (" (months only route, no month crops)" if a.no_month_crops else "")
                    + (" @ learned date selector" if a.date_selector == "xattn" else " @ random")
                    + (", date crops ungated" if a.date_gate == "none" else ""))
    if task.reduction.startswith("pair_"):         # series: dates, months, pairs
        lr_desc = (f"all dates ps{a.lr_patch}"
                   + (f" per {a.lr_tile} px tile" if a.lr_tile != 64 else "")
                   if a.lr_steps == "all"
                   else f"months ps{a.lr_patch}")
        crop_desc = ("no crops" if a.selector == "none" else
                     f"{a.k_frac:.0%} of date-cells as ps4 pair crops inject@L{a.inject_layer} "
                     f"@ {a.selector}" + (f" ({reg})" if reg else ""))
        arm = f"{lr_desc} + {crop_desc}, {task.reduction} head, full FT{readout}"
    else:
        arm = arm_label(a.selector, model.k, a.inject_layer, *model.GT, a.select_unit, a.pick,
                        a.head, reg, a.crop_px, a.epochs > 0) + readout
        if a.lr_patch != 16:                    # ps8 / ps4 seasons
            arm = arm.replace("ps16", f"ps{a.lr_patch}", 1)
        if a.no_seasons:
            arm = "NO season tokens (crops only): " + arm
    print(f"TEST {arm} (best val epoch {best_epoch}, val {best:.4f}): {_brief(test)} | "
          f"{test_crops:.0f} crops/sample")
    if not a.limit_batches:
        torch.save({"args": vars(a), "model": best_state}, f"checkpoints/{name}.pt")
    _append(Path(f"results/{a.task}/inject_fft{smoke}.csv"), {
        "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "arm": arm, "task": a.task,
        "model_size": a.model_size, "selector": a.selector, "inject_layer": a.inject_layer,
        "k_frac": a.k_frac if a.selector != "none" else 0, "select_unit": a.select_unit,
        "pick": a.pick, "head": a.head, "score_norm": a.score_norm, "score_l1": a.score_l1,
        "sel_temporal": a.sel_temporal, "gain_loss": a.gain_loss, "readout": a.readout,
        "frac_slot": a.frac_slot, "date_level": a.date_level,
        "date_per_month": a.date_per_month if a.date_level != "none" else 0,
        "month_crops": not a.no_month_crops, "date_selector": a.date_selector,
        "acq_pool": a.acq_pool, "grid_mode": a.grid_mode if a.selector == "grid" else "",
        "joint_explore": a.joint_explore if a.joint else "",
        "date_gate": a.date_gate, "res_mix": a.res_mix or "", "pos": a.pos, "doy": a.doy, "attn": a.attn, "head_attn": a.head_attn, "joint_feat": a.joint_feat if a.joint else "", "joint_target": a.joint_target if a.joint else "", "joint_in": a.joint_in if a.joint else "", "joint_arch": a.joint_arch if a.joint else "", "joint_embed": a.joint_embed if a.joint else "", "jitter": int(a.jitter), "drop_frac": a.drop_frac if a.joint and a.joint_target == "drop" else "", "drop_passes": a.drop_passes if a.joint and a.joint_target == "drop" else "", "crop_px": a.crop_px, "crop_free": int(a.crop_free), "n_crops": a.n_crops, "vary_crops": int(a.vary_crops), "date_joint": a.date_joint if a.joint else "", "joint_grad": a.joint_grad if a.joint else "", "dps_sigma": a.dps_sigma if a.joint and a.joint_grad != "value" else "",
        "joint": a.joint, "joint_res": a.joint_res, "eps_end": a.eps_end if a.joint else "",
        "init_ckpt": a.init_ckpt,
        "apt_pool": int(a.apt_pool) if a.selector == "apt" else "",
        "apt_merge_mse": a.apt_merge_mse if a.selector == "apt" and a.apt_pool else "",
        "apt_time": a.apt_time if a.selector == "apt" else "",
        "apt_res": int(a.apt_res) if a.selector == "apt" else "",
        **joint_extra,
        "lr_steps": a.lr_steps, "lr_patch": a.lr_patch, "lr_tile": a.lr_tile, "pair_match_random": a.pair_match_random, "self_teach": a.self_teach, "teach_event": a.teach_event, "teach_detach": a.teach_detach if a.self_teach else "", "teach_value": a.teach_value if a.self_teach else "",
        "teach_weight": a.teach_weight if a.self_teach else "", "eps_hold": a.eps_hold if a.joint else "",
        "epoch_samples": a.epoch_samples, "seed": a.seed,
        "pretrained": int(not a.no_pretrain), "warmup_epochs": a.warmup_epochs, "fold": a.fold,
        "lr": a.lr, "epochs": a.epochs, "batch_size": bs, "best_epoch": best_epoch,
        "val_excluded": " ".join(task.val_excluded_events), "hours": round(hours, 2),
        "test_crops_per_sample": round(test_crops, 1),
        **{f"val_{k}": round(v, 5) for k, v in best_val.items()},
        **{f"test_{k}": round(v, 5) for k, v in test.items()}})
    torch.save({"done": True}, state_path)


if __name__ == "__main__":
    main()
