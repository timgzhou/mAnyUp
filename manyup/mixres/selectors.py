"""Selectors = Scorer + Allocator over a fixed set of candidate windows (Geometry).

    sel  = Selector(Geometry(crop_px=8, loc_stride=8), LabelLoss(), SeasonHalves(n=8, keep="better"))
    pick = sel(Context(model, ms, ss, season, present, target))      # -> Selection(plan, logw, aux)

A Scorer reads the context and returns Scores on the 16 x 16 ps4 cell grid (-inf: not a
candidate window). An Allocator turns them into a (B, T, 16, 16) crop mask within a crop budget.
Anything a scorer reads of a model's prediction comes from Context.out0, the uniform pass of the
model being trained, unless a different model is passed to it explicitly (Entropy(source=...)).
Wrong compositions raise; nothing falls back silently.

Ported from exp.upsamplers.finetune_inject (InjectFT._season_mask / _season_pick /
_prior_scores / _window_scores / lr_entropy_scores / _residual_loc and the location path of
forward); exp.mixres.check_selectors replays the legacy reference picks.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

from manyup.encoders.olmoearth import _to_device
from manyup.mixres.model import LR_PS, MixedResModel, Out
from manyup.mixres.plan import LOOKUP_PS, TILE, Plan
from manyup.mixres.residual import ResidualScorer, SeasonResidual, listnet_loss, window_pixels
from manyup.mixres.sign import SignNet, season_spearman
from manyup.timAnyUp import topk_mask

H = TILE // LOOKUP_PS       # the 16 x 16 cell grid every score map lives on
NEG = float("-inf")


def location_grid_scores(B: int, H: int, n: int, crop_px: int, random_phase: bool,
                         device, offset: int = 0) -> torch.Tensor:
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
    on = ((i + c * j)[None] + phase + offset) % m == 0                         # (B, S, S)
    scores = torch.rand(B, H, H, device=device) * 1e-3
    cy = (torch.arange(S, device=device) * crop_px + crop_px // 2) // (TILE // H)
    scores[:, cy[:, None], cy[None, :]] += on.float()
    return scores


# -- candidates ---------------------------------------------------------------------------
class Geometry:
    """The candidate crop windows: crop_px windows, either centred on every ps4 cell (clamped
    into the tile) or, with loc_stride, a lattice of windows spread evenly over the tile, each
    represented by the ps4 cell under its top-left corner (InjectFT.set_loc_stride)."""

    def __init__(self, crop_px: int, loc_stride: int | None = None):
        self.crop_px, self.loc_px = crop_px, None
        if loc_stride is None:
            return
        if loc_stride < crop_px:
            raise ValueError(f"loc_stride {loc_stride} < crop_px {crop_px}: not ported")
        n = (TILE - crop_px) // loc_stride + 1
        tl = sorted({round(i * (TILE - crop_px) / (n - 1)) for i in range(n)})
        cells = [v // LOOKUP_PS for v in tl]
        if len(set(cells)) != len(cells):
            raise ValueError(f"loc_stride {loc_stride}: two windows share an anchor cell")
        self.loc_px = torch.full((H,), -1, dtype=torch.long)
        self.loc_px[cells] = torch.tensor(tl)

    def window_mean(self, px: torch.Tensor) -> torch.Tensor:
        """(N, 64, 64) per-pixel values -> (N, 16, 16): each candidate's mean over the window a
        pick of it would cut (-inf: not a candidate)."""
        P_ = self.crop_px
        if P_ == LOOKUP_PS:
            return F.avg_pool2d(px[:, None], LOOKUP_PS)[:, 0]
        win = F.avg_pool2d(px[:, None], P_, stride=1)[:, 0]
        if self.loc_px is not None:
            lut = self.loc_px.to(px.device)
            c = (lut >= 0).nonzero(as_tuple=True)[0]
            out = px.new_full((px.shape[0], H, H), NEG)
            out[:, c[:, None], c[None, :]] = win[:, lut[c]][:, :, lut[c]]
            return out
        tl = (torch.arange(H, device=px.device) * LOOKUP_PS + LOOKUP_PS // 2
              - P_ // 2).clamp(0, TILE - P_)
        return win[:, tl][:, :, tl]

    def anchors(self, device) -> torch.Tensor | None:
        """(16, 16) bool candidate cells, None when every cell is one."""
        if self.loc_px is None:
            return None
        a = self.loc_px.to(device) >= 0
        return a[:, None] & a[None, :]

    def tiling(self) -> torch.Tensor:
        """Anchor cells of the non-overlapping windows that tile the 64 px tile (the residual's
        candidates); requires every such window to be a candidate."""
        cell = torch.arange(TILE // self.crop_px) * self.crop_px // LOOKUP_PS
        if self.loc_px is not None and not bool((self.loc_px[cell] >= 0).all()):
            raise ValueError("the residual needs candidates that tile the tile (stride = crop)")
        return cell

    def plan(self, mask: torch.Tensor) -> Plan:
        return Plan.from_mask(mask, self.crop_px, self.loc_px)


# -- context ------------------------------------------------------------------------------
class Context:
    """One batch as a selector sees it. out0 is the uniform pass (season tokens only) of the
    model being trained, computed once on first use; target only exists while training or for
    a label oracle."""

    def __init__(self, model: MixedResModel, ms, ss, season, present, target=None, dates=None):
        dev = model.device
        self.dates = dates
        self.model = model
        self.ms, self.ss = _to_device(ms, dev, torch.bfloat16), _to_device(ss, dev, torch.bfloat16)
        self.season, self.present = season.to(dev), present.to(dev)
        self.target = None if target is None else target.to(dev)
        self.B, self.T = self.present.shape
        self.NS = self.ss.timestamps.shape[1]
        self._out0 = None

    @classmethod
    def from_batch(cls, model: MixedResModel, batch) -> "Context":
        """manyup.mixres.data.Batch -> the context (the label included: label scorers read it)."""
        return cls(model, batch.ms, batch.ss, batch.season, batch.present, batch.target,
                   batch.dates)

    @property
    def out0(self) -> Out:
        if self._out0 is None:
            self._out0 = self.model.uniform(self.ms, self.ss, self.season, self.present)
        return self._out0

    @property
    def seasons(self):
        """The uniform pass's embedded season tokens if it ran, for reuse at inference."""
        return None if self._out0 is None else self._out0.seasons

    @property
    def valid(self) -> torch.Tensor:
        """(B, T) month is present and in a season."""
        return self.present & (self.season >= 0)

    @property
    def used(self) -> torch.Tensor:
        """(B, NS) season slot holds a present month."""
        oh = F.one_hot(self.season.clamp(min=0), self.NS).bool() & self.valid[..., None]
        return oh.any(1)


def season_maps(model: MixedResModel, lr_F: torch.Tensor) -> torch.Tensor:
    """(B, NS, h, w, D) season tokens -> (B NS, C, 64, 64): each season token's own head map
    (ps8 tokens area-pooled to 8 px, as the decoder does). Precision follows the caller's
    autocast state."""
    lin = model.head.linear
    NS, gh, gw = lr_F.shape[1:4]
    C = lin.out_features // LR_PS ** 2
    m = lin(lr_F).view(-1, C, LR_PS, LR_PS)
    if model.cfg.lr_patch < LR_PS:
        m = F.avg_pool2d(m, LR_PS // model.cfg.lr_patch)
    return rearrange(m, "(b s gh gw) c i j -> (b s) c (gh i) (gw j)", s=NS, gh=gh, gw=gw)


def pixel_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(1)
    return -(p * p.clamp(min=1e-8).log()).sum(1)                                # (N, 64, 64)


def pixel_label_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N, C, 64, 64) vs the label -> (N, 64, 64); 0 where unlabelled."""
    if logits.shape[1] > 1:
        return F.cross_entropy(logits, target.long(), ignore_index=-1, reduction="none")
    y = target[:, 0] if target.ndim == 4 else target           # YieldSAT: (value, field id)
    return ((logits[:, 0] - y.float()) ** 2).nan_to_num(0.0)


# -- scores -------------------------------------------------------------------------------
@dataclass
class Scores:
    """On the 16 x 16 cell grid, -inf where not a candidate.
    window   (B, 16, 16)       where to crop (higher first)
    season   (B, NS, 16, 16)   per (season, window) pair, for the season allocators
    gate     (B, 16, 16)       a crop's decoder log-weight is logsigmoid(gate[b, y, x])
    season_gate (B, NS, 16, 16) ... or the log-weight itself at its (season, cell)
    aux      scalar loss to add while training (e.g. the residual's ranking loss)
    pooled, lr_F               the prediction and season tokens the scores came from"""
    window: torch.Tensor | None = None
    season: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    season_gate: torch.Tensor | None = None
    aux: torch.Tensor | None = None
    pooled: torch.Tensor | None = None
    lr_F: torch.Tensor | None = None


class Scorer:
    def __call__(self, ctx: Context, geo: Geometry, need=("window", "season")) -> Scores:
        """need: the scores the allocator reads; a scorer may skip computing the others."""
        raise NotImplementedError


class LabelLoss(Scorer):
    """The label oracle: the uniform pass's loss against the label, per window (pooled
    prediction) and per (season, window) (each season token's own map)."""

    def __call__(self, ctx, geo, need=("window", "season")):
        if ctx.target is None:
            raise ValueError("LabelLoss: no target in the context")
        out0 = ctx.out0
        sc = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if "season" in need:
                s_log = season_maps(ctx.model, out0.lr_F)
                sc = geo.window_mean(pixel_label_loss(
                    s_log, ctx.target.repeat_interleave(ctx.NS, 0))).view(ctx.B, ctx.NS, H, H)
            win = geo.window_mean(pixel_label_loss(out0.logits, ctx.target))
        return Scores(win, sc, pooled=out0.logits, lr_F=out0.lr_F)


class Entropy(Scorer):
    """Label-free: the prediction entropy, mean over each window (entropy first, then the
    window mean). source None: the uniform pass of the model being trained (bfloat16, as
    InjectFT --score_src entropy). source a MixedResModel: that FROZEN model's uniform pass,
    decoded in float32 (as the legacy --score_src lr_entropy --scorer)."""

    def __init__(self, source: MixedResModel | None = None):
        self.source = source

    def __call__(self, ctx, geo, need=("window", "season")):
        if ctx.model.task.kind != "classification":
            raise ValueError(f"Entropy: {ctx.model.task.name} is not a classification task")
        if self.source is None:
            out0 = ctx.out0
            sc = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if "season" in need:
                    s_log = season_maps(ctx.model, out0.lr_F).detach()
                    sc = geo.window_mean(pixel_entropy(s_log)).view(ctx.B, ctx.NS, H, H)
                win = geo.window_mean(pixel_entropy(out0.logits.detach()))
            return Scores(win, sc, pooled=out0.logits.detach(), lr_F=out0.lr_F.detach())
        src = self.source
        with torch.no_grad():
            lr_F = src.uniform(ctx.ms, ctx.ss, ctx.season, ctx.present).lr_F
            z = torch.zeros(0, dtype=torch.long, device=lr_F.device)
            pooled = src.readout(src.head.linear, lr_F, ctx.season, ctx.present,
                                 lr_F.new_zeros(0, lr_F.shape[-1]), (z,) * 4).float()
            sc = None
            if "season" in need:
                mu = season_maps(src, lr_F.to(src.head.linear.weight.dtype)).float()
                sc = geo.window_mean(pixel_entropy(mu)).view(ctx.B, ctx.NS, H, H)
        win = geo.window_mean(pixel_entropy(pooled))
        return Scores(win, sc, pooled=pooled, lr_F=lr_F)


class Residual(Scorer):
    """window score = z(base window entropy) + r, r a learned ResidualScorer (last layer zero:
    exactly the base at the start), on the windows that tile the tile.
        train  "e2e"     r learns from the task loss through the crop gate logsigmoid(score),
                         Gumbel(tau) exploration on the picks while training
               "oracle"  r learns from a ListNet ranking to the base pass's own window loss
               "both"    both; the oracle weight `lam` is set from outside (decayed)
    Ported from InjectFT._residual_loc."""

    def __init__(self, base: Entropy, net, train: str = "e2e", tau: float = 0.5,
                 lam: float = 1.0):
        if train not in ("e2e", "oracle", "both"):
            raise ValueError(f"Residual train {train!r}")
        self.base, self.net, self.train, self.tau, self.lam = base, net, train, tau, lam

    def __call__(self, ctx, geo, need=("window", "season")):
        b = self.base(ctx, geo, need)
        pooled, lr_F = b.pooled, b.lr_F
        P_ = geo.crop_px
        S = TILE // P_
        B, NS = lr_F.shape[:2]
        with torch.no_grad():
            lp = pooled.float().log_softmax(1)
            p = lp.exp()
            ent = F.avg_pool2d((-(p * lp).sum(1))[:, None], P_)                    # (B,1,S,S)
            top2 = p.topk(2, dim=1).values
            margin = F.avg_pool2d((top2[:, :1] - top2[:, 1:]), P_)
            pbar = F.avg_pool2d(p, P_)
            js = -(pbar * pbar.clamp(min=1e-8).log()).sum(1, keepdim=True) - ent
            feats = torch.cat([pbar, ent, margin, js], 1)
            used = ctx.used
            w = used.float() / used.float().sum(1, keepdim=True).clamp(min=1)
            tok = torch.einsum("bs,bshwd->bhwd", w, lr_F.float())
            tok = F.interpolate(tok.permute(0, 3, 1, 2), size=(S, S), mode="nearest")
            e = ent[:, 0].flatten(1)
            z = (e - e.mean(1, keepdim=True)) / e.std(1, keepdim=True).clamp(min=1e-6)
        hr_in = {}
        if self.net.arch != "conv":
            hr_in = dict(pix=window_pixels(ctx.ms, P_), mon=ctx.present, lr_F=lr_F, used=used)
        score = z.view(B, S, S) + self.net(feats, tok.permute(0, 2, 3, 1), **hr_in)
        aux = None
        if self.net.training and self.train in ("oracle", "both") and self.lam > 0:
            if ctx.target is None:
                raise ValueError("Residual oracle loss: no target in the context")
            px = pixel_label_loss(pooled.float(), ctx.target)
            lab = ctx.target[:, 0] if ctx.target.ndim == 4 else ctx.target
            lab = (torch.isfinite(lab.float()) & (lab >= 0)).float()[:, None]
            nl = F.avg_pool2d(lab, P_)
            wl = F.avg_pool2d(px[:, None] * lab, P_) / nl.clamp(min=1e-6)
            aux = self.lam * listnet_loss(score.flatten(1), wl.flatten(1), (nl > 0).flatten(1))
        cell = geo.tiling().to(score.device)
        out = score.new_full((B, H, H), NEG)
        out[:, cell[:, None], cell[None, :]] = score
        if self.train == "oracle":
            return Scores(out, b.season, aux=aux, pooled=pooled, lr_F=lr_F)
        pick = out.detach()
        if self.net.training and self.tau > 0:
            g = -torch.log(-torch.log(torch.rand_like(score).clamp(1e-9, 1 - 1e-9)))
            pick = pick.clone()
            pick[:, cell[:, None], cell[None, :]] = (score + self.tau * g).detach()
        return Scores(pick, b.season, gate=out, aux=aux, pooled=pooled, lr_F=lr_F)


class EntropySign(Scorer):
    """Season entropy with a learned per-location flip (manyup.mixres.sign): each location's
    seasons are scored by their window entropy, negated where the SignNet predicts the ranking
    is inverted (p(flip) > 1/2), so the allocator's lowest-score pick becomes the
    highest-entropy one there. While training, the net learns from the sign of each location's
    Spearman r between season entropy and season loss (BCE; the label is read only for this).
    Candidates must tile the tile (stride = crop), as for the residual."""

    def __init__(self, net: SignNet):
        self.net = net
        self.last_flip = None

    def __call__(self, ctx, geo, need=("season",)):
        if ctx.model.task.kind != "classification":
            raise ValueError(f"EntropySign: {ctx.model.task.name} is not a classification task")
        base = Entropy()(ctx, geo, ("window", "season"))
        cell = geo.tiling().to(base.season.device)
        E = base.season[:, :, cell[:, None], cell[None, :]]                      # (B, NS, S, S)
        with torch.no_grad():
            pix = window_pixels(ctx.ms, geo.crop_px)
        logit = self.net(E, ctx.used, base.lr_F, pix, ctx.present)               # (B, S, S)
        flip = logit.detach() > 0
        aux = None
        if self.net.training:
            if ctx.target is None:
                raise ValueError("EntropySign training: no target in the context")
            L = LabelLoss()(ctx, geo, ("season",)).season[:, :, cell[:, None], cell[None, :]]
            r = season_spearman(E, L, ctx.used)
            ok = torch.isfinite(r) & (r != 0)
            if ok.any():
                aux = F.binary_cross_entropy_with_logits(logit[ok], (r[ok] < 0).float())
        self.last_flip = float(flip.float().mean())
        f = torch.zeros(ctx.B, H, H, dtype=torch.bool, device=flip.device)
        f[:, cell[:, None], cell[None, :]] = flip
        sc = torch.where(f[:, None], -base.season, base.season)
        sc = sc.masked_fill(~torch.isfinite(base.season), NEG)
        return Scores(base.window, sc, aux=aux, pooled=base.pooled, lr_F=base.lr_F)


class SeasonResidualScorer(Scorer):
    """Time selection: season score = z(the season's window entropy over the location's used
    seasons) + r, r a learned SeasonResidual per (season, location) (last layer zero: exactly
    the lowest-entropy ranking at the start); the allocator keeps each location's lowest.
        train  "e2e"     r learns from the task loss through each picked crop's decoder weight
                         logsigmoid(-r) - logsigmoid(0) (0 at the start: the decoder too starts
                         as the plain entropy arm); Gumbel(tau) exploration on the picks
               "oracle"  r learns from a ListNet ranking of each location's seasons by their
                         loss (the lowest first), no gate
               "both"    both; the oracle weight `lam` is set from outside (decayed)
    Candidates must tile the tile (stride = crop)."""

    def __init__(self, net: SeasonResidual, train: str = "e2e", tau: float = 0.5,
                 lam: float = 1.0):
        if train not in ("e2e", "oracle", "both"):
            raise ValueError(f"SeasonResidualScorer train {train!r}")
        self.net, self.train, self.tau, self.lam = net, train, tau, lam

    def __call__(self, ctx, geo, need=("season",)):
        if ctx.model.task.kind != "classification":
            raise ValueError(f"SeasonResidual: {ctx.model.task.name} is not a classification task")
        out0, P_ = ctx.out0, geo.crop_px
        cell = geo.tiling().to(ctx.present.device)
        S, B, NS = len(cell), ctx.B, ctx.NS
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_log = season_maps(ctx.model, out0.lr_F).detach()
                E = geo.window_mean(pixel_entropy(s_log))                       # as Entropy
            E = E.view(B, NS, H, H)[:, :, cell[:, None], cell[None, :]].float()   # (B,NS,S,S)
            lp = s_log.float().log_softmax(1)
            p = lp.exp()
            ent = F.avg_pool2d((-(p * lp).sum(1))[:, None], P_)
            top2 = p.topk(2, dim=1).values
            margin = F.avg_pool2d(top2[:, :1] - top2[:, 1:], P_)
            pbar = F.avg_pool2d(p, P_)
            js = -(pbar * pbar.clamp(min=1e-8).log()).sum(1, keepdim=True) - ent
            feats = torch.cat([pbar, ent, margin, js], 1).view(B, NS, -1, S, S)
            u = ctx.used[:, :, None, None].float()
            n = u.sum(1, keepdim=True).clamp(min=1)
            mu = (E * u).sum(1, keepdim=True) / n
            sd = (((E - mu) ** 2) * u).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-6)
            z = (E - mu) / sd
        hr_in = {}
        if self.net.arch != "conv":
            hr_in = dict(pix=window_pixels(ctx.ms, P_), season=ctx.season, valid=ctx.valid)
        r = self.net(feats, out0.lr_F.detach(), ctx.used, **hr_in)               # (B,NS,S,S)
        score = z + r
        aux = None
        if self.net.training and self.train in ("oracle", "both") and self.lam > 0:
            if ctx.target is None:
                raise ValueError("SeasonResidual oracle loss: no target in the context")
            L = LabelLoss()(ctx, geo, ("season",)).season[:, :, cell[:, None], cell[None, :]]
            ok = ctx.used[:, :, None, None].expand_as(L) & torch.isfinite(L)
            aux = self.lam * listnet_loss((-score).permute(0, 2, 3, 1).flatten(0, 2),
                                          (-L).masked_fill(~ok, 0).permute(0, 2, 3, 1).flatten(0, 2),
                                          ok.permute(0, 2, 3, 1).flatten(0, 2))
        pick = score.detach()
        if self.net.training and self.train != "oracle" and self.tau > 0:
            g = -torch.log(-torch.log(torch.rand_like(pick).clamp(1e-9, 1 - 1e-9)))
            pick = pick - self.tau * g                                      # lowest is picked
        sc = pick.new_full((B, NS, H, H), NEG)
        sc[:, :, cell[:, None], cell[None, :]] = pick
        gate = None
        if self.train != "oracle":
            gate = r.new_zeros((B, NS, H, H))
            gate[:, :, cell[:, None], cell[None, :]] = (F.logsigmoid(-r)
                                                          - F.logsigmoid(r.new_zeros(())))
        return Scores(None, sc, season_gate=gate, aux=aux, pooled=out0.logits, lr_F=out0.lr_F)


class RandomWindows(Scorer):
    """Content-blind: a uniform score per (month, window), a window's score its best present
    month's (the legacy --selector random --select_unit location)."""

    def __call__(self, ctx, geo, need=("window",)):
        s = torch.rand(ctx.B, ctx.T, H, H, device=ctx.present.device)
        a = geo.anchors(s.device)
        if a is not None:
            s = s.masked_fill(~a, NEG)
        where = ctx.present[:, :, None, None].expand_as(s)
        return Scores(window=s.masked_fill(~where, NEG).amax(1))


class GridWindows(Scorer):
    """Content-blind: a regular lattice of n crop windows (location_grid_scores; random phase
    while training, fixed at test), each window scored by it for every present month (the legacy
    --selector grid --select_unit location). The lattice marks the cell under each window's
    centre, so the candidates must be centred windows (no loc_stride)."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, ctx, geo, need=("window",)):
        if geo.loc_px is not None:
            raise ValueError("GridWindows: the lattice is of centred windows; set no loc_stride")
        g = location_grid_scores(ctx.B, H, max(1, self.n), geo.crop_px, ctx.model.training,
                                 ctx.present.device)
        s = g[:, None].expand(ctx.B, ctx.T, H, H)
        where = ctx.present[:, :, None, None].expand_as(s)
        return Scores(window=s.masked_fill(~where, NEG).amax(1))


class SeasonGrid(Scorer):
    """Content-blind: in every season a lattice of n windows, shifted one coset per season
    (random phase while training, fixed at test)."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, ctx, geo, need=("season",)):
        rp = ctx.model.training
        dev = ctx.present.device
        sc = torch.stack([location_grid_scores(ctx.B, H, self.n, geo.crop_px, rp, dev, offset=s)
                          for s in range(ctx.NS)], 1)
        return Scores(season=sc)


class NoScores(Scorer):
    """For allocators that read no scores (MonthGrid)."""

    def __call__(self, ctx, geo, need=()):
        if need:
            raise ValueError(f"NoScores: the allocator needs {need} scores")
        return Scores()


class SeasonRandom(Scorer):
    """Content-blind: a uniform score per (season, window) pair."""

    def __call__(self, ctx, geo, need=("season",)):
        return Scores(season=torch.rand(ctx.B, ctx.NS, H, H, device=ctx.present.device))


# -- allocation ---------------------------------------------------------------------------
class Allocator:
    needs: tuple = ("window", "season")     # the scores it reads

    def __call__(self, s: Scores, ctx: Context, geo: Geometry) -> torch.Tensor:
        """-> (B, T, 16, 16) bool crop mask."""
        raise NotImplementedError


class TopWindows(Allocator):
    """The n best windows, every present month of each."""
    needs = ("window",)

    def __init__(self, n: int):
        self.n = n

    def __call__(self, s, ctx, geo):
        if s.window is None:
            raise ValueError("TopWindows needs window scores")
        where = ctx.present[:, :, None, None].expand(ctx.B, ctx.T, H, H)
        return topk_mask(s.window[:, None], max(1, self.n)) & where


def _take(score, cost, budget):
    """flat (B, N) scores / costs -> the best-first prefix within budget (B,), (B, N) bool."""
    order = score.argsort(1, descending=True)
    fin = torch.isfinite(score.gather(1, order))
    spent = (cost.gather(1, order) * fin).cumsum(1)
    keep = fin & (spent <= budget[:, None])
    return torch.zeros_like(keep).scatter(1, order, keep)


class _SeasonBudget(Allocator):
    """Picks (season, window) pairs within the location oracle's budget at n: n windows x the
    sample's present months; a pair costs its season's present months."""

    def __init__(self, n: int):
        self.n = n

    def _setup(self, s, ctx):
        if s.season is None:
            raise ValueError(f"{type(self).__name__} needs per-season scores")
        NS = s.season.shape[1]
        valid, s_idx = ctx.valid, ctx.season.clamp(min=0)
        used = (F.one_hot(s_idx, NS).bool() & valid[..., None]).any(1)
        sc = s.season.masked_fill(~used[:, :, None, None], NEG)
        budget = self.n * valid.sum(1)
        months = (F.one_hot(s_idx, NS) & valid[..., None]).sum(1)
        cost = months[:, :, None, None].expand(ctx.B, NS, H, H)
        return sc, used, budget, cost, s_idx, valid

    @staticmethod
    def _to_months(pair, s_idx, valid):
        B = pair.shape[0]
        mask = pair[torch.arange(B, device=pair.device)[:, None], s_idx]
        return mask & valid[:, :, None, None]


class SeasonPairs(_SeasonBudget):
    """The best (season, window) pairs, every present month of each picked season."""
    needs = ("season",)

    def __call__(self, s, ctx, geo):
        sc, used, budget, cost, s_idx, valid = self._setup(s, ctx)
        B, NS = sc.shape[:2]
        pair = _take(sc.flatten(1), cost.flatten(1), budget).view(B, NS, H, H)
        return self._to_months(pair, s_idx, valid)


class SeasonTop(_SeasonBudget):
    """In every used season its own n best windows by the season scores, every present month
    of that season in each: the season grid's allocation (n windows per season, the same
    budget of n windows x present months) with the windows chosen by score. Seasons may pick
    the same window."""
    needs = ("season",)

    def __call__(self, s, ctx, geo):
        sc, used, budget, cost, s_idx, valid = self._setup(s, ctx)
        B, NS = sc.shape[:2]
        flat = sc.flatten(2)                                                    # (B, NS, 256)
        top = flat.topk(min(self.n, flat.shape[2]), dim=2).indices
        pick = torch.zeros_like(flat, dtype=torch.bool).scatter(2, top, True)
        pair = (pick & torch.isfinite(flat)).view(B, NS, H, H) & used[:, :, None, None]
        return self._to_months(pair, s_idx, valid)


class EveryLocation(Allocator):
    """Time selection with space fixed: EVERY candidate location, each with its lowest-score
    seasons (order "low"; "high": its highest-score seasons) -- keep "half": half its used
    seasons, at least one (the better- / worse-half oracles' rule), "one": a single season --
    every present month of those seasons. Ties broken as SeasonHalves "better" / "worse", so
    with LabelLoss scores this is the better- / worse-half oracle at every location."""
    needs = ("season",)

    def __init__(self, keep: str = "half", order: str = "low"):
        if keep not in ("half", "one"):
            raise ValueError(f"EveryLocation keep {keep!r}")
        if order not in ("low", "high"):
            raise ValueError(f"EveryLocation order {order!r}")
        self.keep, self.order = keep, order

    def __call__(self, s, ctx, geo):
        if s.season is None:
            raise ValueError("EveryLocation needs per-season scores")
        NS = s.season.shape[1]
        valid, s_idx = ctx.valid, ctx.season.clamp(min=0)
        used = (F.one_hot(s_idx, NS).bool() & valid[..., None]).any(1)        # (B, NS)
        sc = s.season.masked_fill(~used[:, :, None, None], NEG)
        nused = used.sum(1)[:, None, None, None]
        k = (nused // 2).clamp(min=1) if self.keep == "half" else torch.ones_like(nused)
        rank = sc.argsort(1, descending=True).argsort(1)    # 0 = highest; unused seasons last
        pick = ((rank >= nused - k) if self.order == "low" else (rank < k)) & (rank < nused)
        a = geo.anchors(sc.device)
        if a is not None:
            pick = pick & a
        mask = pick[torch.arange(ctx.B, device=sc.device)[:, None], s_idx]   # (B, T, 16, 16)
        return mask & valid[:, :, None, None]


class MonthGrid(Allocator):
    """Content-blind time control with space fixed: every candidate location, every `step`-th
    of the sample's PRESENT months in calendar order (so ~1 / step of them, like the season
    arms; phase 0 at test, a random phase per sample while training)."""
    needs = ()

    def __init__(self, step: int):
        if step < 1:
            raise ValueError(f"MonthGrid step {step}")
        self.step = step

    def __call__(self, s, ctx, geo):
        dev = ctx.present.device
        phase = (torch.randint(0, self.step, (ctx.B, 1), device=dev) if ctx.model.training
                 else torch.zeros(ctx.B, 1, dtype=torch.long, device=dev))
        valid = ctx.valid
        order = valid.long().cumsum(1) - 1                  # rank among the present months
        on = ((order + phase) % self.step == 0) & valid
        a = geo.anchors(dev)
        cells = torch.ones(H, H, dtype=torch.bool, device=dev) if a is None else a
        return on[:, :, None, None] & cells


class SeasonHalves(_SeasonBudget):
    """The best windows by the window scores, each with a half of its seasons by the season
    scores: keep "better" (lowest-score half, oracle_bothalf), "worse" (highest, tophalf) or
    "all" (every used season)."""

    def __init__(self, n: int, keep: str = "better"):
        super().__init__(n)
        if keep not in ("better", "worse", "all"):
            raise ValueError(f"SeasonHalves keep {keep!r}")
        self.keep = keep

    def __call__(self, s, ctx, geo):
        if s.window is None:
            raise ValueError("SeasonHalves needs window scores")
        sc, used, budget, cost, s_idx, valid = self._setup(s, ctx)
        B = sc.shape[0]
        nused = used.sum(1)[:, None, None, None]
        half = (nused // 2).clamp(min=1)
        rank = sc.argsort(1, descending=True).argsort(1)     # 0 = worst; unused seasons last
        pick = {"worse": rank < half,
                "better": (rank >= nused - half) & (rank < nused),
                "all": rank < nused}[self.keep]
        pick = pick & torch.isfinite(sc)
        win = _take(s.window.flatten(1), (pick * cost).sum(1).flatten(1), budget).view(B, H, H)
        return self._to_months(pick & win[:, None], s_idx, valid)


# -- selector -----------------------------------------------------------------------------
@dataclass
class Selection:
    plan: Plan | list[Plan]             # one crop group, or several (Level2)
    mask: torch.Tensor                  # (B, T, 16, 16) the (first) group's picks
    logw: torch.Tensor | None = None    # (N,) crop gate log-weights, plan order
    aux: torch.Tensor | None = None


class Selector:
    def __init__(self, geometry: Geometry, scorer: Scorer, allocator: Allocator):
        self.geo, self.scorer, self.allocator = geometry, scorer, allocator

    def modules(self) -> list:
        """The selector's trainable nn.Modules (a residual's network)."""
        net = getattr(self.scorer, "net", None)
        return [net] if net is not None else []

    def __call__(self, ctx: Context) -> Selection:
        s = self.scorer(ctx, self.geo, self.allocator.needs)
        mask = self.allocator(s, ctx, self.geo)
        plan = self.geo.plan(mask)
        logw = None if s.gate is None else F.logsigmoid(s.gate[plan.b, plan.y, plan.x].float())
        if s.season_gate is not None:
            logw = s.season_gate[plan.b, ctx.season[plan.b, plan.t], plan.y, plan.x].float()
        return Selection(plan, mask, logw, s.aux)


class Level2:
    """A second layer of finer month crops INSIDE the first layer's (legacy --level2 m4 / m2):
    the model's own prediction with the first layer's crops (eval, no grad) picks
        where  the n cells of the patch px grid under a first-layer crop with the highest loss
               vs the label (pick "oracle") or random ones (pick "random")
        when   at each, the lower-loss half of the months the first layer cropped there (a
               month crop's own head map vs the label; random: a random half)
    and those months are cut at `patch` px: their composites (source "months", legacy m4 / m2)
    or every raw acquisition in them, each modality on its own dates (source "dates", legacy
    d8 / d4 / d2; n counts cells of the patch grid, so legacy d8 at l2_n is n = l2_n / 4)."""

    def __init__(self, first: Selector, patch: int, n: int, pick: str = "oracle",
                 source: str = "months"):
        if pick not in ("oracle", "random"):
            raise ValueError(f"Level2 pick {pick!r}")
        if source not in ("months", "dates"):
            raise ValueError(f"Level2 source {source!r}")
        self.source = source
        if patch > first.geo.crop_px or (patch == first.geo.crop_px and source == "months"):
            raise ValueError(f"Level2 {source} at {patch} px: not finer than the first layer's "
                             f"{first.geo.crop_px} px month crops")
        self.first, self.patch, self.n, self.pick = first, patch, n, pick
        self.geo, self.scorer = first.geo, first.scorer

    def modules(self) -> list:
        return self.first.modules()

    def __call__(self, ctx: Context) -> Selection:
        s1 = self.first(ctx)
        if s1.logw is not None or not isinstance(s1.plan, Plan):
            raise ValueError("Level2 over a gated or multi-group first layer: not ported")
        if ctx.target is None:
            raise ValueError("Level2: no target in the context (both picks read the label)")
        m, p1, tgt = ctx.model, s1.plan, ctx.target
        P_, su = p1.patch, self.patch
        B, T = ctx.present.shape
        Hs = TILE // su
        dev = ctx.present.device
        was = m.training
        m.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out1 = m(ctx.ms, ctx.ss, ctx.season, ctx.present, p1)
                logits, v = out1.logits.float(), out1.hr
        finally:
            m.train(was)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            b, t, py, px = p1.b, p1.t, p1.py, p1.px
            cell = F.avg_pool2d(pixel_label_loss(logits, tgt)[:, None], su)[:, 0]   # (B,Hs,Hs)
            lin = m.head.linear
            C = lin.out_features // LR_PS ** 2
            maps = F.avg_pool2d(lin(v).view(-1, C, LR_PS, LR_PS), LR_PS // P_)       # (N,C,P,P)
            ar = torch.arange(P_, device=dev)
            ry, rx = (py[:, None] + ar)[:, :, None], (px[:, None] + ar)[:, None, :]
            tb = tgt[b]
            tw = (tb[torch.arange(len(b), device=dev)[:, None, None], ry, rx] if tb.ndim == 3
                  else tb[torch.arange(len(b), device=dev)[:, None, None, None],
                          torch.arange(tb.shape[1], device=dev)[None, :, None, None],
                          ry[:, None], rx[:, None]])
            ml = F.avg_pool2d(pixel_label_loss(maps, tw)[:, None], su)[:, 0]        # (N, f, f)
            f = P_ // su
            ML = torch.full((B, T, Hs, Hs), float("nan"), device=dev)
            for i in range(f):
                for j in range(f):
                    ML[b, t, py // su + i, px // su + j] = ml[:, i, j]
            avail = ~torch.isnan(ML)
            cand = avail.any(1)
            loc = cell if self.pick == "oracle" else torch.rand_like(cell)
            loc = loc.masked_fill(~cand, NEG).flatten(1)
            rank = loc.argsort(1, descending=True).argsort(1)
            loc = ((rank < self.n) & torch.isfinite(loc)).view_as(cand)
            key = ML if self.pick == "oracle" else torch.rand_like(ML)
            r = key.masked_fill(~avail, float("inf")).argsort(1).argsort(1)    # 0 = lowest
            half = (avail.sum(1, keepdim=True) // 2).clamp(min=1)
            mask2 = (r < half) & avail & loc[:, None]
        if self.source == "months":
            return Selection([p1, Plan.from_grid(mask2, su)], s1.mask, None, s1.aux)
        if ctx.dates is None:
            raise ValueError("Level2 source dates: the batch has no dates (loader dates=True)")
        plans, bidx = [p1], torch.arange(B, device=dev)[:, None]
        for mod, (_, d_valid, d_month, d_frac) in ctx.dates.items():
            d_valid, d_month, d_frac = d_valid.to(dev), d_month.to(dev), d_frac.to(dev)
            ok = d_valid & (d_month >= 0) & ctx.present.gather(1, d_month.clamp(min=0))
            c = ok[:, :, None, None] & mask2[bidx, d_month.clamp(min=0)]       # (B,Td,Hs,Hs)
            bb, dd, yy, xx = c.nonzero(as_tuple=True)
            if not len(bb):
                continue
            mo = d_month[bb, dd]
            py, px = yy * su, xx * su
            plans.append(Plan(bb, mo, py // LOOKUP_PS, px // LOOKUP_PS, py, px, su, src=mod,
                              src_t=dd, slot=ctx.season[bb, mo].float() + d_frac[bb, dd]))
        return Selection(plans, s1.mask, None, s1.aux)


def build_selector(cfg, model: MixedResModel, example=None):
    """manyup.mixres.config.SelectConfig -> Selector (or Level2). example: one batch's (monthly,
    season) samples, to size the HR residual arms' pixel embedding and season embedding."""
    geo = Geometry(cfg.crop_px, cfg.loc_stride)
    if cfg.scorer == "season_residual":
        r = cfg.residual
        if r.arch != "conv" and example is None:
            raise ValueError(f"season residual arch {r.arch}: build_selector needs an example")
        net = SeasonResidual(model.mix.encoder.embedding_size,
                             model.head.linear.out_features // LR_PS ** 2, arch=r.arch,
                             C=0 if r.arch == "conv" else window_pixels(example[0], 1).shape[-1],
                             P=cfg.crop_px, S=TILE // cfg.crop_px, gh=TILE // model.cfg.lr_patch,
                             NS=0 if r.arch == "conv" else example[1].timestamps.shape[1]
                             ).to(model.device)
        scorer = SeasonResidualScorer(net, train=r.train, tau=r.tau,
                                      lam=1.0 if r.train != "e2e" else 0.0)
    elif cfg.scorer == "entropy_sign":
        if example is None:
            raise ValueError("scorer entropy_sign: build_selector needs an example batch")
        net = SignNet(model.mix.encoder.embedding_size, example[1].timestamps.shape[1],
                      window_pixels(example[0], 1).shape[-1] * cfg.crop_px ** 2).to(model.device)
        scorer = EntropySign(net)
    elif cfg.scorer == "residual":
        r = cfg.residual
        if r.arch != "conv" and example is None:
            raise ValueError(f"residual arch {r.arch}: build_selector needs an example batch")
        C = 0 if r.arch == "conv" else window_pixels(example[0], 1).shape[-1]
        D = model.mix.encoder.embedding_size
        net = ResidualScorer(D, model.head.linear.out_features // LR_PS ** 2, arch=r.arch, C=C,
                             P=cfg.crop_px, S=TILE // cfg.crop_px,
                             gh=TILE // model.cfg.lr_patch,
                             NS=0 if r.arch == "conv"
                             else example[1].timestamps.shape[1]).to(model.device)
        scorer = Residual(Entropy(), net, train=r.train, tau=r.tau,
                          lam=1.0 if r.train != "e2e" else 0.0)
    else:
        scorer = {"label": LabelLoss, "entropy": Entropy, "random": RandomWindows,
                  "none": NoScores,
                  "grid": lambda: GridWindows(cfg.n),
                  "season_grid": lambda: SeasonGrid(cfg.n),
                  "season_random": SeasonRandom}[cfg.scorer]()
    allocator = {"top": lambda: TopWindows(cfg.n), "pairs": lambda: SeasonPairs(cfg.n),
                 "season_top": lambda: SeasonTop(cfg.n),
                 "every_location": lambda: EveryLocation(cfg.keep, cfg.season_order),
                 "month_grid": lambda: MonthGrid(cfg.month_step),
                 "halves": lambda: SeasonHalves(cfg.n, cfg.keep)}[cfg.allocator]()
    sel = Selector(geo, scorer, allocator)
    if cfg.level2:
        sel = Level2(sel, cfg.level2.patch, cfg.level2.n, cfg.level2.pick, cfg.level2.source)
    return sel
