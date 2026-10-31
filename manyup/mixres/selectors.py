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
import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from manyup.encoders.olmoearth import _to_device
from manyup.mixres.model import LR_PS, MixedResModel, Out
from manyup.mixres.plan import LOOKUP_PS, TILE, Plan
from manyup.mixres.lossup import LossUp
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
    def lr_enc(self):
        """The uniform pass's season tokens through every block (cross attn modes), if it ran,
        for reuse at inference."""
        return None if self._out0 is None else self._out0.lr_enc

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
    sh = getattr(model, "season_head", None)
    if sh is not None:                  # model.season_heads: each season slot its own head
        if len(sh) != NS:
            raise ValueError(f"{len(sh)} season heads for {NS} season slots")
        m = torch.stack([h(lr_F[:, s]) for s, h in enumerate(sh)], 1).view(-1, C, LR_PS, LR_PS)
    else:
        m = lin(lr_F).view(-1, C, LR_PS, LR_PS)
    P = model.cfg.lr_patch
    if P < LR_PS:
        m = F.avg_pool2d(m, LR_PS // P) if LR_PS % P == 0 else F.adaptive_avg_pool2d(m, P)
    m = rearrange(m, "(b s gh gw) c i j -> (b s) c (gh i) (gw j)", s=NS, gh=gh, gw=gw)
    lo = model.lo           # ps6 / ps10 seasons: the tile padded to whole tokens, cropped back
    return m[..., lo:lo + TILE, lo:lo + TILE] if lo else m


def pixel_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(1)
    return -(p * p.clamp(min=1e-8).log()).sum(1)                                # (N, 64, 64)


def pixel_msr(logits: torch.Tensor) -> torch.Tensor:
    """Softmax response uncertainty: 1 - the top class probability, (N, 64, 64)."""
    return 1 - logits.softmax(1).amax(1)


def spatial_uncertainty(ctx: "Context", kind: str = "entropy") -> torch.Tensor:
    """(B, 64, 64) per-pixel uncertainty of the crop-free pass: classification the entropy of
    the pooled prediction (kind "msr": its softmax response, 1 - max p); regression the
    variance over the used seasons of each season token's own map (where the seasons
    disagree)."""
    out0 = ctx.out0
    with torch.no_grad():
        if ctx.model.task.kind == "classification":
            return (pixel_msr if kind == "msr" else pixel_entropy)(out0.logits.float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = season_maps(ctx.model, out0.lr_F)
        m = m.float().view(ctx.B, ctx.NS, -1, *m.shape[-2:]).mean(2)          # (B, NS, 64, 64)
        w = ctx.used.float()[..., None, None]
        n = w.sum(1).clamp(min=1)
        mu = (w * m).sum(1) / n
        return (w * (m - mu[:, None]) ** 2).sum(1) / n


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

    def __init__(self, source: MixedResModel | None = None, temp: "SeasonTemperature | None" = None):
        self.source, self.net = source, temp

    def __call__(self, ctx, geo, need=("window", "season")):
        if ctx.model.task.kind != "classification":
            raise ValueError(f"Entropy: {ctx.model.task.name} is not a classification task")
        if self.source is None:
            out0 = ctx.out0
            sc = aux = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if "season" in need:
                    s_log = season_maps(ctx.model, out0.lr_F).detach()
                    if self.net is not None:         # per-season temperature (select.entropy_T)
                        s_log = s_log.float().view(ctx.B, ctx.NS, *s_log.shape[1:])
                        if self.net.training and self.net.online and ctx.target is not None:
                            aux = self.net.nll(s_log, ctx.target, ctx.used)
                        s_log = self.net.scale(s_log).detach().flatten(0, 1)
                    sc = geo.window_mean(pixel_entropy(s_log)).view(ctx.B, ctx.NS, H, H)
                win = geo.window_mean(pixel_entropy(out0.logits.detach()))
            return Scores(win, sc, aux=aux, pooled=out0.logits.detach(), lr_F=out0.lr_F.detach())
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


class SeasonTemperature(nn.Module):
    """A temperature per season slot for the season maps' entropy (select.entropy_T), fitted
    online by the NLL of each batch's detached season maps (used seasons, labelled pixels):
    temperature scaling, so that entropies compare across seasons. Labels only in training."""

    def __init__(self, NS: int, online: bool = True):
        """online: fitted by the training batches' NLL (an aux loss); False: by fit() only
        (select.entropy_T_fit=val: exp.mixres.train fits it on the val split every epoch)."""
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(NS))
        self.online = online

    def scale(self, m):
        """(B, NS, C, H, W) -> divided by each season's temperature."""
        return m / self.logT.exp()[None, :, None, None, None]

    @torch.enable_grad()
    def fit(self, maps, steps: int = 200):
        """maps: [(season maps (B, NS, C, H, W), target, used)] -> logT by full-batch NLL."""
        opt = torch.optim.LBFGS([self.logT], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = sum(self.nll(m.float(), t, u) for m, t, u in maps) / len(maps)
            loss.backward()
            return loss
        opt.step(closure)

    def nll(self, m, target, used):
        B, NS = m.shape[:2]
        t = target.long()[:, None].expand(B, NS, *target.shape[-2:]).flatten(0, 1)
        ce = F.cross_entropy(self.scale(m.detach()).flatten(0, 1), t, ignore_index=-1,
                             reduction="none")
        ok = (t != -1) & used.flatten()[:, None, None]
        return ce[ok].mean()


class Residual(Scorer):
    """window score = z(base window entropy) + r, r a learned ResidualScorer (last layer zero:
    exactly the base at the start), on the windows that tile the tile.
        train  "e2e"     r learns from the task loss through the crop gate logsigmoid(score),
                         Gumbel(tau) exploration on the picks while training
               "oracle"  r learns from a ListNet ranking to the base pass's own window loss
               "both"    both; the oracle weight `lam` is set from outside (decayed)
    Ported from InjectFT._residual_loc."""

    def __init__(self, base: Entropy, net, train: str = "e2e", tau: float = 0.5,
                 lam: float = 1.0, prior: str = "entropy"):
        if train not in ("e2e", "oracle", "both"):
            raise ValueError(f"Residual train {train!r}")
        if prior not in ("entropy", "none"):
            raise ValueError(f"Residual prior {prior!r}")
        self.base, self.net, self.train, self.tau, self.lam = base, net, train, tau, lam
        self.prior = prior

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
        r = self.net(feats, tok.permute(0, 2, 3, 1), **hr_in)
        score = z.view(B, S, S) + r if self.prior == "entropy" else r
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
            if self.prior == "none":            # r = 0 at the start: break the ties at random
                out = out.detach() + 1e-6 * torch.rand_like(out)
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
    prior "none": score = r alone, no entropy term (r = 0 at the start: ties, broken at random,
    so the start is the random-season arm). Candidates must tile the tile (stride = crop)."""

    def __init__(self, net: SeasonResidual, train: str = "e2e", tau: float = 0.5,
                 lam: float = 1.0, prior: str = "entropy"):
        if train not in ("e2e", "oracle", "both"):
            raise ValueError(f"SeasonResidualScorer train {train!r}")
        if prior not in ("entropy", "none"):
            raise ValueError(f"SeasonResidualScorer prior {prior!r}")
        self.net, self.train, self.tau, self.lam, self.prior = net, train, tau, lam, prior

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
        score = z + r if self.prior == "entropy" else r
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
        if self.prior == "none":                                    # break ties (r = 0 at start)
            pick = pick + 1e-6 * torch.rand_like(pick)
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


class LossUpScorer(Scorer):
    """Both scores from one predicted loss map (manyup.mixres.lossup.LossUp): per season map and
    for the pooled prediction, the predicted per-pixel log(1 + loss) of the crop-free pass,
    window-meaned -> window score (pooled map; highest picked) and season scores (each season's
    map; the allocator keeps the lowest). Trained, model frozen or not, by the residual scorers'
    ListNet rankings to the true loss -- role "space": the sample's windows by the pooled loss,
    "time": each location's used seasons by their loss (lowest first), "both" -- plus the dense
    per-pixel regression (smooth L1 on log(1 + loss), weight `pixel`). One forward per context:
    the same instance may score space and time (SpaceTime), its loss counted once."""

    def __init__(self, net: LossUp, role: str = "both", pixel: float = 1.0):
        if role not in ("space", "time", "both"):
            raise ValueError(f"LossUpScorer role {role!r}")
        self.net, self.role, self.pixel = net, role, pixel

    @staticmethod
    def inputs(ctx):
        """-> logits (B K, n_cls, 64, 64), tokens (B K, D, gh, gw), guide (B K, C, 64, 64), valid
        (B, K); K = NS + 1, the pooled prediction last."""
        out0 = ctx.out0
        B, NS = ctx.B, ctx.NS
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_log = season_maps(ctx.model, out0.lr_F)
            s_log = s_log.float().view(B, NS, *s_log.shape[1:])
            logits = torch.cat([s_log, out0.logits.float()[:, None]], 1).flatten(0, 1)
            tok = out0.lr_F.float().permute(0, 1, 4, 2, 3)                   # (B,NS,D,gh,gw)
            u = ctx.used.float()[:, :, None, None, None]
            pooled = (tok * u).sum(1, keepdim=True) / u.sum(1, keepdim=True).clamp(min=1)
            tok = torch.cat([tok, pooled], 1).flatten(0, 1)
            pix = window_pixels(ctx.ms, 1)                                   # (B,64,64,T,C)
            w = (F.one_hot(ctx.season.clamp(min=0), NS).bool() & ctx.valid[..., None]).float()
            w = torch.cat([w, ctx.valid.float()[..., None]], -1)             # + all months
            g = torch.einsum("bxytc,btk->bkcxy", pix, w)
            g = g / w.sum(1)[:, :, None, None, None].clamp(min=1)
            mu = g.mean((1, 3, 4), keepdim=True)                             # per sample, band
            sd = g.std((1, 3, 4), keepdim=True).clamp(min=1e-6)
            g = ((g - mu) / sd).flatten(0, 1)
            valid = torch.cat([ctx.used, torch.ones_like(ctx.used[:, :1])], 1)
        return logits, tok, g, valid

    def __call__(self, ctx, geo, need=("window", "season")):
        if ctx.model.task.kind != "classification":
            raise ValueError(f"LossUp: {ctx.model.task.name} is not a classification task")
        cache = ctx.__dict__.setdefault("_lossup", {})
        if id(self) in cache:                       # the second role of a shared instance
            return replace(cache[id(self)], aux=None)
        B, NS = ctx.B, ctx.NS
        logits, tok, g, valid = self.inputs(ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = self.net(logits, tok, g).float()                             # (B K, 64, 64)
        wm = geo.window_mean(p).view(B, NS + 1, H, H)
        window, season = wm[:, NS], wm[:, :NS]
        aux = None
        if self.net.training:
            if ctx.target is None:
                raise ValueError("LossUp: no target in the context")
            tgt = ctx.target.repeat_interleave(NS + 1, 0)
            with torch.no_grad():
                px = pixel_label_loss(logits, tgt)                           # (B K, 64, 64)
                L = geo.window_mean(px).view(B, NS + 1, H, H)
            terms = []
            if self.pixel > 0:
                m = (tgt != -1) & valid.flatten()[:, None, None]
                terms.append(self.pixel * F.smooth_l1_loss(p[m], px.log1p()[m]))
            if self.role in ("space", "both"):
                ok = torch.isfinite(L[:, NS]).flatten(1)
                terms.append(listnet_loss(window.flatten(1).masked_fill(~ok, 0),
                                          L[:, NS].flatten(1).masked_fill(~ok, 0), ok))
            if self.role in ("time", "both"):
                Ls = L[:, :NS]
                ok = torch.isfinite(Ls) & ctx.used[:, :, None, None]
                f = lambda x: x.permute(0, 2, 3, 1).flatten(0, 2)               # noqa: E731
                terms.append(listnet_loss(f(-season.masked_fill(~ok, 0)),
                                          f(-Ls.masked_fill(~ok, 0)), f(ok)))
            aux = sum(terms)
        out = Scores(window.detach(), season.detach(), aux=aux, pooled=ctx.out0.logits,
                     lr_F=ctx.out0.lr_F)
        cache[id(self)] = out
        return out


class ReadoutAttention(Scorer):
    """Label-free season scores from the readout itself: its learned attention score
    (NativeSetReadout.score, trained by the task loss alone to weight each cell's season
    tokens) on the crop-free pass's season tokens, each cell its token's value. Minus the score,
    so the allocator's lowest = the season the readout trusts most (exp.mixres.attn_probe)."""

    def __call__(self, ctx, geo, need=("season",)):
        ro = ctx.model.readout0
        if ro.score is None:
            raise ValueError("ReadoutAttention: the readout has no attention score (native_mean)")
        with torch.no_grad():
            a = ro._score(ctx.out0.lr_F.float())                               # (B, NS, gh, gw)
            a = F.interpolate(a, size=(H, H), mode="nearest")
        return Scores(season=-a, pooled=ctx.out0.logits, lr_F=ctx.out0.lr_F)


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


class SpaceTime:
    """Space then time: a spatial selector (window scorer + TopWindows: n locations, every
    present month) AND a time selector over every location (season scorer + EveryLocation /
    MonthGrid: each location's chosen months) -> the n locations, each with only its chosen
    months. The time scorer's gate / aux pass through (a learned season residual trains on every
    location's ranking, picked or not).
    tau_space / tau_time (evaluation ablation, default 0 = top-k): sample instead, without
    replacement, P ~ exp(z / tau) with z the window score z-scored over the sample's candidates
    / minus the season score z-scored over the location's used seasons (Gumbel top-k); inf:
    uniform."""

    tau_space: float = 0.0
    tau_time: float = 0.0

    def __init__(self, geometry: Geometry, space_scorer: Scorer, space_alloc: Allocator,
                 time_scorer: Scorer, time_alloc: Allocator):
        if not isinstance(space_alloc, TopWindows):
            raise ValueError("SpaceTime: the spatial allocator must be TopWindows (allocator top)")
        if not isinstance(time_alloc, (EveryLocation, MonthGrid)):
            raise ValueError("SpaceTime: the time allocator must be every_location or month_grid")
        self.geo, self.space_scorer, self.space_alloc = geometry, space_scorer, space_alloc
        self.scorer, self.allocator = time_scorer, time_alloc   # (train.py reads .scorer.lam)

    def modules(self) -> list:
        nets = [getattr(sc, "net", None) for sc in (self.space_scorer, self.scorer)]
        if self.space_scorer is self.scorer:        # one scorer for both (LossUpScorer)
            nets = nets[:1]
        return [n for n in nets if n is not None]

    @staticmethod
    def _sample(z, ok, dim, tau):
        """z with non-candidates (~ok) -inf -> z-scored over `dim` among ok, + tau x Gumbel
        (tau inf: Gumbel alone); the top-k of the result samples P ~ exp(z / tau)."""
        n = ok.sum(dim, keepdim=True).clamp(min=1)
        zf = z.masked_fill(~ok, 0.0)
        mu = zf.sum(dim, keepdim=True) / n
        sd = (((zf - mu) ** 2) * ok).sum(dim, keepdim=True).div(n).sqrt().clamp(min=1e-6)
        g = -torch.log(-torch.log(torch.rand_like(zf).clamp(1e-9, 1 - 1e-9)))
        out = g if math.isinf(tau) else (zf - mu) / sd + tau * g
        return out.masked_fill(~ok, NEG)

    def __call__(self, ctx: Context) -> Selection:
        s_sp = self.space_scorer(ctx, self.geo, self.space_alloc.needs)
        if self.tau_space:
            w = s_sp.window
            s_sp = replace(s_sp, window=self._sample(w, torch.isfinite(w), (1, 2), self.tau_space))
        where = self.space_alloc(s_sp, ctx, self.geo).any(1)                    # (B, 16, 16)
        s_t = self.scorer(ctx, self.geo, self.allocator.needs)
        if self.tau_time:           # the allocator keeps the lowest: sample on minus the score
            sc = s_t.season
            ok = torch.isfinite(sc) & ctx.used[:, :, None, None]
            s_t = replace(s_t, season=(-self._sample(-sc, ok, 1, self.tau_time)).masked_fill(~ok, NEG))
        mask = self.allocator(s_t, ctx, self.geo) & where[:, None]
        plan = self.geo.plan(mask)
        logw = None
        if s_t.season_gate is not None:
            logw = s_t.season_gate[plan.b, ctx.season[plan.b, plan.t], plan.y, plan.x].float()
        aux = [a for a in (s_sp.aux, s_t.aux) if a is not None]
        return Selection(plan, mask, logw, sum(aux) if aux else None)


class QuadrantState(nn.Module):
    """Quadrant's global thresholds: EMA tables of the training batches' quantiles (101 levels)
    of the two uncertainties, saved with the selector, fixed at evaluation."""

    LEVELS = 101

    def __init__(self, momentum: float = 0.95):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("tab_s", torch.zeros(self.LEVELS))
        self.register_buffer("tab_t", torch.zeros(self.LEVELS))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, u_s, u_t):
        lv = torch.linspace(0, 1, self.LEVELS, device=u_s.device)
        qs = torch.quantile(u_s.flatten().float(), lv)
        qt = torch.quantile(u_t.flatten().float(), lv)
        m = 0.0 if int(self.seen) == 0 else self.momentum
        self.tab_s.mul_(m).add_((1 - m) * qs)
        self.tab_t.mul_(m).add_((1 - m) * qt)
        self.seen += 1

    def threshold(self, tab, q):
        """The value above which a share q of the tracked distribution lies."""
        x = (1 - q) * (self.LEVELS - 1)
        i = min(int(x), self.LEVELS - 2)
        return tab[i] + (x - i) * (tab[i + 1] - tab[i])


class Quadrant:
    """The quadrant policy, per low-res season-token cell (16 px at ps16, 8 px at ps8), from the
    model's crop-free pass:
        spatial uncertainty   mean pixel entropy of the pooled prediction over the cell
        temporal uncertainty  entropy of the readout's attention over the cell's used seasons
                              (softmax of score + log months), / log(used seasons) in [0, 1]
    high / low by the shares q_s / q_t (mode "sample": within each sample; "global": thresholds
    of the training distribution, QuadrantState), then per cell
        high space, high time   high-res crops covering the cell, every present month
        high space, low time    the same crops, the months of the most-attended season
        low space, high time    lowspace "lr_all": a low-res month token per present month
                                (patch = the season tokens'), "skip": nothing
        low space, low time     nothing
    -> two crop groups: the high-res crops (crop_px, on their own grid), the low-res month
    tokens. random: training draws q_s ~ U(0, 2 q_s), q_t ~ U(0, 1) per batch. hs_ht / hs_lt
    swap "every present month" / "the most-attended season" for one / half / all of the cell's
    used seasons by attention (rank "weight": score + log months; "score": the score alone);
    space "crop" decides high space per crop window instead of per cell (so with hs_ht = hs_lt
    and lowspace skip, mode sample, rank score this is entropy-top + attention-every_location)."""

    def __init__(self, geometry: Geometry, cfg, lr_patch: int, device=None):
        self.geo, self.cfg, self.lr_patch = geometry, cfg, lr_patch
        # a patch that does not divide the tile (ps6 seasons, 12 px windows) tiles it padded,
        # centred, to whole patches (66 / 72 px; MixedResOE.embed pads the season tokens the
        # same way): u_s lives on the crop windows' frame, u_t on the season tokens' frame
        self.lo = self.frame_lo(lr_patch)
        cp = geometry.crop_px
        # nested: the crop windows tile each season cell (ps16 / ps8 seasons, 8 / 4 px windows;
        # ps6 seasons with 6 px windows); else each window reads the season cell under its centre
        self.nested = lr_patch % cp == 0 and self.frame_lo(cp) == self.lo
        if (self.lo or not self.nested) and (cfg.lowspace == "lr_all" or cfg.space != "crop"):
            raise ValueError("season tokens / crop windows that do not tile the 64 px tile: "
                             "space crop, lowspace skip")
        self.state = QuadrantState().to(device)
        self.scorer = NoScores()                    # (train.py reads selector.scorer)
        self.last = {}
        # oracle mixing (train.oracle_mix; exp.mixres.train sets oracle_p every epoch): while
        # training, with probability oracle_p (per batch, or per window for unit "crop") the
        # high-space decision is the label oracle's (each sample's q_s share of windows with
        # the highest loss of the crop-free prediction) instead of the entropy rule's
        self.oracle_p, self.oracle_unit = 0.0, "batch"

    def modules(self) -> list:
        return [self.state]

    @staticmethod
    def frame_lo(px: int) -> int:
        """Padding before the tile when px-px patches tile it padded to whole patches."""
        return (-(-TILE // px) * px - TILE) // 2

    def pool(self, u, px):
        """(B, 64, 64) -> the mean per px window (B, frame / px, frame / px), the map
        replicate-padded to px's padded frame first when px does not divide the tile."""
        lo = self.frame_lo(px)
        if TILE % px:
            i = (torch.arange(-(-TILE // px) * px, device=u.device) - lo).clamp(0, TILE - 1)
            u = u[:, i][:, :, i]
        return F.avg_pool2d(u[:, None], px)[:, 0]

    def cell_of_window(self, n_windows: int, gh: int, device) -> torch.Tensor:
        """(n_windows,) the season cell (0..gh-1) under each crop window's centre."""
        cp = self.geo.crop_px
        c = (torch.arange(n_windows, device=device) * cp + cp / 2 - self.frame_lo(cp)
             + self.lo)                                     # centre in the season tokens' frame
        return (c // self.lr_patch).long().clamp(0, gh - 1)

    def threshold_values(self, ctx) -> torch.Tensor:
        """The spatial uncertainties the global threshold applies to (exp.mixres.train
        val_threshold), flat."""
        px = self.geo.crop_px if self.cfg.space == "crop" else self.lr_patch
        return self.pool(spatial_uncertainty(ctx, self.cfg.unc), px).flatten()

    def uncertainties(self, ctx, rank: bool = False):
        """-> u_s, u_t (B, gh, gw), top (B, gh, gw) the most-attended season slot (rank: and the
        season ranking scores (B, NS, gh, gw), -inf at unused seasons)."""
        out0 = ctx.out0
        P = self.lr_patch
        with torch.no_grad():
            u_s = self.pool(spatial_uncertainty(ctx, self.cfg.unc), P)
            lr_F = out0.lr_F.float()
            months = (F.one_hot(ctx.season.clamp(min=0), ctx.NS).float()
                      * ctx.valid.float()[..., None]).sum(1)                     # (B, NS)
            score = ctx.model.readout0._score(lr_F)
            a = score + months.clamp(min=1).log()[:, :, None, None]
            unused = (months == 0)[:, :, None, None]
            a = a.masked_fill(unused, float("-inf"))
            w = a.softmax(1)
            h = -(w * w.clamp(min=1e-12).log()).sum(1)
            nu = (months > 0).sum(1).float()[:, None, None]
            u_t = torch.where(nu > 1, h / nu.clamp(min=2).log(), torch.zeros_like(h))
            top = a.argmax(1)
        if not rank:
            return u_s, u_t, top
        r = score.masked_fill(unused, float("-inf")) if self.cfg.rank == "score" else a
        return u_s, u_t, top, r

    @staticmethod
    def _seasons(r, keep):
        """(B, NS, gh, gw) ranking scores (-inf unused) -> the kept seasons: one / half (at least
        one, EveryLocation's rule) / all of the used ones, best first."""
        used = torch.isfinite(r)
        nused = used.sum(1, keepdim=True)
        k = {"one": torch.ones_like(nused), "half": (nused // 2).clamp(min=1),
             "all": nused}[keep]
        return (r.argsort(1, descending=True).argsort(1) < k) & used

    OFFGRID_ROUNDS = 256

    def offgrid_windows(self, u, k):
        """(B, 64, 64) uncertainty, (B,) long counts -> b, py, px of each sample's k[b] best
        non-overlapping crop_px windows at ANY pixel offset (inside the tile). Rounds of local-
        maximum suppression on the stride-1 window means, each round batched over the whole
        batch: a window is a peak when it is the best within +-(crop_px - 1) px of it, so a
        round's peaks never overlap each other; each sample takes its best peaks (those above
        all its free non-peak windows) up to what it still needs, the windows overlapping a pick
        leave the pool, and the next round finds the peaks of what is left = sequential greedy
        NMS, in a few batched rounds."""
        cp = self.geo.crop_px
        s = F.avg_pool2d(u[:, None].float(), cp, stride=1)[:, 0]               # (B, W, W)
        s = s + torch.rand_like(s) * 1e-6 * (s.abs().amax() + 1e-12)           # (no ties)
        free = torch.ones_like(s, dtype=torch.bool)
        picked = torch.zeros_like(free)
        need = k.clone()
        self.rounds, kmax = 0, int(need.max())
        for _ in range(self.OFFGRID_ROUNDS):
            if kmax <= 0:
                break
            self.rounds += 1
            v = s.masked_fill(~free, float("-inf"))
            peak = free & (v == F.max_pool2d(v[:, None], 2 * cp - 1, 1, cp - 1)[:, 0])
            # a peak is a greedy pick (nothing higher overlaps it); only the peaks at least as
            # high as every free non-peak window are taken now, so the picks come in exactly
            # the order (and set) of sequential greedy NMS
            bar = v.masked_fill(peak, float("-inf")).flatten(1).amax(1)[:, None, None]
            peak &= v >= bar
            pv = v.masked_fill(~peak, float("-inf")).flatten(1)
            top, idx = pv.topk(min(kmax, pv.shape[1]), 1)                     # best first
            ok = (torch.arange(idx.shape[1], device=idx.device) < need[:, None]) & torch.isfinite(top)
            take = torch.zeros_like(pv, dtype=torch.bool).scatter_(1, idx, ok).view_as(peak)
            picked |= take
            got = take.flatten(1).sum(1)
            need = need - got
            free &= F.max_pool2d(picked[:, None].float(), 2 * cp - 1, 1, cp - 1)[:, 0] == 0
            kmax, moved = torch.stack([need.max(), got.sum()]).tolist()        # (one sync)
            if not moved:           # no free window left where one is still needed
                break
        self.short = int(need.clamp(min=0).sum())       # (windows a full tile could not fit)
        return picked.nonzero(as_tuple=True)

    def oracle_months(self, ctx, b, py, px, keep):
        """(N,) windows -> (N, T) the present months of each window's best season(s): lowest
        label loss of each season token's own map, mean over the window's pixels (one: the
        best season; half: the better half of the used ones, at least one)."""
        if ctx.target is None:
            raise ValueError("quadrant rank oracle: no label in the context")
        cp = self.geo.crop_px
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s_log = season_maps(ctx.model, ctx.out0.lr_F)
            loss = pixel_label_loss(s_log.float(), ctx.target.repeat_interleave(ctx.NS, 0))
        loss = F.avg_pool2d(loss.view(ctx.B, ctx.NS, TILE, TILE), cp, stride=1)
        wl = loss[b, :, py, px]                                                    # (N, NS)
        months = (F.one_hot(ctx.season.clamp(min=0), ctx.NS).float()
                  * ctx.valid.float()[..., None]).sum(1)                          # (B, NS)
        used = months[b] > 0
        r = (-wl).masked_fill(~used, float("-inf"))           # (higher = better season)
        nused = used.sum(1, keepdim=True)
        k = {"one": torch.ones_like(nused), "half": (nused // 2).clamp(min=1), "all": nused}[keep]
        best = (r.argsort(1, descending=True).argsort(1) < k) & used               # (N, NS)
        return best.gather(1, ctx.season[b].clamp(min=0)) & ctx.valid[b]           # (N, T)

    def high(self, u, q, tab):
        if self.cfg.mode == "sample":
            B = u.shape[0]
            k = round(q * u[0].numel())
            r = u.flatten(1).argsort(1, descending=True).argsort(1)
            return (r < k).view_as(u)
        return u >= self.state.threshold(tab, q)

    def __call__(self, ctx: Context) -> Selection:
        q = self.cfg
        u_s, u_t, top, r = self.uncertainties(ctx, rank=True)
        f = self.lr_patch // self.geo.crop_px
        up = lambda m: m.repeat_interleave(f, -2).repeat_interleave(f, -1)      # noqa: E731
        if not self.nested:         # each crop window: the season cell under its centre
            n_w = self.pool(spatial_uncertainty(ctx)[:1], self.geo.crop_px).shape[-1]
            ci = self.cell_of_window(n_w, u_t.shape[-1], u_t.device)
            up = lambda m: m[..., ci, :][..., ci]                              # noqa: E731
        if q.space == "crop":       # high space per crop window, time per cell
            u_s = self.pool(spatial_uncertainty(ctx, self.cfg.unc), self.geo.crop_px)
        training = ctx.model.training
        if training and q.mode == "global":
            self.state.update(u_s, u_t)
        qs, qt = q.q_s, q.q_t
        if training and q.random:
            qs, qt = float(torch.rand(()) * 2 * q.q_s), float(torch.rand(()))
        hs, ht = self.high(u_s, qs, self.state.tab_s), self.high(u_t, qt, self.state.tab_t)
        n_oracle = 0.0
        if training and self.oracle_p > 0 and ctx.target is not None:
            px = self.geo.crop_px if q.space == "crop" else self.lr_patch
            with torch.no_grad():
                loss = F.avg_pool2d(pixel_label_loss(ctx.out0.logits.float(),
                                                     ctx.target)[:, None], px)[:, 0]
            k = round(qs * loss[0].numel())
            hs_o = (loss.flatten(1).argsort(1, descending=True).argsort(1) < k).view_as(hs)
            coin = (torch.rand((), device=hs.device) if self.oracle_unit == "batch"
                    else torch.rand(hs.shape, device=hs.device)) < self.oracle_p
            hs = torch.where(coin, hs_o, hs)
            n_oracle = float(coin.float().mean())
        valid = ctx.valid                                                        # (B, T)
        B, T = valid.shape
        gh = ht.shape[-1]
        sidx = ctx.season.clamp(min=0)[:, :, None, None].expand(B, T, gh, gh)
        def months(keep):           # the kept seasons' present months per cell (B, T, gh, gw)
            if keep == "all":
                return valid[:, :, None, None].expand(B, T, gh, gh)
            return self._seasons(r, keep).gather(1, sidx) & valid[:, :, None, None]
        m_hh, m_hl = months(q.hs_ht), months(q.hs_lt)
        if q.offgrid or q.rank == "oracle":     # per window: (sample, top-left px), then months
            if q.space != "crop" or q.lowspace == "lr_all" or n_oracle:
                raise ValueError("quadrant offgrid / rank oracle: space crop, lowspace skip, no "
                                 "oracle mixing")
            cp, P = self.geo.crop_px, self.lr_patch
            if q.offgrid:           # as many windows as on the grid, placed at any pixel offset
                b, py, px = self.offgrid_windows(spatial_uncertainty(ctx, self.cfg.unc),
                                                 hs.flatten(1).sum(1))
            else:                   # the grid windows above the threshold
                b, gy, gx = hs.nonzero(as_tuple=True)
                py, px = ((g * cp - self.frame_lo(cp)).clamp(0, TILE - cp) for g in (gy, gx))
            if q.rank == "oracle":
                mw = self.oracle_months(ctx, b, py, px, q.hs_ht)
            else:
                cy, cx = (((v + cp // 2 + self.lo) // P).clamp(max=gh - 1) for v in (py, px))
                mw = torch.where(ht[b, cy, cx][:, None], m_hh[b, :, cy, cx], m_hl[b, :, cy, cx])
            w, t = mw.nonzero(as_tuple=True)                    # (window, month) of each crop
            plan = Plan(b[w], t, py[w] // LOOKUP_PS, px[w] // LOOKUP_PS, py[w], px[w], cp,
                        offgrid=q.offgrid)
            n = float(hs.numel())
            self.last = {"q_s": qs, "q_t": qt, "oracle": 0.0, "hs": float(hs.sum()) / n,
                         "rounds": getattr(self, "rounds", 0) if q.offgrid else 0,
                         "short": getattr(self, "short", 0) if q.offgrid else 0}
            return Selection([plan], None, None, None)
        if q.space == "crop":       # hs already on the crop grid
            hr = hs[:, None] & up(ht[:, None] & m_hh | ~ht[:, None] & m_hl)
            hs_c = (F.max_pool2d(hs[:, None].float(), f)[:, 0].bool() if self.nested
                    else None)                                  # (low space: no crop; lr_all)
        else:
            hr = up(hs[:, None] & (ht[:, None] & m_hh | ~ht[:, None] & m_hl))  # crop grid
            hs_c = hs
        allm = valid[:, :, None, None].expand(B, T, gh, gh)
        lr = (((~hs_c & ht)[:, None] & allm) if q.lowspace == "lr_all"
              else torch.zeros_like(allm))
        plans = [Plan.from_grid(hr, self.geo.crop_px), Plan.from_grid(lr, self.lr_patch)]
        lo_c = self.frame_lo(self.geo.crop_px)
        if lo_c or not self.nested:     # the windows' padded frame -> the tile (edges clamped)
            plans, mask = [plans[0].shifted(-lo_c), plans[1]], None
        else:
            g = TILE // self.geo.crop_px
            mask = hr.repeat_interleave(H // g, 2).repeat_interleave(H // g, 3)  # (B, T, 16, 16)
        if q.space == "crop":
            ht = up(ht)
        n = float(hs.numel())
        self.last = {"q_s": qs, "q_t": qt, "oracle": n_oracle,
                     "hs_ht": float((hs & ht).sum()) / n,
                     "hs_lt": float((hs & ~ht).sum()) / n, "ls_ht": float((~hs & ht).sum()) / n}
        return Selection(plans, mask, None, None)


class Level2:
    """A second layer of finer month crops INSIDE the first layer's (legacy --level2 m4 / m2):
    the model's own prediction with the first layer's crops (eval, no grad) picks
        where  the n cells of the patch px grid under a first-layer crop with the highest loss
               vs the label (pick "oracle"), the highest prediction entropy (pick "entropy",
               label-free) or random ones (pick "random")
        when   at each, the lower-loss half of the months the first layer cropped there (a
               month crop's own head map vs the label; entropy: the half the readout attends
               to most, NativeSetReadout.score of each month crop's final token; random: a
               random half). Ablations, one axis label-free and the other random: "entropy_r"
               (entropy where, random half), "r_attention" (random where, attention half).
    "entropy0" / "entropy0_r": the same from the crop-free pass layer 1 already ran (ctx.out0, no
    second pass): entropy of its prediction where, and when the months of the cropped season the
    readout attends to most at that cell (season tokens: months of one season tie, broken at
    random) / a random half
    and those months are cut at `patch` px: their composites (source "months", legacy m4 / m2)
    or every raw acquisition in them, each modality on its own dates (source "dates", legacy
    d8 / d4 / d2; n counts cells of the patch grid, so legacy d8 at l2_n is n = l2_n / 4)."""

    def __init__(self, first: Selector, patch: int, n: int, pick: str = "oracle",
                 source: str = "months"):
        if pick not in ("oracle", "random", "entropy", "entropy_r", "r_attention", "entropy0",
                        "entropy0_r"):
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

    def _from_out0(self, ctx: Context, s1: Selection, p1: Plan, su: int, Hs: int) -> Selection:
        if self.source != "months":
            raise ValueError(f"Level2 pick {self.pick}: months only")
        m, out0 = ctx.model, ctx.out0
        B, T = ctx.present.shape
        dev = ctx.present.device
        f = p1.patch // su
        with torch.no_grad():
            cell = F.avg_pool2d(pixel_entropy(out0.logits.float())[:, None], su)[:, 0]  # (B,Hs,Hs)
            a = m.readout0._score(out0.lr_F.float()).float()                         # (B,NS,gh,gw)
            a = F.interpolate(a, size=(Hs, Hs), mode="nearest")
            ML = torch.full((B, T, Hs, Hs), float("nan"), device=dev)
            b, t = p1.b, p1.t
            sn = ctx.season[b, t].long()
            for i in range(f):
                for j in range(f):
                    y, x = p1.py // su + i, p1.px // su + j
                    ML[b, t, y, x] = -a[b, sn, y, x] + 1e-6 * torch.rand(len(b), device=dev)
            avail = ~torch.isnan(ML)
            cand = avail.any(1)
            loc = cell.masked_fill(~cand, NEG).flatten(1)
            rank = loc.argsort(1, descending=True).argsort(1)
            loc = ((rank < self.n) & torch.isfinite(loc)).view_as(cand)
            key = ML if self.pick == "entropy0" else torch.rand_like(ML)
            r = key.masked_fill(~avail, float("inf")).argsort(1).argsort(1)    # 0 = lowest
            half = (avail.sum(1, keepdim=True) // 2).clamp(min=1)
            mask2 = (r < half) & avail & loc[:, None]
        return Selection([p1, Plan.from_grid(mask2, su)], s1.mask, None, s1.aux)

    def __call__(self, ctx: Context) -> Selection:
        s1 = self.first(ctx)
        if s1.logw is not None or not isinstance(s1.plan, Plan):
            raise ValueError("Level2 over a gated or multi-group first layer: not ported")
        if ctx.target is None and self.pick == "oracle":
            raise ValueError("Level2: no target in the context (the oracle pick reads the label)")
        m, p1, tgt = ctx.model, s1.plan, ctx.target
        P_, su = p1.patch, self.patch
        B, T = ctx.present.shape
        Hs = TILE // su
        dev = ctx.present.device
        if self.pick.startswith("entropy0"):
            return self._from_out0(ctx, s1, p1, su, Hs)
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
            f = P_ // su
            if self.pick != "oracle":       # label-free: entropy where, readout attention when
                cell = F.avg_pool2d(pixel_entropy(logits)[:, None], su)[:, 0]          # (B,Hs,Hs)
                ml = (-m.readout._score(v.float()).float())[:, None, None].expand(-1, f, f)
            else:
                cell = F.avg_pool2d(pixel_label_loss(logits, tgt)[:, None], su)[:, 0]
                lin = m.head.linear
                C = lin.out_features // LR_PS ** 2
                maps = F.avg_pool2d(lin(v).view(-1, C, LR_PS, LR_PS), LR_PS // P_)   # (N,C,P,P)
                ar = torch.arange(P_, device=dev)
                ry, rx = (py[:, None] + ar)[:, :, None], (px[:, None] + ar)[:, None, :]
                tb = tgt[b]
                tw = (tb[torch.arange(len(b), device=dev)[:, None, None], ry, rx] if tb.ndim == 3
                      else tb[torch.arange(len(b), device=dev)[:, None, None, None],
                              torch.arange(tb.shape[1], device=dev)[None, :, None, None],
                              ry[:, None], rx[:, None]])
                ml = F.avg_pool2d(pixel_label_loss(maps, tw)[:, None], su)[:, 0]    # (N, f, f)
            ML = torch.full((B, T, Hs, Hs), float("nan"), device=dev)
            for i in range(f):
                for j in range(f):
                    ML[b, t, py // su + i, px // su + j] = ml[:, i, j]
            avail = ~torch.isnan(ML)
            cand = avail.any(1)
            loc = cell if self.pick in ("oracle", "entropy", "entropy_r") else torch.rand_like(cell)
            loc = loc.masked_fill(~cand, NEG).flatten(1)
            rank = loc.argsort(1, descending=True).argsort(1)
            loc = ((rank < self.n) & torch.isfinite(loc)).view_as(cand)
            key = ML if self.pick in ("oracle", "entropy", "r_attention") else torch.rand_like(ML)
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


class Route:
    """Route each crop to ps8 or ps4 (s16m8): one per-pixel prediction entropy map of the
    crop-free pass, pooled over the 8 x 8 ps8 windows and the 16 x 16 ps4 cells; n crops picked
    greedily by value per token, a ps8 window worth its entropy mass (64 px x mean), a ps4 cell
    alpha x its mass (16 px x mean), no two crops overlapping (a region is cropped at ONE
    resolution). alpha 1 never prefers a ps4 cell over its own window; alpha 4 ranks both by mean
    entropy (density); inf: ps4 cells only. Each crop then takes its season(s) the readout
    attends to most (ReadoutAttention + EveryLocation keep one / half), all their months."""

    def __init__(self, n: int, alpha: float, keep: str):
        self.n, self.alpha = n, alpha
        self.geo = Geometry(LOOKUP_PS)
        self.scorer, self.allocator = ReadoutAttention(), EveryLocation(keep)
        g8, g4 = TILE // 8, TILE // LOOKUP_PS
        w = torch.arange(g8 * g8)
        c = torch.arange(g4 * g4)
        cw = (c // g4 // 2) * g8 + (c % g4) // 2            # the window over each cell
        blk = torch.zeros(g8 * g8 + g4 * g4, g8 * g8 + g4 * g4, dtype=torch.bool)
        blk[w, w] = True
        blk[g8 * g8 + c, g8 * g8 + c] = True
        blk[cw, g8 * g8 + c] = True                         # a window blocks its cells
        blk[g8 * g8 + c, cw] = True                         # a cell blocks its window
        self.blk, self.g8, self.g4 = blk, g8, g4
        self.last = {}

    def modules(self) -> list:
        return []

    def __call__(self, ctx: Context) -> Selection:
        out0 = ctx.out0
        dev = ctx.present.device
        g8, g4, B = self.g8, self.g4, ctx.B
        with torch.no_grad():
            e = pixel_entropy(out0.logits.detach().float())[:, None]          # (B,1,64,64)
            v8 = (64 * F.avg_pool2d(e, 8)).flatten(1)
            v4 = (16 * F.avg_pool2d(e, LOOKUP_PS)).flatten(1)
            if math.isinf(self.alpha):
                v8 = torch.full_like(v8, NEG)
            else:
                v4 = self.alpha * v4
            v = torch.cat([v8, v4], 1)                                         # (B, 64 + 256)
            blk = self.blk.to(dev)
            pick = torch.zeros_like(v, dtype=torch.bool)
            ar = torch.arange(B, device=dev)
            for _ in range(self.n):
                i = v.argmax(1)
                ok = torch.isfinite(v[ar, i])
                pick[ar[ok], i[ok]] = True
                v = v.masked_fill(blk[i] & ok[:, None], NEG)
            p8 = pick[:, :g8 * g8].view(B, g8, g8)
            p4 = pick[:, g8 * g8:].view(B, g4, g4)
            s_t = self.scorer(ctx, self.geo, self.allocator.needs)
            months = self.allocator(s_t, ctx, self.geo)                        # (B,T,16,16)
        m8 = months[:, :, ::2, ::2] & p8[:, None]
        m4 = months & p4[:, None]
        mask = m4 | (m8.repeat_interleave(2, 2).repeat_interleave(2, 3))
        n4 = float(p4.sum())
        self.last = {"ps4_share": n4 / max(n4 + float(p8.sum()), 1.0)}
        return Selection([Plan.from_grid(m8, 8), Plan.from_grid(m4, LOOKUP_PS)], mask, None,
                         s_t.aux)


def build_selector(cfg, model: MixedResModel, example=None):
    """manyup.mixres.config.SelectConfig -> Selector (or Level2). example: one batch's (monthly,
    season) samples, to size the HR residual arms' pixel embedding and season embedding."""
    geo = Geometry(cfg.crop_px, cfg.loc_stride)
    if cfg.route_alpha is not None:
        return Route(cfg.n, cfg.route_alpha, cfg.keep)
    if cfg.quadrant is not None:
        return Quadrant(Geometry(cfg.crop_px), cfg.quadrant, model.cfg.lr_patch, model.device)
    if cfg.time_allocator is not None:
        sp = _scorer(cfg.scorer, cfg, model, example,
                     role="both" if cfg.time_scorer == "lossup" else "space")
        sel = SpaceTime(geo, sp, _allocator(cfg.allocator, cfg),
                        sp if cfg.scorer == cfg.time_scorer == "lossup"
                        else _scorer(cfg.time_scorer or "none", cfg, model, example, role="time"),
                        _allocator(cfg.time_allocator, cfg))
    else:
        sel = Selector(geo, _scorer(cfg.scorer, cfg, model, example),
                       _allocator(cfg.allocator, cfg))
    if cfg.level2:
        sel = Level2(sel, cfg.level2.patch, cfg.level2.n, cfg.level2.pick, cfg.level2.source)
    return sel


def _scorer(name: str, cfg, model: MixedResModel, example, role: str = "both") -> Scorer:
    if name == "lossup":
        r = cfg.residual
        if example is None:
            raise ValueError("scorer lossup: build_selector needs an example batch")
        C = window_pixels(example[0], 1).shape[-1]
        net = LossUp(model.mix.encoder.embedding_size,
                     model.head.linear.out_features // LR_PS ** 2, C, mode=r.arch).to(model.device)
        return LossUpScorer(net, role=role, pixel=r.pixel)
    if name == "season_residual":
        r = cfg.residual
        if r.arch != "conv" and example is None:
            raise ValueError(f"season residual arch {r.arch}: build_selector needs an example")
        net = SeasonResidual(model.mix.encoder.embedding_size,
                             model.head.linear.out_features // LR_PS ** 2, arch=r.arch,
                             C=0 if r.arch == "conv" else window_pixels(example[0], 1).shape[-1],
                             P=cfg.crop_px, S=TILE // cfg.crop_px, gh=TILE // model.cfg.lr_patch,
                             NS=0 if example is None else example[1].timestamps.shape[1],
                             hidden=r.hidden, depth=r.depth, pix_enc=r.pix_enc,
                             month_pool=r.month_pool, season_mix=r.season_mix,
                             T=0 if example is None else window_pixels(example[0], 1).shape[3]
                             ).to(model.device)
        scorer = SeasonResidualScorer(net, train=r.train, tau=r.tau,
                                      lam=1.0 if r.train != "e2e" else 0.0, prior=r.prior)
    elif name == "entropy_sign":
        if example is None:
            raise ValueError("scorer entropy_sign: build_selector needs an example batch")
        net = SignNet(model.mix.encoder.embedding_size, example[1].timestamps.shape[1],
                      window_pixels(example[0], 1).shape[-1] * cfg.crop_px ** 2).to(model.device)
        scorer = EntropySign(net)
    elif name == "residual":
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
                          lam=1.0 if r.train != "e2e" else 0.0, prior=r.prior)
    else:
        if name == "entropy" and cfg.entropy_T and role != "space":
            if example is None:
                raise ValueError("select.entropy_T: build_selector needs an example batch")
            return Entropy(temp=SeasonTemperature(example[1].timestamps.shape[1],
                                                  online=cfg.entropy_T_fit == "train"
                                                  ).to(model.device))
        scorer = {"label": LabelLoss, "entropy": Entropy, "random": RandomWindows,
                  "none": NoScores,
                  "grid": lambda: GridWindows(cfg.n),
                  "season_grid": lambda: SeasonGrid(cfg.n),
                  "season_random": SeasonRandom, "attention": ReadoutAttention}[name]()
    return scorer


def _allocator(name: str, cfg) -> Allocator:
    return {"top": lambda: TopWindows(cfg.n), "pairs": lambda: SeasonPairs(cfg.n),
                 "season_top": lambda: SeasonTop(cfg.n),
                 "every_location": lambda: EveryLocation(cfg.keep, cfg.season_order),
                 "month_grid": lambda: MonthGrid(cfg.month_step),
                 "halves": lambda: SeasonHalves(cfg.n, cfg.keep)}[name]()
