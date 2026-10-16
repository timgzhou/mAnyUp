"""Seasonal timAnyUp: upsample every MONTH from its SEASON's low-res tokens, then spend a budget
of high-res lookups where a learned selector expects them to pay off most.

    lr      seasonal low-res features (GroupSteps cache), G steps       (B, G, h, w, D)
    up      each month upsampled by a frozen per-timestep mAnyUp from its season's tokens,
            guided by the month's own image; A = its attention            (B, T, H, W, D)
    hr      high-res low-context features of the month (the lookup)      (B, T, H, W, D)
    fused   up + fuser(up, hr), used at the selected locations only

Why the selector regresses GAIN and nothing fancier: the fuser is per location, so the
reconstruction loss after picking a set S of locations is
    sum_{x not in S} err_up(x) + sum_{x in S} err_fused(x) = const - sum_{x in S} gain(x),
gain = err_up - err_fused. Top-k by gain is exactly the loss-minimizing selection, so the
selector only has to RANK locations by gain -- no relaxed top-k needed. And the fuser, being
per location, can be trained densely on every location, for any selection.

Components are swappable: any Selector maps a SeasonalContext to (B, T, H, W) scores.
A month is only upsampled if it has guidance of its own dates (guidance_present) and its
season has data; `present` marks those, and nothing else is scored, selected or counted.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

EPS = 1e-8


# -- months <-> seasons --------------------------------------------------------------------
def season_index(lr_start: torch.Tensor, lr_end: torch.Tensor, out_start: torch.Tensor,
                 out_end: torch.Tensor) -> torch.Tensor:
    """(B, T) index of the lr step whose [start, end] contains each output step's midpoint;
    -1 where none does."""
    mid = (out_start + out_end).double()[:, :, None] / 2                     # (B, T, 1)
    inside = (lr_start[:, None, :] <= mid) & (mid <= lr_end[:, None, :])     # (B, T, G)
    return torch.where(inside.any(-1), inside.double().argmax(-1),
                       torch.full_like(out_start, -1)).long()


def seasons_used(season: torch.Tensor, present: torch.Tensor, n_seasons: int) -> torch.Tensor:
    """(B, n_seasons) bool: the season holds at least one present step. (Not a scatter_ of
    `present`: a padded step writes False over the real steps of the same season.)"""
    real = present & (season >= 0)
    hot = torch.nn.functional.one_hot(season.clamp(min=0), n_seasons).bool()   # (B, T, S)
    return (hot & real[..., None]).any(1)


def season_mean(x: torch.Tensor, season: torch.Tensor, present: torch.Tensor,
                G: int) -> torch.Tensor:
    """Mean of x (B, T, ...) over the PRESENT months of each month's season -> same shape."""
    B, T = season.shape
    w = torch.zeros(B, T, G, device=x.device, dtype=x.dtype)
    w.scatter_(2, season.clamp(min=0)[..., None], present[..., None].to(x.dtype))
    flat = x.reshape(B, T, -1)
    per_g = torch.einsum("btg,btf->bgf", w, flat) / w.sum(1)[..., None].clamp(min=1)
    return per_g.gather(1, season.clamp(min=0)[..., None].expand(B, T, flat.shape[-1])
                        ).reshape(x.shape)


# -- attention -----------------------------------------------------------------------------
@contextmanager
def capture_attention(model):
    """AnyUp computes attn (features = attn @ v) but only returns it when store_attn is set,
    which CrossAttentionBlock never passes down. Force it with hooks rather than editing the
    vendored repo. Yields a dict whose "attn" is (N, H*W, h*w), head-averaged."""
    store = {}
    ca = model.cross_decode.cross_attn

    def pre(_, args, kwargs):
        return args, {**kwargs, "store_attn": True}

    def post(_, args, kwargs, out):
        store["attn"] = out[1]

    hs = [ca.register_forward_pre_hook(pre, with_kwargs=True),
          ca.register_forward_hook(post, with_kwargs=True)]
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


def upsample_with_attention(model, feats, guide, size):
    """feats (N,D,h,w), guide (N,C,H,W) -> (out (N,D,*size), attn (N, prod(size), h*w))."""
    with capture_attention(model) as st:
        out = model(guide.float(), feats.float(), size)
    return out, st["attn"].float()


def kl(p, q):
    """KL(p || q) over the last dim."""
    p, q = p.clamp_min(EPS), q.clamp_min(EPS)
    return (p * (p.log() - q.log())).sum(-1)


def entropy(p):
    p = p.clamp_min(EPS)
    return -(p * p.log()).sum(-1)


# -- context every selector reads ------------------------------------------------------------
@dataclass
class SeasonalContext:
    up: torch.Tensor            # (B, T, D, H, W) months upsampled from their season's tokens
    attn: torch.Tensor          # (B, T, H*W, h*w) their attention
    attn_season: torch.Tensor   # (B, T, H*W, h*w) the mean over the season's present months
    present: torch.Tensor       # (B, T) bool: upsampled months

    @property
    def grid(self) -> tuple[int, int]:
        return tuple(self.up.shape[-2:])


def attention_features(ctx: SeasonalContext) -> torch.Tensor:
    """(B, T, H, W, 10) per-location summaries of how month t's attention departs from its
    season's. Deliberately invariant to the ORDER of the lr tokens: an attention row is
    indexed by absolute lr token, and "attends to token 5" means something different at every
    query location, so a shared per-location head should not read raw rows."""
    A, Ab = ctx.attn, ctx.attn_season
    B, T = A.shape[:2]
    H, W = ctx.grid
    k_fwd, k_bwd = kl(A, Ab), kl(Ab, A)
    f = torch.stack([
        k_fwd.log1p(), k_bwd.log1p(), entropy(A), entropy(Ab),
        A.amax(-1), Ab.amax(-1), (A - Ab).abs().sum(-1),
        (A.argmax(-1) != Ab.argmax(-1)).float(),
        k_fwd.log1p().mean(-1, keepdim=True).expand_as(k_fwd),        # "this month is odd"
        ctx.present.float()[..., None].expand_as(k_fwd),
    ], -1)
    return f.reshape(B, T, H, W, -1)


# -- selectors: SeasonalContext -> (B, T, H, W) scores; higher = look up first --------------
class Selector(nn.Module):
    learned = True

    def forward(self, ctx: SeasonalContext) -> torch.Tensor:
        raise NotImplementedError


def _mlp(d_in, hidden):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                         nn.GELU(), nn.Linear(hidden, 1))


class AttentionSelector(Selector):
    """Per-location MLP on attention_features: a few thousand parameters, independent of the
    feature dim D and of T."""
    N_FEATURES = 10

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(self.N_FEATURES)
        self.net = _mlp(self.N_FEATURES, hidden)

    def forward(self, ctx):
        return self.net(self.norm(attention_features(ctx).detach()))[..., 0]


class ContentSelector(Selector):
    """Per-location MLP on the upsampled feature itself (what the selection could see without
    attention): the content baseline."""

    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.net = _mlp(feat_dim, hidden)

    def forward(self, ctx):
        return self.net(self.norm(ctx.up.detach().permute(0, 1, 3, 4, 2)))[..., 0]


class KLMeanSelector(Selector):
    """The untrained heuristic: KL(A_month || A_season-mean)."""
    learned = False

    def forward(self, ctx):
        B, T = ctx.attn.shape[:2]
        return kl(ctx.attn, ctx.attn_season).reshape(B, T, *ctx.grid)


class RandomSelector(Selector):
    learned = False

    def forward(self, ctx):
        B, T = ctx.attn.shape[:2]
        return torch.rand(B, T, *ctx.grid, device=ctx.attn.device)


# -- fuser ---------------------------------------------------------------------------------
class ResidualFuser(nn.Module):
    """up + g([up, hr]), g a pointwise map 2D -> D (linear, or an MLP with `hidden`), its last
    layer zero-initialized so fusing starts as the identity on up."""

    def __init__(self, feat_dim: int, hidden: int = 0):
        super().__init__()
        D = feat_dim
        layers = ([nn.Conv2d(2 * D, D, 1)] if not hidden else
                  [nn.Conv2d(2 * D, hidden, 1), nn.GELU(), nn.Conv2d(hidden, D, 1)])
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.g = nn.Sequential(*layers)

    def forward(self, up: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """(B, T, D, H, W) each -> (B, T, D, H, W)."""
        B, T = up.shape[:2]
        x = torch.cat([up, hr], 2).flatten(0, 1)
        return up + self.g(x).reshape(up.shape)


def select(scores: torch.Tensor, present: torch.Tensor, k: int) -> torch.Tensor:
    """Flat top-k over (T, H, W) per sample, never a month that was not upsampled."""
    from manyup.timAnyUp import topk_mask
    where = present[:, :, None, None].expand_as(scores)
    return topk_mask(scores.float().masked_fill(~where, float("-inf")), k) & where
