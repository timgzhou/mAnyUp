"""Heads on per-timestep token features (B, T, gH, gW, D) -> per-pixel outputs (B, C, H, W).

Both are OlmoEarth's pa2px head (one linear map per token to C * p * p sub-pixel outputs,
unfolded to pixels) after a fixed reduction over time:
  time_mean  mean over the timesteps that have data (frame_valid), for a label that holds for
             the whole series (PASTIS crop type, BioMassters biomass)
  diff       post - pre (T must be 2), for a label that describes a change (GEOID-Flood)
  pair_diff      for a series: every pair of consecutive dates (`pairs`) through
                 ONE shared linear map of (later - earlier), the per-pair logits pooled by
                 logsumexp over the sample's pairs (a soft "any pair shows it"). A location
                 where nothing changed gives the head's bias. (GEOID series, flood only)
  pair_diffmean  the same on [later - earlier, (later + earlier) / 2]: the mean half sees
                 what does not change (permanent water). (GEOID series, 3 classes)
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from manyup.data.features import time_mean  # noqa: F401  (re-exported)

REDUCTIONS = ("time_mean", "diff", "pair_diff", "pair_diffmean")


def consecutive_pairs(present: torch.Tensor, orbit: torch.Tensor | None) -> torch.Tensor:
    """(B, T) present steps (time-ordered), (B, T) orbit id or None -> (B, P, 2) long index
    pairs (earlier, later) of consecutive present steps (on the same orbit if `orbit` is
    given), padded with -1."""
    B, T = present.shape
    out = []
    for b in range(B):
        last, pairs = {}, []
        for t in range(T):
            if not present[b, t]:
                continue
            o = 0 if orbit is None else int(orbit[b, t])
            if o in last:
                pairs.append((last[o], t))
            last[o] = t
        out.append(pairs)
    P = max(1, max(len(p) for p in out))
    res = torch.full((B, P, 2), -1, dtype=torch.long)
    for b, pairs in enumerate(out):
        if pairs:
            res[b, :len(pairs)] = torch.tensor(pairs)
    return res


class Pa2PxHead(nn.Module):
    """norm=True standardizes each feature channel before the linear map (non-affine
    BatchNorm, the usual linear-probe choice, e.g. MAE): frozen features of small or uneven
    scale otherwise make the best learning rate depend on the feature source, and low rates
    barely move the head. Still linear in the features at eval (running statistics)."""

    def __init__(self, reduction: str, embed_dim: int, num_outputs: int, patch_size: int,
                 norm: bool = False):
        super().__init__()
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction {reduction!r} not in {REDUCTIONS}")
        self.reduction, self.num_outputs, self.patch_size = reduction, num_outputs, patch_size
        self.norm = nn.BatchNorm1d(embed_dim, affine=False) if norm else None
        d_in = 2 * embed_dim if reduction == "pair_diffmean" else embed_dim
        self.linear = nn.Linear(d_in, num_outputs * patch_size ** 2)

    def forward(self, tokens: torch.Tensor, frame_valid: torch.Tensor,
                pairs: torch.Tensor | None = None) -> torch.Tensor:
        if self.reduction.startswith("pair_"):
            return self._pairs(tokens, pairs)
        if self.reduction == "diff":
            if tokens.shape[1] != 2 or not frame_valid.all():
                raise ValueError("diff head needs exactly 2 valid timesteps (pre, post)")
            x = tokens[:, 1] - tokens[:, 0]
        else:
            x = time_mean(tokens, frame_valid)
        if self.norm is not None:
            x = self.norm(x.flatten(0, 2).float()).view(x.shape)
        p = self.patch_size
        return rearrange(self.linear(x), "b gh gw (c i j) -> b c (gh i) (gw j)",
                         c=self.num_outputs, i=p, j=p)

    def _pairs(self, tokens, pairs):
        if pairs is None:
            raise ValueError(f"{self.reduction} needs the date pairs (consecutive_pairs)")
        B = tokens.shape[0]
        pairs = pairs.to(tokens.device)
        ok = (pairs >= 0).all(-1)                                              # (B, P)
        bi = torch.arange(B, device=tokens.device)[:, None]
        early, late = tokens[bi, pairs[..., 0].clamp(min=0)], tokens[bi, pairs[..., 1].clamp(min=0)]
        x = late - early                                                       # (B,P,gh,gw,D)
        if self.reduction == "pair_diffmean":
            x = torch.cat([x, (late + early) / 2], -1)
        p = self.patch_size
        logits = rearrange(self.linear(x), "b q gh gw (c i j) -> b q c (gh i) (gw j)",
                           c=self.num_outputs, i=p, j=p).float()
        logits = logits.masked_fill(~ok[:, :, None, None, None], float("-inf"))
        none = ~ok.any(1)                                  # a sample without a pair: bias only
        pooled = torch.logsumexp(logits.masked_fill(none[:, None, None, None, None], 0.0), 1)
        bias = self.linear.bias.float().view(self.num_outputs, p, p).repeat(
            1, tokens.shape[2], tokens.shape[3])
        return torch.where(none[:, None, None, None], bias, pooled)
