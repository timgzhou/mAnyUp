"""Heads on per-timestep token features (B, T, gH, gW, D) -> per-pixel outputs (B, C, H, W).

Both are OlmoEarth's pa2px head (one linear map per token to C * p * p sub-pixel outputs,
unfolded to pixels) after a fixed reduction over time:
  time_mean  mean over the timesteps that have data (frame_valid), for a label that holds for
             the whole series (PASTIS crop type, BioMassters biomass)
  diff       post - pre (T must be 2), for a label that describes a change (GEOID-Flood)
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from manyup.data.features import time_mean  # noqa: F401  (re-exported)

REDUCTIONS = ("time_mean", "diff")


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
        self.linear = nn.Linear(embed_dim, num_outputs * patch_size ** 2)

    def forward(self, tokens: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
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
