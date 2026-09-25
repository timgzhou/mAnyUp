"""FeatureSeries: what every encoder emits and every feature upsampler consumes and returns.

Features keep an explicit time axis (T=1 when an encoder collapsed time) with the date range
each step covers, and the ground geometry of the token grid, so upsamplers and probes can
match features to guidance and labels by date and location instead of by position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from manyup.data.geometry import Geometry


@dataclass
class FeatureSeries:
    ids: list[str]                        # sample ids, the join key across streams
    tokens: torch.Tensor                  # (B, T, gH, gW, D)
    t_start: torch.Tensor                 # (B, T) int64 days (start of what step t covers)
    t_end: torch.Tensor                   # (B, T)
    time_precision: str
    frame_valid: torch.Tensor             # (B, T) bool
    geometry: list[Geometry]              # per sample, the (gH, gW) token grid on the ground
    provenance: dict = field(default_factory=dict)  # encoder, data key, temporal_mode, ...

    def __post_init__(self):
        B, T, gH, gW, _ = self.tokens.shape
        for name, t in (("t_start", self.t_start), ("t_end", self.t_end),
                        ("frame_valid", self.frame_valid)):
            if tuple(t.shape) != (B, T):
                raise ValueError(f"{name} shape {tuple(t.shape)} != {(B, T)}")
        if len(self.ids) != B or len(self.geometry) != B:
            raise ValueError("ids / geometry need one entry per sample")
        if any(g.shape != (gH, gW) for g in self.geometry):
            raise ValueError(f"geometry shapes must equal the token grid {(gH, gW)}")

    @property
    def dim(self) -> int:
        return self.tokens.shape[-1]


def time_mean(tokens: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    """(B, T, gH, gW, D), (B, T) bool -> (B, gH, gW, D), averaging valid steps only."""
    w = frame_valid.to(tokens.dtype).view(*frame_valid.shape, 1, 1, 1)
    return (tokens * w).sum(1) / w.sum(1).clamp(min=1)
