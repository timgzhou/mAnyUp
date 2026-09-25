"""Bicubic: the no-guidance baseline. Outputs stay in the input feature space."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from manyup.upsamplers.base import FrameUpsampler


class Bicubic(FrameUpsampler):
    def __init__(self, pooled: bool = False, align_corners: bool = True):
        """pooled: average the feature steps first (one output step); else every step is
        upsampled. align_corners=True, as for Bilinear."""
        super().__init__()
        self._declare("bicubic", None, "pooled" if pooled else "nearest", "input")
        self.align_corners = align_corners

    def upsample_frames(self, feats: torch.Tensor, guide: None,
                        size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(feats.float(), size=size, mode="bicubic",
                             align_corners=self.align_corners)
