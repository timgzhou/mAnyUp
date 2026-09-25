"""AnyUp (wimmerth/anyup), the pretrained multi-backbone model, frozen.

Its weights expect a natural-colour RGB image, min-max normalized per image and then
ImageNet-standardized, so the guidance must have exactly 3 channels in R, G, B order.
Outputs are attention-weighted combinations of the input features ("input" space).
"""
from __future__ import annotations

import torch

from manyup.data.guidance import S2_RGB, GuidanceSpec
from manyup.upsamplers.base import FrameUpsampler

WEIGHTS = ("https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/"
           "anyup_multi_backbone.pth")
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_norm(rgb01: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) in [0, 1] -> ImageNet-standardized."""
    return (rgb01 - _IMAGENET_MEAN.to(rgb01.device)) / _IMAGENET_STD.to(rgb01.device)


class AnyUp(FrameUpsampler):
    def __init__(self, time_pool: str | None = "mean", rgb: dict | None = None,
                 time_matching: str | None = None):
        """time_pool "mean"/"median": one pooled RGB composite and time-pooled features (the
        legacy `anyup` head); None: every feature step with its own date-matched RGB frame
        (`anyup_t1`), matched by `time_matching` (default nearest). rgb: {modality: 3 bands},
        default S2 B04/B03/B02."""
        super().__init__()
        from anyup.model import AnyUp as _AnyUp            # third_party/anyup (manyup/__init__)
        spec = GuidanceSpec(rgb or {"S2": S2_RGB}, time_pool)
        if spec.channels != 3:
            raise ValueError(f"AnyUp's pretrained weights take 3 channels, got {spec.key}")
        tm = "pooled" if time_pool else (time_matching or "nearest")
        self._declare(f"anyup[{spec.key}]", spec, tm, "input")
        self.model = _AnyUp()
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(
            WEIGHTS, map_location="cpu", progress=False))
        self.model.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()                                   # frozen: never in train mode
        return self

    def upsample_frames(self, feats: torch.Tensor, guide: torch.Tensor,
                        size: tuple[int, int]) -> torch.Tensor:
        return self.model(imagenet_norm(guide.float()), feats.float(), output_size=size)
