"""mAnyUp: AnyUp retrained on our features with multispectral (any channel count) guidance.

The guidance encoder is a conv over channels, so the checkpoint fixes only the channel COUNT
(its input_dim); which bands and modalities fill them is the GuidanceSpec. from_checkpoint
rebuilds the spec the checkpoint was trained with (its "guidance" entry, or legacy
train_manyup's guidance_mod and time_pool); pass `guidance` to feed other modality names
(e.g. "S1" instead of PASTIS's "S1A") with the same channel count.

Output space: "input" for a transform_depth=0 checkpoint without projector (a pure
attention-weighted sum of the input features), else "target" (the feature space of the
checkpoint's hr_cfg, which it was trained to predict).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from manyup.data.guidance import LEGACY_GUIDANCE, GuidanceSpec, spec_from_dict
from manyup.upsamplers.base import FrameUpsampler


class MAnyUp(FrameUpsampler):
    def __init__(self, model: nn.Module, guidance: GuidanceSpec, proj: nn.Module | None = None,
                 output_space: str = "target", name: str = "manyup",
                 time_matching: str | None = None):
        super().__init__()
        tm = "pooled" if guidance.time_pool else (time_matching or "nearest")
        self._declare(f"{name}[{guidance.key}]", guidance, tm, output_space)
        self.model, self.proj = model, proj

    @classmethod
    def from_checkpoint(cls, path: str | Path, guidance: GuidanceSpec | None = None,
                        use_proj: bool = True, frozen: bool = True) -> "MAnyUp":
        """A train_manyup checkpoint (arch "manyup", or "anyup" = stock AnyUp trained by us)."""
        import anyup.model

        from manyup.mAnyUp import mAnyUp

        ck = torch.load(path, map_location="cpu", weights_only=False)
        args = ck.get("args", {})
        in_dim = ck["input_dim"]
        trained = (spec_from_dict(ck["guidance"]) if "guidance" in ck else   # exp.upsamplers.train
                   GuidanceSpec(LEGACY_GUIDANCE[args.get("guidance_mod", "s2")],
                                args.get("time_pool", "mean")))
        guidance = guidance or trained
        if guidance.channels != in_dim:
            raise ValueError(f"{Path(path).name} takes {in_dim} guidance channels, "
                             f"{guidance.key} has {guidance.channels}")
        if (guidance.time_pool, guidance.norm) != (trained.time_pool, trained.norm):
            raise ValueError(f"{Path(path).name} was trained on {trained.key} guidance; "
                             f"{guidance.key} pools or normalizes differently")
        kw = dict(input_dim=in_dim, qk_dim=ck.get("qk_dim", 128),
                  window_ratio=ck.get("window_ratio", 0.1))
        arch = ck.get("arch", "anyup")
        if arch == "manyup":
            kw.update(feat_dim=ck.get("feat_dim", 768),
                      transform_depth=ck.get("transform_depth", 1))
            model = mAnyUp(**kw)
        else:
            model = anyup.model.AnyUp(**kw)
        model.load_state_dict(ck["model"])
        proj = None
        if use_proj and ck.get("proj_head") is not None:
            D = ck["proj_head"]["weight"].shape[0]
            proj = nn.Conv2d(D, D, 1)
            proj.load_state_dict(ck["proj_head"])
        pure = arch == "anyup" or ck.get("transform_depth", 1) == 0
        up = cls(model, guidance, proj, "input" if pure and proj is None else "target",
                 name=f"manyup:{Path(path).stem}")
        if frozen:
            up.requires_grad_(False).eval()
        return up

    def upsample_frames(self, feats: torch.Tensor, guide: torch.Tensor,
                        size: tuple[int, int]) -> torch.Tensor:
        # full-resolution guidance whatever the output grid, as train_manyup feeds it
        out = self.model(guide.float(), feats.float(), size)
        return out if self.proj is None else self.proj(out)
