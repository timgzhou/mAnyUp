"""timAnyUp as a FeatureUpsampler (the model: manyup/timAnyUp.py).

    lr   F_lrhc  low-res, high-context features (e.g. ps16 tile64, series)     the input
    hr   F_hrlc  high-res, low-context features (e.g. ps4 tile4, per timestep)  side feature
    ->   F_hrhc  estimate on hr's grid, in the target feature space ("target")

Every lr step is upsampled with its own date-matched guidance frame, the top-k locations the
query head flags are blended with the transformed hr features, and the projector maps the
result into the target space. `hr` must cover the same samples and dates as `lr`, on the
output grid. The query head is Linear(T*C -> T): a checkpoint fixes T (num_frames).

forward is the inference path; the trainer (exp/upsamplers/train_timanyup.py) calls the
model's stages (upsample / query / transform / project) itself for its losses.
"""
from __future__ import annotations

from pathlib import Path

import torch

from manyup.data.features import FeatureSeries
from manyup.data.geometry import Geometry
from manyup.data.guidance import LEGACY_GUIDANCE, GuidanceSpec, spec_from_dict
from manyup.data.sample import MTMMBatch
from manyup.timAnyUp import TimAnyUp as _TimAnyUp
from manyup.timAnyUp import blend, topk_mask
from manyup.upsamplers.base import FeatureUpsampler


class TimAnyUp(FeatureUpsampler):
    side_features = ("hr",)

    def __init__(self, model: _TimAnyUp, guidance: GuidanceSpec, k: int,
                 name: str = "timanyup", time_matching: str = "nearest"):
        super().__init__()
        if guidance.time_pool is not None:
            raise ValueError("timAnyUp upsamples every step with its own guidance frame")
        if guidance.channels != model.upsampler.image_encoder[0].in_channels:
            raise ValueError(f"model takes {model.upsampler.image_encoder[0].in_channels} "
                             f"guidance channels, {guidance.key} has {guidance.channels}")
        self._declare(f"{name}[{guidance.key}]", guidance, time_matching, "target")
        self.model, self.k = model, k

    @classmethod
    def from_checkpoint(cls, path: str | Path, guidance: GuidanceSpec | None = None,
                        k: int | None = None, frozen: bool = True) -> "TimAnyUp":
        """A train_timanyup checkpoint. k defaults to the TRAINED lookup budget."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        model = _TimAnyUp(input_dim=ck["input_dim"], qk_dim=a.get("qk_dim", 128),
                          feat_dim=ck["feat_dim"], num_frames=ck["num_frames"],
                          transform_depth=a.get("transform_depth", 0),
                          window_ratio=a.get("window_ratio", 1.0),
                          query_input=a.get("query_input", "bilinear"))
        model.load_state_dict(ck["model"])
        trained = (spec_from_dict(ck["guidance"]) if "guidance" in ck else
                   GuidanceSpec(LEGACY_GUIDANCE[a.get("guidance_mod", "s2")], None))
        guidance = guidance or trained
        up = cls(model, guidance, k if k is not None else a.get("k", 512),
                 name=f"timanyup:{Path(path).stem}")
        if frozen:
            up.requires_grad_(False).eval()
        return up

    def forward(self, lr: FeatureSeries, guidance: MTMMBatch | None,
                target: list[Geometry], *, hr: FeatureSeries) -> FeatureSeries:
        size = self.check_inputs(lr, guidance, target, {"hr": hr})
        T = lr.tokens.shape[1]
        if T != self.model.num_frames:
            raise ValueError(f"{self.key}: the query head was built for T="
                             f"{self.model.num_frames}, got T={T}")
        if tuple(hr.tokens.shape[2:4]) != tuple(size):
            raise ValueError(f"{self.key}: hr grid {tuple(hr.tokens.shape[2:4])} must be the "
                             f"output grid {tuple(size)} (the blend is per location)")
        lrhc = lr.tokens.float().permute(0, 1, 4, 2, 3).contiguous()     # (B, T, D, h, w)
        hrlc = hr.tokens.float().permute(0, 1, 4, 2, 3).contiguous().to(lrhc.device)
        guide = self.build_guidance(guidance, lr).to(lrhc.device)        # (B, T, C, H, W)
        up = self.model.upsample(lrhc, guide, size)
        mask = topk_mask(self.model.query(lrhc, size, lrhc_up=up).float(), self.k)
        fused = self.model.project(blend(up, self.model.transform(hrlc), mask))
        return self.output(lr, fused.permute(0, 1, 3, 4, 2), target)
