"""FeatureUpsampler: low-res features + guidance -> features on a finer grid.

Upsamplers do not see images on their own; they refine an encoder's FeatureSeries using
guidance. Both sides may be multimodal and multitemporal, and they need NOT share modalities
or timesteps (e.g. monthly S2+S1 features guided by per-date S2). What an upsampler reads is
declared, not implied by its class:

  guidance       a GuidanceSpec (modalities, bands by name, normalization, time_pool), or
                 None for an upsampler that reads no image (bilinear). Single- vs
                 multi-modal guidance is just a longer spec.
  time_matching  "pooled": features are averaged over their valid steps and guidance is
                 pooled by spec.time_pool -> one output step (T = 1).
                 "nearest" | "within_range": every feature step is upsampled with guidance
                 picked BY DATE for that step (manyup.data.guidance.MATCHINGS); T is kept.
                 Pooled iff the spec has a time_pool. A step that some guidance modality
                 has no frame for (guidance_present) is NOT upsampled: its output tokens are
                 zero and its frame_valid False, so heads and losses skip it rather than
                 read a step guided by another date's image.
  side_features  extra FeatureSeries the upsampler reads, by role name (timAnyUp's
                 high-res low-context arm "hr"); passed to forward as keyword arguments.
  output_space   "input": outputs are (convex) combinations of the input features, so a
                 probe fitted on the low-res features applies unchanged. "target": a learned
                 map moved them to another feature space (the space it was trained to predict).

FrameUpsampler is the template for upsamplers that treat every (feature map, guidance image)
pair independently (bilinear, AnyUp, UPA/UPMA, mAnyUp): the base does the time handling and
the FeatureSeries bookkeeping, a subclass implements upsample_frames only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from manyup.data.dataset import MTMMDataset
from manyup.data.features import FeatureSeries, time_mean
from manyup.data.geometry import Geometry
from manyup.data.guidance import (MATCHINGS, GuidanceSpec, build_guidance_batch,
                                  guidance_present)
from manyup.data.sample import MTMMBatch

TIME_MATCHING = ("pooled",) + MATCHINGS
OUTPUT_SPACES = ("input", "target")


class FeatureUpsampler(nn.Module, ABC):
    key: str
    guidance: GuidanceSpec | None
    time_matching: str                      # one of TIME_MATCHING
    side_features: tuple[str, ...] = ()
    output_space: str                       # one of OUTPUT_SPACES

    def _declare(self, key: str, guidance: GuidanceSpec | None, time_matching: str,
                 output_space: str) -> None:
        """Set and validate the declarations (call from __init__)."""
        if time_matching not in TIME_MATCHING:
            raise ValueError(f"time_matching {time_matching!r} not in {TIME_MATCHING}")
        if output_space not in OUTPUT_SPACES:
            raise ValueError(f"output_space {output_space!r} not in {OUTPUT_SPACES}")
        if guidance is not None and (guidance.time_pool is None) == (time_matching == "pooled"):
            raise ValueError(f"{key}: guidance time_pool={guidance.time_pool!r} does not fit "
                             f"time_matching={time_matching!r} (pooled iff a time_pool is set)")
        self.key, self.guidance = key, guidance
        self.time_matching, self.output_space = time_matching, output_space

    @abstractmethod
    def forward(self, lr: FeatureSeries, guidance: MTMMBatch | None,
                target: list[Geometry], **side: FeatureSeries) -> FeatureSeries:
        """Return features on `target` (one Geometry per sample, all the same shape), same ids
        as `lr`; same time axis, or one pooled step when time_matching is "pooled"."""

    # -- checks ---------------------------------------------------------------------------
    def check(self, dataset: MTMMDataset) -> None:
        """Fail at setup, before any data is loaded, if the dataset cannot give the guidance."""
        if self.guidance is None:
            return
        spec = dataset.SPEC
        for mod, entries in self.guidance.bands.items():
            if mod not in dataset.modalities:
                raise ValueError(f"{self.key} needs guidance modality {mod!r}; dataset gives "
                                 f"{sorted(dataset.modalities)}")
            have = spec.modalities[mod].bands
            missing = sorted({e.split("<-")[-1] for e in entries} - set(have))
            if missing:
                raise ValueError(f"{self.key}: {spec.name}/{mod} lacks bands {missing}")

    def check_inputs(self, lr: FeatureSeries, guidance: MTMMBatch | None,
                     target: list[Geometry], side: dict[str, FeatureSeries]) -> tuple[int, int]:
        """Validate one call; returns the output (H, W)."""
        if (guidance is None) != (self.guidance is None):
            raise ValueError(f"{self.key} " + ("needs guidance" if self.guidance
                                               else "reads no guidance; pass None"))
        if guidance is not None and list(guidance.ids) != list(lr.ids):
            raise ValueError(f"{self.key}: guidance ids differ from feature ids")
        if set(side) != set(self.side_features):
            raise ValueError(f"{self.key} takes side features {self.side_features}, "
                             f"got {tuple(side)}")
        for role, f in side.items():
            if list(f.ids) != list(lr.ids):
                raise ValueError(f"{self.key}: side features {role!r} ids differ from lr")
            if not (torch.equal(f.t_start, lr.t_start) and torch.equal(f.t_end, lr.t_end)):
                raise ValueError(f"{self.key}: side features {role!r} cover other dates than lr")
        if len(target) != len(lr.ids):
            raise ValueError(f"{self.key}: one target geometry per sample needed")
        shapes = {g.shape for g in target}
        if len(shapes) != 1:
            raise ValueError(f"{self.key}: target shapes differ within the batch {shapes}")
        return next(iter(shapes))

    # -- helpers for subclasses -----------------------------------------------------------
    def build_guidance(self, guidance: MTMMBatch, lr: FeatureSeries) -> torch.Tensor:
        """(B, C, H, W) pooled, or (B, T, C, H, W) with one frame per feature step."""
        if self.time_matching == "pooled":
            return build_guidance_batch(guidance, self.guidance)
        return build_guidance_batch(guidance, self.guidance, match=self.time_matching,
                                    f_start=lr.t_start, f_end=lr.t_end)

    def guidance_present(self, guidance: MTMMBatch | None, lr: FeatureSeries) -> torch.Tensor:
        """(B, T') bool: the output steps that have guidance of their own dates (T' = 1 when
        pooled: pooled guidance is a reduction over whatever frames have data)."""
        B, T = lr.tokens.shape[:2]
        if self.time_matching == "pooled":
            return torch.ones(B, 1, dtype=torch.bool)
        if guidance is None:
            return torch.ones(B, T, dtype=torch.bool)
        return guidance_present(guidance, self.guidance, lr.t_start, lr.t_end)

    def output(self, lr: FeatureSeries, tokens: torch.Tensor, target: list[Geometry],
               present: torch.Tensor | None = None) -> FeatureSeries:
        """Wrap (B, T, H, W, D) outputs; T = 1 means the steps of lr were pooled. `present`
        (B, T) marks the steps that were upsampled; the rest become invalid."""
        t_start, t_end, valid = lr.t_start, lr.t_end, lr.frame_valid
        if tokens.shape[1] == 1 and lr.tokens.shape[1] != 1:
            big = torch.iinfo(t_start.dtype).max
            t_start = t_start.masked_fill(~valid, big).amin(1, keepdim=True)
            t_end = t_end.masked_fill(~valid, -1).amax(1, keepdim=True)
            valid = valid.any(1, keepdim=True)
        if present is not None:
            valid = valid & present.to(valid.device)
        prov = {**lr.provenance, "upsampler": self.key, "space": self.output_space,
                "time_matching": self.time_matching}
        return FeatureSeries(ids=list(lr.ids), tokens=tokens, t_start=t_start, t_end=t_end,
                             time_precision=lr.time_precision, frame_valid=valid,
                             geometry=list(target), provenance=prov)


class FrameUpsampler(FeatureUpsampler):
    """Upsamples every (feature map, guidance image) pair on its own; time is folded into
    the batch. Subclasses implement upsample_frames.

    Only the steps with guidance of their own dates reach upsample_frames; during that call
    `_frames` is the (B * T,) bool selection, for subclasses that keep per-frame state in
    batch order (JBU's kernel cache)."""

    _frames: torch.Tensor | None = None

    @abstractmethod
    def upsample_frames(self, feats: torch.Tensor, guide: torch.Tensor | None,
                        size: tuple[int, int]) -> torch.Tensor:
        """(N, D, h, w) features + (N, C, H, W) guidance (or None) -> (N, D', *size)."""

    def forward(self, lr: FeatureSeries, guidance: MTMMBatch | None,
                target: list[Geometry], **side: FeatureSeries) -> FeatureSeries:
        size = self.check_inputs(lr, guidance, target, side)
        tok = lr.tokens
        if self.time_matching == "pooled":
            tok = time_mean(tok.float(), lr.frame_valid.to(tok.device))[:, None]
        B, T, h, w, D = tok.shape
        present = self.guidance_present(guidance, lr)                  # (B, T)
        sel = present.flatten().to(tok.device)
        g = None
        if guidance is not None:
            g = self.build_guidance(guidance, lr).to(tok.device)
            g = g.reshape(B * T, *g.shape[-3:])[sel]     # pooled (B,C,H,W) has T == 1
        # contiguous: a channels-last view sends convs down another kernel (rounding differs)
        feats = tok.permute(0, 1, 4, 2, 3).reshape(B * T, D, h, w).contiguous()
        if bool(sel.all()):
            out = self._upsample_selected(feats, g, size, None)       # (B*T, D', H, W)
        else:
            part = self._upsample_selected(feats[sel], g, size, sel)
            out = part.new_zeros(B * T, *part.shape[1:])
            out[sel] = part
        out = out.reshape(B, T, *out.shape[1:]).permute(0, 1, 3, 4, 2)
        return self.output(lr, out, target, present)

    def _upsample_selected(self, feats, g, size, sel):
        self._frames = sel
        try:
            return self.upsample_frames(feats, g, size)
        finally:
            self._frames = None
