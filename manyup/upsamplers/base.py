"""FeatureUpsampler: low-res features + guidance -> features on a finer grid.

Upsamplers do not see images on their own; they refine an encoder's FeatureSeries using
guidance. Both sides may be multimodal and multitemporal, and they need NOT share modalities
or timesteps (e.g. monthly S2+S1 features guided by per-date S2). How feature steps are
matched to guidance frames is the upsampler's declared `time_matching`, applied by date --
never by position.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from manyup.data.features import FeatureSeries
from manyup.data.geometry import Geometry
from manyup.data.sample import MTMMBatch

TIME_MATCHING = ("nearest", "within_range", "pooled")


class FeatureUpsampler(nn.Module, ABC):
    key: str
    time_matching: str                          # one of TIME_MATCHING
    guidance_bands: dict[str, tuple[str, ...]]  # guidance modality -> bands it reads

    @abstractmethod
    def forward(self, lr: FeatureSeries, guidance: MTMMBatch,
                target: list[Geometry]) -> FeatureSeries:
        """Return features on `target` (one Geometry per sample), same ids and time axis
        as `lr`."""

    def check_guidance(self, guidance: MTMMBatch) -> None:
        for mod, bands in self.guidance_bands.items():
            if mod not in guidance.modalities:
                raise ValueError(f"{self.key} needs guidance modality {mod!r}")
            missing = set(bands) - set(guidance.modalities[mod].bands)
            if missing:
                raise ValueError(f"{self.key}: guidance {mod} lacks bands {sorted(missing)}")
