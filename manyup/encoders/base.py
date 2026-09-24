"""Encoder: an image(-time-series) model that turns an MTMMBatch into a FeatureSeries.

Each encoder declares what it needs (Requirements) and how it treats time:
  - TEMPORAL_MODES: which of "series" (the whole time series in one pass, so timesteps can
    attend to each other) and "per_timestep" (every frame encoded independently) it supports;
  - KEEPS_TIME: whether "series" output keeps one feature per input step (OlmoEarth) or
    collapses time to T=1 (UTAE).
`check()` compares the requirements with a dataset configuration before any data is loaded,
so a mismatch fails at setup, not mid-run. Normalization and packing into the model's own
input format are the encoder's job; datasets deliver physical units.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn

from manyup.data.dataset import MTMMDataset
from manyup.data.features import FeatureSeries
from manyup.data.sample import MTMMBatch

TEMPORAL_MODES = ("series", "per_timestep")


@dataclass(frozen=True)
class Requirements:
    bands: dict[str, tuple[str, ...]]   # modality -> bands the encoder reads (by name)
    aligned_time: bool                  # all timed modalities must share one time axis
    max_frames: int | None              # per modality; None = no limit
    size_multiple: int                  # H and W must be multiples of this


class Encoder(nn.Module, ABC):
    TEMPORAL_MODES: tuple[str, ...]
    KEEPS_TIME: bool
    key: str

    @abstractmethod
    def requirements(self) -> Requirements: ...

    @abstractmethod
    def encode(self, batch: MTMMBatch, temporal_mode: str) -> FeatureSeries: ...

    def check(self, dataset: MTMMDataset, temporal_mode: str) -> None:
        if temporal_mode not in self.TEMPORAL_MODES:
            raise ValueError(f"{self.key} supports temporal_mode {self.TEMPORAL_MODES}, "
                             f"got {temporal_mode!r}")
        req, spec = self.requirements(), dataset.SPEC
        for mod, bands in req.bands.items():
            if mod not in dataset.modalities:
                raise ValueError(f"{self.key} needs modality {mod!r}; dataset gives "
                                 f"{sorted(dataset.modalities)}")
            missing = set(bands) - set(spec.modalities[mod].bands)
            if missing:
                raise ValueError(f"{self.key}: {spec.name}/{mod} lacks bands {sorted(missing)}")
        size = dataset.tiling.size if dataset.tiling else min(spec.native_shape)
        if size % req.size_multiple:
            raise ValueError(f"{self.key}: sample size {size} not a multiple of "
                             f"{req.size_multiple}")
