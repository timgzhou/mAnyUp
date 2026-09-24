"""One sample / one batch of multimodal, multitemporal data, and the shared collate.

Each modality keeps its OWN time axis: S2 and S1 of the same sample may have different frame
counts and dates. Putting them on one axis is an explicit temporal policy, never implied.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from manyup.data.geometry import Geometry
from manyup.data.time import TimeAxis


@dataclass
class ModalityData:
    data: torch.Tensor                    # (T, C, H, W); static modalities have T == 1
    bands: tuple[str, ...]                # names of the C channels, in order
    time: TimeAxis | None                 # None for static modalities
    frame_valid: torch.Tensor             # (T,) bool; False = missing frame (data is filler)
    geometry: Geometry
    pixel_valid: torch.Tensor | None = None                       # (T, H, W) bool
    aux: dict[str, torch.Tensor] = field(default_factory=dict)    # name -> (T, H, W)
    history: tuple[str, ...] = ()         # temporal/spatial ops applied, in order

    @property
    def num_frames(self) -> int:
        return self.data.shape[0]

    def select_frames(self, idx) -> "ModalityData":
        """Subset frames; every per-frame field moves together."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        return replace(
            self, data=self.data[idx], time=None if self.time is None else self.time.select(idx),
            frame_valid=self.frame_valid[idx],
            pixel_valid=None if self.pixel_valid is None else self.pixel_valid[idx],
            aux={k: v[idx] for k, v in self.aux.items()})

    def crop(self, row: int, col: int, h: int, w: int) -> "ModalityData":
        sl = (..., slice(row, row + h), slice(col, col + w))
        return replace(
            self, data=self.data[sl], geometry=self.geometry.crop(row, col, h, w),
            pixel_valid=None if self.pixel_valid is None else self.pixel_valid[sl],
            aux={k: v[sl] for k, v in self.aux.items()})


@dataclass
class MTMMSample:
    id: str                               # stable id; caches are keyed by this, never by index
    split: str
    modalities: dict[str, ModalityData]
    target: torch.Tensor | None           # (H, W) labels / values on target_geometry
    target_geometry: Geometry | None
    meta: dict = field(default_factory=dict)


@dataclass
class ModalityBatch:
    data: torch.Tensor                    # (B, T, C, H, W); T = max over the batch, padded
    bands: tuple[str, ...]
    t_start: torch.Tensor | None          # (B, T) int64 days; -1 on padding
    t_end: torch.Tensor | None
    time_precision: str | None
    frame_valid: torch.Tensor             # (B, T) bool; padding is False
    geometry: list[Geometry]
    pixel_valid: torch.Tensor | None      # (B, T, H, W)
    aux: dict[str, torch.Tensor]          # name -> (B, T, H, W)
    history: tuple[str, ...]


@dataclass
class MTMMBatch:
    ids: list[str]
    split: str
    modalities: dict[str, ModalityBatch]
    target: torch.Tensor | None           # (B, H, W)
    target_geometry: list[Geometry | None]
    meta: list[dict]


def _pad_t(x: torch.Tensor, T: int, value) -> torch.Tensor:
    if x.shape[0] == T:
        return x
    pad = torch.full((T - x.shape[0], *x.shape[1:]), value, dtype=x.dtype)
    return torch.cat([x, pad])


def _collate_modality(name: str, items: list[ModalityData]) -> ModalityBatch:
    first = items[0]
    for it in items[1:]:
        if it.bands != first.bands or it.data.shape[1:] != first.data.shape[1:]:
            raise ValueError(f"{name}: bands/shape differ within the batch "
                             f"({first.bands} {tuple(first.data.shape[1:])} vs "
                             f"{it.bands} {tuple(it.data.shape[1:])})")
        if it.history != first.history or set(it.aux) != set(first.aux):
            raise ValueError(f"{name}: samples were processed differently within the batch")
    T = max(it.num_frames for it in items)
    timed = first.time is not None
    stack = lambda f, v: torch.stack([_pad_t(f(it), T, v) for it in items])  # noqa: E731
    return ModalityBatch(
        data=stack(lambda it: it.data, 0),
        bands=first.bands,
        t_start=stack(lambda it: it.time.start, -1) if timed else None,
        t_end=stack(lambda it: it.time.end, -1) if timed else None,
        time_precision=first.time.precision if timed else None,
        frame_valid=stack(lambda it: it.frame_valid, False),
        geometry=[it.geometry for it in items],
        pixel_valid=(stack(lambda it: it.pixel_valid, False)
                     if first.pixel_valid is not None else None),
        aux={k: stack(lambda it, k=k: it.aux[k], 0) for k in first.aux},
        history=first.history)


def mtmm_collate(samples: list[MTMMSample]) -> MTMMBatch:
    """Stack samples; each modality's time axis is padded separately and marked invalid."""
    names = list(samples[0].modalities)
    splits = {s.split for s in samples}
    if len(splits) != 1:
        raise ValueError(f"batch mixes splits {splits}")
    for s in samples:
        if list(s.modalities) != names:
            raise ValueError(f"sample {s.id} has modalities {list(s.modalities)}, expected {names}")
    has_target = samples[0].target is not None
    return MTMMBatch(
        ids=[s.id for s in samples], split=splits.pop(),
        modalities={n: _collate_modality(n, [s.modalities[n] for s in samples]) for n in names},
        target=torch.stack([s.target for s in samples]) if has_target else None,
        target_geometry=[s.target_geometry for s in samples],
        meta=[s.meta for s in samples])
