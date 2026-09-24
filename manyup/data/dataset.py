"""MTMMDataset: the base class every multimodal, multitemporal dataset subclasses.

A subclass declares SPEC (a DatasetSpec) and implements raw I/O only:
    _native_ids(split) -> list[str]         stable ids of the native samples in a split
    _load(native_id, modalities) -> MTMMSample   native data for exactly those modalities

The base class owns the rest, in a fixed order:
    load -> temporal policy (per modality) -> spatial tiling -> validate

Usage (everything explicit: which modalities, how each is sampled in time, how space is cut):
    ds = PastisDataset(split="train",
                       modalities={"S2": Pool("month", "median", window=...),
                                   "S1A": Pool("month", "mean", window=...)},
                       tiling=Tiling(64, "col-major"))
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import torch
from torch.utils.data import Dataset

from manyup.data.sample import ModalityData, MTMMSample
from manyup.data.spatial import Tiling
from manyup.data.spec import DatasetSpec
from manyup.data.temporal import TemporalPolicy
from manyup.data.time import from_days


class MTMMDataset(Dataset, ABC):
    SPEC: DatasetSpec

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy],
                 tiling: Tiling | None, validate: bool = True):
        spec = self.SPEC
        if split not in spec.split_sizes:
            raise ValueError(f"{spec.name}: split {split!r} not in {sorted(spec.split_sizes)}")
        unknown = set(modalities) - set(spec.modalities)
        if not modalities or unknown:
            raise ValueError(f"{spec.name}: modalities must be a non-empty subset of "
                             f"{sorted(spec.modalities)}; unknown {sorted(unknown)}")
        for name, policy in modalities.items():
            if not isinstance(policy, TemporalPolicy):
                raise TypeError(f"{name}: expected a TemporalPolicy, got {policy!r}")
        self.split, self.modalities, self.tiling, self.validate = split, dict(modalities), tiling, validate

        natives = self._native_ids(split)
        if len(natives) != spec.split_sizes[split]:
            raise ValueError(f"{spec.name}/{split}: found {len(natives)} samples, "
                             f"spec declares {spec.split_sizes[split]}")
        if len(set(natives)) != len(natives):
            raise ValueError(f"{spec.name}/{split}: duplicate sample ids")
        offsets = tiling.offsets(spec.native_shape) if tiling else [None]
        self._index = [(nid, off) for nid in natives for off in offsets]
        # Tiles of one native sample are consecutive, so sequential access (e.g. feature
        # extraction) reuses the loaded + temporally processed native sample.
        self._last: tuple[str, MTMMSample] | None = None

    # --- subclass API -------------------------------------------------------------------
    @abstractmethod
    def _native_ids(self, split: str) -> list[str]: ...

    @abstractmethod
    def _load(self, native_id: str, modalities: list[str]) -> MTMMSample: ...

    # --- public -------------------------------------------------------------------------
    @property
    def key(self) -> str:
        """Names this dataset configuration, e.g. for feature caches."""
        mods = "_".join(f"{m}-{p.resolved_key(self.SPEC.modalities[m])}"
                        for m, p in self.modalities.items())
        return f"{self.SPEC.name}_{mods}_{self.tiling.key if self.tiling else 'native'}"

    def __len__(self) -> int:
        return len(self._index)

    def sample_id(self, i: int) -> str:
        nid, off = self._index[i]
        return nid if off is None else Tiling.tile_id(nid, *off)

    def __getitem__(self, i: int) -> MTMMSample:
        nid, off = self._index[i]
        if self._last is not None and self._last[0] == nid:
            s = self._last[1]
        else:
            s = self._load(nid, list(self.modalities))
            if self.validate:
                self._check(s, native=True)
            for name, policy in self.modalities.items():
                s.modalities[name] = policy(s.modalities[name], self.SPEC.modalities[name])
            self._last = (nid, s)
        if off is not None:
            s = self._crop(s, *off)
        if self.validate:
            self._check(s, native=False)
        return s

    def _crop(self, s: MTMMSample, row: int, col: int) -> MTMMSample:
        t, mods = self.tiling.size, {}
        for name, md in s.modalities.items():
            if md.geometry.shape != self.SPEC.native_shape:
                raise NotImplementedError(f"{name}: tiling across different pixel grids "
                                          f"({md.geometry.shape} vs {self.SPEC.native_shape})")
            mods[name] = md.crop(row, col, t, t)   # new object; the cached native stays whole
        target = None if s.target is None else s.target[row:row + t, col:col + t]
        tg = None if s.target_geometry is None else s.target_geometry.crop(row, col, t, t)
        return MTMMSample(Tiling.tile_id(s.id, row, col), s.split, mods, target, tg,
                          dict(s.meta))

    # --- validation ---------------------------------------------------------------------
    def _check(self, s: MTMMSample, native: bool) -> None:
        spec, where = self.SPEC, f"{self.SPEC.name}/{s.id}"
        if list(s.modalities) != list(self.modalities):
            raise AssertionError(f"{where}: modalities {list(s.modalities)} != requested "
                                 f"{list(self.modalities)}")
        for name, md in s.modalities.items():
            ms = spec.modalities[name]
            _check_modality(f"{where}/{name}", md, ms, spec.georeferenced)
            if native and not ms.frames[0] <= md.num_frames <= ms.frames[1]:
                raise AssertionError(f"{where}/{name}: {md.num_frames} native frames outside "
                                     f"declared {ms.frames}")
        if s.target is not None:
            if s.target_geometry is None or tuple(s.target.shape) != s.target_geometry.shape:
                raise AssertionError(f"{where}: target shape does not match its geometry")

    def describe(self, n: int | None = None) -> str:
        """Per-modality frame counts, date range and missing-frame rate over n samples."""
        idx = range(len(self)) if n is None else range(min(n, len(self)))
        frames, missing, lo, hi = Counter(), Counter(), {}, {}
        for i in idx:
            for name, md in self[i].modalities.items():
                frames[(name, md.num_frames)] += 1
                missing[name] += int((~md.frame_valid).sum())
                if md.time is not None:
                    lo[name] = min(lo.get(name, 10**9), int(md.time.start.min()))
                    hi[name] = max(hi.get(name, -10**9), int(md.time.end.max()))
        lines = [f"{self.key}: {len(self)} samples ({self.split}), described {len(idx)}"]
        for name in self.modalities:
            counts = sorted((t, c) for (m, t), c in frames.items() if m == name)
            span = (f"{from_days(lo[name])}..{from_days(hi[name])}" if name in lo else "static")
            lines.append(f"  {name}: frames {counts} | {span} | missing frames {missing[name]}")
        return "\n".join(lines)


def _check_modality(where: str, md: ModalityData, ms, georeferenced: bool) -> None:
    T, C, H, W = md.data.shape
    if md.bands != ms.bands:
        raise AssertionError(f"{where}: bands {md.bands} != spec {ms.bands}")
    if C != len(md.bands):
        raise AssertionError(f"{where}: {C} channels but {len(md.bands)} band names")
    if md.frame_valid.shape != (T,) or md.frame_valid.dtype != torch.bool:
        raise AssertionError(f"{where}: frame_valid must be bool (T,)")
    if md.geometry.shape != (H, W):
        raise AssertionError(f"{where}: geometry {md.geometry.shape} != data {(H, W)}")
    if md.geometry.known != georeferenced:
        raise AssertionError(f"{where}: geometry known={md.geometry.known}, spec "
                             f"georeferenced={georeferenced}")
    if md.pixel_valid is not None and md.pixel_valid.shape != (T, H, W):
        raise AssertionError(f"{where}: pixel_valid shape {tuple(md.pixel_valid.shape)}")
    for k, v in md.aux.items():
        if k not in ms.aux_layers or v.shape != (T, H, W):
            raise AssertionError(f"{where}: aux {k!r} undeclared or shaped {tuple(v.shape)}")
    if ms.is_static:
        if md.time is not None or T != 1:
            raise AssertionError(f"{where}: static modality must have T=1 and no time axis")
        return
    if md.time is None or len(md.time) != T:
        raise AssertionError(f"{where}: time axis length != {T} frames")
    if (md.time.end < md.time.start).any():
        raise AssertionError(f"{where}: a frame ends before it starts")
    if T > 1 and (md.time.start[1:] < md.time.start[:-1]).any():
        raise AssertionError(f"{where}: frames not in time order")
    if md.frame_valid.any() and not torch.isfinite(md.data[md.frame_valid].float()).all():
        raise AssertionError(f"{where}: non-finite values in a valid frame")
