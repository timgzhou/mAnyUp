"""Temporal policies: which frames of ONE modality a model sees.

Each policy maps ModalityData -> ModalityData and never loses timing: selected frames keep
their dates; pooled bins record their date range and the source dates that went in. Every
policy has a `key` for cache / checkpoint / CSV names (`resolved_key` fills in anything taken
from the modality's spec). Chain with `+`:

    Every(10) + Pool("month", "mean", window=("2018-09-01", "2019-10-31"))

Policies are given per modality (see MTMMDataset); there is no shared default.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import replace

import torch

from manyup.data.sample import ModalityData
from manyup.data.spec import ModalitySpec
from manyup.data.time import TimeAxis, month_bounds, to_days

REDUCERS = ("mean", "median")


class TemporalPolicy(ABC):
    key: str

    @abstractmethod
    def _apply(self, md: ModalityData, spec: ModalitySpec) -> ModalityData: ...

    def resolved_key(self, spec: ModalitySpec) -> str:
        return self.key

    def __call__(self, md: ModalityData, spec: ModalitySpec) -> ModalityData:
        if spec.is_static and not isinstance(self, All):
            raise ValueError(f"{spec.name} is static; only All() applies, got {self.key}")
        out = self._apply(md, spec)
        return replace(out, history=md.history + (self.resolved_key(spec),))

    def __add__(self, other: "TemporalPolicy") -> "Compose":
        return Compose(self, other)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.key}>"


class Compose(TemporalPolicy):
    """Policies applied left to right; each records its own history entry."""

    def __init__(self, *policies: TemporalPolicy):
        self.policies = [q for p in policies
                         for q in (p.policies if isinstance(p, Compose) else [p])]
        self.key = "+".join(p.key for p in self.policies)

    def resolved_key(self, spec):
        return "+".join(p.resolved_key(spec) for p in self.policies)

    def _apply(self, md, spec):
        for p in self.policies:
            md = p(md, spec)
        return md

    def __call__(self, md, spec):
        return self._apply(md, spec)


class All(TemporalPolicy):
    """Every native frame, unchanged (the only policy allowed on static modalities)."""
    key = "all"

    def _apply(self, md, spec):
        return md


class Indices(TemporalPolicy):
    """Frames at these positions of the current time axis."""

    def __init__(self, indices: list[int]):
        self.indices = list(indices)
        self.key = "idx" + "-".join(map(str, self.indices))

    def _apply(self, md, spec):
        bad = [i for i in self.indices if not 0 <= i < md.num_frames]
        if bad:
            raise IndexError(f"{spec.name}: indices {bad} out of range for {md.num_frames} frames")
        return md.select_frames(self.indices)


class Every(TemporalPolicy):
    """Every k-th frame, starting at `offset`."""

    def __init__(self, k: int, offset: int = 0):
        if k < 1 or offset < 0:
            raise ValueError("Every needs k >= 1 and offset >= 0")
        self.k, self.offset = k, offset
        self.key = f"every{k}" + (f"o{offset}" if offset else "")

    def _apply(self, md, spec):
        return md.select_frames(list(range(self.offset, md.num_frames, self.k)))


class DateRange(TemporalPolicy):
    """Frames whose interval lies inside [start, end] (ISO dates, inclusive)."""

    def __init__(self, start: str, end: str):
        self.lo = to_days(dt.date.fromisoformat(start))
        self.hi = to_days(dt.date.fromisoformat(end))
        self.key = f"dates{start.replace('-', '')}-{end.replace('-', '')}"

    def _apply(self, md, spec):
        keep = (md.time.start >= self.lo) & (md.time.end <= self.hi)
        return md.select_frames(keep.nonzero().flatten())


class MaxFrames(TemporalPolicy):
    """At most n frames, spread uniformly over the axis (unchanged if there are <= n)."""

    def __init__(self, n: int):
        self.n = n
        self.key = f"max{n}"

    def _apply(self, md, spec):
        if md.num_frames <= self.n:
            return md
        return md.select_frames(torch.linspace(0, md.num_frames - 1, self.n).round().long())


class Pool(TemporalPolicy):
    """Reduce frames into calendar months.

    window=(first, last) ISO dates -> a FIXED grid: every month overlapping the window, the
      same for all samples; months with no valid frame stay as invalid (masked) slots.
    window=None -> only the months this sample has frames in, so T varies per sample.
    reducer=None -> the modality's declared ModalitySpec.pool_reducer.

    A frame belongs to the month its interval starts in. Invalid frames and invalid pixels
    never enter a reduction; "median" is torch's (the lower of the two middle values, so the
    result is always an observed value). Output is float32.
    """

    def __init__(self, freq: str = "month", reducer: str | None = None,
                 window: tuple[str, str] | None = None):
        if freq != "month":
            raise NotImplementedError(f"Pool freq {freq!r}; only 'month' is implemented")
        if reducer is not None and reducer not in REDUCERS:
            raise ValueError(f"reducer {reducer!r} not in {REDUCERS}")
        self.freq, self.reducer = freq, reducer
        self.window = None if window is None else tuple(dt.date.fromisoformat(w) for w in window)
        self._win = "" if window is None else "@" + "-".join(d.strftime("%Y%m") for d in self.window)
        self.key = f"{freq}-{reducer or 'specreducer'}{self._win}"

    def _reducer(self, spec: ModalitySpec) -> str:
        r = self.reducer or spec.pool_reducer
        if r is None:
            raise ValueError(f"Pool on {spec.name}: no reducer given and the spec declares none")
        return r

    def resolved_key(self, spec):
        return f"{self.freq}-{self._reducer(spec)}{self._win}"

    def _months(self, md: ModalityData) -> list[tuple[int, int]]:
        if self.window is not None:
            a, b = self.window
            ym = [(y, m) for y in range(a.year, b.year + 1) for m in range(1, 13)
                  if (a.year, a.month) <= (y, m) <= (b.year, b.month)]
        else:
            ym = sorted({(s.year, s.month) for s, _ in md.time.dates()})
        return [month_bounds(y, m) for y, m in ym]

    def _apply(self, md, spec):
        if md.time is None:
            raise ValueError(f"{spec.name}: Pool needs a time axis")
        reducer = self._reducer(spec)
        T, C, H, W = md.data.shape
        valid = md.frame_valid.view(T, 1, 1).expand(T, H, W)       # (T, H, W)
        if md.pixel_valid is not None:
            valid = valid & md.pixel_valid
        data, fv, pv, starts, ends, sources = [], [], [], [], [], []
        aux = {k: [] for k in md.aux}
        for lo, hi in self._months(md):
            members = ((md.time.start >= lo) & (md.time.start <= hi)).nonzero().flatten()
            v = valid[members]
            data.append(_reduce(md.data[members].float(), v, reducer))
            for k in aux:
                aux[k].append(_reduce(md.aux[k][members].float().unsqueeze(1), v, reducer)[0])
            pv.append(v.any(0) if len(members) else torch.zeros((H, W), dtype=torch.bool))
            fv.append(bool(v.any()))
            starts.append(lo)
            ends.append(hi)
            sources.append(md.time.start[members][md.frame_valid[members]].tolist())
        return replace(
            md, data=torch.stack(data),
            time=TimeAxis(torch.tensor(starts), torch.tensor(ends), "month", sources),
            frame_valid=torch.tensor(fv), pixel_valid=torch.stack(pv),
            aux={k: torch.stack(v) for k, v in aux.items()})


def _reduce(x: torch.Tensor, valid: torch.Tensor, reducer: str) -> torch.Tensor:
    """Reduce (n, C, H, W) over n using only entries valid in (n, H, W); 0 where none are."""
    n, C, H, W = x.shape
    if n == 0:
        return torch.zeros((C, H, W))
    m = valid.unsqueeze(1).expand_as(x)
    if reducer == "mean":
        cnt = m.sum(0)
        return torch.where(cnt > 0, (x * m).sum(0) / cnt.clamp(min=1), torch.zeros(()))
    med = torch.where(m, x, torch.full((), float("nan"))).nanmedian(dim=0).values
    return torch.nan_to_num(med, nan=0.0)
