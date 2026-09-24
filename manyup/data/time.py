"""Frame timing: every frame covers a date interval [start, end], as int days since 1970-01-01.

A single acquisition is [d, d]. A pooled bin is [first day, last day] of the bin, and keeps
which source dates went into it. `precision` says what the dates can be trusted to:
  "day"            an acquisition date;
  "month"          the month is known (e.g. a monthly composite of a known year);
  "month_of_year"  only the calendar month is known, not the year (e.g. BioMassters). Days
                   then sit on a documented PLACEHOLDER year; never read the year back.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import torch

EPOCH = dt.date(1970, 1, 1)
PRECISIONS = ("day", "month", "month_of_year")


def to_days(d: dt.date) -> int:
    return (d - EPOCH).days


def from_days(n: int) -> dt.date:
    return EPOCH + dt.timedelta(days=int(n))


def yyyymmdd_to_days(v: int | str) -> int:
    s = str(v)
    return to_days(dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])))


def month_bounds(year: int, month: int) -> tuple[int, int]:
    first = dt.date(year, month, 1)
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return to_days(first), to_days(nxt) - 1


@dataclass
class TimeAxis:
    start: torch.Tensor                  # (T,) int64 days
    end: torch.Tensor                    # (T,) int64 days, end >= start
    precision: str
    sources: list[list[int]] | None = field(default=None)  # per frame: source start days

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision {self.precision!r} not in {PRECISIONS}")
        if self.start.shape != self.end.shape or self.start.ndim != 1:
            raise ValueError(f"start {tuple(self.start.shape)} / end {tuple(self.end.shape)}")
        if self.sources is not None and len(self.sources) != len(self.start):
            raise ValueError("sources must have one entry per frame")

    @staticmethod
    def instants(days: list[int], precision: str = "day") -> "TimeAxis":
        t = torch.tensor(days, dtype=torch.long)
        return TimeAxis(t, t.clone(), precision)

    def __len__(self) -> int:
        return len(self.start)

    def select(self, idx: torch.Tensor | list[int]) -> "TimeAxis":
        idx = torch.as_tensor(idx, dtype=torch.long)
        src = None if self.sources is None else [self.sources[i] for i in idx.tolist()]
        return TimeAxis(self.start[idx], self.end[idx], self.precision, src)

    def dates(self) -> list[tuple[dt.date, dt.date]]:
        return [(from_days(s), from_days(e)) for s, e in zip(self.start.tolist(), self.end.tolist())]
