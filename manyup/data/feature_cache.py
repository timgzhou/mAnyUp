"""Read a feature cache (one encoder's per-sample features for one split) by sample ID.

Layout, as written by the extractors:
    <root>/<name>/meta.json
    <root>/<name>/<split_dir>/<idx>.pt           (T, gH, gW, D) fp16
    <root>/<name>/<split_dir>/ids.txt            sample id of every idx
    <root>/<name>/<split_dir>/frame_valid.pt     (N, T) bool
    <root>/<name>/<split_dir>/t_start.pt, t_end.pt   (N, T) int64 days

Caches from before the data layer lack ids / validity / dates. For those the caller passes
`legacy`, a reconstruction (ids, frame_valid, t_start, t_end) it vouches for; `reconstructed`
then names what came from it, so it is never mistaken for recorded provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LegacyReconstruction:
    ids: list[str]
    frame_valid: torch.Tensor   # (N, T) bool
    t_start: torch.Tensor       # (N, T) int64 days
    t_end: torch.Tensor


class FeatureCache:
    def __init__(self, root: str | Path, name: str, split_dir: str,
                 legacy: LegacyReconstruction | None = None):
        self.dir = Path(root).expanduser() / name
        self.split_dir = self.dir / split_dir
        self.name = name
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.n = len(list(self.split_dir.glob("[0-9]*.pt")))
        self.reconstructed: list[str] = []

        ids_file = self.split_dir / "ids.txt"
        if ids_file.exists():
            ids = ids_file.read_text().split()
        else:
            ids = self._legacy(legacy, "ids").ids
        fv = self._load("frame_valid", legacy)
        ts, te = self._load("t_start", legacy), self._load("t_end", legacy)
        if not (len(ids) == len(fv) == len(ts) == len(te) == self.n):
            raise ValueError(f"{name}/{split_dir}: {self.n} feature files but ids {len(ids)}, "
                             f"frame_valid {len(fv)}, dates {len(ts)}")
        self.ids, self.frame_valid, self.t_start, self.t_end = ids, fv, ts, te
        self._pos = {sid: i for i, sid in enumerate(ids)}
        if len(self._pos) != len(ids):
            raise ValueError(f"{name}/{split_dir}: duplicate sample ids")

    def _legacy(self, legacy, what):
        if legacy is None:
            raise FileNotFoundError(f"{self.split_dir} has no {what} and no legacy "
                                    "reconstruction was given")
        if self.meta.get("temporal", "legacy") != "legacy":
            raise FileNotFoundError(f"{self.split_dir} lacks {what} but is not a legacy cache")
        if what not in self.reconstructed:
            self.reconstructed.append(what)
        return legacy

    def _load(self, field: str, legacy) -> torch.Tensor:
        f = self.split_dir / f"{field}.pt"
        if f.exists():
            return torch.load(f)
        return getattr(self._legacy(legacy, field), field)

    def __len__(self) -> int:
        return self.n

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._pos

    def get(self, sample_id: str) -> dict:
        i = self._pos[sample_id]
        return {"tokens": torch.load(self.split_dir / f"{i}.pt"),
                "frame_valid": self.frame_valid[i], "t_start": self.t_start[i],
                "t_end": self.t_end[i]}
