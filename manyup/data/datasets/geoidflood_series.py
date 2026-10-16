"""GEOID-Flood as a Sentinel-1 TIME SERIES: every acquisition in a 6-month window around the
flood, instead of GEOID's pre/post pair.

Same chips, splits and labels as GeoidFloodDataset (c64 index); the imagery is the year of
Planetary Computer `sentinel-1-rtc` fetched around each tile's GEOID post-event scene on the
relative orbits of GEOID's post and pre scenes (exp/geoidflood/fetch_s1_series.py), packed
chip-major (exp/geoidflood/pack_s1_series.py). Where Planetary Computer lacks GEOID's post
scene, GEOID's own is in the series at its date.

Window: `window_days` (183) of acquisitions that starts at least `before_event_days` (14)
before the tile's CEMS event date (GEOID's `event_date`, per tile) -- so the series has dry
dates before the flood as CEMS recorded it -- and ends at least `margin_days` (12) after
GEOID's post scene (a median 4 days after the event date, at most 108). Within those bounds
the window is placed at random, so the flood can sit anywhere in it. Train draws the window
afresh every time a chip is read; val and test use one fixed window per chip (position from
a hash of its id), so they are reproducible. CEMS's event date is when the event was
recorded: a long flood can predate it.

Returned: modality "S1" (VV, VH in dB, one step per acquisition, day precision), aux "orbit"
(relative orbit per step; the inserted GEOID scene takes the orbit it repeats), and the
3-class label. meta: event, post_day, event_day, window_start.
"""
from __future__ import annotations

import csv
import datetime as dt
import functools
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from manyup.data.dataset import MTMMDataset
from manyup.data.datasets.geoidflood import (DB_FLOOR, DEFAULT_ROOT, ChipSelection,
                                             GeoidFloodDataset)
from manyup.data.geometry import Geometry
from manyup.data.sample import ModalityData, MTMMSample
from manyup.data.spec import DatasetSpec, ModalitySpec
from manyup.data.temporal import TemporalPolicy
from manyup.data.time import TimeAxis

PACKED = Path(__file__).resolve().parents[3] / "data" / "GEOID-Flood-series-packed"
NODATA = -32768


@functools.lru_cache(maxsize=256)
def _meta(path: str) -> dict:
    m = np.load(path)
    out = {k: m[k] for k in m.files}
    out["chip_row"] = {(int(r), int(c)): i for i, (r, c) in enumerate(out["chips"])}
    tod = np.array([dt.datetime.fromisoformat(str(t)[:19]).hour * 3600
                    + dt.datetime.fromisoformat(str(t)[:19]).minute * 60 for t in out["time"]])
    orbit = out["rel_orbit"].copy()
    for i in np.nonzero(orbit < 0)[0]:          # inserted GEOID scene: the orbit it repeats
        cand = [j for j in range(len(orbit)) if orbit[j] >= 0
                and abs(((tod[j] - tod[i]) + 43200) % 86400 - 43200) < 180
                and (out["day"][j] - out["day"][i]) % 6 == 0]
        orbit[i] = orbit[cand[0]] if cand else 0
    out["orbit_filled"] = orbit
    return out


@functools.lru_cache(maxsize=64)
def _array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


class GeoidSeriesDataset(MTMMDataset):
    SPEC = DatasetSpec(
        name="geoidflood_series",
        modalities={"S1": ModalitySpec("S1", ("VV", "VH"), "dB", 10.0, "irregular", (1, 160),
                                       "day", pool_reducer="mean", nodata=DB_FLOOR,
                                       aux_layers=("orbit",))},
        target=GeoidFloodDataset.SPEC.target,
        native_shape=(64, 64),
        split_sizes={},                   # the c64 chips whose tile has a series; set per instance
        georeferenced=True,
        notes=("samples are GEOID-Flood c64 chips with a packed S1 series",
               "S1: every acquisition in a window around the post-event scene"))

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy],
                 window_days: int = 183, margin_days: int = 12, before_event_days: int = 14,
                 validate: bool = True,
                 root: str | Path = DEFAULT_ROOT, packed: str | Path = PACKED):
        self.root, self.packed = Path(root), Path(packed)
        self.window_days, self.margin_days = window_days, margin_days
        self.before_event_days = before_event_days
        self._event_day = {}                       # tile -> CEMS event date (days), per tile
        with open(self.root / "data_tiles_s256_st128.csv") as f:
            for r in csv.DictReader(f):
                self._event_day[r["label_id"][:-len("_label")]] = (
                    dt.date.fromisoformat(r["event_date"][:10]) - dt.date(1970, 1, 1)).days
        chips = ChipSelection(64)
        idx = json.loads((self.root / f"chip_index_{chips.key}.json").read_text())
        self._tiles = idx["tiles"]
        self._ids = {}
        for s, ids in idx["splits"].items():
            self._ids[s] = [c for c in ids if (self._tile_path(c.split("@")[0])).exists()]
        self.SPEC = replace(type(self).SPEC, split_sizes={s: len(v) for s, v in self._ids.items()})
        super().__init__(split, modalities, tiling=None, validate=validate)

    def _tile_path(self, tile: str) -> Path:
        return self.packed / tile.rsplit("-", 1)[0] / f"{tile}.npy"

    @property
    def key(self) -> str:
        return super().key + f"_w{self.window_days}"

    def _native_ids(self, split):
        return list(self._ids.get(split, []))

    def window_start(self, chip_id: str, post_day: int, event_day: int) -> int:
        lo = post_day + self.margin_days - self.window_days + 1   # post >= margin before the end
        hi = min(event_day, post_day) - self.before_event_days     # >= 2 weeks before the event
        hi = max(hi, lo)                     # (never binds: event -> post is <= 108 days)
        if self.split == "train":
            return int(np.random.randint(lo, hi + 1))
        u = int(hashlib.md5(chip_id.encode()).hexdigest()[:8], 16) / 16 ** 8
        return lo + int(u * (hi - lo + 1))

    def _load(self, chip_id, modalities):
        tile, rc = chip_id.split("@r")
        row, col = map(int, rc.split("c"))
        path = self._tile_path(tile)
        meta = _meta(str(path.with_name(path.stem + "_meta.npz")))
        x = np.asarray(_array(str(path))[meta["chip_row"][(row, col)]])   # (A, 2, S, S)
        post = int(meta["post_day"])
        event = self._event_day.get(tile, post)
        start = self.window_start(chip_id, post, event)
        sel = np.nonzero((meta["day"] >= start) & (meta["day"] < start + self.window_days))[0]
        x = x[sel]
        ok = x != NODATA
        db = np.where(ok, x.astype(np.float32) / 100, DB_FLOOR)
        pixel_valid = ok.all(1)                                          # (T, S, S)
        s = x.shape[-1]
        with rasterio.open(self._tiles[tile]["label"]) as r:
            window = Window(col, row, s, s)
            label = r.read(1, window=window).astype(np.int64)
            geo = Geometry(str(r.crs), tuple(r.window_transform(window))[:6], (s, s))
        orbit = torch.from_numpy(meta["orbit_filled"][sel].astype(np.float32))
        s1 = ModalityData(torch.from_numpy(db), ("VV", "VH"),
                          TimeAxis.instants([int(d) for d in meta["day"][sel]]),
                          torch.from_numpy(pixel_valid.reshape(len(sel), -1).any(1)), geo,
                          pixel_valid=torch.from_numpy(pixel_valid),
                          aux={"orbit": orbit[:, None, None].expand(len(sel), s, s).clone()})
        return MTMMSample(chip_id, self.split, {"S1": s1}, torch.from_numpy(label), geo,
                          {"event": tile.rsplit("-", 1)[0], "post_day": post, "event_day": event,
                           "window_start": start})
