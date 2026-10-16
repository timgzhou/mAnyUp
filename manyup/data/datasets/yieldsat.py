"""YieldSAT (Miranda et al., CVPR 2026; https://yieldsat.github.io): pixel-level crop yield.

2,173 field-seasons in Argentina, Brazil, Uruguay and Germany (corn, rapeseed, soybean,
wheat; harvests 2016-2024), each a bounding box on its own UTM grid (median 94 x 96 px, up
to 578 x 408): 10 m combine-harvester yield (t/ha, moisture-corrected) and every Sentinel-2
L2A acquisition from about a month before seeding to about a month after harvest (11-153
per field). The S2 rasters are clipped to the field polygon (0 outside).

Two views of the same fields, tiles, targets and splits:

  YieldSatDataset    ("raw", the original-preprocessed release): every acquisition, "S2"
                     with the SCL layer as aux "scl"; pixel_valid = SCL 4 (vegetation) or 5
                     (bare soil), the same clear-pixel test the ML-ready release applies.
                     Cloudy pixels keep their values (only masked by pixel_valid).
  YieldSat24Dataset  ("ml24", the preprocessed-24-ts release, rasterized AS-IS): the release's
                     24 steps per field (calendar-month slots of the two years ending with the
                     harvest year, one least-cloudy acquisition each -- sometimes from the
                     month before -- or none). A value exists
                     only at a LABELLED, clear pixel -- the release holds nothing else --
                     so every other pixel is 0 with pixel_valid False. A slot without an
                     acquisition is an invalid frame dated at its month's first day.

The ml24 values are read from the same packed acquisitions as the raw view: at every
labelled pixel the release's S2 equals the raw acquisition of the same date bit for bit,
and it is finite exactly where SCL is 4 or 5 (checked on every field by
`python -m exp.yieldsat.prepare verify`). Only its dates, targets and pixel set come from the
NetCDF (exp/yieldsat/prepare.py).

Samples are 64 x 64 px tiles of a field's bounding box: a grid of ceil(H / 64) x ceil(W / 64)
tiles centred on the box (pixels past it are padding: 0, invalid, unlabelled), keeping the
tiles with at least one labelled pixel. Tiles do not overlap, so every labelled pixel is
scored exactly once. Sample id "<field_shared_name>@r<row>c<col>" (row / col of the tile's top
left corner in the field's box; negative in the padding).

Target: (64, 64) float32 t/ha, NaN = unlabelled (the release's target, i.e. the raw
mean_scaled_yield_masked_regional_statistical_outlier.tif at its labelled pixels).

Splits: the paper's protocol (field-grouped 10-fold CV, stratified) made fixed: within each
country x crop, fields sorted by md5 of their name take folds 0..9 in turn; test = fold
`test_fold` (0), val = the next fold, train = the other 8. meta: field, country, crop, year.
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from manyup.data.dataset import MTMMDataset
from manyup.data.geometry import Geometry
from manyup.data.sample import ModalityData, MTMMSample
from manyup.data.spec import DatasetSpec, ModalitySpec, TargetSpec
from manyup.data.temporal import TemporalPolicy
from manyup.data.time import TimeAxis, month_bounds

ROOT = Path(__file__).resolve().parents[3] / "data" / "YieldSAT"
PACKED = Path(os.environ.get("YIELDSAT_PACKED", ROOT / "packed"))   # env: e.g. a test build
TILE = 64
FOLDS = 10
# raw GeoTIFF band order (= the packed arrays')
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
CLEAR_SCL = (4, 5)                    # vegetation, bare soil: finite in the ML-ready release
COUNTRIES = ("Argentina", "Brazil", "Germany", "Uruguay")


@functools.lru_cache(maxsize=4)
def load_index(packed: str) -> dict:
    return json.loads((Path(packed) / "index.json").read_text())


@functools.lru_cache(maxsize=512)
def _array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def read_window(a: np.ndarray, r0: int, c0: int, size: int, fill=0) -> np.ndarray:
    """a[..., r0:r0+size, c0:c0+size] with `fill` wherever the window leaves the array."""
    H, W = a.shape[-2:]
    out = np.full(a.shape[:-2] + (size, size), fill, dtype=a.dtype)
    r1, c1 = max(r0, 0), max(c0, 0)
    r2, c2 = min(r0 + size, H), min(c0 + size, W)
    if r2 > r1 and c2 > c1:
        out[..., r1 - r0:r2 - r0, c1 - c0:c2 - c0] = a[..., r1:r2, c1:c2]
    return out


def tile_origins(H: int, W: int, size: int = TILE) -> list[tuple[int, int]]:
    """Top-left corners of the ceil(H/size) x ceil(W/size) grid centred on an H x W box."""
    nr, nc = -(-H // size), -(-W // size)
    r0, c0 = -((nr * size - H) // 2), -((nc * size - W) // 2)
    return [(r0 + i * size, c0 + j * size) for i in range(nr) for j in range(nc)]


def fold_splits(test_fold: int) -> dict[str, set[int]]:
    val = (test_fold + 1) % FOLDS
    return {"test": {test_fold}, "val": {val},
            "train": set(range(FOLDS)) - {test_fold, val}}


def parse_id(sample_id: str) -> tuple[str, int, int]:
    field, rc = sample_id.rsplit("@r", 1)
    r, c = rc.split("c")
    return field, int(r), int(c)


class YieldSatDataset(MTMMDataset):
    """Every acquisition (the raw release). See the module docstring."""
    SPEC = DatasetSpec(
        name="yieldsat",
        modalities={"S2": ModalitySpec("S2", S2_BANDS, "reflectance*1e4", 10.0, "irregular",
                                       (1, 160), "day", pool_reducer="mean", nodata=0,
                                       aux_layers=("scl",))},
        target=TargetSpec("regression", "t/ha"),
        native_shape=(TILE, TILE),
        split_sizes={},                   # tiles per split; set per instance from the index
        georeferenced=True,
        notes=("samples: 64 px tiles of a field's box, >= 1 labelled pixel each",
               "S2 clipped to the field polygon; pixel_valid = SCL in (4, 5)",
               "splits: md5-ordered folds per country x crop; test fold 0, val fold 1"))

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy],
                 test_fold: int = 0, validate: bool = True, packed: str | Path = PACKED):
        self.packed, self.test_fold = Path(packed), test_fold
        idx = load_index(str(self.packed))
        self.fields = idx["fields"]
        folds = fold_splits(test_fold)
        self._ids = {s: [f"{name}@r{r}c{c}" for name, f in sorted(self.fields.items())
                         if f["fold"] in fs for r, c in f["tiles"]]
                     for s, fs in folds.items()}
        self.SPEC = replace(type(self).SPEC,
                            split_sizes={s: len(v) for s, v in self._ids.items()})
        super().__init__(split, modalities, tiling=None, validate=validate)

    @property
    def key(self) -> str:
        return super().key + ("" if self.test_fold == 0 else f"_fold{self.test_fold}")

    def _native_ids(self, split):
        return list(self._ids[split])

    def _path(self, kind: str, field: str) -> str:
        return str(self.packed / kind / self.fields[field]["country"] / f"{field}.npy")

    def _geometry(self, f: dict, r0: int, c0: int) -> Geometry:
        a, b, x0, d, e, y0 = f["transform"]
        return Geometry(f["crs"], (a, b, x0 + a * c0 + b * r0, d, e, y0 + d * c0 + e * r0),
                        (TILE, TILE))

    def _target(self, field: str, r0: int, c0: int) -> torch.Tensor:
        return torch.from_numpy(read_window(_array(self._path("target", field)), r0, c0, TILE,
                                            np.float32("nan")))

    def _load_target(self, sample_id):
        return self._target(*parse_id(sample_id))

    def _meta(self, field: str, f: dict) -> dict:
        return {"field": field, "country": f["country"], "crop": f["crop"], "year": f["year"]}

    def _load(self, sample_id, modalities):
        field, r0, c0 = parse_id(sample_id)
        f = self.fields[field]
        s2 = read_window(_array(self._path("s2", field)), r0, c0, TILE)          # (A, 12, S, S)
        scl = read_window(_array(self._path("scl", field)), r0, c0, TILE)        # (A, S, S)
        clear = np.isin(scl, CLEAR_SCL)
        geo = self._geometry(f, r0, c0)
        md = ModalityData(torch.from_numpy(s2.astype(np.float32)), S2_BANDS,
                          TimeAxis.instants(f["raw_days"]),
                          torch.from_numpy(clear.reshape(len(clear), -1).any(1)), geo,
                          pixel_valid=torch.from_numpy(clear),
                          aux={"scl": torch.from_numpy(scl.astype(np.float32))})
        return MTMMSample(sample_id, self.split, {"S2": md}, self._target(field, r0, c0), geo,
                          self._meta(field, f))


class YieldSat24Dataset(YieldSatDataset):
    """The ML-ready release's 24 steps, rasterized as-is. See the module docstring."""
    SPEC = replace(YieldSatDataset.SPEC, name="yieldsat24", modalities={
        "S2": ModalitySpec("S2", S2_BANDS, "reflectance*1e4", 10.0, "fixed", (24, 24), "day",
                           pool_reducer="mean", nodata=0)},
        notes=YieldSatDataset.SPEC.notes + (
            "ml24: the release's 24 month slots; values only at labelled clear pixels",))

    @staticmethod
    def slot_days(f: dict) -> list[int]:
        """The 24 slot dates: the acquisition's day, or (no acquisition) the first day of the
        slot's month -- slot i = month i of the two years ending with the HARVEST year --
        moved as needed to stay between its neighbours, so the axis is in time order. Real
        acquisition dates are never moved."""
        y0 = int(f["harvest"][:4]) - 1
        days = list(f["ml24_days"])
        out, prev = [], -10**9
        for i, d in enumerate(days):
            if d < 0:
                nxt = min((x for x in days[i + 1:] if x >= 0), default=10**9)
                d = min(max(month_bounds(y0 + i // 12, i % 12 + 1)[0], prev), nxt)
            out.append(d)
            prev = d
        return out

    def _load(self, sample_id, modalities):
        field, r0, c0 = parse_id(sample_id)
        f = self.fields[field]
        raw = f["raw_days"]
        acq = [raw.index(d) if d >= 0 else -1 for d in f["ml24_days"]]
        have = [i for i, a in enumerate(acq) if a >= 0]
        target = self._target(field, r0, c0)
        s2_all = _array(self._path("s2", field))
        scl_all = _array(self._path("scl", field))
        take = [acq[i] for i in have]
        s2 = np.zeros((24, len(S2_BANDS), TILE, TILE), np.float32)
        valid = np.zeros((24, TILE, TILE), bool)
        if take:
            # the release holds a value only at a labelled pixel whose SCL is clear
            # window of each taken acquisition only (a field's full stack can be ~0.5 GB)
            scl = np.stack([read_window(scl_all[a], r0, c0, TILE) for a in take])
            ok = np.isin(scl, CLEAR_SCL) & np.isfinite(target.numpy())[None]
            win = np.stack([read_window(s2_all[a], r0, c0, TILE) for a in take])
            s2[have] = np.where(ok[:, None], win, 0)
            valid[have] = ok
        geo = self._geometry(f, r0, c0)
        md = ModalityData(torch.from_numpy(s2), S2_BANDS,
                          TimeAxis.instants(self.slot_days(f)),
                          torch.from_numpy(valid.reshape(24, -1).any(1)), geo,
                          pixel_valid=torch.from_numpy(valid))
        return MTMMSample(sample_id, self.split, {"S2": md}, target, geo, self._meta(field, f))
