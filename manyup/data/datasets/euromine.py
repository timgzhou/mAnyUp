"""EuroMineNet (Yu et al. 2025, arXiv 2510.14661; RODARE doi 10.14278/rodare.4656): annual
mine-footprint mapping at 133 mine sites in 14 EU countries, 2015-2024.

Per site: one Sentinel-2 L2A image PER YEAR (cloud-free, peak vegetation season; 10 bands, the
20 m ones resampled to 10 m, no B01 / B09 / B10) and one expert binary mine mask PER YEAR, so
the label itself changes along the series (the footprint grows / shrinks ~1 % of the site area
per year). Packed once by exp/euromine/prepare.py into PACKED:
    index.json         sites -> split, shape, crs, transform, tiles; years; bands
    s2/<site>.npy      (10, 10, H, W) uint16 (year, band)
    label/<site>.npy   (10, H, W) uint8 in {0, 1}

Samples are 64 x 64 px tiles of a site raster: a non-overlapping grid from the top-left, the last
row / column moved back to end at the raster's edge (the paper's own 160 px cropping rule, so
every tile is full and real; edge tiles overlap their neighbour). Sample id
"<site>@r<row>c<col>".

Target: (64, 64) int64 bitmask, bit t = mine in year YEARS[t] (decode_years -> (B, T, H, W));
it keeps the dataset's target 2-D like every other task's.

Splits: the release's train_val_test_splits.py site lists (sites whose rasters the release
excludes from its own preprocessing dropped, see SPLITS). Time: the release gives no
acquisition date, so each year's image is dated 1 July (the month embedding is constant).
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
from manyup.data.time import TimeAxis, to_days

ROOT = Path(__file__).resolve().parents[3] / "data" / "EuroMineNet"
PACKED = Path(os.environ.get("EUROMINE_PACKED", ROOT / "packed"))
TILE = 64
YEARS = tuple(range(2015, 2025))
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
DATE = (7, 1)                         # placeholder acquisition day of every year's image
# exp.euromine.fetch_seasonal: cloud-masked Planetary Computer L2A medians per calendar quarter,
# 2015 Q3 (the archive's start) .. 2024 Q4, on the release grid; packed by prepare pack_seasonal
SEASONAL = Path(os.environ.get("EUROMINE_SEASONAL", ROOT / "seasonal_packed"))
QUARTERS = ((2015, 3), (2015, 4)) + tuple((y, q) for y in range(2016, 2025) for q in (1, 2, 3, 4))
# year t's label belongs to its Jul-Sep quarter: the release images are August-dominated summer
# mosaics (a Q3 composite correlates 0.81-0.96 per band with them, a Q1 one 0.14-0.41)
LABEL_QUARTERS = tuple(QUARTERS.index((y, 3)) for y in YEARS)
SERIES = ("annual", "seasonal", "q3")


@functools.lru_cache(maxsize=4)
def load_index(packed: str) -> dict:
    return json.loads((Path(packed) / "index.json").read_text())


@functools.lru_cache(maxsize=512)
def _array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def tile_origins(H: int, W: int, size: int = TILE, pad: bool = False) -> list[tuple[int, int]]:
    """Top-left corners: a grid of `size` steps from (0, 0), plus a last row / column ending
    at the edge when the raster does not divide evenly. A side shorter than `size` raises,
    or with pad gets one tile at 0 running past the edge (read_window pads it)."""
    if (H < size or W < size) and not pad:
        raise ValueError(f"raster {H} x {W} smaller than a {size} px tile")
    def axis(n):
        if n <= size:
            return [0]
        o = list(range(0, n - size + 1, size))
        return o + ([n - size] if o[-1] != n - size else [])
    return [(r, c) for r in axis(H) for c in axis(W)]


def read_window(a: np.ndarray, r0: int, c0: int, size: int, fill=0) -> np.ndarray:
    """a[..., r0:r0+size, c0:c0+size], `fill` wherever the window leaves the array."""
    H, W = a.shape[-2:]
    if r0 + size <= H and c0 + size <= W:
        return np.asarray(a[..., r0:r0 + size, c0:c0 + size])
    out = np.full(a.shape[:-2] + (size, size), fill, dtype=a.dtype)
    h, w = min(size, H - r0), min(size, W - c0)
    out[..., :h, :w] = a[..., r0:r0 + h, c0:c0 + w]
    return out


UNLABELLED = 1 << len(YEARS)        # target bit of a padding pixel (outside the site raster)


def encode_years(labels: np.ndarray, unlabelled: np.ndarray | None = None) -> np.ndarray:
    """(T, H, W) {0, 1} [+ (H, W) bool padding] -> (H, W) int64 bitmask."""
    w = (1 << np.arange(labels.shape[0], dtype=np.int64))[:, None, None]
    out = (labels.astype(np.int64) * w).sum(0)
    return out if unlabelled is None else out | np.where(unlabelled, UNLABELLED, 0)


def decode_years(target: torch.Tensor, T: int = len(YEARS)) -> torch.Tensor:
    """(..., H, W) bitmask -> (..., T, H, W) long in {0, 1}, -1 (ignored) at padding."""
    bits = torch.arange(T, device=target.device)
    t = target.long()[..., None, :, :]
    y = (t >> bits[:, None, None]) & 1
    return torch.where((t & UNLABELLED) > 0, -1, y)


def parse_id(sample_id: str) -> tuple[str, int, int]:
    site, rc = sample_id.rsplit("@r", 1)
    r, c = rc.split("c")
    return site, int(r), int(c)


class EuroMineDataset(MTMMDataset):
    SPEC = DatasetSpec(
        name="euromine",
        modalities={"S2": ModalitySpec("S2", BANDS, "reflectance*1e4", 10.0, "fixed",
                                       (len(YEARS), len(YEARS)), "day", pool_reducer="mean",
                                       nodata=0)},
        target=TargetSpec("segmentation", "class", ("background", "mine")),
        native_shape=(TILE, TILE),
        split_sizes={},                   # tiles per split; set per instance from the index
        georeferenced=True,
        notes=("samples: 64 px (or `tile` px) tiles of a site raster (edge tiles snapped to the "
               "edge; a site side shorter than the tile is padded: no data, unlabelled)",
               "target: per-year mine masks as a bitmask (decode_years)",
               "one image per year, dated 1 July (the release has no dates)"))

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy], validate: bool = True,
                 packed: str | Path = PACKED, tile: int = TILE, series: str = "annual",
                 seasonal: str | Path = SEASONAL):
        """series  annual    the release's 10 yearly images (top-of-atmosphere summer mosaics)
                   seasonal  our 38 quarterly L2A composites (fetch_seasonal), 2015 Q3 .. 2024 Q4
                   q3        only their 10 Jul-Sep composites, one per label year
        The tiles and the per-year labels are the same for every series."""
        if series not in SERIES:
            raise ValueError(f"series {series!r} not in {SERIES}")
        self.packed, self.tile, self.series = Path(packed), tile, series
        self.seasonal = Path(seasonal)
        idx = load_index(str(self.packed))
        self.sites = idx["sites"]
        if tuple(idx["bands"]) != BANDS or tuple(idx["years"]) != YEARS:
            raise ValueError(f"{self.packed}: bands / years differ from the dataset's")
        self._ids = {s: [f"{name}@r{r}c{c}" for name, f in sorted(self.sites.items())
                         if f["split"] == s
                         for r, c in (f["tiles"] if tile == TILE
                                      else tile_origins(*f["shape"], tile, pad=True))]
                     for s in ("train", "val", "test")}
        if series == "annual":
            self.steps = None
            start = end = [to_days(dt.date(y, *DATE)) for y in YEARS]
        else:
            self.steps = (list(range(len(QUARTERS))) if series == "seasonal"
                          else list(LABEL_QUARTERS))
            qs = [QUARTERS[i] for i in self.steps]
            start = [to_days(dt.date(y, 3 * q - 2, 1)) for y, q in qs]
            end = [to_days(dt.date(y + (q == 4), 3 * q % 12 + 1, 1)) - 1 for y, q in qs]
        self.time = TimeAxis(torch.tensor(start), torch.tensor(end), "day")
        T = len(start)
        self.SPEC = replace(type(self).SPEC, native_shape=(tile, tile),
                            name="euromine" if series == "annual" else f"euromine_{series}",
                            split_sizes={s: len(v) for s, v in self._ids.items()},
                            modalities={"S2": replace(type(self).SPEC.modalities["S2"],
                                                      frames=(T, T))})
        super().__init__(split, modalities, tiling=None, validate=validate)

    @property
    def key(self) -> str:
        """<name>_<modalities>_tile<px>: a tile is a native sample here (no Tiling), so the
        base class would say "native"; the tile store is named from this."""
        k = super().key
        if not k.endswith("_native"):
            raise ValueError(f"unexpected dataset key {k!r}")
        return k[:-len("native")] + f"tile{self.tile}"

    def _native_ids(self, split):
        return list(self._ids[split])

    def _path(self, kind: str, site: str) -> str:
        return str(self.packed / kind / f"{site}.npy")

    def _geometry(self, f: dict, r0: int, c0: int) -> Geometry:
        a, b, x0, d, e, y0 = f["transform"]
        return Geometry(f["crs"], (a, b, x0 + a * c0 + b * r0, d, e, y0 + d * c0 + e * r0),
                        (self.tile, self.tile))

    def _target(self, site: str, r0: int, c0: int) -> torch.Tensor:
        a = _array(self._path("label", site))
        lab = read_window(a, r0, c0, self.tile)
        H, W = a.shape[-2:]
        pad = None
        if r0 + self.tile > H or c0 + self.tile > W:
            pad = np.ones((self.tile, self.tile), bool)
            pad[:H - r0, :W - c0] = False
        return torch.from_numpy(encode_years(lab, pad))

    def _load_target(self, sample_id):
        return self._target(*parse_id(sample_id))

    def _load(self, sample_id, modalities):
        site, r0, c0 = parse_id(sample_id)
        f = self.sites[site]
        if self.steps is None:
            x = read_window(_array(self._path("s2", site)), r0, c0, self.tile)
        else:   # a quarter pixel without a clear observation is 0 -> pixel_valid False
            a = _array(str(self.seasonal / "s2" / f"{site}.npy"))
            x = (read_window(a, r0, c0, self.tile) if self.series == "seasonal" else
                 np.stack([read_window(a[i], r0, c0, self.tile) for i in self.steps]))
        ok = (x != 0).any(1)                                                  # (T, S, S)
        geo = self._geometry(f, r0, c0)
        md = ModalityData(torch.from_numpy(x.astype(np.float32)), BANDS, self.time,
                          torch.from_numpy(ok.reshape(len(x), -1).any(1)), geo,
                          pixel_valid=torch.from_numpy(ok))
        return MTMMSample(sample_id, self.split, {"S2": md}, self._target(site, r0, c0), geo,
                          {"site": site, "country": site.rsplit("_", 1)[0]})
