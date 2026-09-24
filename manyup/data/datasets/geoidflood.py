"""GEOID-Flood (links-ads/geoid-flood): flood segmentation from S1 pre/post pairs.

Raw layout (data/GEOID-Flood-full/geoid-flood): <event>/<layer>/<event>-<tile>_<layer>[_<pre|post>_<datetime>].tif,
1024x1024 tiles at 10 m in UTM. This download has s1grd (VV, VH, LINEAR sigma0), label and
validity; s1rtc / s2l2a / dem are not on disk, so they are not offered.

Samples are CHIPS of a tile, chosen by a named ChipSelection (chip size, minimum mapped
fraction, required class) and listed in a chip index built once:
    python -m manyup.data.datasets.geoidflood --build_index --size 128
Splits come from the dataset's per-TILE assignment in data_tiles_s256_st128.csv (51 of 210
events span several splits, so the split is not an event property). Chip ids are
"<event>-<tile>@r<row>c<col>". Reads are windowed, so a chip never loads its whole tile.

S1 is returned in dB (units the OlmoEarth S1 encoder was pretrained on): 10*log10(linear),
nodata (linear <= 0) set to DB_FLOOR and marked invalid in pixel_valid.
Labels: 0 background, 1 permanent water, 2 flood, 255 outside the mapped area (ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from manyup.data.dataset import MTMMDataset
from manyup.data.geometry import Geometry
from manyup.data.sample import ModalityData, MTMMSample
from manyup.data.spec import DatasetSpec, ModalitySpec, TargetSpec
from manyup.data.temporal import TemporalPolicy
from manyup.data.time import TimeAxis, yyyymmdd_to_days

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "GEOID-Flood-full" / "geoid-flood"
SPLIT_CSV = "data_tiles_s256_st128.csv"
TILE_PX = 1024
UNMAPPED = 255
DB_FLOOR = -50.0
_DATE = re.compile(r"_(\d{8})T\d{6}")


@dataclass(frozen=True)
class ChipSelection:
    size: int                     # chip side in pixels; divides 1024
    min_mapped: float = 0.5       # keep chips with >= this fraction of labelled pixels
    require_class: int | None = 2  # ... and at least one pixel of this class (2 = flood)

    @property
    def key(self) -> str:
        req = "" if self.require_class is None else f"_class{self.require_class}"
        return f"c{self.size}_mapped{self.min_mapped:g}{req}"


# Chips per split for each selection, recorded when its index was built. The dataset refuses
# a selection whose counts are not declared here.
CHIP_COUNTS: dict[str, dict[str, int]] = {
    "c128_mapped0.5_class2": {"train": 79738, "val": 10957, "test": 17908},
    "c64_mapped0.5_class2": {"train": 211725, "val": 32259, "test": 46048},
    "c16_mapped0.5_class2": {"train": 1739110, "val": 323043, "test": 341410},
}


def to_db(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear sigma0 -> (dB clipped to [DB_FLOOR, 50], valid mask of linear > 0)."""
    x = np.asarray(linear, dtype=np.float32)
    valid = np.isfinite(x) & (x > 0)
    out = np.full_like(x, DB_FLOOR)
    out[valid] = 10.0 * np.log10(x[valid])
    return np.clip(out, DB_FLOOR, 50.0), valid


def _tile_files(root: Path, event: str, tile: str) -> dict | None:
    def one(layer, pas=None):
        pat = f"{event}-{tile}_{layer}" + (f"_{pas}_*" if pas else "") + ".tif"
        hits = sorted((root / event / layer).glob(pat))
        return str(hits[0]) if hits else None
    files = {"label": one("label"), "pre": one("s1grd", "pre"), "post": one("s1grd", "post")}
    return files if all(files.values()) else None


def build_index(root: Path, sel: ChipSelection) -> Path:
    """Scan every labelled tile once and list the chips `sel` keeps, per split."""
    split_of = {}
    with open(root / SPLIT_CSV) as f:
        for row in csv.DictReader(f):
            if row["label_id"].endswith("_label"):
                split_of[row["label_id"][:-len("_label")]] = row["split"]
    if TILE_PX % sel.size:
        raise ValueError(f"chip size {sel.size} must divide {TILE_PX}")
    index = {"selection": sel.key, "splits": {}, "tiles": {}, "excluded": []}
    for lab in sorted(root.glob("*/label/*_label.tif")):
        stem = lab.name[:-len("_label.tif")]
        split = split_of.get(stem)
        event, tile = stem.rsplit("-", 1)
        files = _tile_files(root, event, tile)
        if split is None or files is None:
            index["excluded"].append([stem, "no split" if split is None else "missing s1grd pass"])
            continue
        try:
            with rasterio.open(files["label"]) as r:
                label = r.read(1)
        except Exception as e:                       # corrupt / truncated COG
            index["excluded"].append([stem, f"unreadable label: {e}"])
            continue
        index["tiles"][stem] = files
        n = TILE_PX // sel.size
        for r_ in range(n):
            for c_ in range(n):
                y, x = r_ * sel.size, c_ * sel.size
                lc = label[y:y + sel.size, x:x + sel.size]
                if (lc != UNMAPPED).mean() < sel.min_mapped:
                    continue
                if sel.require_class is not None and not (lc == sel.require_class).any():
                    continue
                index["splits"].setdefault(split, []).append(f"{stem}@r{y}c{x}")
    out = root / f"chip_index_{sel.key}.json"
    out.write_text(json.dumps(index))
    print(f"{out}: " + ", ".join(f"{s} {len(v)}" for s, v in index["splits"].items())
          + f"; {len(index['excluded'])} tiles excluded")
    return out


class GeoidFloodDataset(MTMMDataset):
    SPEC = DatasetSpec(
        name="geoidflood",
        modalities={"S1GRD": ModalitySpec("S1GRD", ("VV", "VH"), "dB", 10.0, "fixed", (2, 2),
                                          "day", pool_reducer="mean", nodata=DB_FLOOR)},
        target=TargetSpec("segmentation", "class", ("background", "permanent water", "flood"),
                          ignore_index=UNMAPPED),
        native_shape=(TILE_PX, TILE_PX),
        split_sizes={},                   # per chip selection, see CHIP_COUNTS
        georeferenced=True,
        notes=("samples are chips listed in chip_index_<selection>.json",
               "S1 frames: pre-event, post-event (dates from the file names)"))

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy], chips: ChipSelection,
                 validate: bool = True, root: str | Path = DEFAULT_ROOT):
        self.root, self.chips = Path(root), chips
        if chips.key not in CHIP_COUNTS:
            raise ValueError(f"chip selection {chips.key} has no declared counts; build its "
                             f"index and add them to CHIP_COUNTS")
        path = self.root / f"chip_index_{chips.key}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run python -m "
                                    f"manyup.data.datasets.geoidflood --build_index --size {chips.size}")
        self._index_data = json.loads(path.read_text())
        # this instance's spec: chip-sized samples, counts of this selection
        self.SPEC = replace(type(self).SPEC, native_shape=(chips.size, chips.size),
                            split_sizes=CHIP_COUNTS[chips.key])
        super().__init__(split, modalities, tiling=None, validate=validate)

    @property
    def key(self) -> str:
        return super().key.replace("_native", f"_{self.chips.key}")

    def _native_ids(self, split):
        return list(self._index_data["splits"].get(split, []))

    def _load(self, chip_id, modalities):
        stem, rc = chip_id.split("@r")
        row, col = map(int, rc.split("c"))
        files, s = self._index_data["tiles"][stem], self.chips.size
        window = Window(col, row, s, s)
        frames, valid, days = [], [], []
        for pas in ("pre", "post"):
            with rasterio.open(files[pas]) as r:
                db, ok = to_db(r.read(window=window))              # (2, s, s)
                crs, transform = str(r.crs), r.window_transform(window)
            frames.append(db)
            valid.append(ok.all(0))
            days.append(yyyymmdd_to_days(_DATE.search(Path(files[pas]).name).group(1)))
        with rasterio.open(files["label"]) as r:
            label = r.read(1, window=window).astype(np.int64)
        geo = Geometry(crs, tuple(transform)[:6], (s, s))
        s1 = ModalityData(torch.from_numpy(np.stack(frames)), ("VV", "VH"),
                          TimeAxis.instants(days), torch.ones(2, dtype=torch.bool), geo,
                          pixel_valid=torch.from_numpy(np.stack(valid)))
        return MTMMSample(chip_id, self.split, {"S1GRD": s1}, torch.from_numpy(label), geo,
                          {"event": stem.rsplit("-", 1)[0]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a GEOID-Flood chip index.")
    ap.add_argument("--build_index", action="store_true", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--min_mapped", type=float, default=0.5)
    ap.add_argument("--require_class", type=int, default=2)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    a = ap.parse_args()
    build_index(Path(a.root), ChipSelection(a.size, a.min_mapped, a.require_class))
