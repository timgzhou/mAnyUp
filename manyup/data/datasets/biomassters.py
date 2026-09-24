"""BioMassters (Nascetti et al. 2023): above-ground forest biomass regression, Finland.

Read from GeoBench-v2's tortilla shards (data/BioMassters -> the copy in
ever-changing-modalities), directly with tacoreader + rasterio, so geobench_v2 itself is not
needed. Each sample holds per-month GeoTIFFs plus the AGBM target:
    S1  (4, 256, 256) float32 dB: VV/VH ascending, VV/VH descending; -9999 = nodata (pixels)
    S2  (11, 256, 256) uint16: 10 L2A bands + cloud probability 0-255 (band 11 -> aux layer)
    AGBM (1, 256, 256) float32, t/ha
Facts the spec encodes (checked by tests/test_biomassters.py):
  - 12 monthly slots, September -> August of ONE year per chip; S1 has all 12, S2 has 5-12
    (missing months are absent rows -> invalid frames).
  - WHICH year (2016-2021) is not in the data, so time precision is "month_of_year" on the
    placeholder year PLACEHOLDER_YEAR (Sep-Dec) / +1 (Jan-Aug). Never read the year back.
  - GeoTIFFs carry a placeholder EPSG:4326 transform in pixel units: geometry is unknown.
  - GeoBench-v2 packs 13 samples twice (9 train, 1 val, 3 test; byte-identical copies in the
    same split). The index keeps the first copy and lists the rest under "duplicates".
Chip ids (the 8-hex BioMassters id) and their shard positions come from an index built once:
    python -m manyup.data.datasets.biomassters --build_index
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from functools import cached_property
from pathlib import Path

import numpy as np
import rasterio
import tacoreader
import torch

from manyup.data.dataset import MTMMDataset
from manyup.data.geometry import Geometry
from manyup.data.sample import ModalityData, MTMMSample
from manyup.data.spec import DatasetSpec, ModalitySpec, TargetSpec
from manyup.data.temporal import TemporalPolicy
from manyup.data.time import TimeAxis, month_bounds

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "BioMassters"
INDEX_NAME = "biomassters_index.json"
MONTHS = ("September", "October", "November", "December", "January", "February", "March",
          "April", "May", "June", "July", "August")
PLACEHOLDER_YEAR = 2000          # Sep-Dec in 2000, Jan-Aug in 2001; the real year is unknown
S1_NODATA = -9999.0
SPLITS = {"train": "train", "val": "validation", "test": "test"}   # ours -> tortilla's
S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
S1_BANDS = ("VV_asc", "VH_asc", "VV_desc", "VH_desc")


def _month_days(i: int) -> tuple[int, int]:
    """Slot i (0 = September) on the placeholder biomass year."""
    m = dt.datetime.strptime(MONTHS[i], "%B").month
    return month_bounds(PLACEHOLDER_YEAR + (m < 9), m)


def _shards(root: Path) -> list[str]:
    return sorted(str(p) for p in root.glob("*.tortilla"))


def build_index(root: Path) -> Path:
    """chip id -> (split, position in the concatenated shards), read once from every sample.
    A chip id seen again is recorded under "duplicates" instead of being indexed twice."""
    ds = tacoreader.load(_shards(root))
    inv = {v: k for k, v in SPLITS.items()}
    index, seen, dups = {k: [] for k in SPLITS}, set(), []
    for pos in range(len(ds)):
        inner = ds.read(pos)
        chip = inner["tortilla:id"].iloc[0].split("_")[0]
        split = inv[inner["tortilla:data_split"].iloc[0]]
        if chip in seen:
            dups.append([chip, split, pos])
            continue
        seen.add(chip)
        index[split].append([chip, pos])
    out = root / INDEX_NAME
    out.write_text(json.dumps({"splits": index, "duplicates": dups}))
    print(f"{out}: " + ", ".join(f"{k} {len(v)}" for k, v in index.items())
          + f"; {len(dups)} duplicate copies skipped")
    return out


class BiomasstersDataset(MTMMDataset):
    SPEC = DatasetSpec(
        name="biomassters",
        modalities={
            "S1": ModalitySpec("S1", S1_BANDS, "dB", 10.0, "monthly", (12, 12), "month_of_year",
                               pool_reducer="mean", nodata=S1_NODATA),
            "S2": ModalitySpec("S2", S2_BANDS, "reflectance*1e4", 10.0, "monthly", (12, 12),
                               "month_of_year", pool_reducer="median",
                               aux_layers=("cloud_prob",)),
        },
        target=TargetSpec("regression", "t/ha"),
        native_shape=(256, 256),
        split_sizes={"train": 4002, "val": 1738, "test": 2773},   # after de-duplication
        georeferenced=False,
        notes=("year per chip unknown (2016-2021): month_of_year on a placeholder year",
               "13 GeoBench-v2 duplicates dropped (byte-identical, same split)",
               "GeoBench-v2 splits", "S2 cloud probability (0-255) as aux layer cloud_prob"))

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy],
                 validate: bool = True, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        path = self.root / INDEX_NAME
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run python -m "
                                    "manyup.data.datasets.biomassters --build_index")
        self._pos = {chip: pos for chip, pos in json.loads(path.read_text())["splits"][split]}
        super().__init__(split, modalities, tiling=None, validate=validate)

    @cached_property
    def _shards(self):
        return tacoreader.load(_shards(self.root))

    def _native_ids(self, split):
        return list(self._pos)

    def _load(self, chip, modalities):
        inner = self._shards.read(self._pos[chip])
        rows = {(r["modality"], r["month"]): r["internal:subfile"] for _, r in inner.iterrows()}
        geo = Geometry.unknown(self.SPEC.native_shape)
        starts, ends = zip(*(_month_days(i) for i in range(12)))
        out = {}
        for m in modalities:
            C = len(self.SPEC.modalities[m].bands)
            data = torch.zeros(12, C, 256, 256)
            present = torch.zeros(12, dtype=torch.bool)
            pixel_valid = torch.zeros(12, 256, 256, dtype=torch.bool)
            cloud = torch.zeros(12, 256, 256)
            for i, month in enumerate(MONTHS):
                src = rows.get((m, month))
                if src is None:
                    continue
                with rasterio.open(src) as f:
                    a = torch.from_numpy(f.read().astype(np.float32))
                present[i] = True
                if m == "S1":
                    ok = (a != S1_NODATA).all(0)
                    data[i] = torch.where(ok, a, torch.zeros(()))
                    pixel_valid[i] = ok
                else:
                    data[i], cloud[i] = a[:C], a[C]
                    pixel_valid[i] = True
            aux = {"cloud_prob": cloud} if m == "S2" else {}
            out[m] = ModalityData(data, self.SPEC.modalities[m].bands,
                                  TimeAxis(torch.tensor(starts), torch.tensor(ends), "month_of_year"),
                                  present, geo, pixel_valid=pixel_valid, aux=aux)
        agbm = inner[inner["modality"] == "AGBM"]["internal:subfile"].iloc[0]
        with rasterio.open(agbm) as f:
            target = torch.from_numpy(f.read(1).astype(np.float32))
        return MTMMSample(chip, self.split, out, target, geo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the BioMassters chip index.")
    ap.add_argument("--build_index", action="store_true", required=True)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    build_index(Path(ap.parse_args().root))
