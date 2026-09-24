"""PASTIS-R (Garnot et al. 2022): crop-type semantic segmentation from S2 + S1 time series.

Raw layout (data/PASTIS-R): DATA_S2/S2_<id>.npy (T, 10, 128, 128) int16 reflectance*1e4,
DATA_S1A/S1A_<id>.npy (T, 3, 128, 128) float16 dB, ANNOTATIONS/TARGET_<id>.npy (3, 128, 128)
with the semantic class in channel 0, per-modality acquisition dates in metadata.geojson.

Facts the spec encodes (checked by tests/test_pastis.py):
  - S2 and S1A have their OWN dates: S2 38-61 frames (2018-09-17..2019-10-27), S1A 65-71
    (2018-10-01..2019-12-04). They never share a date list.
  - S1D is not on disk here, so it is not offered.
  - Splits are the benchmark folds: train = folds 1-3, val = 4, test = 5, enumerated fold by
    fold in metadata order -- the order OlmoEarth's PASTIS prep used, so legacy feature-cache
    indices map to the same samples.
  - metadata.geojson gives each patch as a ~1376 m bounding box in EPSG:2154 (the 1280 m UTM
    patch, reprojected), so geometry is approximate: centred on that box, 10 m pixels.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from manyup.data.dataset import MTMMDataset
from manyup.data.geometry import Geometry
from manyup.data.sample import ModalityData, MTMMSample
from manyup.data.spatial import Tiling
from manyup.data.spec import DatasetSpec, ModalitySpec, TargetSpec
from manyup.data.temporal import Indices, Pool, TemporalPolicy
from manyup.data.time import TimeAxis, yyyymmdd_to_days

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "PASTIS-R"
FOLDS = {"train": (1, 2, 3), "val": (4,), "test": (5,)}
CLASS_NAMES = ("Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
               "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
               "Winter triticale", "Winter durum wheat", "Fruits, vegetables, flowers",
               "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
               "Sorghum", "Void label")

# The one 12-month window in which both S2 and S1A have data (S1A starts 2018-10). S2
# December 2018 is missing for 22% of patches; those samples get a masked slot there.
AGRI_YEAR = ("2018-10-01", "2019-09-30")


def monthly(window: tuple[str, str] = AGRI_YEAR) -> dict[str, TemporalPolicy]:
    """The default: calendar months on a fixed window, each modality with its spec reducer
    (S2 median, S1A mean)."""
    return {"S2": Pool("month", window=window), "S1A": Pool("month", window=window)}


# What OlmoEarth's PASTIS prep did for S2: mean over each month that HAS frames, then the
# first 12 such months -- so for 531 of 2433 patches (S2 Dec 2018 missing) step t is a
# different calendar month than for the rest. Kept only to reproduce legacy caches.
LEGACY_OLMOEARTH_S2 = Pool("month", "mean") + Indices(list(range(12)))


class PastisDataset(MTMMDataset):
    SPEC = DatasetSpec(
        name="pastis",
        modalities={
            "S2": ModalitySpec("S2", ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                                      "B11", "B12"),
                               "reflectance*1e4", 10.0, "irregular", (38, 61), "day",
                               pool_reducer="median"),
            "S1A": ModalitySpec("S1A", ("VV", "VH", "VV-VH"), "dB", 10.0, "irregular",
                                (65, 71), "day", pool_reducer="mean"),
        },
        target=TargetSpec("segmentation", "class", CLASS_NAMES, ignore_index=19),
        native_shape=(128, 128),
        split_sizes={"train": 1455, "val": 482, "test": 496},
        georeferenced=True,
        notes=("geometry approximate: EPSG:2154 bounding box of a reprojected UTM patch",
               "S1D not downloaded", "S1A channel 3 is VV-VH in dB"))

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy],
                 tiling: Tiling | None = Tiling(64, "col-major"), validate: bool = True,
                 root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        super().__init__(split, modalities, tiling, validate)

    @cached_property
    def _meta(self) -> dict[str, dict]:
        feats = json.load(open(self.root / "metadata.geojson"))["features"]
        return {str(f["properties"]["ID_PATCH"]): f for f in feats}

    def _native_ids(self, split):
        return [pid for fold in FOLDS[split] for pid, f in self._meta.items()
                if f["properties"]["Fold"] == fold]

    def _geometry(self, pid: str) -> Geometry:
        g = self._meta[pid]["geometry"]      # Polygon or single-part MultiPolygon
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        cx = (min(p[0] for p in ring) + max(p[0] for p in ring)) / 2
        cy = (min(p[1] for p in ring) + max(p[1] for p in ring)) / 2
        H, W = self.SPEC.native_shape
        return Geometry("EPSG:2154", (10.0, 0.0, cx - 5 * W, 0.0, -10.0, cy + 5 * H), (H, W),
                        approximate=True)

    def _load_target(self, pid):
        return torch.from_numpy(np.load(self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0]
                                .astype(np.int64))

    def _load(self, pid, modalities):
        props, geo = self._meta[pid]["properties"], self._geometry(pid)
        out = {}
        for m in modalities:
            dates = props[f"dates-{m}"]
            days = [yyyymmdd_to_days(dates[k]) for k in sorted(dates, key=int)]
            data = torch.from_numpy(np.load(self.root / f"DATA_{m}" / f"{m}_{pid}.npy"))
            if data.shape[0] != len(days):
                raise ValueError(f"pastis/{pid}/{m}: {data.shape[0]} frames, {len(days)} dates")
            out[m] = ModalityData(data, self.SPEC.modalities[m].bands, TimeAxis.instants(days),
                                  torch.ones(len(days), dtype=torch.bool), geo)
        return MTMMSample(pid, self.split, out, self._load_target(pid), geo,
                          {"fold": props["Fold"], "tile": props["TILE"]})
