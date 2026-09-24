"""PASTIS data setup shared by the OlmoEarth scripts (extraction, fine-tuning).

CLI/config modality names ("sentinel2_l2a", "sentinel1") are the ones in cache names and
past runs; they map to PastisDataset modalities and OlmoEarth modalities here. `temporal`
names a time setup:
  monthly  S2 median / S1A mean on the Oct 2018 - Sep 2019 month grid, empty months masked
  legacy   OlmoEarth's own PASTIS prep for S2 (month means, first 12 non-empty months);
           S2 only, since that prep aligned S1 by S2's dates
"""
from __future__ import annotations

import torch

from exp.common.paths import DATA
from manyup.data import Tiling
from manyup.data.feature_cache import LegacyReconstruction
from manyup.data.time import month_bounds
from manyup.data.datasets.pastis import LEGACY_OLMOEARTH_S2, PastisDataset, monthly
from manyup.data.sample import MTMMBatch

# config name -> (PastisDataset modality, OlmoEarth modality)
MODALITIES = {"sentinel2_l2a": ("S2", "sentinel2_l2a"), "sentinel1": ("S1A", "sentinel1")}
TEMPORAL = ("monthly", "legacy")
IGNORE_LABEL = -1          # what PASTIS's void class (19) becomes for the losses/metrics
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}   # feature-cache folder names


def temporal_setup(name: str, modalities: list[str]) -> dict:
    """PastisDataset modality -> TemporalPolicy for these config modalities."""
    if name == "monthly":
        full = monthly()
        return {MODALITIES[m][0]: full[MODALITIES[m][0]] for m in modalities}
    if name == "legacy":
        if modalities != ["sentinel2_l2a"]:
            raise ValueError("temporal 'legacy' is S2-only: OlmoEarth's prep aligned S1 by "
                             "S2's dates, so legacy S1 cannot be reproduced; use 'monthly'")
        return {"S2": LEGACY_OLMOEARTH_S2}
    raise ValueError(f"temporal {name!r} not in {TEMPORAL}")


def make_dataset(split: str, modalities: list[str], temporal: str, image_size: int,
                 with_s2: bool = False) -> PastisDataset:
    """image_size 64: 64 px tiles in the legacy order; 128: whole patches. with_s2 adds S2
    (on the same time setup) when a head needs it as guidance but the model does not."""
    mods = list(modalities)
    if with_s2 and "sentinel2_l2a" not in mods:
        mods = ["sentinel2_l2a"] + mods
    tiling = {64: Tiling(64, "col-major"), 128: None}[image_size]
    return PastisDataset(split, temporal_setup(temporal, mods), tiling)


def labels(batch: MTMMBatch) -> torch.Tensor:
    """(B, H, W) targets with the void class as IGNORE_LABEL."""
    t = batch.target.clone()
    t[t == PastisDataset.SPEC.target.ignore_index] = IGNORE_LABEL
    return t



def legacy_reconstruction(split: str, image_size: int) -> LegacyReconstruction:
    """ids / validity / dates for a feature cache made before caches recorded them. Those
    were all extracted from OlmoEarth's legacy prep, in the order PastisDataset reproduces,
    with the prep's months (first 12 non-empty, every step valid) as the time axis."""
    ds = make_dataset(split, ["sentinel2_l2a"], "legacy", image_size)
    prep = DATA / ("pastis_olmoearth" if image_size == 64 else "pastis128_olmoearth")
    months = torch.load(prep / f"pastis_r_{SPLIT_DIRS[split]}" / "months.pt")   # (N, 12) yyyymm
    if len(months) != len(ds):
        raise ValueError(f"legacy prep has {len(months)} samples, dataset {len(ds)}")
    bounds = [[month_bounds(int(m) // 100, int(m) % 100) for m in row] for row in months.tolist()]
    t = torch.tensor(bounds, dtype=torch.long)                                  # (N, 12, 2)
    return LegacyReconstruction(ids=[ds.sample_id(i) for i in range(len(ds))],
                                frame_valid=torch.ones(months.shape, dtype=torch.bool),
                                t_start=t[..., 0], t_end=t[..., 1])
