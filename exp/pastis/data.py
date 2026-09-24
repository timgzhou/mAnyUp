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
from manyup.data.feature_cache import FeatureCache, LegacyReconstruction
from manyup.data.guidance import LEGACY_S1_2, LEGACY_S2_13, GuidanceSpec, build_guidance
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


# ---- guidance for upsamplers / guided probes ------------------------------------------------
# guidance_mod (as in checkpoints and CLI: s2 | s1 | s2s1) -> PASTIS modalities, channel order
GUIDANCE_MODS = {"s2": ("S2",), "s1": ("S1A",), "s2s1": ("S2", "S1A")}
RGB = ("B04", "B03", "B02")
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def guidance_spec(kind: str, guidance_mod: str, time_pool: str | None) -> GuidanceSpec:
    """kind "rgb": natural-colour S2 for stock AnyUp; "full": the 13-band S2 stack (+ S1
    VV/VH for s1/s2s1) mAnyUp/timAnyUp were trained on. time_pool None = per timestep."""
    if kind == "rgb":
        return GuidanceSpec({"S2": RGB}, time_pool)
    if kind == "full":
        full = {"S2": LEGACY_S2_13, "S1A": LEGACY_S1_2}
        return GuidanceSpec({m: full[m] for m in GUIDANCE_MODS[guidance_mod]}, time_pool)
    raise ValueError(f"guidance kind {kind!r} not in (rgb, full)")


def imagenet_norm(rgb01: torch.Tensor) -> torch.Tensor:
    """[0,1] RGB, (...,3,H,W) -> ImageNet-normalized, as stock AnyUp expects."""
    return (rgb01 - _IMAGENET_MEAN) / _IMAGENET_STD


class PastisGuidance:
    """Guidance by sample id, from the SAME time setup and sample size as a feature cache
    (read from its meta.json), so features and guidance describe the same data."""

    def __init__(self, split: str, cache_meta: dict, spec: GuidanceSpec, anyup_rgb: bool = False):
        temporal = cache_meta.get("temporal", "legacy")
        mods = [m for m, (pm, _) in MODALITIES.items() if pm in spec.bands]
        self.ds = make_dataset(split, mods, temporal, cache_meta.get("image_size", 64))
        self.spec, self.anyup_rgb = spec, anyup_rgb

    def __call__(self, sample_id: str) -> torch.Tensor:
        g = build_guidance(self.ds[self.ds.index_of(sample_id)].modalities, self.spec)
        return imagenet_norm(g) if self.anyup_rgb else g


def open_cache(root, name: str, split: str) -> FeatureCache:
    """A PASTIS feature cache by name; caches made before ids/dates were recorded get the
    legacy reconstruction (their meta has no "temporal", i.e. they are legacy caches)."""
    import json
    from pathlib import Path
    meta = json.loads((Path(root).expanduser() / name / "meta.json").read_text())
    split_dir = f"pastis_r_{SPLIT_DIRS[split]}"
    has_ids = (Path(root).expanduser() / name / split_dir / "ids.txt").exists()
    legacy = None if has_ids else legacy_reconstruction(split, meta.get("image_size", 64))
    return FeatureCache(root, name, split_dir, legacy)
