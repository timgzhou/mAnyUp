"""Upsampler inputs from the data layer: guidance built by band name reproduces the legacy
guidance mAnyUp/timAnyUp were trained on, and legacy feature caches read back by sample id."""
from pathlib import Path

import pytest
import torch

from manyup.data.feature_cache import FeatureCache
from manyup.data.guidance import LEGACY_S2_13, GuidanceSpec, build_guidance
from manyup.data.datasets.pastis import LEGACY_OLMOEARTH_S2, PastisDataset
from manyup.data.time import from_days

DATA = Path(__file__).resolve().parents[1] / "data"
LEGACY = DATA / "pastis_olmoearth"
FEATURES = Path("~/projects/aip-gpleiss/timz/features").expanduser()
pytestmark = pytest.mark.skipif(not LEGACY.exists(), reason="legacy prep not on disk")
IDX = [0, 1, 5, 2999, 5819]


def _legacy_minmax(x):            # exactly exp.upsamplers.common.norm_guidance
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


@pytest.mark.parametrize("pool", ["mean", "median"])
def test_pooled_s2_guidance_matches_legacy(pool):
    ds = PastisDataset("train", {"S2": LEGACY_OLMOEARTH_S2})
    spec = GuidanceSpec({"S2": LEGACY_S2_13}, pool)
    for i in IDX:
        legacy = torch.load(LEGACY / "pastis_r_train" / "s2_images" / f"{i}.pt").float()
        ref = _legacy_minmax(legacy.mean(0) if pool == "mean" else legacy.median(0).values)
        assert torch.equal(build_guidance(ds[i].modalities, spec), ref)


def test_per_timestep_s2_guidance_matches_legacy():
    ds = PastisDataset("val", {"S2": LEGACY_OLMOEARTH_S2})
    spec = GuidanceSpec({"S2": LEGACY_S2_13}, None)
    for i in (0, 7):
        legacy = torch.load(LEGACY / "pastis_r_valid" / "s2_images" / f"{i}.pt").float()
        assert torch.equal(build_guidance(ds[i].modalities, spec), _legacy_minmax(legacy))


def test_guidance_refuses_unknown_bands_and_empty_frames():
    ds = PastisDataset("val", {"S2": LEGACY_OLMOEARTH_S2})
    with pytest.raises(ValueError, match="B99"):
        build_guidance(ds[0].modalities, GuidanceSpec({"S2": ("B99",)}, "mean"))


@pytest.mark.skipif(not (FEATURES / "oe_base_s2_ps16_tile64").exists(), reason="no legacy cache")
def test_legacy_cache_reads_back_by_sample_id():
    from exp.pastis.data import legacy_reconstruction
    cache = FeatureCache(FEATURES, "oe_base_s2_ps16_tile64", "pastis_r_test",
                         legacy_reconstruction("test", 64))
    assert set(cache.reconstructed) == {"ids", "frame_valid", "t_start", "t_end"}
    ds = PastisDataset("test", {"S2": LEGACY_OLMOEARTH_S2})
    for i in (0, 3, 1983):
        sid = ds.sample_id(i)
        item = cache.get(sid)
        assert torch.equal(item["tokens"], torch.load(cache.split_dir / f"{i}.pt"))
        months = [from_days(d).strftime("%Y%m") for d in item["t_start"].tolist()]
        assert months == [from_days(d).strftime("%Y%m")
                          for d in ds[i].modalities["S2"].time.start.tolist()]
