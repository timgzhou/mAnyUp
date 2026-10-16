"""OlmoEarthEncoder on PASTIS: input packing rules, missing-frame handling, and agreement
with the ORIGINAL OlmoEarth v1 feature cache (made under torch 2.7 / olmoearth-pretrain 0.1.0;
a linear probe scores the same mIoU on it and on the new stack's features)."""
from pathlib import Path

import pytest
import torch

from manyup.data import mtmm_collate
from manyup.data.datasets.pastis import LEGACY_OLMOEARTH_S2, PastisDataset, monthly

ROOT = Path(__file__).resolve().parents[1] / "data" / "PASTIS-R"
CACHE = Path("~/projects/aip-gpleiss/timz/features/oe_base_s2_ps4_tile64/pastis_r_train").expanduser()
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
              pytest.mark.skipif(not ROOT.exists(), reason="PASTIS-R not on disk")]


@pytest.fixture(scope="module")
def E():
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()
    import manyup.encoders.olmoearth as E
    return E


def _enc(E, modality_map, ps=4, tile=64, impute=None, attention="flash"):
    impute = {"S2": E.LEGACY_S2_IMPUTE} if impute is None else impute
    return E.OlmoEarthEncoder("OLMOEARTH_V1_BASE", E.OlmoEarthInput(modality_map, impute),
                              ps, tile, name="base", attention=attention).cuda().eval()


def _complete_samples(ds, n):
    """n tiles whose S2 has data in December 2018, the only month PASTIS patches miss within
    the default window -- found from the metadata, without loading any imagery."""
    idx = [i for i in range(0, len(ds), 4)
           if any(str(d).startswith("201812")
                  for d in ds._meta[ds._index[i][0]]["properties"]["dates-S2"].values())]
    return [ds[i] for i in idx[:n]]


def test_packs_bands_by_name_in_olmoearth_order(E):
    enc = _enc(E, {"S2": "sentinel2_l2a", "S1A": "sentinel1"})
    ds = PastisDataset("val", monthly())
    b = mtmm_collate([ds[0]])
    sample, _ = enc.inputs.to_olmoearth(b)
    raw = b.modalities["S2"].data[0, 0, :, 0, 0]              # t=0, pixel (0,0), dataset order
    bands = b.modalities["S2"].bands
    order = enc.inputs.olmo["S2"].band_order                            # B02 B03 B04 B08 ... B01 B09
    got = torch.from_numpy(enc.inputs.normalizer.normalize(
        enc.inputs.olmo["S2"], raw[[bands.index(E.LEGACY_S2_IMPUTE.get(x, x)) for x in order]]
        .numpy()[None])).float()[0]
    assert torch.allclose(sample.sentinel2_l2a[0, 0, 0, 0], got)
    assert sample.sentinel1.shape[-1] == 2                       # VV, VH only (VV-VH dropped)
    enc.check(ds, "series")


def test_missing_band_without_imputation_fails_at_setup(E):
    enc = _enc(E, {"S2": "sentinel2_l2a"}, impute={})
    with pytest.raises(ValueError, match="B01"):
        enc.check(PastisDataset("val", {"S2": LEGACY_OLMOEARTH_S2}), "series")


def test_misaligned_modalities_are_refused(E):
    from manyup.data import All
    enc = _enc(E, {"S2": "sentinel2_l2a", "S1A": "sentinel1"})
    b = mtmm_collate([PastisDataset("val", {"S2": All(), "S1A": All()}, None)[0]])
    with pytest.raises(ValueError, match="aligned"):
        enc.inputs.to_olmoearth(b)


def test_missing_s2_month_uses_s1_alone(E):
    ds = PastisDataset("train", monthly())
    s = ds[0]                                    # patch 10000: S2 has no December 2018
    dec = (~s.modalities["S2"].frame_valid).nonzero().flatten().tolist()
    assert dec == [2] and s.modalities["S1A"].frame_valid[2]
    b = mtmm_collate([s])
    both = _enc(E, {"S2": "sentinel2_l2a", "S1A": "sentinel1"}).encode(b, "series")
    assert both.frame_valid[0].all() and torch.isfinite(both.tokens).all()
    sample, _ = E.OlmoEarthInput({"S2": "sentinel2_l2a", "S1A": "sentinel1"},
                                 {"S2": E.LEGACY_S2_IMPUTE}).to_olmoearth(b)
    assert (sample.sentinel2_l2a_mask[0, :, :, 2] == E.MaskValue.MISSING.value).all()
    assert E._has_missing(sample)
    s2_only = _enc(E, {"S2": "sentinel2_l2a"}).encode(b, "series")
    assert not s2_only.frame_valid[0, 2] and s2_only.frame_valid[0].sum() == 11


@pytest.mark.skipif(not CACHE.exists(), reason="original v1 cache not available")
@pytest.mark.parametrize("mode", ["series", "per_timestep"])
def test_matches_original_v1_cache(E, mode):
    idx = [0, 1, 2, 3, 2999, 5819]
    ds = PastisDataset("train", {"S2": LEGACY_OLMOEARTH_S2})
    fs = _enc(E, {"S2": "sentinel2_l2a"}).encode(mtmm_collate([ds[i] for i in idx]), mode)
    assert fs.ids == [ds.sample_id(i) for i in idx]
    assert fs.geometry[0].shape == (16, 16) and fs.geometry[0].gsd == (40.0, 40.0)
    if mode != "series":             # the cache was made in series mode
        return
    new = fs.tokens.half().float().cpu()
    cache = torch.stack([torch.load(CACHE / f"{i}.pt") for i in idx]).float()
    cos = torch.nn.functional.cosine_similarity(new.reshape(-1, 768), cache.reshape(-1, 768), dim=-1)
    assert cos.min() > 0.995 and ((new - cache).norm() / cache.norm()) < 0.02


def test_flash_features_do_not_depend_on_the_batch(E):
    """A sample with a missing month must get the same features alone and batched: this is
    what OlmoEarth's SDPA masked path gets wrong in eval mode (padding leaks in)."""
    ds = PastisDataset("train", monthly())
    missing, complete = ds[0], _complete_samples(ds, 3)
    enc = _enc(E, {"S2": "sentinel2_l2a", "S1A": "sentinel1"})
    alone = enc.encode(mtmm_collate([missing]), "series").tokens[0]
    batched = enc.encode(mtmm_collate([missing] + complete), "series").tokens[0]
    assert torch.equal(alone, batched)


def test_sdpa_refuses_batched_missing_tokens_and_matches_flash_otherwise(E):
    ds = PastisDataset("train", monthly())
    sdpa = _enc(E, {"S2": "sentinel2_l2a", "S1A": "sentinel1"}, attention="sdpa")
    with pytest.raises(ValueError, match="flash"):
        sdpa.encode(mtmm_collate([ds[0], ds[4]]), "series")
    complete = mtmm_collate(_complete_samples(ds, 2))
    a = sdpa.encode(complete, "series").tokens.float()
    b = _enc(E, {"S2": "sentinel2_l2a", "S1A": "sentinel1"}).encode(complete, "series").tokens.float()
    assert ((a - b).norm() / a.norm()) < 0.01          # kernel numerics only (~0.3%)


def test_mid_timestamp_is_the_middle_of_each_step(E):
    b = mtmm_collate([PastisDataset("val", monthly())[0]])
    m = {"S2": "sentinel2_l2a", "S1A": "sentinel1"}
    start, _ = E.OlmoEarthInput(m, {"S2": E.LEGACY_S2_IMPUTE}).to_olmoearth(b)
    mid, _ = E.OlmoEarthInput(m, {"S2": E.LEGACY_S2_IMPUTE}, timestamp="mid").to_olmoearth(b)
    assert (start.timestamps[..., 0] == 1).all()                       # months start on the 1st
    assert ((mid.timestamps[..., 0] >= 14) & (mid.timestamps[..., 0] <= 16)).all()
    assert torch.equal(mid.timestamps[..., 1:], start.timestamps[..., 1:])   # same month, year
