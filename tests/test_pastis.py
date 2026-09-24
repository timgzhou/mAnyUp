"""PastisDataset: conformance, the declared facts, and parity with OlmoEarth's legacy prep
(data/pastis_olmoearth), which every S2 result so far was computed from."""
from pathlib import Path

import pytest
import torch

from manyup.data import All, Tiling
from manyup.data.datasets.pastis import (AGRI_YEAR, LEGACY_OLMOEARTH_S2, PastisDataset,
                                         monthly)
from manyup.data.time import from_days
from tests.mtmm_conformance import check_dataset

ROOT = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (ROOT / "PASTIS-R").exists(), reason="PASTIS-R not on disk")

LEGACY = ROOT / "pastis_olmoearth"
LEGACY_SPLIT = {"train": "train", "val": "valid", "test": "test"}
# legacy 13-band order: B01<-B02, B02..B08, B8A, B09<-B8A, B10<-B11, B11, B12 (3 imputed)
LEGACY_REAL_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]


def make(split, modalities, tiling):
    return PastisDataset(split, modalities, tiling)


def test_conformance():
    check_dataset(make, PastisDataset.SPEC, Tiling(64, "col-major"), AGRI_YEAR)


def test_default_is_64px_tiles_and_monthly_with_spec_reducers():
    ds = PastisDataset("val", monthly())
    assert len(ds) == 4 * 482
    s = ds[0]
    assert s.modalities["S2"].history == ("month-median@201810-201909",)
    assert s.modalities["S1A"].history == ("month-mean@201810-201909",)
    for md in s.modalities.values():   # both on the same 12 calendar months
        assert [from_days(d).strftime("%Y%m") for d in md.time.start.tolist()] == \
               [f"2018{m}" for m in ("10", "11", "12")] + [f"2019{m:02d}" for m in range(1, 10)]


def test_s1_and_s2_keep_their_own_dates():
    s = PastisDataset("test", {"S2": All(), "S1A": All()}, None)[0]
    s2, s1 = s.modalities["S2"].time.start.tolist(), s.modalities["S1A"].time.start.tolist()
    assert len(s2) != len(s1) and s2 != s1[:len(s2)]


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy prep not on disk")
@pytest.mark.parametrize("split, idx", [("train", 0), ("train", 1), ("train", 2), ("train", 3),
                                        ("train", 2999), ("train", 5819), ("val", 7),
                                        ("test", 0), ("test", 1983)])
def test_s2_matches_legacy_olmoearth_prep(split, idx):
    ds = PastisDataset(split, {"S2": LEGACY_OLMOEARTH_S2})
    s = ds[idx]
    d = LEGACY / f"pastis_r_{LEGACY_SPLIT[split]}"
    legacy = torch.load(d / "s2_images" / f"{idx}.pt")[:, LEGACY_REAL_BANDS]
    assert torch.equal(s.modalities["S2"].data, legacy), "S2 values differ from legacy prep"
    months = [int(from_days(x).strftime("%Y%m")) for x in s.modalities["S2"].time.start.tolist()]
    assert months == torch.load(d / "months.pt")[idx].tolist()
    target = s.target.clone()
    target[target == 19] = -1
    assert torch.equal(target, torch.load(d / "targets.pt")[idx])
