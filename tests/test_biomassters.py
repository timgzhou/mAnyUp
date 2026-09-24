"""BiomasstersDataset: conformance and the declared facts (12 Sep->Aug month slots with the
year unknown, S2 months missing as invalid frames, S1 nodata as invalid pixels, cloud
probability as an aux layer, no georeferencing), plus raw-value checks against the tifs."""
import datetime as dt

import numpy as np
import pytest
import rasterio
import torch

from manyup.data import All
from manyup.data.datasets.biomassters import (DEFAULT_ROOT, INDEX_NAME, MONTHS,
                                              BiomasstersDataset)
from manyup.data.time import from_days
from tests.mtmm_conformance import check_dataset

pytestmark = pytest.mark.skipif(not (DEFAULT_ROOT / INDEX_NAME).exists(),
                                reason="BioMassters index not built")
WINDOW = ("2000-09-01", "2001-08-31")      # the placeholder biomass year


def make(split, modalities, tiling):
    assert tiling is None
    return BiomasstersDataset(split, modalities)


def test_conformance():
    check_dataset(make, BiomasstersDataset.SPEC, None, WINDOW)


def test_twelve_month_slots_september_to_august_year_unknown():
    s = make("val", {"S1": All(), "S2": All()}, None)[0]
    for md in s.modalities.values():
        assert md.time.precision == "month_of_year"
        assert [from_days(d).strftime("%B") for d in md.time.start.tolist()] == list(MONTHS)
    assert s.modalities["S1"].frame_valid.all()
    assert not s.modalities["S1"].geometry.known and s.target.shape == (256, 256)


def test_s2_missing_months_are_invalid_zero_frames():
    ds = make("val", {"S2": All()}, None)
    for i in range(10):
        s2 = ds[i].modalities["S2"]
        assert 5 <= int(s2.frame_valid.sum()) <= 12
        assert (s2.data[~s2.frame_valid] == 0).all()
        assert s2.aux["cloud_prob"].shape == (12, 256, 256)
        assert float(s2.aux["cloud_prob"].max()) <= 255


def test_values_match_the_raw_tifs():
    ds = make("train", {"S1": All(), "S2": All()}, None)
    s = ds[0]
    inner = ds._shards.read(ds._pos[s.id])
    for _, r in inner.iterrows():
        if r["modality"] not in ("S1", "S2"):
            continue
        i = MONTHS.index(r["month"])
        with rasterio.open(r["internal:subfile"]) as f:
            raw = torch.from_numpy(f.read().astype(np.float32))
        md = s.modalities[r["modality"]]
        if r["modality"] == "S2":
            assert torch.equal(md.data[i], raw[:10]) and torch.equal(md.aux["cloud_prob"][i], raw[10])
        else:
            ok = (raw != -9999).all(0)
            assert torch.equal(md.pixel_valid[i], ok) and torch.equal(md.data[i][:, ok], raw[:, ok])
