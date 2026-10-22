"""PlanteurDataset: conformance and the facts its docstring declares."""
from pathlib import Path

import numpy as np
import pytest
import torch

from manyup.data import All, Pool, Tiling
from manyup.data.datasets.planteur import DB_FLOOR, VOID, YEAR_2019, PlanteurDataset
from manyup.data.time import from_days
from tests.mtmm_conformance import check_dataset

ROOT = Path(__file__).resolve().parents[1] / "data" / "PLANTEUR"
pytestmark = pytest.mark.skipif(not (ROOT / "DATA_S1").exists(), reason="PLANTEUR not on disk")


def make(split, modalities, tiling):
    return PlanteurDataset(split, modalities, tiling)


def raw(split="val"):
    return PlanteurDataset(split, {"S2": All(), "S1": All()}, None)


def test_conformance():
    check_dataset(make, PlanteurDataset.SPEC, Tiling(64, "col-major"), YEAR_2019)


def test_default_is_64px_tiles_on_calendar_2019_months():
    month = Pool("month", "mean", window=YEAR_2019)
    ds = PlanteurDataset("val", {"S2": month, "S1": month})
    assert len(ds) == 4 * 411
    s = ds[0]
    for md in s.modalities.values():
        assert md.history == ("month-mean@201901-201912",)
        assert [from_days(d).strftime("%Y%m") for d in md.time.start.tolist()] == \
               [f"2019{m:02d}" for m in range(1, 13)]


def test_s1_and_s2_keep_their_own_dates():
    s = raw("test")[0]
    s2, s1 = s.modalities["S2"].time.start.tolist(), s.modalities["S1"].time.start.tolist()
    assert s2 != s1[:len(s2)]
    for d in (s2, s1):
        assert d == sorted(set(d))            # unique, chronological (one frame per date)


@pytest.mark.parametrize("i", [0, 17, 200])
def test_s1_is_db_and_its_zeros_are_invalid(i):
    ds = raw()
    pid = ds.sample_id(i)
    lin = np.load(ROOT / "DATA_S1" / f"S1_{pid}.npy")
    md = ds[i].modalities["S1"]
    ok = (lin > 0).all(1)
    assert torch.equal(md.pixel_valid, torch.from_numpy(ok))
    db, valid = md.data.numpy(), np.broadcast_to(ok[:, None], lin.shape)
    np.testing.assert_allclose(db[valid], 10 * np.log10(lin[valid]), rtol=1e-5, atol=1e-4)
    assert (db[~valid] == DB_FLOOR).all()
    assert -40 < float(np.median(db[:, 0][ok])) < 5            # VV, a plausible dB value


def test_targets_and_geometry():
    import rasterio
    ds = raw()
    for i in (0, 100, 410):
        s, pid = ds[i], ds.sample_id(i)
        assert set(s.target.unique().tolist()) <= set(range(8)) | {VOID}
        with rasterio.open(ROOT / "ANNOTATIONS_GEOTIFF" / f"TARGET_{pid}.tif") as r:
            assert torch.equal(s.target, torch.from_numpy(r.read(1).astype(np.int64)))
            assert s.target_geometry.crs == str(r.crs)
            assert tuple(s.target_geometry.transform)[:6] == tuple(r.transform)[:6]
        assert s.modalities["S2"].geometry == s.target_geometry


def test_splits_are_the_default_folds_and_disjoint():
    ids = {sp: set(make(sp, {"S2": All()}, None)._native_ids(sp)) for sp in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    assert sum(map(len, ids.values())) == 2750
