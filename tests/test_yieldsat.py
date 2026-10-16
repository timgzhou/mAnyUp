"""YieldSAT: conformance of both views, the ml24 view against the release's NetCDF, tiles,
splits, and the target / loss / meter the tasks build (exp/common/tasks.py _YieldSat).

YIELDSAT_PACKED=<dir> points the data tests at another packed store (e.g. a one-country
test build); YIELDSAT_ML_DIR at the NetCDF release the store was built from."""
import collections
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from exp.common.metrics import YieldMeter
from manyup.data import All, PoolYears, mtmm_collate
from manyup.data.datasets.yieldsat import (CLEAR_SCL, PACKED, ROOT, S2_BANDS, TILE,
                                           YieldSat24Dataset, YieldSatDataset, fold_splits,
                                           load_index, tile_origins)
from tests.mtmm_conformance import check_dataset

STORE = Path(os.environ.get("YIELDSAT_PACKED", PACKED))
ML_DIR = Path(os.environ.get("YIELDSAT_ML_DIR", ROOT / "preprocessed-24-ts"))
data = pytest.mark.skipif(not (STORE / "index.json").exists(), reason="YieldSAT store not built")


def factory(cls):
    def make(split, modalities, tiling):
        assert tiling is None
        return cls(split, modalities, packed=STORE)
    return make


@data
@pytest.mark.parametrize("cls", [YieldSatDataset, YieldSat24Dataset])
def test_conformance(cls):
    make = factory(cls)
    check_dataset(make, make("val", {"S2": All()}, None).SPEC, None, None)


@data
def test_ml24_matches_the_netcdf_at_labelled_pixels():
    h5py = pytest.importorskip("h5py")
    from exp.yieldsat.prepare import NC_NAME, codes, decode, field_rows
    ds = YieldSat24Dataset("test", {"S2": All()}, packed=STORE)
    by_country = collections.defaultdict(list)
    for i in range(0, len(ds), max(1, len(ds) // 12)):
        by_country[ds[i].meta["country"]].append(i)
    checked = 0
    for country, items in by_country.items():
        path = ML_DIR / country / NC_NAME
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            names = {v: k for k, v in codes(f["field_shared_name"]).items()}
            bands = [b.decode() if isinstance(b, bytes) else b for b in f["band"][:]]
            cols = [bands.index(b) for b in S2_BANDS]
            rows = field_rows(f["field_shared_name"][:])
            row, col = decode(f["row"]), decode(f["col"])
            for i in items:
                s = ds[i]
                field, r0, c0 = s.meta["field"], *map(int, s.id.rsplit("@r", 1)[1].split("c"))
                idx = rows[names[field]]
                inside = ((row[idx] >= r0) & (row[idx] < r0 + TILE)
                          & (col[idx] >= c0) & (col[idx] < c0 + TILE))
                idx = idx[inside]
                nc = f["sample"][idx][:, :, cols]                        # (n, 24, 12)
                r, c = row[idx] - r0, col[idx] - c0
                md = s.modalities["S2"]
                got = md.data[:, :, r, c].permute(2, 0, 1).numpy()        # (n, 24, 12)
                valid = md.pixel_valid[:, r, c].T.numpy()                # (n, 24)
                fin = np.isfinite(nc).all(-1)
                assert (valid == fin).all()
                assert (got[fin] == nc[fin]).all()
                assert np.allclose(s.target[r, c].numpy(), f["target"][idx])
                # nothing outside the release's pixels
                lab = torch.isfinite(s.target)
                assert not md.pixel_valid[:, ~lab].any() and not md.data[:, :, ~lab].any()
                checked += 1
    if not checked:
        pytest.skip("no NetCDF for the sampled countries")


@data
def test_ml24_frames_are_masked_raw_acquisitions():
    raw = YieldSatDataset("val", {"S2": All()}, packed=STORE)
    ml = YieldSat24Dataset("val", {"S2": All()}, packed=STORE)
    for i in range(0, len(raw), max(1, len(raw) // 10)):
        a, b = raw[i].modalities["S2"], ml[i].modalities["S2"]
        lab = torch.isfinite(raw[i].target)
        days = a.time.start.tolist()
        for t, d in enumerate(ml.fields[ml[i].meta["field"]]["ml24_days"]):
            if d < 0:
                assert not b.frame_valid[t]
                continue
            j = days.index(d)
            assert int(b.time.start[t]) == d
            ok = a.pixel_valid[j] & lab
            assert torch.equal(b.pixel_valid[t], ok)
            assert torch.equal(b.data[t], torch.where(ok, a.data[j], torch.zeros(())))
            assert torch.equal(a.pixel_valid[j], torch.isin(a.aux["scl"][j].long(),
                                                            torch.tensor(CLEAR_SCL)))


@data
def test_tiles_cover_every_labelled_pixel_once_and_splits_are_grouped():
    idx = load_index(str(STORE))
    for name, f in list(idx["fields"].items())[::25]:
        tgt = np.load(STORE / "target" / f["country"] / f"{name}.npy")
        count = np.zeros_like(tgt, dtype=int)
        for r, c in f["tiles"]:
            count[max(r, 0):r + TILE, max(c, 0):c + TILE] += 1
        assert (count[np.isfinite(tgt)] == 1).all()
        assert set(map(tuple, f["tiles"])) <= set(tile_origins(f["H"], f["W"]))
    fields = {s: {n for n, f in idx["fields"].items() if f["fold"] in folds}
              for s, folds in fold_splits(0).items()}
    assert not (fields["train"] & fields["val"]) and not (fields["train"] & fields["test"])
    groups = collections.Counter((f["country"], f["crop"]) for f in idx["fields"].values())
    for (country, crop), n in groups.items():
        if n >= 10:                     # stratified: every subset in every split
            for s in fields.values():
                assert any(idx["fields"][x]["country"] == country
                           and idx["fields"][x]["crop"] == crop for x in s)


@data
def test_raw_months_are_the_24_slots_of_the_harvest_years():
    ds = YieldSatDataset("val", {"S2": PoolYears(2, "mean")}, packed=STORE)
    s = ds[0]
    md = s.modalities["S2"]
    f = ds.fields[s.meta["field"]]
    assert md.num_frames == 24
    first = md.time.dates()[0][0]
    assert (first.year, first.month) == (int(f["harvest"][:4]) - 1, 1)
    assert md.frame_valid.any()
    batch = mtmm_collate([ds[0], ds[1]])
    assert batch.modalities["S2"].data.shape[1] == 24


def test_meter_scores_each_field_once_across_batches():
    # two fields of group a, one of group b; field 0 is cut into two batches
    m = YieldMeter(mean=5.0, std=2.0, field_group=("a", "a", "b"))
    pred = {0: [6.0, 8.0, 6.0], 1: [4.0, 4.0], 2: [9.0]}
    true = {0: [7.0, 7.0, 5.0], 1: [3.0, 5.0], 2: [10.0]}

    def batch(fid, p, t):
        z = lambda v: (torch.tensor(v) - 5.0) / 2.0          # noqa: E731
        target = torch.stack([z(t), torch.full((len(t),), float(fid))]).view(1, 2, 1, len(t))
        return z(p).view(1, 1, 1, len(p)), target

    for fid, sl in ((0, slice(0, 2)), (1, slice(None)), (0, slice(2, None)), (2, slice(None))):
        m.update(*batch(fid, pred[fid][sl], true[fid][sl]))
    nan_target = torch.tensor([[[[float("nan")]], [[0.0]]]])
    m.update(torch.zeros(1, 1, 1, 1), nan_target)           # unlabelled: ignored
    out = m.compute()
    p = np.concatenate([pred[k] for k in range(3)])
    t = np.concatenate([true[k] for k in range(3)])
    assert out["rmse"] == pytest.approx(np.sqrt(((p - t) ** 2).mean()))
    assert out["r2"] == pytest.approx(1 - ((p - t) ** 2).mean() / t.var())
    fp = np.array([np.mean(pred[k]) for k in range(3)])
    ft = np.array([np.mean(true[k]) for k in range(3)])
    assert out["field_rmse"] == pytest.approx(np.sqrt(((fp - ft) ** 2).mean()))
    assert out["fields"] == 3 and out["a_fields"] == 2 and out["b_fields"] == 1
    assert out["b_field_rmse"] == pytest.approx(1.0)
    assert out["primary"] == out["rmse"]
    other = YieldMeter(5.0, 2.0, ("a", "a", "b"))
    other.update(*batch(2, [9.0], [10.0]))
    assert m.merge(other).compute()["b_rmse"] == pytest.approx(1.0)


def test_masked_mse_skips_unlabelled_pixels():
    from exp.common.tasks import _MaskedMSE
    pred = torch.tensor([1.0, 2.0, 100.0]).view(1, 1, 1, 3)
    target = torch.stack([torch.tensor([0.0, 0.0, float("nan")]), torch.zeros(3)]).view(1, 2, 1, 3)
    assert _MaskedMSE()(pred, target).item() == pytest.approx((1 + 4) / 2)
    empty = torch.full((1, 2, 1, 3), float("nan"))
    assert _MaskedMSE()(pred, empty).item() == 0.0
