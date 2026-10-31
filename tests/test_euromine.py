"""EuroMineNet: the per-year target encoding, tiles, the per-step meter, the per-step decoder
(manyup.mixres.model MixedResModel._decode_steps) and the per-step selectors' budgets
(manyup.mixres.steps).

EUROMINE_PACKED=<dir> points the data tests at another packed store."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from exp.common.metrics import StepMeter
from manyup.data import All
from manyup.data.datasets.euromine import (PACKED, TILE, YEARS, EuroMineDataset, decode_years,
                                           encode_years, load_index, tile_origins)

STORE = Path(os.environ.get("EUROMINE_PACKED", PACKED))
data = pytest.mark.skipif(not (STORE / "index.json").exists(), reason="EuroMineNet not packed")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (FlashAttention)")


def test_year_bits_roundtrip():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 2, (len(YEARS), 5, 7)).astype(np.uint8)
    t = torch.from_numpy(encode_years(lab))
    assert torch.equal(decode_years(t), torch.from_numpy(lab).long())
    assert torch.equal(decode_years(t[None])[0], torch.from_numpy(lab).long())   # batched


def test_tile_origins_cover_and_stay_inside():
    for H, W in [(64, 64), (109, 586), (128, 200), (65, 64)]:
        o = tile_origins(H, W)
        cov = np.zeros((H, W), bool)
        for r, c in o:
            assert 0 <= r <= H - TILE and 0 <= c <= W - TILE
            cov[r:r + TILE, c:c + TILE] = True
        assert cov.all()
        assert len(o) == -(-H // TILE) * -(-W // TILE)
    with pytest.raises(ValueError):
        tile_origins(63, 100)


def _logits(pred):
    """(B, T, H, W) {0, 1} -> logits (B, T, 2, H, W) whose argmax is pred."""
    return torch.stack([1 - pred, pred], 2).float()


def test_step_meter_by_hand():
    T = len(YEARS)
    g = torch.zeros(1, T, 2, 2, dtype=torch.long)
    g[0, 3:, 0, 0] = 1                     # pixel (0, 0) becomes mine in year 3
    g[0, :, 1, 1] = 1                      # pixel (1, 1) mine every year
    p = g.clone()
    m = StepMeter(tuple(map(str, YEARS)))
    m.update(_logits(p), g)
    out = m.compute()
    assert out["of1"] == pytest.approx(1.0) and out["cf1"] == pytest.approx(1.0)
    assert out["dyn_f1"] == pytest.approx(1.0) and out["gca_tiou"] == pytest.approx(1.0)
    # predicting the change one year late: one missed change at interval 1 (2->3), ...
    p2 = g.clone()
    p2[0, 3, 0, 0] = 0
    m = StepMeter(tuple(map(str, YEARS)))
    m.update(_logits(p2), g)
    out = m.compute()
    tp = T + (T - 3) - 1                   # mine pixel-years found
    assert out["of1"] == pytest.approx(2 * tp / (2 * tp + 1))
    assert out["f1_2018"] == pytest.approx(2 / 3)          # year 3: 1 of 2 mine pixels found
    assert out["dyn_f1"] == pytest.approx(2 * 6 / (2 * 6 + 1))
    assert out["cf1_1y"] < 1 and out["cf1_9y"] == pytest.approx(1.0)
    # ignored pixels count nowhere
    g3 = g.clone()
    g3[0, :, 0, 0] = -1
    m = StepMeter(tuple(map(str, YEARS)))
    m.update(_logits(p2), g3)
    assert m.compute()["of1"] == pytest.approx(1.0)


def test_step_grid_exact_budget():
    from manyup.mixres.selectors import Geometry, SeasonPairs
    from manyup.mixres.steps import StepGrid, StepRandom

    class Ctx:          # what the scorers / allocator read
        B, T, NS = 3, len(YEARS), len(YEARS)
        present = torch.ones(3, len(YEARS), dtype=torch.bool)
        season = torch.arange(len(YEARS)).expand(3, -1)
        valid = present

        class model:
            training = True

    for crop, ns in [(8, (8, 16, 32)), (4, (32, 64, 128))]:
        geo = Geometry(crop, crop)
        for n in ns:
            for sc in (StepGrid(n), StepRandom()):
                s = sc(Ctx, geo)
                mask = SeasonPairs(n)(s, Ctx, geo)
                assert (mask.sum((1, 2, 3)) == n * len(YEARS)).all(), (crop, n, type(sc))
                assert bool((geo.plan(mask).py >= 0).all())          # tiling windows only
            if isinstance(sc, StepRandom):
                continue
            per_year = mask.sum((2, 3))
            assert (per_year == n).all()                             # the grid: n per year


@data
def test_dataset_reads_packed_tiles():
    idx = load_index(str(STORE))
    ds = EuroMineDataset("val", {"S2": All()}, packed=STORE)
    s = ds[0]
    md = s.modalities["S2"]
    assert md.data.shape == (len(YEARS), 10, TILE, TILE)
    assert s.target.shape == (TILE, TILE)
    site = s.meta["site"]
    assert idx["sites"][site]["split"] == "val"
    lab = decode_years(s.target)
    assert set(lab.unique().tolist()) <= {0, 1}
    sizes = {k: len(EuroMineDataset(k, {"S2": All()}, packed=STORE)) for k in ("train", "val", "test")}
    assert sizes["train"] > sizes["test"] > sizes["val"] > 0


@gpu
def test_per_step_decode_routes_crops_to_their_year():
    """No crops: year t's map is year t's season token's own head map. A crop of year t changes
    year t's map only."""
    from exp.common.tasks import TASKS
    from manyup.mixres import MixedResModel
    from manyup.mixres.model import ModelConfig
    from manyup.mixres.plan import Plan
    from manyup.mixres.selectors import season_maps

    task = TASKS["euromine"]
    torch.manual_seed(0)
    m = MixedResModel(task, ModelConfig("tiny", 16), pretrained=False).cuda().eval()
    m.readout.score[1].weight.data.normal_(std=0.01)   # non-trivial candidate weights
    B, T, NS, D = 2, len(YEARS), len(YEARS), m.mix.encoder.embedding_size
    lr_F = torch.randn(B, NS, 4, 4, D, device="cuda")
    season = torch.arange(T, device="cuda").expand(B, -1)
    present = torch.ones(B, T, dtype=torch.bool, device="cuda")
    z = torch.zeros(0, D, device="cuda")
    with torch.no_grad():
        out = m.decode(lr_F, z, [], season, present)
        assert out.shape == (B, T, 2, TILE, TILE)
        ref = season_maps(m, lr_F).view(B, NS, 2, TILE, TILE)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        t = torch.tensor([3], device="cuda")
        one = torch.tensor([1], device="cuda")
        p = Plan(one, t, torch.tensor([2], device="cuda"), torch.tensor([4], device="cuda"),
                 torch.tensor([8], device="cuda"), torch.tensor([16], device="cuda"), 8)
        out2 = m.decode(lr_F, torch.randn(1, D, device="cuda"), [p], season, present)
        diff = (out2 - out).abs().amax((2,))                         # (B, T, 64, 64)
        assert diff[1, 3, 8:16, 16:24].min() > 0                     # the crop's window, year 3
        diff[1, 3, 8:16, 16:24] = 0
        assert diff.max() == 0                                       # nothing else moves


@gpu
@data
def test_independent_steps_do_not_see_other_years():
    """independent_steps: changing year 3's image changes year 3's map only; the joint model's
    other years move too (it attends across years)."""
    from exp.common.tasks import TASKS
    from manyup.mixres import MixedResModel
    from manyup.mixres.data import loader
    from manyup.mixres.model import ModelConfig

    task = TASKS["euromine"]
    b = next(iter(loader(task, "val", 2, 0, False)))
    ss2 = b.ss._replace(sentinel2_l2a=b.ss.sentinel2_l2a.clone())
    ss2.sentinel2_l2a[:, :, :, 3] = ss2.sentinel2_l2a[:, :, :, 3].flip(1)
    for indep in (True, False):
        torch.manual_seed(0)
        m = MixedResModel(task, ModelConfig("tiny", 16, independent_steps=indep)).cuda()
        a = m.uniform(b.ms, b.ss, b.season, b.present).logits
        c = m.uniform(b.ms, ss2, b.season, b.present).logits
        d = (a - c).abs().flatten(2).amax(2)                          # (B, T)
        assert d[:, 3].min() > 0
        others = torch.cat([d[:, :3], d[:, 4:]], 1)
        assert (others.max() == 0) == indep, (indep, others.max())


def test_padding_bit_is_ignored():
    lab = np.ones((len(YEARS), 4, 4), np.uint8)
    pad = np.zeros((4, 4), bool)
    pad[2:] = True
    y = decode_years(torch.from_numpy(encode_years(lab, pad)))
    assert (y[:, 2:] == -1).all() and (y[:, :2] == 1).all()


@data
@pytest.mark.parametrize("tile", [128, 256])
def test_large_tiles_cover_every_site_pixel(tile):
    from manyup.data.datasets.euromine import UNLABELLED
    idx = load_index(str(STORE))
    ds = EuroMineDataset("test", {"S2": All()}, packed=STORE, tile=tile)
    site = sorted(s for s, f in idx["sites"].items() if f["split"] == "test")[0]
    H, W = idx["sites"][site]["shape"]
    cov = np.zeros((H, W), bool)
    for i in range(len(ds)):
        sid = ds.sample_id(i)
        if not sid.startswith(site + "@"):
            continue
        s = ds[i]
        assert s.modalities["S2"].data.shape == (len(YEARS), 10, tile, tile)
        r, c = (int(v) for v in sid.rsplit("@r", 1)[1].split("c"))
        real = (s.target & UNLABELLED) == 0
        cov[r:r + tile, c:c + tile] |= real.numpy()[:H - r, :W - c]
    assert cov.all()


@gpu
def test_ps32_decodes_full_resolution():
    from exp.common.tasks import TASKS
    from manyup.mixres import MixedResModel
    from manyup.mixres.model import ModelConfig
    m = MixedResModel(TASKS["euromine_t128"], ModelConfig("tiny", 32), pretrained=False).cuda()
    B, T, D = 2, len(YEARS), m.mix.encoder.embedding_size
    season = torch.arange(T, device="cuda").expand(B, -1)
    present = torch.ones(B, T, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        out = m.decode(torch.randn(B, T, 4, 4, D, device="cuda"), torch.zeros(0, D, device="cuda"),
                       [], season, present)
    assert out.shape == (B, T, 2, 128, 128)
    assert out[0, 0, 0, :32, :32].std() > 0          # a per-pixel map inside one token


def test_label_quarters_are_jul_sep_of_their_year():
    from manyup.data.datasets.euromine import LABEL_QUARTERS, QUARTERS
    assert len(QUARTERS) == 38 and QUARTERS[0] == (2015, 3)
    assert [QUARTERS[i] for i in LABEL_QUARTERS] == [(y, 3) for y in YEARS]


seasonal = pytest.mark.skipif(
    not (Path(os.environ.get("EUROMINE_SEASONAL", PACKED.parent / "seasonal_packed"))
         / "index.json").exists(), reason="seasonal series not packed")


@data
@seasonal
def test_q3_series_is_the_seasonal_series_at_the_label_quarters():
    from manyup.data.datasets.euromine import LABEL_QUARTERS
    sea = EuroMineDataset("val", {"S2": All()}, packed=STORE, series="seasonal")
    q3 = EuroMineDataset("val", {"S2": All()}, packed=STORE, series="q3")
    ann = EuroMineDataset("val", {"S2": All()}, packed=STORE)
    i = len(sea) // 2
    a, b = sea[i], q3[i]
    assert a.id == b.id == ann[i].id
    assert a.modalities["S2"].data.shape[0] == 38 and b.modalities["S2"].data.shape[0] == 10
    assert torch.equal(a.modalities["S2"].data[list(LABEL_QUARTERS)], b.modalities["S2"].data)
    assert torch.equal(a.target, ann[i].target)                  # the same yearly labels
    t = b.modalities["S2"].time
    import datetime as dt
    from manyup.data.time import from_days
    assert [from_days(int(v)).month for v in t.start] == [7] * 10
    assert [from_days(int(v)) for v in t.end][:1] == [dt.date(2015, 9, 30)]
