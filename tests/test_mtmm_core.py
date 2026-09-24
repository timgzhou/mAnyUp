"""Core MTMM data layer on a synthetic dataset: two timed modalities with DIFFERENT dates
and frame counts, one static modality, real-looking georeferencing."""
import datetime as dt

import pytest
import torch

from manyup.data import (All, DatasetSpec, Every, Geometry, Indices, MaxFrames, ModalityData,
                         ModalitySpec, MTMMDataset, MTMMSample, Pool, TargetSpec, Tiling,
                         TimeAxis, mtmm_collate)
from manyup.data.time import to_days
from tests.mtmm_conformance import check_dataset

H = W = 8
WINDOW = ("2020-01-01", "2020-04-30")


def _days(*iso):
    return [to_days(dt.date.fromisoformat(d)) for d in iso]


class ToyDataset(MTMMDataset):
    SPEC = DatasetSpec(
        name="toy",
        modalities={
            "S2": ModalitySpec("S2", ("B02", "B03", "B04"), "reflectance", 10.0, "irregular",
                               (3, 5), "day", pool_reducer="median"),
            "S1": ModalitySpec("S1", ("VV", "VH"), "dB", 10.0, "irregular", (4, 4), "day",
                               pool_reducer="mean"),
            "DEM": ModalitySpec("DEM", ("elevation",), "m", 30.0, "static", (1, 1), None),
        },
        target=TargetSpec("segmentation", "class", ("bg", "a", "b"), ignore_index=-1),
        native_shape=(H, W), split_sizes={"train": 3, "test": 2}, georeferenced=True)

    S2_DATES = [_days("2020-01-03", "2020-01-20", "2020-03-02"),
                _days("2020-01-05", "2020-02-10", "2020-02-25", "2020-03-15", "2020-04-01")]
    S1_DATES = _days("2020-01-02", "2020-01-14", "2020-02-07", "2020-04-20")

    def _native_ids(self, split):
        return [f"{split}{i}" for i in range(self.SPEC.split_sizes[split])]

    def _load(self, nid, modalities):
        k = int(nid[-1])
        geo = Geometry("EPSG:32631", (10.0, 0.0, 500000.0 + 1000 * k, 0.0, -10.0, 4.0e6), (H, W))
        g = torch.Generator().manual_seed(k)
        out = {}
        for m in modalities:
            if m == "DEM":
                out[m] = ModalityData(torch.rand(1, 1, H, W, generator=g), ("elevation",), None,
                                      torch.ones(1, dtype=torch.bool), geo)
                continue
            days = self.S2_DATES[k % 2] if m == "S2" else self.S1_DATES
            C = len(self.SPEC.modalities[m].bands)
            out[m] = ModalityData(torch.rand(len(days), C, H, W, generator=g),
                                  self.SPEC.modalities[m].bands, TimeAxis.instants(days),
                                  torch.ones(len(days), dtype=torch.bool), geo)
        target = torch.randint(0, 3, (H, W), generator=g)
        return MTMMSample(nid, self.split, out, target, geo)


def make(split, modalities, tiling):
    return ToyDataset(split, modalities, tiling)


def test_conformance():
    check_dataset(make, ToyDataset.SPEC, Tiling(4, "col-major"), WINDOW)


def test_modalities_must_be_explicit():
    with pytest.raises(ValueError):
        ToyDataset("train", {}, None)
    with pytest.raises(ValueError):
        ToyDataset("train", {"S3": All()}, None)
    with pytest.raises(TypeError):
        ToyDataset("train", {"S2": "month"}, None)
    with pytest.raises(ValueError):
        ToyDataset("val", {"S2": All()}, None)


def test_static_modality_only_accepts_all():
    ds = ToyDataset("train", {"DEM": Every(2)}, None)
    with pytest.raises(ValueError, match="static"):
        ds[0]


def test_modalities_keep_separate_time_axes():
    s = ToyDataset("train", {"S2": All(), "S1": All()}, None)[1]
    assert s.modalities["S2"].num_frames == 5 and s.modalities["S1"].num_frames == 4
    assert s.modalities["S2"].time.start.tolist() != s.modalities["S1"].time.start.tolist()


def test_pool_fixed_window_masks_empty_months_and_keeps_sources():
    ds = ToyDataset("train", {"S2": Pool("month", "mean", window=WINDOW)}, None)
    s2 = ds[0].modalities["S2"]          # native dates: Jan 3, Jan 20, Mar 2
    assert s2.num_frames == 4             # Jan..Apr, fixed grid
    assert s2.frame_valid.tolist() == [True, False, True, False]
    assert s2.time.sources == [_days("2020-01-03", "2020-01-20"), [], _days("2020-03-02"), []]
    assert s2.time.precision == "month"
    assert s2.time.start[0] == to_days(dt.date(2020, 1, 1))
    assert s2.time.end[0] == to_days(dt.date(2020, 1, 31))
    raw = ToyDataset("train", {"S2": All()}, None)[0].modalities["S2"].data
    assert torch.allclose(s2.data[0], raw[:2].mean(0))
    assert (s2.data[1] == 0).all()


def test_pool_reducer_comes_from_spec_and_is_recorded():
    ds = ToyDataset("train", {"S2": Pool("month", window=WINDOW), "S1": Pool("month", window=WINDOW)}, None)
    s = ds[1]
    assert s.modalities["S2"].history == ("month-median@202001-202004",)
    assert s.modalities["S1"].history == ("month-mean@202001-202004",)
    assert "S2-month-median@202001-202004" in ds.key and "S1-month-mean" in ds.key


def test_pool_ignores_invalid_frames_and_pixels():
    md = ToyDataset("train", {"S2": All()}, None)[0].modalities["S2"]
    md.frame_valid[1] = False                       # drop Jan 20
    md.pixel_valid = torch.ones(3, H, W, dtype=torch.bool)
    md.pixel_valid[0, 0, 0] = False                 # and one pixel of Jan 3
    out = Pool("month", "mean", window=WINDOW)(md, ToyDataset.SPEC.modalities["S2"])
    assert torch.allclose(out.data[0, :, 1:, 1:], md.data[0, :, 1:, 1:])
    assert (out.data[0, :, 0, 0] == 0).all() and not out.pixel_valid[0, 0, 0]
    assert out.time.sources[0] == _days("2020-01-03")


def test_median_is_an_observed_value():
    md = ToyDataset("test", {"S2": All()}, None)[1].modalities["S2"]   # Feb 10, Feb 25 in Feb
    out = Pool("month", "median", window=WINDOW)(md, ToyDataset.SPEC.modalities["S2"])
    feb = md.data[1:3]
    assert ((out.data[1] == feb[0]) | (out.data[1] == feb[1])).all()


def test_selection_policies_compose_and_keep_dates():
    ds = ToyDataset("test", {"S2": Every(2) + MaxFrames(2), "S1": Indices([0, 3])}, None)
    s = ds[1]
    native = ToyDataset.S2_DATES[1]
    assert s.modalities["S2"].time.start.tolist() == [native[0], native[4]]
    assert s.modalities["S2"].history == ("every2", "max2")
    assert s.modalities["S1"].time.start.tolist() == [ToyDataset.S1_DATES[i] for i in (0, 3)]
    with pytest.raises(IndexError):
        ToyDataset("test", {"S1": Indices([7])}, None)[0]


def test_tiling_ids_order_and_geometry():
    t = Tiling(4, "col-major")
    assert t.offsets((8, 8)) == [(0, 0), (4, 0), (0, 4), (4, 4)]
    assert Tiling(4, "row-major").offsets((8, 8)) == [(0, 0), (0, 4), (4, 0), (4, 4)]
    ds = ToyDataset("train", {"S2": All()}, t)
    s = ds[1]                                        # bottom-left tile of train0
    assert s.id == "train0@r4c0" and Tiling.parse(s.id) == ("train0", 4, 0)
    geo = s.modalities["S2"].geometry
    assert geo.shape == (4, 4) and geo.transform[2] == 500000.0 and geo.transform[5] == 4.0e6 - 40
    assert s.target_geometry == geo


def test_collate_pads_each_modality_separately():
    ds = ToyDataset("train", {"S2": All(), "S1": All(), "DEM": All()}, None)
    b = mtmm_collate([ds[0], ds[1]])                 # S2 has 3 and 5 frames; S1 4 and 4
    s2, s1 = b.modalities["S2"], b.modalities["S1"]
    assert s2.data.shape == (2, 5, 3, H, W) and s1.data.shape == (2, 4, 2, H, W)
    assert s2.frame_valid.tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    assert (s2.t_start[0, 3:] == -1).all()
    assert b.modalities["DEM"].t_start is None and b.target.shape == (2, H, W)


def test_geometry_rescale_to_token_grid():
    g = Geometry("EPSG:32631", (10.0, 0.0, 0.0, 0.0, -10.0, 80.0), (8, 8))
    tok = g.rescale((2, 2))
    assert tok.gsd == (40.0, 40.0) and tok.transform[2] == 0.0 and tok.transform[5] == 80.0
    assert Geometry.unknown((8, 8)).crop(0, 0, 4, 4).known is False


@pytest.mark.parametrize("breakage, match", [
    ("unsorted", "time order"),
    ("too_many_frames", "outside declared"),
    ("wrong_bands", "bands"),
    ("fake_georef", "georeferenced"),
])
def test_validation_rejects_broken_samples(breakage, match):
    class Broken(ToyDataset):
        def _load(self, nid, modalities):
            s = super()._load(nid, modalities)
            md = s.modalities["S1"]
            if breakage == "unsorted":
                md.time = TimeAxis.instants(list(reversed(md.time.start.tolist())))
            elif breakage == "too_many_frames":
                s.modalities["S1"] = md.select_frames([0, 1, 2, 3, 3])
                s.modalities["S1"].time = TimeAxis.instants(sorted(md.time.start.tolist() + [md.time.start[-1].item()]))
            elif breakage == "wrong_bands":
                md.bands = ("VH", "VV")
            elif breakage == "fake_georef":
                md.geometry = Geometry.unknown((H, W))
            return s

    with pytest.raises(AssertionError, match=match):
        Broken("train", {"S1": All()}, None)[0]
