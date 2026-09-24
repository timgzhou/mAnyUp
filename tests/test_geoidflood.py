"""GeoidFloodDataset: conformance, chip geometry/time facts, and parity with the chips the
old prep_tiles.py wrote (data/geoidflood_tiles_t128, a --limit 40 smoke prefix)."""
from pathlib import Path

import pytest
import torch

from manyup.data import All
from manyup.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset, DEFAULT_ROOT
from manyup.data.time import from_days
from tests.mtmm_conformance import check_dataset

SEL = ChipSelection(128)
INDEX = DEFAULT_ROOT / f"chip_index_{SEL.key}.json"
LEGACY = Path(__file__).resolve().parents[1] / "data" / "geoidflood_tiles_t128"
pytestmark = pytest.mark.skipif(not INDEX.exists(), reason="GEOID chip index not built")


def make(split, modalities, tiling):
    assert tiling is None
    return GeoidFloodDataset(split, modalities, SEL)


def test_conformance():
    ds = make("val", {"S1GRD": All()}, None)
    check_dataset(make, ds.SPEC, None, None)


def test_chips_are_windows_of_their_tile_with_real_geometry():
    s = make("val", {"S1GRD": All()}, None)[0]
    stem, rc = s.id.split("@r")
    row, col = map(int, rc.split("c"))
    geo = s.modalities["S1GRD"].geometry
    assert geo.shape == (128, 128) and geo.crs.startswith("EPSG:")
    assert geo.gsd == pytest.approx((10.0, 10.0))
    assert s.target_geometry == geo
    pre, post = s.modalities["S1GRD"].time.start.tolist()
    assert pre < post                                        # pre-event, then post-event


def test_selection_keeps_only_mapped_flooded_chips():
    ds = make("val", {"S1GRD": All()}, None)
    for i in range(0, len(ds), max(1, len(ds) // 20)):
        t = ds[i].target
        assert (t != 255).float().mean() >= 0.5 and (t == 2).any()


def test_unregistered_selection_is_refused():
    with pytest.raises(ValueError, match="declared counts"):
        GeoidFloodDataset("val", {"S1GRD": All()}, ChipSelection(96))


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy prep not on disk")
@pytest.mark.parametrize("split", ["train", "val"])
def test_matches_legacy_prep_chips(split):
    ds = make(split, {"S1GRD": All()}, None)
    n = len(list((LEGACY / split).glob("[0-9]*.pt")))
    for i in list(range(min(n, 40))) + [n - 1]:
        old = torch.load(LEGACY / split / f"{i}.pt")
        s = ds[i]
        md = s.modalities["S1GRD"]
        stem, rc = s.id.split("@r")
        r, c = map(int, rc.split("c"))
        label = s.target.clone()
        label[label == 255] = -1
        assert (stem, (r // 128, c // 128)) == (old["src"], tuple(old["pos"]))
        assert torch.equal(md.data.half(), old["s1"]) and torch.equal(label.short(), old["label"])
        assert [from_days(d).strftime("%Y%m%d") for d in md.time.start.tolist()] == \
               [old["date_pre"], old["date_post"]]
