"""UTAEInput: the official PASTIS normalization, dates in days since 2018-09-01, and invalid
frames as all-zero (UTAE's padding signal). Bit-parity with the official loader was checked
when it was replaced (commit message of the UTAE port)."""
from pathlib import Path

import pytest
import torch

from exp.utae.inputs import REFERENCE_DAY, UTAEInput
from manyup.data import All, mtmm_collate
from manyup.data.datasets.pastis import PastisDataset

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "PASTIS-R").exists(),
    reason="PASTIS-R not on disk")


def test_padding_frames_are_zero_and_dates_are_relative():
    ds = PastisDataset("test", {"S2": All(), "S1A": All()}, tiling=None)
    batch = mtmm_collate([ds[0], ds[3]])
    data, dates = UTAEInput(["S2", "S1A"], "test")(batch)
    for s in ("S2", "S1A"):
        mb = batch.modalities[s]
        pad = ~mb.frame_valid
        assert (data[s][pad] == 0).all() and (dates[s][pad] == 0).all()
        assert torch.equal(dates[s][~pad], (mb.t_start - REFERENCE_DAY)[~pad])
        assert not (data[s][~pad] == 0).all(dim=(1, 2, 3)).any(), "a real frame looks like padding"


def test_each_split_uses_its_own_folds_statistics():
    tr, te = UTAEInput(["S2"], "train"), UTAEInput(["S2"], "test")
    assert not torch.equal(tr.norm["S2"][0], te.norm["S2"][0])
