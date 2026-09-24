"""Feature caches with missing timesteps: probes average only the timesteps that have data,
and legacy caches (no frame_valid.pt) are all-valid by construction."""
import json

import pytest
import torch

from exp.pastis.lp_cached_features import _frame_valid, _time_mean


def _cache(tmp_path, temporal, valid=None):
    split = tmp_path / "pastis_r_train"
    split.mkdir()
    (tmp_path / "meta.json").write_text(json.dumps({"temporal": temporal} if temporal else {}))
    torch.save(torch.ones(3, 2, 2, 4, dtype=torch.float16), split / "0.pt")
    if valid is not None:
        torch.save(valid, split / "frame_valid.pt")
    return split


def test_legacy_cache_without_file_is_all_valid(tmp_path):
    split = _cache(tmp_path, None)
    assert _frame_valid(tmp_path, split, [0, 1]).tolist() == [[True] * 3] * 2


def test_non_legacy_cache_must_carry_frame_valid(tmp_path):
    split = _cache(tmp_path, "monthly")
    with pytest.raises(FileNotFoundError):
        _frame_valid(tmp_path, split, [0])


def test_frame_valid_follows_selected_ids(tmp_path):
    fv = torch.tensor([[True, True, True], [True, False, True], [False, True, True]])
    split = _cache(tmp_path, "monthly", fv)
    assert torch.equal(_frame_valid(tmp_path, split, [2, 1]), fv[[2, 1]])


def test_time_mean_skips_missing_steps_and_is_unchanged_when_all_valid():
    feat = torch.stack([torch.full((2, 2, 4), v) for v in (1.0, 0.0, 3.0)]).half()
    assert torch.equal(_time_mean(feat, torch.tensor([True, True, True])),
                       feat.float().mean(dim=0, keepdim=True))
    assert torch.all(_time_mean(feat, torch.tensor([True, False, True])) == 2.0)
