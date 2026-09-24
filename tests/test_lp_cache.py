"""Feature caches with missing timesteps: FeatureCache refuses to guess validity/dates for a
non-legacy cache, reconstructs legacy ones only when told how, and the probes' time mean
covers the valid timesteps only (unchanged when every step is valid)."""
import json

import pytest
import torch

from exp.pastis.lp_cached_features import _time_mean
from manyup.data.feature_cache import FeatureCache, LegacyReconstruction


def _cache(tmp_path, temporal, n=2, T=3, files=("ids", "frame_valid", "t_start", "t_end")):
    split = tmp_path / "c" / "s"
    split.mkdir(parents=True)
    (tmp_path / "c" / "meta.json").write_text(json.dumps({"temporal": temporal} if temporal else {}))
    for i in range(n):
        torch.save(torch.ones(T, 2, 2, 4, dtype=torch.float16), split / f"{i}.pt")
    fv = torch.tensor([[True, False, True], [True, True, True]])[:n]
    if "ids" in files:
        (split / "ids.txt").write_text("a\nb\n")
    if "frame_valid" in files:
        torch.save(fv, split / "frame_valid.pt")
    for f in ("t_start", "t_end"):
        if f in files:
            torch.save(torch.zeros(n, T, dtype=torch.long), split / f"{f}.pt")
    return fv


def test_recorded_fields_are_read_by_sample_id(tmp_path):
    fv = _cache(tmp_path, "monthly")
    cache = FeatureCache(tmp_path, "c", "s")
    assert cache.reconstructed == [] and torch.equal(cache.get("a")["frame_valid"], fv[0])
    assert torch.equal(cache.frame_valid, fv)


def test_non_legacy_cache_without_validity_is_refused(tmp_path):
    _cache(tmp_path, "monthly", files=("ids", "t_start", "t_end"))
    rec = LegacyReconstruction(["a", "b"], torch.ones(2, 3, dtype=torch.bool),
                               torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(FileNotFoundError, match="not a legacy cache"):
        FeatureCache(tmp_path, "c", "s", rec)


def test_legacy_cache_needs_and_reports_its_reconstruction(tmp_path):
    _cache(tmp_path, None, files=())
    with pytest.raises(FileNotFoundError, match="no legacy"):
        FeatureCache(tmp_path, "c", "s")
    rec = LegacyReconstruction(["x", "y"], torch.ones(2, 3, dtype=torch.bool),
                               torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
    cache = FeatureCache(tmp_path, "c", "s", rec)
    assert cache.ids == ["x", "y"] and set(cache.reconstructed) == {"ids", "frame_valid", "t_start", "t_end"}


def test_time_mean_skips_missing_steps_and_is_unchanged_when_all_valid():
    feat = torch.stack([torch.full((2, 2, 4), v) for v in (1.0, 0.0, 3.0)]).half()
    assert torch.equal(_time_mean(feat, torch.tensor([True, True, True])),
                       feat.float().mean(dim=0, keepdim=True))
    assert torch.all(_time_mean(feat, torch.tensor([True, False, True])) == 2.0)
