"""FeatureUpsampler wrappers (manyup/upsamplers): guidance picked by date, the declarations
and input checks, and every wrapper reproducing its legacy code path on the same inputs."""
import glob
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from manyup.data.features import FeatureSeries
from manyup.data.geometry import Geometry
from manyup.data.guidance import (LEGACY_S2_13, GuidanceSpec, build_guidance,
                                  build_guidance_batch, match_frames, percentile_per_image)
from manyup.data.sample import ModalityBatch, ModalityData, MTMMBatch
from manyup.data.time import TimeAxis
from manyup.upsamplers import JBU, Bilinear, MAnyUp, TimAnyUp

REPO = Path(__file__).resolve().parents[1]
S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mb(data, t_start, valid=None, bands=S2_BANDS, t_end=None):
    B, T = data.shape[:2]
    t_start = torch.as_tensor(t_start)
    return ModalityBatch(data=data, bands=bands, t_start=t_start,
                         t_end=t_start if t_end is None else torch.as_tensor(t_end),
                         time_precision="day",
                         frame_valid=torch.ones(B, T, dtype=torch.bool) if valid is None
                         else torch.as_tensor(valid),
                         pixel_valid=None, geometry=[Geometry.unknown(data.shape[-2:])] * B,
                         aux={}, history=())


def _batch(mods: dict, ids=None):
    B = next(iter(mods.values())).data.shape[0]
    ids = ids or [f"s{i}" for i in range(B)]
    return MTMMBatch(ids=ids, split="train", modalities=mods, target=None,
                     target_geometry=[None] * B, meta=[{}] * B)


def _series(tokens, t_start, valid=None, ids=None):
    B, T, h, w, _ = tokens.shape
    t_start = torch.as_tensor(t_start)
    return FeatureSeries(ids=ids or [f"s{i}" for i in range(B)], tokens=tokens,
                         t_start=t_start, t_end=t_start.clone(), time_precision="day",
                         frame_valid=torch.ones(B, T, dtype=torch.bool) if valid is None
                         else torch.as_tensor(valid),
                         geometry=[Geometry.unknown((h, w))] * B)


# -- guidance by date --------------------------------------------------------------------
def test_nearest_matches_by_date_and_skips_missing_frames():
    mb = _mb(torch.zeros(1, 4, 10, 2, 2), [[10, 40, 70, 100]], [[True, True, False, True]])
    w = match_frames(mb, torch.tensor([[12, 68, 99]]), torch.tensor([[12, 68, 99]]), "nearest")
    assert w.argmax(-1).tolist() == [[0, 1, 3]]          # 68 -> 40, not the missing 70


def test_within_range_averages_overlapping_frames_else_nearest():
    mb = _mb(torch.zeros(1, 3, 10, 2, 2), [[1, 10, 40]])
    f0, f1 = torch.tensor([[0, 30]]), torch.tensor([[15, 35]])    # [0,15] and [30,35]
    w = match_frames(mb, f0, f1, "within_range")
    assert torch.allclose(w[0, 0], torch.tensor([0.5, 0.5, 0], dtype=w.dtype))
    assert w[0, 1].tolist() == [0, 0, 1]                           # none inside: nearest


def test_batched_guidance_equals_per_sample_guidance():
    torch.manual_seed(0)
    data = torch.rand(2, 3, 10, 8, 8) * 1e4
    t = [[5, 35, 65], [5, 35, 65]]
    mb = _mb(data, t)
    spec13 = {"S2": LEGACY_S2_13}
    for norm in ("minmax", "percentile"):
        per_t = build_guidance_batch(_batch({"S2": mb}), GuidanceSpec(spec13, None, norm),
                                     match="nearest", f_start=torch.tensor(t),
                                     f_end=torch.tensor(t))
        pooled = build_guidance_batch(_batch({"S2": mb}), GuidanceSpec(spec13, "median", norm))
        for b in range(2):
            md = ModalityData(data=data[b], bands=S2_BANDS,
                              time=TimeAxis(torch.tensor(t[b]), torch.tensor(t[b]), "day"),
                              frame_valid=torch.ones(3, dtype=torch.bool),
                              geometry=Geometry.unknown((8, 8)))
            assert torch.equal(per_t[b], build_guidance({"S2": md}, GuidanceSpec(spec13, None,
                                                                                  norm)))
            assert torch.equal(pooled[b], build_guidance({"S2": md},
                                                         GuidanceSpec(spec13, "median", norm)))


def test_percentile_norm_matches_legacy_stretch():
    from exp.upsamplers.upa_anyup import percentile_stretch
    x = torch.rand(13, 64, 64) * 3000
    ref = torch.from_numpy(percentile_stretch(x.numpy()))
    assert torch.allclose(percentile_per_image(x), ref, atol=1e-5)


# -- declarations and checks -------------------------------------------------------------
def test_bilinear_keeps_dates_or_pools_them():
    tok = torch.randn(2, 3, 4, 4, 5)
    lr = _series(tok, [[10, 20, 30]] * 2, [[True, True, False]] * 2)
    target = [Geometry.unknown((16, 16))] * 2
    out = Bilinear()(lr, None, target)
    ref = F.interpolate(tok[:, 1].permute(0, 3, 1, 2), (16, 16), mode="bilinear",
                        align_corners=True)
    assert torch.allclose(out.tokens[:, 1].permute(0, 3, 1, 2), ref)
    assert torch.equal(out.t_start, lr.t_start) and out.provenance["space"] == "input"
    pooled = Bilinear(pooled=True)(lr, None, target)
    assert pooled.tokens.shape == (2, 1, 16, 16, 5)
    assert pooled.t_start.tolist() == [[10]] * 2 and pooled.t_end.tolist() == [[20]] * 2


def test_declarations_and_inputs_are_checked():
    with pytest.raises(ValueError, match="pooled iff"):
        JBU(GuidanceSpec({"S2": ("B04",)}, None), time_matching="pooled")
    lr = _series(torch.randn(1, 12, 4, 4, 8), [list(range(12))])
    with pytest.raises(ValueError, match="reads no guidance"):
        Bilinear()(lr, _batch({"S2": _mb(torch.zeros(1, 1, 10, 4, 4), [[0]])}),
                   [Geometry.unknown((8, 8))])
    up = TimAnyUp(_tiny_timanyup(8), GuidanceSpec({"S2": ("B04",)}, None), k=4)
    g = _batch({"S2": _mb(torch.rand(1, 12, 10, 16, 16), [list(range(12))])})
    with pytest.raises(ValueError, match="side features"):
        up.check_inputs(lr, g, [Geometry.unknown((16, 16))], {})
    shifted = _series(torch.randn(1, 12, 16, 16, 8), [list(range(1, 13))])
    with pytest.raises(ValueError, match="other dates"):
        up(lr, g, [Geometry.unknown((16, 16))], hr=shifted)


def _tiny_timanyup(D):
    from manyup.timAnyUp import TimAnyUp as Model
    return Model(input_dim=1, qk_dim=16, feat_dim=D, num_frames=12, num_heads=4)


@pytest.mark.skipif(not (REPO / "data" / "pastis_olmoearth").exists(), reason="no PASTIS")
def test_check_rejects_missing_guidance_modality():
    from manyup.data.datasets.pastis import LEGACY_OLMOEARTH_S2, PastisDataset
    ds = PastisDataset("val", {"S2": LEGACY_OLMOEARTH_S2})
    Bilinear().check(ds)
    JBU.upma().check(ds)
    with pytest.raises(ValueError, match="S1A"):
        JBU(GuidanceSpec({"S1A": ("VV",)}, "mean")).check(ds)


# -- the wrappers reproduce the legacy code paths ----------------------------------------
def _pastis_like(B=2, T=12, H=64, D=768, g=4, seed=0):
    torch.manual_seed(seed)
    t = torch.arange(T).repeat(B, 1) * 30 + 5
    guid = _batch({"S2": _mb(torch.rand(B, T, 10, H, H) * 1e4, t)})
    return _series(torch.randn(B, T, g, g, D), t), guid


def _one(pattern):
    hits = sorted(glob.glob(str(REPO / "checkpoints" / pattern)))
    if not hits:
        pytest.skip(f"no checkpoint {pattern}")
    return hits[-1]


@torch.no_grad()
def test_manyup_matches_legacy_cached_head():
    from exp.pastis.lp_cached_features import CachedManyUp
    ckpt = _one("manyup/manyup_0transform_oe_base_s2_ps16_tile64_to_oe_base_s2_ps4_tile64_*ep31.pth")
    lr, guid = _pastis_like()
    # one feature step: the legacy head's feats.mean(1) and our valid-weighted time_mean agree
    # only to rounding (~1e-6) over 12 steps; with one step both are exact
    lr = _series(lr.tokens[:, :1], lr.t_start[:, :1])
    up = MAnyUp.from_checkpoint(ckpt)
    out = up(lr, guid, [Geometry.unknown((64, 64))] * 2)
    legacy = CachedManyUp(768, 19, 16, ckpt).eval()
    rgb = build_guidance_batch(guid, GuidanceSpec({"S2": LEGACY_S2_13}, "mean"))
    ref = legacy.features(lr.tokens, rgb)                                # (B, D, 64, 64)
    assert torch.equal(out.tokens[:, 0].permute(0, 3, 1, 2), ref)
    assert out.provenance["space"] == "input" and up.time_matching == "pooled"


@torch.no_grad()
def test_timanyup_matches_legacy_cached_head():
    from exp.pastis.lp_cached_features import CachedTimAnyUp
    ckpt = _one("timanyup/*to__oe_base_s2_ps4_tile64__fixdetach/*ep19.pth")
    lr, guid = _pastis_like(B=1)
    hr = _series(torch.randn(1, 12, 16, 16, 768), lr.t_start)
    up = TimAnyUp.from_checkpoint(ckpt)
    out = up(lr, guid, [Geometry.unknown((16, 16))], hr=hr)
    legacy = CachedTimAnyUp(768, 19, 16, ckpt).eval()
    per_t = build_guidance_batch(guid, GuidanceSpec({"S2": LEGACY_S2_13}, None),
                                 match="nearest", f_start=lr.t_start, f_end=lr.t_end)
    ref = legacy.features((lr.tokens, hr.tokens), per_t)                 # (B, T, D, 16, 16)
    assert torch.equal(out.tokens.permute(0, 1, 4, 2, 3), ref)
    assert out.tokens.shape[1] == 12 and out.provenance["space"] == "target"


@torch.no_grad()
def test_anyup_matches_hub_model_with_legacy_normalization():
    hub = Path(torch.hub.get_dir()) / "wimmerth_anyup_main"
    if not hub.exists():
        pytest.skip("AnyUp hub repo not cached")
    from exp.upsamplers.upa_anyup import _norm_rgb
    from manyup.upsamplers import AnyUp
    lr, guid = _pastis_like(B=1, D=64)
    out = AnyUp(time_pool=None)(lr, guid, [Geometry.unknown((64, 64))])
    ref_model = torch.hub.load(str(hub), "anyup_multi_backbone", source="local").eval()
    s2 = guid.modalities["S2"]
    rgb = s2.data[0, :, [S2_BANDS.index(b) for b in ("B04", "B03", "B02")]].float()
    for t in (0, 7):
        ref = ref_model(_norm_rgb(rgb[t])[None], lr.tokens[:, t].permute(0, 3, 1, 2),
                        output_size=(64, 64))
        assert torch.allclose(out.tokens[:, t].permute(0, 3, 1, 2), ref, atol=1e-5)


@cuda
def test_upma_matches_legacy_function():
    from exp.upsamplers.upa_anyup import UPMA, percentile_stretch
    lr, guid = _pastis_like(B=1, D=32)
    lr = _series(lr.tokens.cuda(), lr.t_start)
    out = JBU.upma(fit_steps=5)(lr, guid, [Geometry.unknown((64, 64))])
    raw = guid.modalities["S2"].data[0].float().mean(0)
    idx = [S2_BANDS.index(e.split("<-")[-1]) for e in LEGACY_S2_13]
    guide = percentile_stretch(raw[idx].numpy())                         # legacy composite
    ours = build_guidance_batch(guid, GuidanceSpec({"S2": LEGACY_S2_13}, "mean", "percentile"))
    assert np.allclose(guide, ours[0].numpy(), atol=1e-5)
    ref = UPMA(guide, lr.tokens.mean(1).permute(0, 3, 1, 2), fit_steps=5)
    assert torch.allclose(out.tokens[:, 0].permute(0, 3, 1, 2), ref.float(), atol=2e-2)
