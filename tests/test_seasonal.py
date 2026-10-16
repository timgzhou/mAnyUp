"""Seasonal timAnyUp components (manyup/seasonal.py): month -> season mapping, season means
over present months only, the fuser starting as the identity, selection never touching a
month that was not upsampled, and attention features blind to lr-token order."""
import torch

from manyup.seasonal import (AttentionSelector, KLMeanSelector, ResidualFuser, SeasonalContext,
                             attention_features, season_index, season_mean, select)


def test_season_index_maps_each_month_to_the_season_containing_it():
    months_s = torch.tensor([[0, 31, 59, 90, 120, 151]])
    months_e = torch.tensor([[30, 58, 89, 119, 150, 180]])
    seasons_s, seasons_e = torch.tensor([[0, 90]]), torch.tensor([[89, 150]])
    idx = season_index(seasons_s, seasons_e, months_s, months_e)
    assert idx.tolist() == [[0, 0, 0, 1, 1, -1]]           # June is in no season


def test_season_mean_averages_present_months_of_the_same_season():
    x = torch.arange(4.0).reshape(1, 4, 1) + 1             # months 1, 2, 3, 4
    season = torch.tensor([[0, 0, 1, 1]])
    present = torch.tensor([[True, False, True, True]])
    m = season_mean(x, season, present, 2)
    assert m.flatten().tolist() == [1.0, 1.0, 3.5, 3.5]      # month 2 does not count


def _ctx(B=2, T=3, H=4, W=4, hw=16, D=8, present=None):
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(B, T, H * W, hw), -1)
    return SeasonalContext(torch.randn(B, T, D, H, W), A, A.mean(1, keepdim=True).expand_as(A),
                           torch.ones(B, T, dtype=torch.bool) if present is None else present)


def test_fuser_starts_as_the_identity_on_the_upsample():
    up, hr = torch.randn(2, 3, 8, 4, 4), torch.randn(2, 3, 8, 4, 4)
    for hidden in (0, 16):
        assert torch.equal(ResidualFuser(8, hidden)(up, hr), up)


def test_selection_never_picks_a_month_that_was_not_upsampled():
    present = torch.tensor([[True, False, True]] * 2)
    ctx = _ctx(present=present)
    m = select(KLMeanSelector()(ctx) + 100 * (~present)[:, :, None, None], present, 20)
    assert not m[:, 1].any() and (m.flatten(1).sum(1) == 20).all()


def test_attention_features_ignore_the_order_of_lr_tokens():
    ctx = _ctx()
    perm = torch.randperm(ctx.attn.shape[-1])
    shuffled = SeasonalContext(ctx.up, ctx.attn[..., perm], ctx.attn_season[..., perm],
                               ctx.present)
    assert torch.allclose(attention_features(ctx), attention_features(shuffled), atol=1e-6)
    assert AttentionSelector()(ctx).shape == (2, 3, 4, 4)


def test_crop_patches_takes_the_block_at_that_location_and_date():
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    from manyup.encoders.olmoearth import crop_patches
    B, H, W, T, C = 2, 16, 16, 3, 2
    x = torch.arange(B * H * W * T * C, dtype=torch.float).reshape(B, H, W, T, C)
    ts = torch.arange(B * T * 3).reshape(B, T, 3)
    s = MaskedOlmoEarthSample(timestamps=ts, sentinel1=x, sentinel1_mask=x.clone())
    b, t, y, xx = (torch.tensor(v) for v in ([1, 0], [2, 0], [3, 0], [1, 2]))
    c = crop_patches(s, b, t, y, xx, 4)
    assert c.sentinel1.shape == (2, 4, 4, 1, C) and c.timestamps.shape == (2, 1, 3)
    assert torch.equal(c.sentinel1[0, :, :, 0], x[1, 12:16, 4:8, 2])
    assert torch.equal(c.sentinel1[1, :, :, 0], x[0, 0:4, 8:12, 0])
    assert torch.equal(c.timestamps[0, 0], ts[1, 2])


def test_geoid_val_ignores_excluded_events_and_nothing_else():
    from types import SimpleNamespace
    from exp.common.tasks import IGNORE, TASKS
    task = TASKS["geoidflood"]
    t = torch.zeros(3, 4, 4, dtype=torch.long)
    ids = ["EMSR342-12_3@r0c0", "EMSR203-1_1@r0c0", "EMSR3420-1_1@r0c0"]   # prefix look-alike
    for split in ("val", "train", "test"):
        out = task.target(SimpleNamespace(target=t, split=split, ids=ids))
        assert (out[0] == IGNORE).all() == (split == "val")
        assert (out[1:] == 0).all()
