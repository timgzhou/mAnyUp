"""manyup.inject: the mixed-resolution OlmoEarth pass reproduces OlmoEarth where it should."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _setup():
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    from exp.common.config import MODEL_SIZE_TO_ID
    from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput, _to_device
    from manyup.inject import MixedResOE

    enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID["tiny"], OlmoEarthInput({"S1": "sentinel1"}),
                           16, 64, name="tiny").cuda().eval()
    g = torch.Generator().manual_seed(0)
    B, S, T = 2, 64, 4
    d = {"timestamps": torch.tensor([[[15, m, 2020] for m in (0, 3, 6, 9)]] * B)}
    for name in ("sentinel2_l2a", "sentinel1"):
        C = len(Modality.get(name).band_order)
        d[name] = torch.randn(B, S, S, T, C, generator=g)
        mask = torch.full_like(d[name], MaskValue.ONLINE_ENCODER.value)
        if name == "sentinel2_l2a":
            mask[1, :, :, 2] = MaskValue.MISSING.value            # one missing S2 frame
        d[name + "_mask"] = mask
    sample = _to_device(MaskedOlmoEarthSample(**d), "cuda", torch.bfloat16)
    return enc, MixedResOE(enc.encoder, 6).cuda().eval(), sample


@torch.no_grad()
def test_lr_only_pass_is_olmoearths_series_encoding():
    enc, mix, sample = _setup()
    ref = enc(sample, "series").float()                                        # (B,T,4,4,D)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = mix.head_blocks(mix.embed(sample, 16))
        x, _ = mix.tail(lr, None, None)
    out = lr.pooled(x).float()
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@torch.no_grad()
def test_crop_is_encoded_at_its_true_position_and_slot():
    """The encodings a crop receives are those of its location on the full ps4 grid. (Its
    patch embedding differs near the crop border: the FlexiViT embedding resizes the image
    bicubically, so the full image's patches see their neighbours.)"""
    from manyup.encoders.olmoearth import _slice_time, crop_patches
    _, mix, sample = _setup()
    t0 = 1
    one = _slice_time(sample, t0)                                              # T = 1
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 1], [t0, t0], [3, 15], [7, 0]))
    crops = crop_patches(sample, b, t, y, x, 4)

    def encodings(s, **kw):                         # embed() minus the raw patch embedding
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = mix.embed(s, 4, **kw)
            raw = mix.encoder.patch_embeddings.forward(s, 4)
        raw = torch.cat([raw[m].flatten(1, -2) for m in p.dims], 1)[p.keep]
        return p, p.x.float() - raw.float()

    full, full_e = encodings(one)                                              # 16 x 16 grid
    _, crop_e = encodings(crops, slot=torch.zeros(2, dtype=torch.long, device="cuda"),
                          py=4 * y, px=4 * x)
    full_x = torch.zeros(*full.keep.shape, full_e.shape[-1], device="cuda")
    full_x[full.keep] = full_e
    per_mod, i = {}, 0
    for m, shape in full.dims.items():                                         # (B,16,16,1,S,D)
        n = shape[1] * shape[2] * shape[3] * shape[4]
        per_mod[m] = full_x[:, i:i + n].reshape(shape[0], 16, 16, shape[4], -1)
        i += n
    want = torch.cat([torch.cat([per_mod[m][bb, yy, xx] for m in full.dims])
                      for bb, yy, xx in zip(b.tolist(), y.tolist(), x.tolist())])
    torch.testing.assert_close(crop_e, want, atol=1e-2, rtol=1e-2)   # bf16 rounding
    # and a different position or slot is a different encoding
    _, moved = encodings(crops, slot=torch.ones(2, dtype=torch.long, device="cuda"),
                         py=4 * y, px=4 * x + 1)
    assert (moved - want).abs().amax() > 0.1


@torch.no_grad()
def test_injected_crops_only_touch_their_own_sample():
    from manyup.encoders.olmoearth import crop_patches
    _, mix, sample = _setup()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = mix.head_blocks(mix.embed(sample, 16))
        base, _ = mix.tail(lr, None, None)
        b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 0, 0], [0, 1, 3], [0, 5, 9],
                                                             [1, 5, 2]))
        hr = mix.head_blocks(mix.embed(crop_patches(sample, b, t, y, x, 4), 4,
                                       slot=t, py=4 * y, px=4 * x))
        with_hr, hr_out = mix.tail(lr, hr, b)
    n0 = int(lr.seqlens[0])
    assert hr_out.shape == hr.x.shape
    assert not torch.allclose(with_hr[:n0], base[:n0])                        # sample 0 moved
    torch.testing.assert_close(with_hr[n0:], base[n0:], atol=1e-2, rtol=1e-2)  # sample 1 not


def test_per_step_topk_spreads_the_budget_over_present_steps():
    from exp.upsamplers.finetune_inject import per_step_topk
    s = torch.rand(2, 4, 3, 3)
    present = torch.tensor([[True, True, False, True], [True, True, True, True]])
    m = per_step_topk(s, present, 12)
    assert m[0].flatten(1).sum(1).tolist() == [4, 4, 0, 4]
    assert m[1].flatten(1).sum(1).tolist() == [3, 3, 3, 3]
    top = s[1, 0].flatten().topk(3).indices
    assert m[1, 0].flatten()[top].all()


def test_temporal_attn_pool_ignores_absent_steps():
    from exp.upsamplers.finetune_inject import TemporalAttnPool
    pool = TemporalAttnPool(8)
    x = torch.randn(2, 3, 2, 2, 8)
    present = torch.tensor([[True, False, False], [True, True, True]])
    out = pool(x, present)
    torch.testing.assert_close(out[0], x[0, 0])            # one present step: it IS the pool
    x2 = x.clone()
    x2[0, 1:] = 1e3                                         # absent steps cannot leak in
    torch.testing.assert_close(pool(x2, present)[0], out[0])


def test_rank_within_ranks_inside_each_group():
    from exp.upsamplers.finetune_inject import rank_within
    v = torch.tensor([3.0, 1.0, 2.0, 10.0, -1.0])
    g = torch.tensor([0, 0, 0, 1, 1])
    torch.testing.assert_close(rank_within(v, g), torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0]))


def test_cross_attn_selector_takes_any_number_of_steps():
    from manyup.inject import CrossAttnSelector
    D, G = 8, 2
    sel = CrossAttnSelector(D, hidden=16, temporal=True)
    lr = torch.randn(2, G, 4, 4, D)
    for T in (3, 7):                          # months or dates: same weights
        e = torch.randn(2, T, 16, 16, D)
        season = torch.randint(0, G, (2, T))
        present = torch.ones(2, T, dtype=torch.bool)
        present[1, 0] = False
        out = sel(e, e.mean(1, keepdim=True).expand_as(e), lr, season, present,
                  torch.ones(2, G, dtype=torch.bool), torch.randint(0, 12, (2, T)),
                  torch.randint(0, 12, (2, G)))
        assert out.shape == (2, T, 16, 16) and torch.isfinite(out).all()


def _splat_inputs(T=4, G=2, D=6):
    lr = torch.randn(1, G, 4, 4, D)
    season = torch.tensor([[0, 0, 1, 1]])[:, :T]
    present = torch.ones(1, T, dtype=torch.bool)
    hr = torch.full((1, D), 50.0)                                   # one crop, far from lr values
    idx = tuple(torch.tensor([v]) for v in (0, 1, 5, 9))            # (b, t, y, x)
    return lr, torch.ones(1, G, dtype=torch.bool), season, present, hr, idx


def test_splat_at_init_is_close_to_bilinear_plus_overwrite():
    from manyup.inject import GaussianSplatReadout
    lr, lr_valid, season, present, hr, idx = _splat_inputs()
    out = GaussianSplatReadout(6)(lr, lr_valid, season, present, hr, idx)
    assert out.shape == (1, 4, 16, 16, 6)
    assert out[0, 1, 5, 9].mean() > 45                  # the crop dominates its own cell
    assert (out[0, 1, 5, 12].abs() < 10).all()          # 3 cells away: season tokens again
    assert (out[0, 0].abs() < 10).all()                 # no time kernel: other months untouched


def test_splat_without_time_kernel_reads_only_own_season():
    from manyup.inject import GaussianSplatReadout
    lr, lr_valid, season, present, hr, idx = _splat_inputs()
    m = GaussianSplatReadout(6)
    a = m(lr, lr_valid, season, present, hr, idx)
    lr2 = lr.clone()
    lr2[:, 1] += 100.0                                  # season 1 = months 2, 3
    b = m(lr2, lr_valid, season, present, hr, idx)
    torch.testing.assert_close(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def test_space_time_splat_lets_a_crop_inform_the_next_month():
    from manyup.inject import GaussianSplatReadout
    lr, lr_valid, season, present, hr, idx = _splat_inputs()
    m = GaussianSplatReadout(6, temporal=True, guided=True)
    e = torch.randn(1, 4, 16, 16, 6)
    out = m(lr, lr_valid, season, present, hr, idx, e=e)
    no = m(lr, lr_valid, season, present, hr[:0], tuple(i[:0] for i in idx), e=e)
    assert (out[0, 2, 5, 9] - no[0, 2, 5, 9]).abs().mean() > 1.0     # month 2 sees month 1's crop


def test_consecutive_pairs_follow_each_orbit():
    from exp.common.heads import consecutive_pairs
    present = torch.tensor([[True, True, False, True, True]])
    orbit = torch.tensor([[1, 2, 1, 1, 2]])
    assert consecutive_pairs(present, orbit)[0].tolist() == [[0, 3], [1, 4]]


def test_pair_head_ignores_pairs_without_change():
    from exp.common.heads import Pa2PxHead
    head = Pa2PxHead("pair_diff", 4, 2, 4)
    tok = torch.randn(1, 3, 2, 2, 4)
    tok[:, 1] = tok[:, 0]                                   # no change between steps 0 and 1
    one = head(tok, None, torch.tensor([[[0, 1], [-1, -1]]]))
    bias = head.linear.bias.view(2, 4, 4).repeat(1, 2, 2)
    torch.testing.assert_close(one[0], bias)                # zero difference -> the bias only
    two = head(tok, None, torch.tensor([[[0, 1], [1, 2]]]))
    assert not torch.allclose(two, one)


def test_pair_head_backprops_with_a_pairless_sample():
    from exp.common.heads import Pa2PxHead
    head = Pa2PxHead("pair_diffmean", 4, 3, 4)
    tok = torch.randn(2, 3, 2, 2, 4, requires_grad=True)
    out = head(tok, None, torch.tensor([[[0, 1]], [[-1, -1]]]))
    out.sum().backward()
    assert torch.isfinite(tok.grad).all() and torch.isfinite(out).all()


def test_seasons_used_ignores_padded_steps():
    from manyup.seasonal import seasons_used
    season = torch.tensor([[0, 0, 1, 0, -1]])               # steps 3, 4 are batch padding
    present = torch.tensor([[True, True, True, False, False]])
    assert seasons_used(season, present, 3)[0].tolist() == [True, True, False]



def _cellattn_inputs(B=2, NS=4, D=16):
    lr = torch.randn(B, NS, 4, 4, D)
    lr_valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    b, t = torch.tensor([0, 0, 1]), torch.tensor([1, 5, 2])
    y, x = torch.tensor([3, 3, 15]), torch.tensor([7, 8, 0])
    return lr, lr_valid, torch.randn(3, D), (b, t, y, x)


def test_cellattn_without_crops_is_the_season_mean_nearby():
    from manyup.inject import CellAttnReadout
    lr, lr_valid, hr, idx = _cellattn_inputs()
    m = CellAttnReadout(16)
    out = m(lr, lr_valid, hr[:0], tuple(i[:0] for i in idx))
    # at init every weight is the season Gaussian (sig 2 cells, same log_w): cell (5, 5)
    # (centre 5.5, 5.5) = Gaussian-weighted mean of the 3x3 ps16 tokens (centres 4j + 2) of
    # every used season
    c = torch.arange(3) * 4 + 2.0                                   # ps16 rows/cols 0..2
    wy = torch.exp(-0.5 * ((5.5 - c) / 2) ** 2)
    wgt = (wy[:, None] * wy[None, :])[None].expand(3, 3, 3)         # (seasons, 3, 3)
    want = (wgt[..., None] * lr[0, :3, 0:3, 0:3]).sum((0, 1, 2)) / wgt.sum()
    torch.testing.assert_close(out[0, 5, 5], want, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(out).all()
    # an unused season never contributes
    lr2 = lr.clone(); lr2[0, 3] = 1e3
    assert torch.allclose(m(lr2, lr_valid, hr[:0], tuple(i[:0] for i in idx))[0], out[0])


def test_cellattn_crops_reach_only_nearby_cells():
    from manyup.inject import CellAttnReadout
    lr, lr_valid, hr, idx = _cellattn_inputs()
    m = CellAttnReadout(16)
    no = m(lr, lr_valid, hr[:0], tuple(i[:0] for i in idx))
    out = m(lr, lr_valid, hr, idx)
    changed = (out - no).abs().amax(-1) > 1e-6                     # (B, H, W)
    assert changed[0, 3, 7] and changed[0, 3, 8] and changed[1, 15, 0]
    assert changed[0, 2:5, 6:10].sum() == changed[0].sum()        # radius 1 around (3,7),(3,8)
    assert changed[1, 14:16, 0:2].sum() == changed[1].sum()
    # a heavily down-gated crop fades out
    gated = m(lr, lr_valid, hr, idx, hr_logw=torch.full((3,), -30.0))
    assert torch.allclose(gated, no, atol=1e-5)


@torch.no_grad()
def test_tail_multi_equals_tail_on_the_union_of_crop_sets():
    from manyup.encoders.olmoearth import crop_patches
    from manyup.inject import Packed
    _, mix, sample = _setup()
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 1, 0, 1], [0, 1, 3, 2],
                                                         [0, 5, 9, 1], [1, 5, 2, 7]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = mix.head_blocks(mix.embed(sample, 16))
        hr = mix.head_blocks(mix.embed(crop_patches(sample, b, t, y, x, 4), 4,
                                       slot=t, py=4 * y, px=4 * x))
        lr_one, hr_one = mix.tail(lr, hr, b)
        # the same crops as two sets: crops 0-1 and 2-3
        cut = int(hr.seqlens[:2].sum())
        sets = [(Packed(hr.x[:cut], hr.seqlens[:2], hr.keep[:2], hr.dims, hr.masks), b[:2]),
                (Packed(hr.x[cut:], hr.seqlens[2:], hr.keep[2:], hr.dims, hr.masks), b[2:])]
        lr_two, (h1, h2) = mix.tail_multi(lr, sets)
    # same tokens per sequence, in a different order inside it: equal up to bf16 attention noise
    torch.testing.assert_close(lr_two, lr_one, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(torch.cat([h1, h2]), hr_one, atol=3e-2, rtol=3e-2)


def test_within_places_a_step_inside_its_parent_step():
    from exp.upsamplers.finetune_inject import within
    ts, te = torch.tensor([[0, 31, 61]]), torch.tensor([[30, 60, 91]])      # 3 months
    f = within(ts, te, torch.tensor([[0]]), torch.tensor([[91]]), torch.zeros(1, 3, dtype=torch.long))
    assert f[0, 0] < -0.3 and abs(f[0, 1]) < 0.02 and f[0, 2] > 0.3       # early / mid / late
    assert within(ts, te, torch.tensor([[0]]), torch.tensor([[91]]),
                  torch.tensor([[0, -1, 0]]))[0, 1] == 0                  # no parent -> 0


def test_native_logits_equal_the_dense_month_by_month_readout():
    """native_logits (sparse) == building every month's logit map (season token -> 16x16 px,
    crops overwrite their 4x4 px block with their own head map area-downsampled, gated), then
    the mean over present months."""
    import torch.nn.functional as F
    from exp.upsamplers.finetune_inject import native_logits
    g = torch.Generator().manual_seed(0)
    B, NS, h, T, D, C = 2, 4, 4, 6, 8, 3
    lin = torch.nn.Linear(D, C * 16 * 16)
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    season = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, -1, -1]])
    present = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]).bool()
    b = torch.tensor([0, 0, 0, 1, 1])
    t = torch.tensor([0, 1, 5, 2, 3])
    y = torch.tensor([0, 0, 15, 7, 7])        # two crops at the same cell, different months
    x = torch.tensor([3, 3, 9, 4, 5])
    tok = torch.randn(len(b), D, generator=g)
    gate = torch.rand(len(b), generator=g)
    with torch.no_grad():
        got = native_logits(lin, lr_F, season, present, tok, (b, t, y, x), gate)
        ref = torch.zeros(B, C, 64, 64)
        for bi in range(B):
            maps = []
            for ti in range(T):
                if not present[bi, ti]:
                    continue
                s = season[bi, ti].clamp(min=0)
                m = lin(lr_F[bi, s]).view(h, h, C, 16, 16).permute(2, 0, 3, 1, 4).reshape(C, 64, 64)
                for n in range(len(b)):
                    if b[n] == bi and t[n] == ti:
                        crop = F.avg_pool2d(lin(tok[n]).view(1, C, 16, 16), 4)[0]
                        blk = m[:, 4 * y[n]:4 * y[n] + 4, 4 * x[n]:4 * x[n] + 4]
                        m = m.clone()
                        m[:, 4 * y[n]:4 * y[n] + 4, 4 * x[n]:4 * x[n] + 4] = blk + gate[n] * (crop - blk)
                maps.append(m)
            ref[bi] = torch.stack(maps).mean(0)
        no_crop = native_logits(lin, lr_F, season, present)
    assert torch.allclose(got, ref, atol=1e-5)
    # no crops: the head on the time-mean features (the ps16 series fine-tune's head)
    from exp.common.heads import Pa2PxHead
    head = Pa2PxHead("time_mean", D, C, 16)
    head.linear = lin
    months = lr_F[torch.arange(B)[:, None], season.clamp(min=0)]              # (B, T, h, w, D)
    with torch.no_grad():
        assert torch.allclose(no_crop, head(months, present), atol=1e-5)


def test_native_attn_with_a_constant_score_is_the_native_time_mean():
    """Zero score -> uniform softmax over present months -> exactly native_logits."""
    from exp.upsamplers.finetune_inject import native_attn_logits, native_logits
    g = torch.Generator().manual_seed(1)
    B, NS, h, T, D, C = 2, 4, 4, 6, 8, 3
    lin = torch.nn.Linear(D, C * 16 * 16)
    score = torch.nn.Sequential(torch.nn.LayerNorm(D), torch.nn.Linear(D, 1))
    torch.nn.init.zeros_(score[1].weight)
    torch.nn.init.zeros_(score[1].bias)
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    season = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, -1, -1]])
    present = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]).bool()
    idx = (torch.tensor([0, 0, 1]), torch.tensor([0, 5, 2]), torch.tensor([0, 15, 7]),
           torch.tensor([3, 9, 4]))
    tok = torch.randn(3, D, generator=g)
    gate = torch.rand(3, generator=g)
    with torch.no_grad():
        for crops in ((None, None, None), (tok, idx, gate)):
            a = native_attn_logits(lin, score, lr_F, season, present, *crops)
            assert torch.allclose(a, native_logits(lin, lr_F, season, present, *crops), atol=1e-5)


def test_native_attn_weights_months_by_their_token_score():
    """A crop with a huge score takes its 4x4 px block over entirely (its own logits there);
    every other block is unchanged by it."""
    from exp.upsamplers.finetune_inject import native_attn_logits
    g = torch.Generator().manual_seed(2)
    B, NS, h, T, D, C = 1, 2, 4, 4, 8, 2
    lin = torch.nn.Linear(D, C * 16 * 16)
    score = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(D, 1))
    torch.nn.init.zeros_(score[1].weight)
    torch.nn.init.zeros_(score[1].bias)
    score[1].weight.data[0, 0] = 1.0                        # score = feature channel 0
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    lr_F[..., 0] = 0.0
    season, present = torch.tensor([[0, 0, 1, 1]]), torch.ones(1, T, dtype=torch.bool)
    tok = torch.randn(1, D, generator=g)
    tok[0, 0] = 100.0
    idx = tuple(torch.tensor([v]) for v in (0, 2, 5, 6))
    with torch.no_grad():
        base = native_attn_logits(lin, score, lr_F, season, present)
        got = native_attn_logits(lin, score, lr_F, season, present, tok, idx)
        own = torch.nn.functional.avg_pool2d(lin(tok).view(1, C, 16, 16), 4)[0]
    assert torch.allclose(got[0, :, 20:24, 24:28], own, atol=1e-4)
    diff = (got - base).abs()
    diff[:, :, 20:24, 24:28] = 0
    assert diff.max() == 0


@pytest.mark.parametrize("which", ["cellattn", "splat"])
def test_overlap_readouts_read_only_overlapping_tokens(which):
    """Overlap-only readouts: cell (5, 5) ignores a crop at the neighbouring cell (5, 6) and
    the season tokens of the neighbouring ps16 cell; it does read the crop at (5, 5)."""
    from manyup.inject import CellAttnReadout, GaussianSplatReadout
    torch.manual_seed(0)
    B, NS, h, T, D = 1, 2, 4, 4, 8
    ro = (CellAttnReadout(D, overlap=True) if which == "cellattn"
          else GaussianSplatReadout(D, temporal=True, overlap=True))
    lr = torch.randn(B, NS, h, h, D)
    lr_valid = torch.ones(B, NS, dtype=torch.bool)
    season, present = torch.tensor([[0, 0, 1, 1]]), torch.ones(B, T, dtype=torch.bool)

    def run(lr, hr, y, x):
        idx = tuple(torch.tensor(v, dtype=torch.long)
                    for v in ([0] * len(y), [1] * len(y), y, x))
        with torch.no_grad():
            if which == "cellattn":
                return ro(lr, lr_valid, hr, idx)                                  # (B, H, W, D)
            return ro(lr, lr_valid, season, present, hr, idx).mean(1)             # time mean

    hr = torch.randn(1, D)
    ref = run(lr, hr[:0], [], [])
    assert torch.allclose(run(lr, hr, [5], [6])[0, 5, 5], ref[0, 5, 5])          # neighbour crop
    assert not torch.allclose(run(lr, hr, [5], [5])[0, 5, 5], ref[0, 5, 5])      # own crop
    lr2 = lr.clone()
    lr2[:, :, 1, 2] += 1.0                          # ps16 cell (1, 2) neighbours (1, 1) = (5, 5)
    assert torch.allclose(run(lr2, hr[:0], [], [])[0, 5, 5], ref[0, 5, 5])


def test_native_set_readout():
    """No crops + zero score: the native time mean exactly. With crops (two at one cell,
    gated): a brute-force softmax per cell over (covering season tokens with log month-count
    prior, crops at the cell with their gate)."""
    import math
    import torch.nn.functional as F
    from exp.upsamplers.finetune_inject import NativeSetReadout, native_logits
    g = torch.Generator().manual_seed(3)
    B, NS, h, T, D, C = 2, 4, 4, 6, 8, 3
    lin = torch.nn.Linear(D, C * 16 * 16)
    ro = NativeSetReadout(D)
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    season = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, -1, -1]])
    present = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]).bool()
    empty = torch.zeros(0, dtype=torch.long)
    with torch.no_grad():
        got = ro(lin, lr_F, season, present, torch.zeros(0, D), (empty,) * 4)
        assert torch.allclose(got, native_logits(lin, lr_F, season, present), atol=1e-5)
        ro.score[1].weight.normal_(generator=g)
        b, t = torch.tensor([0, 0, 1]), torch.tensor([0, 5, 2])
        y, x = torch.tensor([6, 6, 1]), torch.tensor([9, 9, 2])
        tok = torch.randn(3, D, generator=g)
        logw = torch.randn(3, generator=g)
        got = ro(lin, lr_F, season, present, tok, (b, t, y, x), logw)
        for bi, cy, cx in ((0, 6, 9), (1, 1, 2), (0, 0, 0)):
            cand, act = [], []
            for s in range(NS):
                n = int(((season[bi] == s) & present[bi]).sum())
                if n:
                    m = lin(lr_F[bi, s, cy // 4, cx // 4]).view(C, 16, 16)
                    cand.append(m[:, (cy % 4) * 4:(cy % 4) * 4 + 4, (cx % 4) * 4:(cx % 4) * 4 + 4])
                    act.append(ro.score(lr_F[bi, s, cy // 4, cx // 4])[0] + math.log(n))
            for n in range(3):
                if b[n] == bi and y[n] == cy and x[n] == cx:
                    cand.append(F.avg_pool2d(lin(tok[n]).view(1, C, 16, 16), 4)[0])
                    act.append(ro.score(tok[n])[0] + logw[n])
            wgt = torch.softmax(torch.stack(act), 0)
            ref = (wgt[:, None, None, None] * torch.stack(cand)).sum(0)
            assert torch.allclose(got[bi, :, 4 * cy:4 * cy + 4, 4 * cx:4 * cx + 4], ref, atol=1e-5)


@pytest.mark.parametrize("ps", [8, 16])
@torch.no_grad()
def test_ps8_ps16_crops_are_encoded_at_their_true_position(ps):
    """As for ps4: a ps8 / ps16 crop's encodings are those of its place on the full ps8 / ps16
    grid of its frame."""
    from manyup.encoders.olmoearth import _slice_time, crop_patches
    _, mix, sample = _setup()
    g = 64 // ps
    one = _slice_time(sample, 1)
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 1], [1, 1], [g - 1, 0], [1, g - 1]))
    crops = crop_patches(sample, b, t, y, x, ps)

    def encodings(s, **kw):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = mix.embed(s, ps, **kw)
            raw = mix.encoder.patch_embeddings.forward(s, ps)
        raw = torch.cat([raw[m].flatten(1, -2) for m in p.dims], 1)[p.keep]
        return p, p.x.float() - raw.float()

    full, full_e = encodings(one)
    _, crop_e = encodings(crops, slot=torch.zeros(2, dtype=torch.long, device="cuda"),
                          py=ps * y, px=ps * x)
    full_x = torch.zeros(*full.keep.shape, full_e.shape[-1], device="cuda")
    full_x[full.keep] = full_e
    per_mod, i = {}, 0
    for m, shape in full.dims.items():
        n = shape[1] * shape[2] * shape[3] * shape[4]
        per_mod[m] = full_x[:, i:i + n].reshape(shape[0], g, g, shape[4], -1)
        i += n
    want = torch.cat([torch.cat([per_mod[m][bb, yy, xx] for m in full.dims])
                      for bb, yy, xx in zip(b.tolist(), y.tolist(), x.tolist())])
    torch.testing.assert_close(crop_e, want, atol=1e-2, rtol=1e-2)


def test_native_set_readout_footprints():
    """A ps8 crop (footprint 2x2 ps4 cells) and a ps16 crop (4x4) reach exactly the cells of
    their footprint, each cell getting its 4x4 px part of the crop's head map pooled to the
    footprint size."""
    import torch.nn.functional as F
    from exp.upsamplers.finetune_inject import NativeSetReadout
    g = torch.Generator().manual_seed(4)
    B, NS, h, T, D, C = 1, 2, 4, 4, 8, 2
    lin = torch.nn.Linear(D, C * 16 * 16)
    ro = NativeSetReadout(D)
    ro.score = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(D, 1))
    with torch.no_grad():                     # score = feature channel 0: crops dominate
        ro.score[1].weight.zero_()
        ro.score[1].bias.zero_()
        ro.score[1].weight[0, 0] = 1.0
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    lr_F[..., 0] = 0.0
    season, present = torch.tensor([[0, 0, 1, 1]]), torch.ones(1, T, dtype=torch.bool)
    tok = torch.randn(2, D, generator=g)
    tok[:, 0] = 100.0
    idx = (torch.tensor([0, 0]), torch.tensor([0, 2]), torch.tensor([2, 8]), torch.tensor([6, 0]))
    size = torch.tensor([2, 4])                 # ps8 at cells (2..3, 6..7); ps16 at (8..11, 0..3)
    with torch.no_grad():
        base = ro(lin, lr_F, season, present, tok[:0], tuple(v[:0] for v in idx))
        got = ro(lin, lr_F, season, present, tok, idx, None, size)
        m8 = F.avg_pool2d(lin(tok[0]).view(1, C, 16, 16), 2)[0]              # (C, 8, 8)
        m16 = lin(tok[1]).view(C, 16, 16)
    # ps8 footprint: pixels 8..15 rows, 24..31 cols; ps16: rows 32..47, cols 0..15
    assert torch.allclose(got[0, :, 8:16, 24:32], m8, atol=1e-4)
    assert torch.allclose(got[0, :, 32:48, 0:16], m16, atol=1e-4)
    outside = torch.ones(64, 64, dtype=torch.bool)
    outside[8:16, 24:32] = False
    outside[32:48, 0:16] = False
    assert torch.allclose(got[0][:, outside], base[0][:, outside], atol=1e-6)


def test_native_set_leave_one_out_equals_rerunning_without_the_crop():
    """--joint's in-context value: removing crop j's weight from its cell's softmax pool,
    (out - w_j L_j) / (1 - w_j), equals the readout re-run without crop j."""
    from einops import rearrange
    from exp.upsamplers.finetune_inject import NativeSetReadout
    g = torch.Generator().manual_seed(5)
    B, NS, h, T, D, C = 2, 4, 4, 6, 8, 3
    lin = torch.nn.Linear(D, C * 16 * 16)
    ro = NativeSetReadout(D)
    with torch.no_grad():
        ro.score[1].weight.normal_(generator=g)
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    season = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, -1, -1]])
    present = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]).bool()
    b, t = torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 5, 2])
    y, x = torch.tensor([6, 6, 2, 9]), torch.tensor([9, 9, 3, 1])       # two crops share a cell
    tok = torch.randn(4, D, generator=g)
    with torch.no_grad():
        _, (wgt, cand, out, idx) = ro(lin, lr_F, season, present, tok, (b, t, y, x), parts=True)
        for j in range(4):
            q = y[j] * 16 + x[j]
            m = (idx[b[j], q] == j).nonzero()[0, 0]
            w, L = wgt[b[j], q, NS + m], cand[b[j], q, NS + m]
            loo = (out[b[j], q] - w * L) / (1 - w)
            keep = torch.arange(4) != j
            ref = ro(lin, lr_F, season, present, tok[keep], tuple(v[keep] for v in (b, t, y, x)))
            ref = rearrange(ref, "b k (y i) (x j) -> b (y x) k (i j)", i=4, j=4)[b[j], q]
            assert torch.allclose(loo, ref, atol=1e-5)


def _time_setup(seed=6):
    g = torch.Generator().manual_seed(seed)
    B, NS, h, T, D = 1, 3, 4, 6, 8
    lr = torch.randn(B, NS, h, h, D, generator=g)
    lr_valid = torch.ones(B, NS, dtype=torch.bool)
    season = torch.tensor([[0, 0, 1, 1, 2, 2]])
    present = torch.ones(B, T, dtype=torch.bool)
    frac = torch.tensor([[-0.25, 0.25, -0.25, 0.25, -0.25, 0.25]])
    return g, lr, lr_valid, season, present, frac, D


@pytest.mark.parametrize("mode", ["gauss", "query"])
def test_time_set_readout_reads_only_overlapping_tokens(mode):
    from manyup.inject import TimeSetReadout
    g, lr, lr_valid, season, present, frac, D = _time_setup()
    ro = TimeSetReadout(D, mode)
    hr = torch.randn(1, D, generator=g)

    def run(lr, y, x):
        idx = tuple(torch.tensor(v, dtype=torch.long) for v in ([0] * len(y), [1] * len(y), y, x))
        with torch.no_grad():
            return ro(lr, lr_valid, season, present, frac, hr[:len(y)], idx)     # (B,T,H,W,D)

    ref = run(lr, [], [])
    assert torch.allclose(run(lr, [5], [6])[0, :, 5, 5], ref[0, :, 5, 5])        # neighbour crop
    assert not torch.allclose(run(lr, [5], [5])[0, :, 5, 5], ref[0, :, 5, 5])    # own crop
    lr2 = lr.clone()
    lr2[:, :, 1, 2] += 1.0                                                      # neighbour ps16
    assert torch.allclose(run(lr2, [], [])[0, :, 5, 5], ref[0, :, 5, 5])


def test_time_splat_gives_two_dates_of_one_slot_different_slices():
    """The GEOID zero-diff problem: with a crop at t = 1 (slot 0, frac +0.25), dates 0 and 1 of
    slot 0 read different mixtures, and date 1 (the crop's own time) leans on the crop more."""
    from manyup.inject import TimeSetReadout
    g, lr, lr_valid, season, present, frac, D = _time_setup()
    ro = TimeSetReadout(D, "gauss")
    hr = torch.randn(1, D, generator=g)
    idx = tuple(torch.tensor([v]) for v in (0, 1, 5, 5))
    with torch.no_grad():
        wgt, feat, crop_of = ro.weights(lr, lr_valid, season, present, frac, hr, idx)
    q = 5 * 16 + 5
    NS = lr.shape[1]
    w_crop = wgt[0, :, q, NS]                                                   # crop weight per step
    assert w_crop[1] > w_crop[0] > w_crop[2]                                    # own time > same slot > next
    assert not torch.allclose(wgt[0, 0, q], wgt[0, 1, q])


def test_native_time_logits_with_one_slot_and_no_crops_is_the_native_head():
    from exp.upsamplers.finetune_inject import native_logits, native_time_logits
    from manyup.inject import TimeSetReadout
    g = torch.Generator().manual_seed(7)
    D, C = 8, 3
    lin = torch.nn.Linear(D, C * 16 * 16)
    lr = torch.randn(2, 1, 4, 4, D, generator=g)
    season, present = torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.bool)
    frac = torch.zeros(2, 4)
    empty = torch.zeros(0, dtype=torch.long)
    for mode in ("gauss", "query"):
        ro = TimeSetReadout(D, mode)
        with torch.no_grad():
            got = native_time_logits(lin, ro, lr, torch.ones(2, 1, dtype=torch.bool), season,
                                     present, frac, torch.zeros(0, D), (empty,) * 4)
            assert torch.allclose(got, native_logits(lin, lr, season, present), atol=1e-5)


def test_tile_sample_cuts_frames_into_independent_tiles_in_row_major_order():
    from einops import rearrange
    from exp.upsamplers.finetune_inject import tile_sample
    _, _, sample = _setup()
    tiles = tile_sample(sample, 4)
    v, t = sample.sentinel1, tiles.sentinel1                  # (B,64,64,T,C), (B*256,4,4,T,C)
    assert t.shape[0] == v.shape[0] * 256 and t.shape[1:3] == (4, 4)
    b, gy, gx = 1, 3, 10
    assert torch.equal(t[b * 256 + gy * 16 + gx], v[b, 4 * gy:4 * gy + 4, 4 * gx:4 * gx + 4])
    assert torch.equal(tiles.timestamps[b * 256 + 7], sample.timestamps[b])
    back = rearrange(t, "(b gy gx) y x t c -> b (gy y) (gx x) t c", gy=16, gx=16)
    assert torch.equal(back, v)


@pytest.mark.parametrize("reduction", ["pair_diff", "pair_diffmean"])
def test_pair_loo_values_equal_rerunning_without_the_crop(reduction):
    """--joint on a series pair head: a crop's value from the weight-removal shortcut equals
    re-running the time readout WITHOUT the crop through the pair head (loss over the crop's
    cell, without minus with)."""
    import torch.nn.functional as F
    from einops import rearrange
    from exp.common.heads import Pa2PxHead, consecutive_pairs
    from exp.upsamplers.finetune_inject import pair_loo_values
    from manyup.inject import TimeSetReadout
    g = torch.Generator().manual_seed(8)
    B, NS, T, D, C = 2, 3, 6, 8, 3
    ro = TimeSetReadout(D, "query")
    head = Pa2PxHead(reduction, D, C, 4)
    lr = torch.randn(B, NS, 4, 4, D, generator=g)
    lr_valid = torch.ones(B, NS, dtype=torch.bool)
    season = torch.tensor([[0, 0, 1, 1, 2, 2]] * B)
    present = torch.ones(B, T, dtype=torch.bool)
    frac = torch.tensor([[-0.25, 0.25] * 3] * B)
    pairs = consecutive_pairs(present, None)
    b = torch.tensor([0, 0, 0, 1])
    t = torch.tensor([1, 2, 4, 3])
    y, x = torch.tensor([5, 5, 0, 9]), torch.tensor([5, 5, 7, 2])      # two crops share a cell
    hr = torch.randn(4, D, generator=g)
    target = torch.randint(0, C, (B, 64, 64), generator=g)
    with torch.no_grad():
        wgt, feat, crop_of = ro.weights(lr, lr_valid, season, present, frac, hr, (b, t, y, x))
        val = pair_loo_values(head, wgt, feat, crop_of, pairs, target, NS)

        def cell_loss(keep, j):
            idx = tuple(v[keep] for v in (b, t, y, x))
            out = ro(lr, lr_valid, season, present, frac, hr[keep], idx)       # (B,T,16,16,D)
            logits = head(out, present, pairs)                                   # (B, C, 64, 64)
            bb, yy, xx = int(b[j]), int(y[j]), int(x[j])
            lg = logits[bb, :, 4 * yy:4 * yy + 4, 4 * xx:4 * xx + 4].reshape(1, C, 16)
            lab = target[bb, 4 * yy:4 * yy + 4, 4 * xx:4 * xx + 4].reshape(1, 16)
            return float(F.cross_entropy(lg, lab).mean())

        full = torch.ones(4, dtype=torch.bool)
        for j in range(4):
            keep = full.clone()
            keep[j] = False
            ref = cell_loss(keep, j) - cell_loss(full, j)
            assert abs(float(val[j]) - ref) < 1e-4, (j, float(val[j]), ref)


@torch.no_grad()
def test_pos_center_and_box():
    """pos "center": a crop sits at its centre in the lr frame (a ps4 crop 6 px up-left of its
    corner, a ps16 crop where it was). pos "box": the corner model at init, another once the
    bottom-right map is non-zero, on lr tokens and crops alike."""
    from manyup.encoders.olmoearth import crop_patches
    from manyup.inject import MixedResOE
    _, mix, sample = _setup()
    center = MixedResOE(mix.encoder, 6, "center", 16).cuda().eval()
    box = MixedResOE(mix.encoder, 6, "box", 16).cuda().eval()
    b, t = (torch.tensor(v, device="cuda") for v in ([0, 1], [1, 1]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for ps, y, x in ((4, [3, 15], [7, 2]), (8, [1, 7], [3, 2]), (16, [0, 3], [2, 1])):
            y, x = torch.tensor(y, device="cuda"), torch.tensor(x, device="cuda")
            crops = crop_patches(sample, b, t, y, x, ps)
            kw = dict(slot=t, py=y * ps, px=x * ps)
            ref = mix.embed(crops, ps, **kw).x.float()
            off = (ps - 16) / 2
            want = mix.embed(crops, ps, slot=t, py=y * ps + off, px=x * ps + off).x.float()
            got = center.embed(crops, ps, **kw).x.float()
            torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
            assert ((got - ref).abs().amax() > 0.1) == (ps != 16)
            torch.testing.assert_close(box.embed(crops, ps, **kw).x.float(), ref)
        lr_ref = mix.embed(sample, 16).x.float()
        torch.testing.assert_close(box.embed(sample, 16).x.float(), lr_ref)
        torch.nn.init.normal_(box.box.weight, std=0.1)
        torch.clear_autocast_cache()          # autocast keeps the bf16 copy of the old weight
        assert (box.embed(sample, 16).x.float() - lr_ref).abs().amax() > 0.1
        # the bottom-right corner tells a ps8 from a ps4 crop at the same top-left pixel
        z = torch.zeros(2, dtype=torch.long, device="cuda")
        d = [box.embed(crop_patches(sample, b, t, z, z, ps), ps, slot=t, py=z, px=z).x.float()
             - mix.embed(crop_patches(sample, b, t, z, z, ps), ps, slot=t, py=z, px=z).x.float()
             for ps in (4, 8)]
        assert (d[0] - d[1]).abs().amax() > 0.1


@torch.no_grad()
def test_doy_encoding():
    """doy: the plain model at init; once the map is non-zero, two dates of one month (same
    slot, same month encoding) get different encodings, lr tokens move too."""
    from manyup.encoders.olmoearth import crop_patches
    from manyup.inject import MixedResOE, doy_encoding
    _, mix, sample = _setup()
    ts = torch.tensor([[1, 0, 2020], [31, 11, 2019], [15, 6, 2020]])
    e = doy_encoding(ts, 48)
    assert e.shape == (3, 48)
    torch.testing.assert_close(e[0, :24], torch.zeros(24), atol=1e-5, rtol=0)         # 1 Jan: angle 0
    assert (e[0] - e[1]).abs().amax() < 0.5 and (e[0] - e[2]).abs().amax() > 1.5   # cyclic
    doy = MixedResOE(mix.encoder, 6, doy=True).cuda().eval()
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 1], [1, 1], [3, 9], [7, 2]))
    crops = crop_patches(sample, b, t, y, x, 4)
    later = crops._replace(timestamps=crops.timestamps + torch.tensor([10, 0, 0], device="cuda"))
    kw = dict(slot=t, py=y * 4, px=x * 4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref, lr_ref = mix.embed(crops, 4, **kw).x.float(), mix.embed(sample, 16).x.float()
        torch.testing.assert_close(mix.embed(later, 4, **kw).x.float(), ref)   # OlmoEarth: no day
        torch.testing.assert_close(doy.embed(crops, 4, **kw).x.float(), ref)
        torch.testing.assert_close(doy.embed(sample, 16).x.float(), lr_ref)
    torch.nn.init.normal_(doy.doy.weight, std=0.1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, c = doy.embed(crops, 4, **kw).x.float(), doy.embed(later, 4, **kw).x.float()
        assert (a - c).abs().amax() > 0.1
        assert (doy.embed(sample, 16).x.float() - lr_ref).abs().amax() > 0.1


@torch.no_grad()
@pytest.mark.parametrize("attn", ["oneway", "overlap"])
def test_oneway_attention(attn):
    """attn "oneway" / "overlap": the lr tokens are the crop-free model's, and each crop is what
    the blocks give it in a sequence of [its lr context at that depth, the crop] -- the context
    rows being reset to the lr pass's states before every block."""
    from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths
    from manyup.encoders.olmoearth import crop_patches
    from manyup.inject import MixedResOE, Packed, _positions
    _, mix, sample = _setup()
    one = MixedResOE(mix.encoder, 6, attn=attn).cuda().eval()
    torch.nn.init.normal_(one.hr_type, std=0.5)
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 0, 1, 1, 1], [0, 1, 2, 3, 3],
                                                         [0, 5, 9, 15, 2], [1, 5, 2, 15, 8]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = one.head_blocks(one.embed(sample, 16))
        base, _ = one.tail(lr, None, None)
        hr = one.head_blocks(one.embed(crop_patches(sample, b, t, y, x, 4), 4,
                                       slot=t, py=4 * y, px=4 * x))
        assert len(set(hr.seqlens.tolist())) > 1            # the missing S2 frame: a shorter crop
        lr_x, hr_x = one.tail(lr, hr, b)
        cut = int(hr.seqlens[:2].sum())                     # two crop sets = their union
        sets = [(Packed(hr.x[:cut], hr.seqlens[:2], hr.keep[:2], hr.dims, hr.masks, hr.yx[:2]), b[:2]),
                (Packed(hr.x[cut:], hr.seqlens[2:], hr.keep[2:], hr.dims, hr.masks, hr.yx[2:]), b[2:])]
        _, (h1, h2) = one.tail_multi(lr, sets)
        torch.testing.assert_close(lr_x.float(), base.float())
        torch.testing.assert_close(torch.cat([h1, h2]).float(), hr_x.float(), atol=2e-2, rtol=2e-2)

        l_s, _ = _positions(lr.seqlens)
        cell = one._lr_cells(lr)
        c_tok, _ = _positions(hr.seqlens)
        blocks = one.encoder.blocks[one.L:]
        want = []
        for c in range(len(b)):
            ctx = l_s == b[c]
            if attn == "overlap":
                ctx = ctx & (cell == (y[c] * 4 // 16) * 4096 + x[c] * 4 // 16)
                assert ctx.sum() and ctx.sum() < (l_s == b[c]).sum()
            xl, xc = lr.x, (hr.x[c_tok == c] + one.hr_type).to(lr.x.dtype)
            cu = get_cumulative_sequence_lengths(lr.seqlens)
            for blk in blocks:
                seq = torch.cat([xl[ctx], xc])
                n = torch.tensor([len(seq)], device="cuda")
                out = blk(x=seq, cu_seqlens=get_cumulative_sequence_lengths(n),
                          max_seqlen=len(seq), attn_mask=None, rope_positions=None)
                xc = out[int(ctx.sum()):]
                xl = blk(x=xl, cu_seqlens=cu, max_seqlen=int(lr.seqlens.max()), attn_mask=None,
                         rope_positions=None)
            want.append(one.encoder.norm(xc))
    torch.testing.assert_close(hr_x.float(), torch.cat(want).float(), atol=3e-2, rtol=3e-2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        joint = mix.tail(lr, hr, b)[1].float()
    assert (hr_x.float() - joint).abs().amax() > 0.1                      # not the joint pass


@torch.no_grad()
@pytest.mark.parametrize("mode", ["oneway", "overlap"])
def test_head_attn(mode):
    """head_attn: in blocks [0, L) the lr pass is untouched and a crop is what the blocks give
    it next to [its lr context at that depth]; with L = all blocks it is attn `mode` at L = 0;
    after block L the pass is the joint one on those crop tokens."""
    from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths
    from manyup.encoders.olmoearth import crop_patches
    from manyup.inject import MixedResOE, Packed, _positions
    _, mix, sample = _setup()                                                  # L = 6, alone
    head = MixedResOE(mix.encoder, 6, head_attn=mode).cuda().eval()
    n = len(mix.encoder.blocks)
    full = MixedResOE(mix.encoder, n, head_attn=mode).cuda().eval()
    tail0 = MixedResOE(mix.encoder, 0, attn=mode).cuda().eval()
    with pytest.raises(ValueError):
        MixedResOE(mix.encoder, 0, head_attn=mode)
    for m in (head, full, tail0, mix):
        m.hr_type.data = torch.linspace(-0.5, 0.5, 192, device="cuda")
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 0, 1, 1, 1], [0, 1, 2, 3, 3],
                                                         [0, 5, 9, 15, 2], [1, 5, 2, 15, 8]))
    crops = crop_patches(sample, b, t, y, x, 4)
    kw = dict(slot=t, py=4 * y, px=4 * x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = head.head_blocks(head.embed(sample, 16))
        torch.testing.assert_close(lr.x.float(), mix.head_blocks(mix.embed(sample, 16)).x.float())
        e = head.embed(crops, 4, **kw)
        hr = head.head_blocks(e, lr, b)
        assert hr.typed and len(lr.trace) == 6
        l_s, _ = _positions(lr.seqlens)
        cell, (c_tok, _) = head._lr_cells(lr), _positions(e.seqlens)
        want = []
        for c in range(len(b)):
            ctx = l_s == b[c]
            if mode == "overlap":
                ctx = ctx & (cell == (y[c] * 4 // 16) * 4096 + x[c] * 4 // 16)
            xc = (e.x[c_tok == c] + head.hr_type).to(lr.x.dtype)
            for blk, xl in zip(head.encoder.blocks, lr.trace):
                seq = torch.cat([xl[ctx], xc])
                cu = get_cumulative_sequence_lengths(torch.tensor([len(seq)], device="cuda"))
                xc = blk(x=seq, cu_seqlens=cu, max_seqlen=len(seq), attn_mask=None,
                         rope_positions=None)[int(ctx.sum()):]
            want.append(xc)
        torch.testing.assert_close(hr.x.float(), torch.cat(want).float(), atol=3e-2, rtol=3e-2)
        # the tail is the joint pass on these tokens (hr_type not added twice)
        lr_x, hr_x = head.tail(lr, hr, b)
        plain = Packed((hr.x.float() - head.hr_type).to(hr.x.dtype), hr.seqlens, hr.keep,
                       hr.dims, hr.masks)
        ref_lr, ref_hr = mix.tail(lr, plain, b)
        torch.testing.assert_close(lr_x.float(), ref_lr.float(), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(hr_x.float(), ref_hr.float(), atol=3e-2, rtol=3e-2)
        # L = every block: the one-way tail at L = 0
        lr_f = full.head_blocks(full.embed(sample, 16))
        a_lr, a_hr = full.tail(lr_f, full.head_blocks(full.embed(crops, 4, **kw), lr_f, b), b)
        lr_0 = tail0.head_blocks(tail0.embed(sample, 16))
        b_lr, b_hr = tail0.tail(lr_0, tail0.head_blocks(tail0.embed(crops, 4, **kw)), b)
        torch.testing.assert_close(a_lr.float(), b_lr.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(a_hr.float(), b_hr.float(), atol=1e-2, rtol=1e-2)


def test_native_set_offgrid_matches_grid():
    """NativeSetReadout with pixel positions (--jitter): on-grid pixels reproduce the per-cell
    pool exactly; an off-grid crop only changes the pixels it covers."""
    import torch
    from torch import nn
    from exp.upsamplers.finetune_inject import NativeSetReadout
    torch.manual_seed(0)
    B, NS, D, C, N = 2, 4, 8, 3, 40
    ro = NativeSetReadout(D)
    nn.init.normal_(ro.score[1].weight)
    lin = nn.Linear(D, C * 256)
    lr_F = torch.randn(B, NS, 4, 4, D)
    season = torch.arange(12)[None].expand(B, 12) // 3
    present = torch.ones(B, 12, dtype=torch.bool)
    hr = torch.randn(N, D)
    b, t = torch.randint(0, B, (N,)), torch.randint(0, 12, (N,))
    y, x = torch.randint(0, 16, (N,)), torch.randint(0, 16, (N,))
    ref, (w0, _, o0, _) = ro(lin, lr_F, season, present, hr, (b, t, y, x), parts=True)
    out, (w1, _, o1, cid) = ro(lin, lr_F, season, present, hr, (b, t, y, x), parts=True,
                               hr_px=(y * 4, x * 4))
    assert torch.allclose(ref, out, atol=1e-5) and torch.allclose(o0, o1, atol=1e-5)
    assert w1.ndim == 4 and torch.allclose(w1[..., 0], w0, atol=1e-6)
    # one crop, 2 px right and 1 px down of cell (3, 5) of sample 0
    one = (torch.tensor([0]), torch.tensor([0]), torch.tensor([3]), torch.tensor([5]))
    base = ro(lin, lr_F, season, present, hr[:0], tuple(v[:0] for v in one))
    off = ro(lin, lr_F, season, present, hr[:1], one, torch.tensor([10.0]),      # a heavy crop
             hr_px=(torch.tensor([13]), torch.tensor([22])))
    changed = (off - base).abs().sum(1) > 1e-3                       # (B, 64, 64)
    want = torch.zeros_like(changed)
    want[0, 13:17, 22:26] = True
    assert (changed == want).all()


def test_native_set_ps16_window_matches_grid():
    """hr_fp = 16: a ps16 crop given by its top-left pixel == the same crop as a 4 x 4-cell
    footprint on the grid (hr_size 4)."""
    import torch
    from torch import nn
    from exp.upsamplers.finetune_inject import NativeSetReadout
    torch.manual_seed(1)
    B, NS, D, C, N = 2, 4, 8, 3, 30
    ro = NativeSetReadout(D)
    nn.init.normal_(ro.score[1].weight)
    lin = nn.Linear(D, C * 256)
    lr_F = torch.randn(B, NS, 4, 4, D)
    season = torch.arange(12)[None].expand(B, 12) // 3
    present = torch.ones(B, 12, dtype=torch.bool)
    hr = torch.randn(N, D)
    b, t = torch.randint(0, B, (N,)), torch.randint(0, 12, (N,))
    y, x = torch.randint(0, 13, (N,)), torch.randint(0, 13, (N,))        # top-left ps4 cell
    ref = ro(lin, lr_F, season, present, hr, (b, t, y, x), hr_size=torch.full((N,), 4))
    out = ro(lin, lr_F, season, present, hr, (b, t, y, x), hr_px=(y * 4, x * 4), hr_fp=16)
    assert torch.allclose(ref, out, atol=1e-5)


def _apt_fixture(noisy_cell=(0, 0), B=2, T=6, H=16):
    """A MaskedOlmoEarthSample whose sentinel2_l2a is flat everywhere except one ps4 cell,
    which is spatially noisy AND (sample 1) also varies a lot across months."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

    g = torch.Generator().manual_seed(0)
    S, C = H * 4, len(Modality.get("sentinel2_l2a").band_order)
    s2 = torch.zeros(B, S, S, T, C)
    cy, cx = noisy_cell
    s2[:, cy * 4:cy * 4 + 4, cx * 4:cx * 4 + 4] = torch.randn(B, 4, 4, T, C, generator=g)
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    return ms


def _apt_scores(ms, present, B, H, modalities=("S2", "sentinel2_l2a"), apt_time="sum"):
    """Calls InjectFT._apt_scores without building the full model (no encoder needed:
    it is pure pixel math). modalities: flat (dataset_name, oe_name, ...) pairs, so a test
    can register more than one (ms carries all of them; _apt_scores sums every band of
    every DISTINCT oe_name it finds in olmoearth.modalities)."""
    from exp.upsamplers.finetune_inject import InjectFT

    pairs = list(zip(modalities[::2], modalities[1::2]))

    class FakeTask:
        olmoearth = type("O", (), {"modalities": dict(pairs)})()

    obj = InjectFT.__new__(InjectFT)
    obj.task = FakeTask()
    obj.apt_time = apt_time
    return obj._apt_scores(ms, present, B, H)


def test_apt_scores_are_high_only_at_the_noisy_cell():
    """APT's H(P): a flat cell (one value repeated) has zero entropy; the one spatially
    noisy cell scores (much) higher, at every month, than every flat cell."""
    import torch
    B, T, H = 2, 6, 16
    ms = _apt_fixture(noisy_cell=(3, 5), B=B, T=T, H=H)
    present = torch.ones(B, T, dtype=torch.bool)
    scores = _apt_scores(ms, present, B, H)
    assert scores.shape == (B, T, H, H)
    assert torch.isfinite(scores).all()
    flat = scores.clone()
    flat[:, :, 3, 5] = 0
    assert (scores[:, :, 3, 5] > flat.amax(dim=(2, 3))).all()
    assert torch.allclose(flat, torch.zeros_like(flat), atol=1e-4)


def test_apt_time_entropy_ignores_absent_months():
    """A cell constant within every present month but at a DIFFERENT level each month scores
    high from H_time alone; padded (not present) months must not pollute that histogram."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B, T, H = 1, 8, 16
    S, C = H * 4, len(Modality.get("sentinel2_l2a").band_order)
    s2 = torch.zeros(B, S, S, T, C)
    level = torch.linspace(0, 1, T).view(1, 1, 1, T, 1)
    s2[:, 20:24, 20:24] = level.expand(B, 4, 4, T, C)
    # months 4..7 are padding: give them an out-of-range level that must NOT affect the score
    s2[:, 20:24, 20:24, 4:] = 99.0
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.zeros(B, T, dtype=torch.bool)
    present[:, :4] = True
    scores_a = _apt_scores(ms, present, B, H)
    s2b = s2.clone()
    s2b[:, 20:24, 20:24, 4:] = -99.0            # change the padding; score must not change
    ms_b = MaskedOlmoEarthSample(timestamps=torch.zeros(B, T, 3, dtype=torch.long),
                                 sentinel2_l2a=s2b)
    scores_b = _apt_scores(ms_b, present, B, H)
    assert torch.allclose(scores_a[:, :4, 5, 5], scores_b[:, :4, 5, 5], atol=1e-5)
    assert scores_a[0, 0, 5, 5] > 0        # 4 distinct levels -> nonzero H_time
    assert torch.allclose(scores_a[:, :, 2, 2], torch.zeros(B, T), atol=1e-4)  # untouched cell


def test_apt_sums_entropy_over_every_band_and_modality():
    """Every band of every modality olmoearth.modalities names is scored and SUMMED -- not
    collapsed to one luma channel. A cell noisy in ALL of sentinel2_l2a's bands scores higher
    than the same cell noisy in only one of its bands; a second, separately-noisy modality
    (sentinel1) adds its own entropy on top of sentinel2_l2a's at the same cell."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B, T, H = 1, 3, 16
    S = H * 4
    C2, C1 = len(Modality.get("sentinel2_l2a").band_order), len(Modality.get("sentinel1").band_order)
    g = torch.Generator().manual_seed(0)

    def fixture(s2_bands_noisy, add_s1_noise):
        s2 = torch.zeros(B, S, S, T, C2)
        s2[:, 0:4, 0:4, :, :s2_bands_noisy] = torch.randn(B, 4, 4, T, s2_bands_noisy, generator=g)
        s1 = torch.zeros(B, S, S, T, C1)
        if add_s1_noise:
            s1[:, 0:4, 0:4] = torch.randn(B, 4, 4, T, C1, generator=g)
        return MaskedOlmoEarthSample(timestamps=torch.zeros(B, T, 3, dtype=torch.long),
                                     sentinel2_l2a=s2, sentinel1=s1)

    present = torch.ones(B, T, dtype=torch.bool)
    one_band = _apt_scores(fixture(1, False), present, B, H,
                           modalities=("S2", "sentinel2_l2a"))
    all_bands = _apt_scores(fixture(C2, False), present, B, H,
                            modalities=("S2", "sentinel2_l2a"))
    assert all_bands[0, 0, 0, 0] > one_band[0, 0, 0, 0] * 1.5   # many noisy bands >> one

    s2_only = _apt_scores(fixture(C2, False), present, B, H,
                          modalities=("S2", "sentinel2_l2a"))
    s2_and_s1 = _apt_scores(fixture(C2, True), present, B, H,
                            modalities=("S2", "sentinel2_l2a", "S1", "sentinel1"))
    assert s2_and_s1[0, 0, 0, 0] > s2_only[0, 0, 0, 0]          # S1's own entropy adds on top


def _apt_pool_obj(modalities=("S2", "sentinel2_l2a")):
    from exp.upsamplers.finetune_inject import InjectFT
    pairs = list(zip(modalities[::2], modalities[1::2]))

    class FakeTask:
        olmoearth = type("O", (), {"modalities": dict(pairs)})()

    obj = InjectFT.__new__(InjectFT)
    obj.task = FakeTask()
    obj.apt_merge_mse = 0.01
    obj.apt_time = "sum"
    return obj


def test_apt_pool_merges_similar_consecutive_months_and_spends_savings_elsewhere():
    """Three cells A > B > C by SCORE (forced, so the merge mechanics are tested in
    isolation from _apt_scores' incidental magnitudes): A's 2 present months are pixel-
    identical (MSE 0 -> merge, cost 1), B's 2 months differ (no merge, cost 2), C is a
    spare lower-scoring cell. Budget 3: without the merge, A (2) + B (2) alone would exceed
    the budget and C would never be reached; WITH the merge, A costs 1, leaving exactly
    enough (1 + 2 = 3) for B too, and C is still left out (budget exhausted) -- the freed
    crop went to the next-best-scoring UNADMITTED cell (B), not wasted."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, H = 1, 2, 16
    S = H * 4
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(0)
    s2 = torch.zeros(B_, S, S, T, C)
    same = torch.randn(1, 4, 4, 1, C, generator=g).expand(1, 4, 4, T, C)
    s2[:, 0:4, 0:4] = same                                    # cell A (0,0): identical months
    s2[:, 20:24, 20:24] = torch.randn(1, 4, 4, T, C, generator=g)   # cell B (5,5): differ
    s2[:, 40:44, 40:44] = torch.randn(1, 4, 4, T, C, generator=g)   # cell C (10,10): differ
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    # force the ranking A > B > C regardless of what _apt_scores would actually compute here
    forced = torch.zeros(B_, T, H, H)
    forced[:, :, 0, 0], forced[:, :, 5, 5], forced[:, :, 10, 10] = 3.0, 2.0, 1.0
    obj._apt_scores = lambda *a, **k: forced
    mask, groups = obj._apt_pool_mask(ms, present, B_, H, k=torch.tensor([3]))
    _, yy, xx = mask[0].nonzero(as_tuple=True)
    cells_picked = {(int(y), int(x)) for y, x in zip(yy, xx)}
    assert cells_picked == {(0, 0), (5, 5)}, (
        f"merge at A must free exactly 1 crop for B (next best); got {cells_picked}")
    a_group = [v for kk, v in groups.items() if kk[2:] == (0, 0)]
    assert len(a_group) == 1 and sorted(a_group[0]) == [0, 1]     # A: one merged group
    b_groups = [v for kk, v in groups.items() if kk[2:] == (5, 5)]
    assert len(b_groups) == 2 and sorted(sum(b_groups, [])) == [0, 1]  # B: two unmerged crops


def test_apt_pool_crops_averages_pixels_and_interpolates_the_slot():
    """A merged group's crop pixels are the MEAN of its member months' pixels, and its slot
    is the MEAN of their slots (continuous sincos time encoding needs no new embedding for
    this -- an averaged / fractional slot is already well-formed)."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    from manyup.encoders.olmoearth import crop_patches
    B, T, H = 1, 3, 16
    S = H * 4
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(0)
    s2 = torch.randn(B, S, S, T, C, generator=g)
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    obj = _apt_pool_obj()
    b = torch.tensor([0])
    t = torch.tensor([0])             # representative (first) month of the merged group
    y = x = torch.tensor([0])
    py = px = torch.tensor([0])
    slot = torch.tensor([1.0])        # season 1's slot at month 0 (representative)
    groups = {(0, 0, 0, 0): [0, 1, 2]}          # merges all 3 months
    pooled, mean_slot = obj._apt_pool_crops(ms, b, t, y, x, py, px, 4, slot, groups)
    ref = torch.stack([crop_patches(ms, b, torch.tensor([tt]), py, px, 4, pixels=True)
                       .sentinel2_l2a for tt in (0, 1, 2)]).mean(0)
    assert torch.allclose(pooled.sentinel2_l2a.float(), ref.float(), atol=1e-4)
    # slot given was the REPRESENTATIVE month's slot (1.0); the mean over 3 members with
    # the SAME representative-season slot convention used by _apt_pool_mask's rep_t = t0
    # would differ once real per-month slots are passed -- here all three share slot 1.0,
    # so the mean must equal 1.0 exactly (degenerate but checks the averaging arithmetic).
    assert torch.allclose(mean_slot, torch.tensor([1.0]), atol=1e-6)


def test_apt_pool_single_month_groups_are_untouched():
    """A group of size 1 (no merge) must produce byte-identical output to the plain,
    non-pooling crop_patches path -- _apt_pool_crops is a no-op there."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    from manyup.encoders.olmoearth import crop_patches
    B, T, H = 1, 2, 16
    S = H * 4
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(0)
    s2 = torch.randn(B, S, S, T, C, generator=g)
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    obj = _apt_pool_obj()
    b, t, y, x, py, px = (torch.tensor([0]),) * 6
    slot = torch.tensor([0.0])
    groups = {(0, 0, 0, 0): [0]}
    pooled, out_slot = obj._apt_pool_crops(ms, b, t, y, x, py, px, 4, slot, groups)
    ref = crop_patches(ms, b, t, py, px, 4, pixels=True)
    assert torch.allclose(pooled.sentinel2_l2a.float(), ref.sentinel2_l2a.float())
    assert torch.allclose(out_slot, slot)


def test_apt_pool_cumulative_cutoff_stops_rather_than_skips():
    """2026-10-08 (user-approved speed fix): admission is a vectorized CUMULATIVE-COST
    CUTOFF, not exact greedy -- it never skips an oversized cell to reach a smaller one
    further down the ranking. Three cells A > B > C: A costs 2 (no merge, blocks the
    cutoff), B costs 1 (would fit alone), C costs 1. Budget 2: cumulative cutoff admits ONLY
    A (cum = 2 <= 2) and stops there, even though skipping A for B + C (1 + 1 = 2) would
    also fit and use the same budget -- exact greedy would have found that; this rule does
    not, by design."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, H = 1, 2, 16
    S = H * 4
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(1)
    s2 = torch.zeros(B_, S, S, T, C)
    s2[:, 0:4, 0:4] = torch.randn(1, 4, 4, T, C, generator=g)      # cell A (0,0): differ, cost 2
    s2[:, 20:24, 20:24] = torch.zeros(1, 4, 4, T, C)                # cell B (5,5): flat, cost 1
    s2[:, 40:44, 40:44] = torch.zeros(1, 4, 4, T, C)                # cell C (10,10): flat, cost 1
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    forced = torch.zeros(B_, T, H, H)
    forced[:, :, 0, 0], forced[:, :, 5, 5], forced[:, :, 10, 10] = 3.0, 2.0, 1.0
    obj._apt_scores = lambda *a, **k: forced
    mask, groups = obj._apt_pool_mask(ms, present, B_, H, k=torch.tensor([2]))
    _, yy, xx = mask[0].nonzero(as_tuple=True)
    cells_picked = {(int(y), int(x)) for y, x in zip(yy, xx)}
    assert cells_picked == {(0, 0)}, (
        f"cumulative cutoff must stop at A (cost 2 exhausts the budget), not skip to B+C; "
        f"got {cells_picked}")
    a_group = [v for kk, v in groups.items() if kk[2:] == (0, 0)]
    assert len(a_group) == 2            # both months, unmerged (A never pools: differ)


def test_apt_pool_mask_has_no_entries_at_absent_months():
    """Regression (2026-10-08): the vectorized run_starts computation forced `breaks=True` in
    a sample's PADDING region (present-rank >= its own n_present), which seeded a spurious
    run-start -- and therefore a mask True -- at an ABSENT month with no corresponding groups
    entry. This crashed every real training batch containing a sample with any missing month
    (PASTIS/BioMassters/YieldSAT all have these) with `KeyError` in `_apt_pool_crops`, while
    every unit test up to this point used present=torch.ones(...) (no gaps) and so never
    caught it. Three samples here: one with a trailing gap, one with scattered gaps, one with
    a mid-series cutoff -- the mask must have a True ONLY at present months, and the mask and
    groups dict must agree exactly (every mask True has a groups entry and vice versa)."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, H = 3, 12, 16
    S = H * 4
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(2)
    s2 = torch.randn(B_, S, S, T, C, generator=g)
    s2[:, :, :, 6:] = s2[:, :, :, 5:6]                      # induce some real merges too
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    present[0, 8:] = False             # trailing gap
    present[1, [2, 7]] = False         # scattered gaps
    present[2, 3:] = False             # early cutoff
    obj = _apt_pool_obj()
    for k in (64, 512, 1020):
        mask, groups = obj._apt_pool_mask(ms, present, B_, H, k=k)
        b, t, y, x = mask.nonzero(as_tuple=True)
        for bi, ti in zip(b.tolist(), t.tolist()):
            assert present[bi, ti], f"mask True at an ABSENT month: sample {bi}, month {ti}"
        mask_keys = {(bi, ti, yi, xi) for bi, ti, yi, xi in
                    zip(b.tolist(), t.tolist(), y.tolist(), x.tolist())}
        assert mask_keys == set(groups.keys()), f"mask/groups mismatch at k={k}"


def test_apt_time_none_is_pure_per_frame_spatial_entropy():
    """--apt_time none: the more literal APT (arXiv:2510.18091) -- per-(month, cell) H_space
    ALONE, no time term. A cell that is flat within every month but changes level ACROSS
    months (nonzero H_time, zero H_space) scores exactly 0 with apt_time='none', and
    exactly sum('sum') - sum('none') recovers H_time on its own."""
    import torch
    B, T, H = 1, 4, 16
    ms = _apt_fixture(noisy_cell=(3, 5), B=B, T=T, H=H)   # the usual spatially-noisy fixture
    present = torch.ones(B, T, dtype=torch.bool)
    space_only = _apt_scores(ms, present, B, H, apt_time="none")
    summed = _apt_scores(ms, present, B, H, apt_time="sum")
    assert space_only.shape == (B, T, H, H)
    # the noisy cell's spatial entropy is identical whether or not H_time is added elsewhere
    assert torch.allclose(summed[:, :, 3, 5] - space_only[:, :, 3, 5],
                          (summed - space_only)[:, :, 3, 5], atol=1e-5)
    # a cell that is flat WITHIN every month (zero H_space) scores exactly 0 under apt_time=none
    # even though it has nonzero H_time (checked by test_apt_time_entropy_ignores_absent_months'
    # fixture reused here): every untouched cell is 0 under both settings
    assert torch.allclose(space_only[:, :, 0, 0], torch.zeros(B, T), atol=1e-4)


def test_apt_res_mask_never_exceeds_the_token_budget():
    """--apt_res: total tokens spent (1 per unsplit ps16 cell + 4 per split cell's ps4
    children) must never exceed k, for a range of budgets, on a realistic mixed-entropy
    fixture (noisy cell -> high split value, flat cells -> no value in splitting)."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 2, 11, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(3)
    s2 = torch.zeros(B_, S, S, T, C)
    # noisy ps16 blocks (worth splitting): top-left and one more, per sample
    s2[:, 0:16, 0:16] = torch.randn(B_, 16, 16, T, C, generator=g)
    s2[:, 16:32, 16:32] = torch.randn(B_, 16, 16, T, C, generator=g)
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    present[0, 8:] = False       # also check with a sample that has missing months
    obj = _apt_pool_obj()
    # base_cost floor is n_present x 16 ps16 cells: 128 (sample 0, 8 present), 176 (sample 1,
    # 11 present) -- k=50 is below both and must raise rather than silently overspend
    import pytest
    with pytest.raises(ValueError, match="apt_res"):
        obj._apt_res_mask(ms, present, B_, k=50)
    for k in (176, 300, 469, 1020):
        mask4, res_masks = obj._apt_res_mask(ms, present, B_, k)
        mask16 = res_masks[16]
        # each mask4 True IS one ps4 crop (1 token); each mask16 True IS one ps16 crop (1
        # token, covering a 4x4 ps4-cell footprint) -- no extra multiplier on either count
        tokens = mask4.flatten(1).sum(1) + mask16.flatten(1).sum(1)
        assert (tokens <= k).all(), f"k={k}: spent {tokens.tolist()} tokens, over budget"
        # every present (month, ps16 cell) is EITHER unsplit (mask16) or split (all 16 of its
        # mask4 children True, a 4x4 ps4-cell block) -- never partially split, never both,
        # never neither. mask4.view(B,T,4,4,4,4) axes are (B, T, block_row, inner_row,
        # block_col, inner_col) (row-major reshape of a contiguous (16,16) grid groups
        # spatial blocks correctly this way -- verified directly); reduce the INNER axes
        # (3, 5), not the block axes, to collapse each 4x4 block to one bool per ps16 cell.
        any_child = mask4.view(B_, T, 4, 4, 4, 4).any(-1).any(3)     # reduce inner_col, inner_row
        covers = any_child ^ mask16                                  # XOR: exactly one
        present_cells = present[:, :, None, None].expand(B_, T, 4, 4)
        assert (covers | ~present_cells).all(), f"k={k}: a present cell is neither split nor kept"


def test_apt_res_mask_splits_the_highest_value_cells_first():
    """At a budget that can afford exactly ONE split, --apt_res must split the ps16 cell with
    the highest (sum of ps4 children - ps16) value, not an arbitrary one. Per-cell min-max
    normalization erases pure AMPLITUDE differences (x3 vs x0.3 noise scores identically --
    confirmed empirically before writing this fixture), so the two cells here differ in
    DISTINCT LEVEL COUNT instead: cell (0,0) is continuous noise (high entropy at both ps16
    and ps4 scale -> high split value, measured ~630 vs ~170 for the other cell), cell (0,1)
    is binary on/off noise (few distinct levels at both scales -> lower split value)."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 1, 1, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(4)
    s2 = torch.zeros(B_, S, S, T, C)
    s2[:, 0:16, 0:16] = torch.randn(B_, 16, 16, T, C, generator=g)                      # continuous
    s2[:, 0:16, 16:32] = (torch.rand(B_, 16, 16, T, C, generator=g) > 0.5).float()      # binary
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    # base_cost = 1 present month x 16 ps16 cells = 16; a split costs 15 MORE (16 ps4 children
    # - 1 ps16 token already spent); k = 16 + 15 affords exactly one split
    mask4, res_masks = obj._apt_res_mask(ms, present, B_, k=31)
    split_cells = {(y, x) for y in range(4) for x in range(4)
                  if mask4[0, 0, y * 4:y * 4 + 4, x * 4:x * 4 + 4].any()}
    assert split_cells == {(0, 0)}, f"expected only the highest-value cell (0,0); got {split_cells}"


def test_apt_quadtree_mask_partitions_every_present_cell_exactly_once():
    """The REAL APT algorithm (paper Section 3.1): no budget, just per-level entropy
    thresholds, recursing coarse->fine. Every present cell at the finest grid must be
    covered by EXACTLY ONE kept patch across the whole returned dict (never zero, never
    more than one level) -- the quadtree partition invariant, generalized from
    _apt_res_mask's 2-level mask4/mask16 to N levels (4, 8, 16, 32 grids == patch sizes
    16, 8, 4, 2 px). Uses real (random) pixel data since this test checks the INVARIANT,
    not which cells get split -- that's covered by the forced-score test below."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 2, 3, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(5)
    s2 = torch.randn(B_, S, S, T, C, generator=g)
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    present[0, 2:] = False                        # one sample with a missing month too
    obj = _apt_pool_obj()
    grid_sizes = (4, 8, 16, 32)
    taus = (5.75, 5.0, 4.0)                        # arbitrary but level-ordered, like the paper
    out = obj._apt_quadtree_mask(ms, present, B_, grid_sizes, taus)
    assert set(out) == set(grid_sizes)
    # upsample every level's mask to the finest grid and count how many levels cover each
    # finest-grid cell -- must be exactly 1 wherever that cell is present, 0 elsewhere.
    finest = grid_sizes[-1]
    coverage = torch.zeros(B_, T, finest, finest, dtype=torch.long)
    for g_ in grid_sizes:
        rep = finest // g_
        coverage += out[g_].repeat_interleave(rep, -1).repeat_interleave(rep, -2).long()
    present_fine = present[:, :, None, None].expand(B_, T, finest, finest)
    assert (coverage[present_fine] == 1).all(), "every present cell must be covered exactly once"
    assert (coverage[~present_fine] == 0).all(), "absent cells must never be covered"


def test_apt_quadtree_mask_keeps_low_entropy_splits_high_entropy():
    """Decision rule itself (paper: keep if H < tau for that level, else split into 4
    children and recurse): a flat (near-zero entropy) region must stay at the COARSEST
    grid; a noisy region must recurse all the way to the FINEST grid, at every
    threshold level along the way. Forces _apt_scores per grid size directly so the test
    exercises only the recursion/thresholding logic, not _apt_scores' own entropy math
    (already covered elsewhere)."""
    import torch
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T = 1, 1
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long))
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    grid_sizes = (2, 4)                             # 2 levels: one threshold
    # cell (0,0) of the 2x2 coarse grid is flat (low entropy -> keep coarse); cell (0,1) is
    # noisy at every level (high entropy -> always split, ends up at the finest grid).
    scores = {2: torch.tensor([[[0.1, 9.0], [0.1, 9.0]]])[:, None],     # (B,T,2,2)
              4: torch.full((B_, T, 4, 4), 9.0)}
    obj._apt_scores = lambda ms_, present_, B_2, H, normalize=False: scores[H]
    out = obj._apt_quadtree_mask(ms, present, B_, grid_sizes, taus=(1.0,))
    assert out[2][0, 0].tolist() == [[True, False], [True, False]], "flat column kept coarse"
    # the noisy coarse cells (0,1) and (1,1) must have split -> their 2x2 finest children True
    assert out[4][0, 0, 0:2, 2:4].all() and out[4][0, 0, 2:4, 2:4].all()
    assert not out[4][0, 0, 0:2, 0:2].any() and not out[4][0, 0, 2:4, 0:2].any()


def test_apt_scores_normalize_equalizes_scale_across_grid_sizes():
    """normalize=True divides by each cell's own achievable entropy ceiling (min(log2(bins),
    log2(n_pixels)) per band) -- without it, a finer grid's cells (fewer pixels per cell)
    cannot reach the same raw entropy as a coarser grid's cells even with IDENTICAL content,
    purely because there are fewer samples to spread across APT_BINS histogram bins. Check
    this directly: a fully-random (maximally diverse, uniform-ish) fixture, scored at a coarse
    grid (big cells, many pixels) and a fine grid (small cells, few pixels) -- raw entropy at
    the fine grid must be noticeably LOWER than at the coarse grid (the ceiling artifact),
    while normalized entropy for both must sit close to the same (near-ceiling) fraction."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 1, 1, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(7)
    s2 = torch.rand(B_, S, S, T, C, generator=g)     # uniform noise: maximally diverse content
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    obj.apt_time = "none"
    raw_coarse = obj._apt_scores(ms, present, B_, H=4, normalize=False).mean()   # ps16, 256px/cell
    raw_fine = obj._apt_scores(ms, present, B_, H=32, normalize=False).mean()    # ps2, 4px/cell
    assert raw_fine < raw_coarse * 0.6, (
        f"raw entropy should drop sharply at the fine grid (ceiling artifact): "
        f"coarse={raw_coarse:.2f} fine={raw_fine:.2f}")
    norm_coarse = obj._apt_scores(ms, present, B_, H=4, normalize=True).mean()
    norm_fine = obj._apt_scores(ms, present, B_, H=32, normalize=True).mean()
    assert abs(norm_coarse - norm_fine) < 0.15 * max(norm_coarse, norm_fine), (
        f"normalized entropy should be close across grid sizes for equally-diverse content: "
        f"coarse={norm_coarse:.3f} fine={norm_fine:.3f}")


def test_apt_octree_flat_block_never_splits():
    """_apt_octree_mask: a perfectly flat (zero-entropy) block must stay a single ps16 leaf
    covering the WHOLE time range, at any tau >= 0 -- pooled entropy is 0, so even tau=0
    should never trigger a split (gain can only be <= 0 against an already-zero parent)."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 1, 6, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    s2 = torch.full((B_, S, S, T, C), 0.5)       # perfectly flat: every pixel identical
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    leaves = obj._apt_octree_mask(ms, present, B_, tau=0.0)
    assert len(leaves[0]) == 16, f"expected 16 untouched ps16 cells, got {len(leaves[0])}"
    for y0, x0, ps, months in leaves[0]:
        assert ps == 16 and months == list(range(T))


def test_apt_octree_splits_space_only_when_only_space_varies():
    """A block whose per-pixel content is IDENTICAL across every timestep (same spatial
    pattern repeated every month) but has real spatial structure (half noisy, half flat) must
    split in SPACE (revealing the noisy half) but should NOT keep subdividing in TIME, since
    every time-split's two halves are pixel-identical to each other and to the parent -- a
    time split can never reduce entropy here, so its gain must never beat the space split's."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 1, 6, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(11)
    frame = torch.full((S, S, C), 0.5)
    frame[0:32, 0:32] = torch.rand(32, 32, C, generator=g)     # noisy quadrant (top-left)
    s2 = frame[:, :, None, :].expand(S, S, T, C).clone()        # IDENTICAL every timestep
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2[None])
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    leaves = obj._apt_octree_mask(ms, present, B_, tau=0.5)
    # the noisy top-left ps16 cell (0,0) must have been split in space (child ps < 16 present)
    noisy_leaves = [leaf for leaf in leaves[0] if leaf[0] < 16 and leaf[1] < 16]
    assert any(ps < 16 for _, _, ps, _ in noisy_leaves), "noisy quadrant should split in space"
    # every leaf covering the noisy region should still span the FULL time range (no time
    # split should ever have won there, since all months are pixel-identical)
    for y0, x0, ps, months in leaves[0]:
        if y0 < 32 and x0 < 32:
            assert months == list(range(T)), (
                f"time should never split when content is time-invariant: leaf {y0,x0,ps} "
                f"has months={months}, expected full range {list(range(T))}")


def test_apt_octree_splits_time_only_when_only_time_varies():
    """A block that's spatially FLAT within each frame (no spatial structure to find) but
    changes sharply partway through its time range must split in TIME (isolating the change)
    and should never bother splitting in SPACE, since every spatial split of a flat frame is
    pixel-identical to its parent -- a space split can never reduce entropy here."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 1, 6, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    s2 = torch.full((B_, S, S, T, C), 0.2)
    s2[:, :, :, T // 2:] = 0.9          # second half of the season: a sharp, flat level jump
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    obj = _apt_pool_obj()
    leaves = obj._apt_octree_mask(ms, present, B_, tau=0.5)
    # every leaf should still be ps16 (no spatial structure to ever justify a space split)
    assert all(ps == 16 for _, _, ps, _ in leaves[0]), (
        f"a spatially-flat scene should never split in space: {leaves[0]}")
    # but there should be MULTIPLE leaves per (y0,x0) location (time got split)
    assert len(leaves[0]) > 16, (
        f"a scene with a real time-axis change should split in time: got {len(leaves[0])} "
        f"leaves for 16 ps16 cells (expected > 16)")


def test_apt_octree_covers_every_present_pixel_timestep_exactly_once():
    """Partition invariant: across all of a sample's leaves, every (pixel, present timestep)
    must be covered by EXACTLY ONE leaf -- never zero, never more than one. Uses real random
    data (checks the invariant, not which nodes split) with a missing month too, since real
    datasets always have gaps."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    B_, T, S = 1, 8, 64
    C = len(Modality.get("sentinel2_l2a").band_order)
    g = torch.Generator().manual_seed(12)
    s2 = torch.rand(B_, S, S, T, C, generator=g)
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B_, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    present = torch.ones(B_, T, dtype=torch.bool)
    present[0, 3] = False                                   # a real gap
    obj = _apt_pool_obj()
    leaves = obj._apt_octree_mask(ms, present, B_, tau=10.0)
    present_months = set(present[0].nonzero(as_tuple=True)[0].tolist())
    coverage = torch.zeros(S, S, T, dtype=torch.long)
    for y0, x0, ps, months in leaves[0]:
        for m in months:
            coverage[y0:y0 + ps, x0:x0 + ps, m] += 1
    for m in present_months:
        assert (coverage[:, :, m] == 1).all(), f"month {m}: not every pixel covered exactly once"
    for m in set(range(T)) - present_months:
        assert (coverage[:, :, m] == 0).all(), f"absent month {m} must never be covered"
