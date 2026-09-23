"""Unit tests for the timAnyUp building blocks.

These pin the properties the training loop silently depends on: exact top-k budgets, the
blend leaving unselected locations untouched, the rank transform being a true per-sample
ordering, and gradient reaching the heads that should learn (and NOT the ones that shouldn't).

    source env_setup/env_olmo.sh && python tests/test_timanyup.py
"""
import sys
import torch

sys.path.insert(0, "/scratch/timz/rs-change-detection/third_party/anyup")
from anyup.timAnyUp import (TimAnyUp, QueryHead, TransformHead, topk_mask,   # noqa: E402
                            random_mask, blend, rank_normalize)

B, T, C, H, W, h, w = 2, 12, 32, 16, 16, 4, 4


def test_query_head_shape_and_temporal_coupling():
    qh = QueryHead(C, T)
    x = torch.randn(B, T, C, H, W)
    out = qh(x)
    assert out.shape == (B, T, H, W)
    # A T*C->T head must let one timestep's features change ANOTHER timestep's score --
    # that temporal coupling is the whole reason we chose it over C->1.
    x2 = x.clone(); x2[:, 0] += 5.0
    d = (qh(x2) - qh(x)).abs()
    assert d[:, 1:].max() > 1e-6, "no temporal coupling: head is behaving like per-timestep C->1"


def test_topk_selects_exactly_k_and_the_worst():
    scores = torch.randn(B, T, H, W)
    for k in (1, 512, T * H * W):
        m = topk_mask(scores, k)
        assert m.sum().item() == B * k, k
    m = topk_mask(scores, 512)
    # per-SAMPLE: comparing across the batch is meaningless, samples have different scales
    for b in range(B):
        assert scores[b][m[b]].min() >= scores[b][~m[b]].max(), \
            f"sample {b}: selected locations must all outrank rejected"


def test_topk_k_larger_than_grid_is_clamped():
    m = topk_mask(torch.randn(B, T, H, W), 99999)
    assert m.all(), "k > n should select everything, not error"


def test_topk_budget_is_flat_over_time():
    """k is a per-SAMPLE budget: the model may spend it all on one timestep."""
    scores = torch.full((1, T, H, W), -10.0)
    scores[0, 3] = 10.0                       # timestep 3 is uniformly the worst
    m = topk_mask(scores, H * W)
    assert m[0, 3].all() and m.sum() == H * W, "budget did not concentrate on one timestep"


def test_random_mask_matches_budget():
    m = random_mask(torch.randn(B, T, H, W), 512)
    assert m.sum().item() == B * 512
    assert m.reshape(B, -1).sum(1).tolist() == [512, 512]


def test_blend_touches_only_selected_locations():
    up = torch.randn(B, T, C, H, W)
    tr = torch.randn(B, T, C, H, W)
    m = topk_mask(torch.randn(B, T, H, W), 512)
    out = blend(up, tr, m)
    mm = m.unsqueeze(2).expand_as(up)
    assert torch.equal(out[~mm], up[~mm]), "unselected locations must be untouched"
    assert torch.allclose(out[mm], (0.5 * (up + tr))[mm]), "selected must be the channel-wise mean"


def test_rank_normalize_is_a_per_sample_ordering():
    err = torch.randn(B, T, H, W)
    r = rank_normalize(err)
    assert r.min() >= 0 and r.max() <= 1
    for b in range(B):
        assert torch.equal(err[b].flatten().argsort(), r[b].flatten().argsort()), "order changed"
        assert abs(r[b].max() - 1.0) < 1e-6 and abs(r[b].min()) < 1e-6, "not per-sample scaled"
    # per-sample: a 100x larger error in one sample must not shift the other's ranks
    err2 = err.clone(); err2[0] *= 100
    assert torch.allclose(rank_normalize(err2)[1], r[1])


def test_query_is_independent_of_upsampler_by_default():
    """The point of query_input='bilinear': the mask must not depend on mAnyUp's output."""
    m = TimAnyUp(input_dim=13, qk_dim=32, feat_dim=C, num_frames=T)
    lrhc = torch.randn(B, T, C, h, w)
    q1 = m.query(lrhc, (H, W))
    q2 = m.query(lrhc, (H, W), lrhc_up=torch.randn(B, T, C, H, W))
    assert torch.equal(q1, q2), "bilinear query must ignore lrhc_up entirely"


def test_forward_pieces_run_and_shapes_line_up():
    m = TimAnyUp(input_dim=13, qk_dim=32, feat_dim=C, num_frames=T)
    lrhc = torch.randn(B, T, C, h, w)
    hrlc = torch.randn(B, T, C, H, W)
    guide = torch.rand(B, T, 13, 64, 64)
    up = m.upsample(lrhc, guide, (H, W))
    assert up.shape == (B, T, C, H, W)
    assert m.query(lrhc, (H, W)).shape == (B, T, H, W)
    assert m.transform(hrlc).shape == (B, T, C, H, W)


def test_gradient_reaches_upsampler_and_transform_but_not_query_via_blend():
    """Gradient structure is the design: the query head learns ONLY from its own selection
    loss, because topk is non-differentiable in the indices. If a future change accidentally
    makes the blend differentiable w.r.t. the mask, this test fails and says so."""
    m = TimAnyUp(input_dim=13, qk_dim=32, feat_dim=C, num_frames=T)
    lrhc = torch.randn(B, T, C, h, w)
    hrlc = torch.randn(B, T, C, H, W)
    guide = torch.rand(B, T, 13, 64, 64)
    up = m.upsample(lrhc, guide, (H, W))
    tr = m.transform(hrlc)
    scores = m.query(lrhc, (H, W))
    fused = blend(up, tr, topk_mask(scores, 256))
    fused.sum().backward()
    assert m.upsampler.transform is not None
    g_up = [p.grad for p in m.upsampler.parameters() if p.grad is not None]
    g_tr = [p.grad for p in m.transform_head.parameters() if p.grad is not None]
    g_q = [p.grad for p in m.query_head.parameters() if p.grad is not None]
    assert g_up, "upsampler must receive gradient from the reconstruction path"
    assert g_tr, "transform head must receive gradient from the reconstruction path"
    assert not g_q, "query head must NOT receive gradient through the blend"


def test_projector_is_always_present_and_linear():
    """The projector is unconditional (not a flag): we reweight low-res tokens, so the fused
    map cannot be expected to land natively in the target's feature space -- we require only
    that it LINEARLY predicts it."""
    m = TimAnyUp(input_dim=13, qk_dim=32, feat_dim=C, num_frames=T)
    assert hasattr(m, "projector"), "projector must always exist"
    x = torch.randn(B, T, C, H, W)
    assert m.project(x).shape == x.shape
    # pointwise linear: additive in its input up to the bias
    a, b = torch.randn(B, T, C, H, W), torch.randn(B, T, C, H, W)
    lhs = m.project(a + b) - m.project(torch.zeros_like(a))
    rhs = (m.project(a) - m.project(torch.zeros_like(a))) + \
          (m.project(b) - m.project(torch.zeros_like(b)))
    assert torch.allclose(lhs, rhs, atol=1e-4), "projector must be linear"
    # and pointwise: location (0,0) must not depend on any other location
    a2 = a.clone(); a2[..., 1:, :] += 10.0
    assert torch.allclose(m.project(a)[..., 0, :], m.project(a2)[..., 0, :], atol=1e-5)


def test_projector_receives_gradient():
    m = TimAnyUp(input_dim=13, qk_dim=32, feat_dim=C, num_frames=T)
    m.project(torch.randn(B, T, C, H, W)).sum().backward()
    assert any(p.grad is not None for p in m.projector.parameters())


def test_query_loss_never_reaches_the_upsampler():
    """query_input='upsampled' lets the head READ mAnyUp's output, but its loss must not
    RESHAPE it. Without the detach, L_query pushes the upsampler toward being easy to predict
    error from -- competing with reconstruction. Measured cost when this leaked: upsample-only
    0.1629 vs 0.1542, and ~0.010 lower mIoU at every k including k=0, where no lookup happens.
    Both query_input modes are checked so a future refactor cannot reintroduce it in either."""
    for qi in ("upsampled", "bilinear"):
        m = TimAnyUp(input_dim=13, qk_dim=32, feat_dim=C, num_frames=T, query_input=qi)
        lrhc = torch.randn(B, T, C, h, w)
        guide = torch.rand(B, T, 13, 64, 64)
        up = m.upsample(lrhc, guide, (H, W))
        m.query(lrhc, (H, W), lrhc_up=up).sum().backward()
        assert any(p.grad is not None for p in m.query_head.parameters()), \
            f"{qi}: query head must receive its own loss"
        assert not any(p.grad is not None for p in m.upsampler.parameters()), \
            f"{qi}: query loss leaked into the upsampler"


def test_query_input_upsampled_reads_the_upsampled_map():
    """The detach must cut the GRADIENT, not the signal: scores still depend on lrhc_up."""
    m = TimAnyUp(input_dim=13, qk_dim=32, feat_dim=C, num_frames=T, query_input="upsampled")
    lrhc = torch.randn(B, T, C, h, w)
    a = torch.randn(B, T, C, H, W)
    s1 = m.query(lrhc, (H, W), lrhc_up=a)
    s2 = m.query(lrhc, (H, W), lrhc_up=a + 5.0)
    assert not torch.allclose(s1, s2), "scores must still respond to the upsampled map"


if __name__ == "__main__":
    torch.manual_seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  PASS {f.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
