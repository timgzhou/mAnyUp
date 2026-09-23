"""cosmse_map must be Cosine_MSE, just unreduced.

timAnyUp supervises its query mask on cosmse_map and picks the transform head's fit locations
by it, while the reconstruction objective uses Cosine_MSE. If the two ever compute different
quantities, the mask learns to predict an error the loss is not actually minimizing -- a silent
failure. These tests make that loud instead.

    source env_setup/env_olmo.sh
    python -m pytest tests/test_cosmse_map.py -q
"""
import sys
import torch

sys.path.insert(0, "/scratch/timz/rs-change-detection/third_party/anyup")
from anyup.loss import Cosine_MSE, cosmse_map    # noqa: E402


def _pair(b=2, c=768, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, c, h, w, generator=g), torch.randn(b, c, h, w, generator=g))


def test_mean_matches_cosine_mse():
    """The headline invariant: mean over locations == the scalar loss."""
    pred, target = _pair()
    assert torch.allclose(cosmse_map(pred, target).mean(),
                          Cosine_MSE()(pred, target)["total"], atol=1e-5)


def test_shape_is_per_location():
    pred, target = _pair(b=3, c=64, h=4, w=8)
    assert cosmse_map(pred, target).shape == (3, 4, 8)


def test_identical_inputs_are_near_zero():
    """cos term -> 0 and mse term -> 0 when pred == target, at every location."""
    _, target = _pair()
    assert cosmse_map(target, target).abs().max() < 1e-5


def test_locates_a_planted_error():
    """A corrupted location must score worse than its neighbours -- the property the query mask
    actually depends on. A mean-reduced loss cannot express this."""
    pred, target = _pair()
    pred = target.clone()
    pred[0, :, 5, 7] = torch.randn(pred.shape[1])
    err = cosmse_map(pred, target)
    assert err[0, 5, 7] > 10 * err[err > 0].median()


def test_matches_across_shapes_and_seeds():
    for seed, (b, c, h, w) in enumerate([(1, 32, 4, 4), (4, 768, 16, 16), (2, 128, 8, 3)]):
        pred, target = _pair(b, c, h, w, seed=seed)
        assert torch.allclose(cosmse_map(pred, target).mean(),
                              Cosine_MSE()(pred, target)["total"], atol=1e-5), (b, c, h, w)
