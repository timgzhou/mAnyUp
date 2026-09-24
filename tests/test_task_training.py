"""Pieces of the task trainers (exp.olmoearth.train, exp.utae.train) that are easy to get
silently wrong: the heads' time reductions and pixel layout, the meters' arithmetic, and the
UTAE packing of non-PASTIS data."""
import torch

from exp.common.heads import Pa2PxHead, time_mean
from exp.common.metrics import RegMeter, SegMeter
from exp.utae.inputs import UTAEInput
from manyup.data import ModalityBatch, MTMMBatch


def test_time_mean_ignores_missing_steps():
    tok = torch.stack([torch.full((2, 2, 3), v) for v in (1.0, 100.0, 3.0)]).unsqueeze(0)
    out = time_mean(tok, torch.tensor([[True, False, True]]))
    assert torch.all(out == 2.0)


def test_pa2px_unfolds_each_token_to_its_own_patch():
    head = Pa2PxHead("time_mean", embed_dim=2, num_outputs=1, patch_size=2)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))   # sub-pixel (i, j) = 2i + j
        head.linear.weight[:, 0] = 10.0                                 # + 10 * token value
    tok = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).view(1, 1, 2, 2, 1).repeat(1, 1, 1, 1, 2)
    tok[..., 1] = 0
    out = head(tok, torch.ones(1, 1, dtype=torch.bool))[0, 0]          # (4, 4)
    for gh in range(2):
        for gw in range(2):
            patch = out[2 * gh:2 * gh + 2, 2 * gw:2 * gw + 2]
            assert torch.equal(patch, 10 * tok[0, 0, gh, gw, 0] + torch.tensor([[0., 1.], [2., 3.]]))


def test_diff_head_is_post_minus_pre():
    head = Pa2PxHead("diff", embed_dim=3, num_outputs=2, patch_size=1)
    pre, post = torch.randn(1, 1, 4, 4, 3), torch.randn(1, 1, 4, 4, 3)
    out = head(torch.cat([pre, post], 1), torch.ones(1, 2, dtype=torch.bool))
    want = head(torch.cat([torch.zeros_like(pre), post - pre], 1), torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(out, want, atol=1e-6)


def test_seg_meter_headline_and_ignore():
    m = SegMeter(3, ("a", "b", "c"), headline=(0, 2))
    target = torch.tensor([[[0, 1, 2, -1]]])
    logits = torch.nn.functional.one_hot(torch.tensor([[[0, 2, 2, 1]]]), 3).permute(0, 3, 1, 2).float()
    m.update(logits, target)
    r = m.compute()
    assert abs(r["iou_a"] - 1) < 1e-6 and r["iou_b"] == 0.0 and abs(r["iou_c"] - 0.5) < 1e-6
    assert abs(r["primary"] - 0.75) < 1e-6          # ignored pixel (-1 -> predicted 1) not counted


def test_reg_meter_reports_in_target_units():
    m = RegMeter(scale=10.0)
    m.update(torch.tensor([[[[1.0, 3.0]]]]), torch.tensor([[[0.0, 1.0]]]))   # errors 1, 2
    r = m.compute()
    assert abs(r["rmse"] - 10 * (2.5 ** 0.5)) < 1e-4 and abs(r["mae"] - 15.0) < 1e-4


def test_utae_input_custom_norm_and_per_sample_reference():
    data = torch.ones(2, 2, 1, 4, 4) * 3
    valid = torch.tensor([[True, True], [True, False]])
    t0 = torch.tensor([[100, 112], [500, -1]])
    mb = ModalityBatch(data=data, bands=("x",), t_start=t0, t_end=t0, time_precision="day",
                       frame_valid=valid, pixel_valid=None, geometry=[None, None], aux={},
                       history=())
    batch = MTMMBatch(ids=["a", "b"], split="train", modalities={"M": mb}, target=None,
                      target_geometry=[None, None], meta=[{}, {}])
    x, d = UTAEInput(["M"], "train", norm={"M": ((1.0,), (2.0,))}, reference_day=None)(batch)
    assert torch.all(x["M"][valid] == 1.0) and torch.all(x["M"][~valid] == 0)
    assert d["M"].tolist() == [[0, 12], [0, 0]]
