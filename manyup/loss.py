# Adapted from JAFAR (https://github.com/PaulCouairon/JAFAR)
import torch
from torch import nn


class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

        gt = torch.ones_like(target[:, 0])

        min_val = torch.min(target, dim=1, keepdim=True).values
        max_val = torch.max(target, dim=1, keepdim=True).values
        pred_normalized = (pred - min_val) / (max_val - min_val + 1e-6)
        target_normalized = (target - min_val) / (max_val - min_val + 1e-6)

        loss = self.cosine_loss(pred, target, gt) + self.mse_loss(pred_normalized, target_normalized)
        return {"total": loss}


def cosmse_map(pred, target):
    """Per-location Cosine_MSE: (B,C,H,W) x2 -> (B,H,W), NO reduction.

    Same quantity Cosine_MSE computes, but kept per spatial location instead of averaged, so a
    caller can ask WHERE the prediction is bad (timAnyUp supervises its query mask on exactly
    this, and picks the transform head's fit locations by it).

    Mirrors Cosine_MSE term for term:
      - cosine part: 1 - cos(pred, target), matching CosineEmbeddingLoss with target=+1
      - mse part:    squared error on the min-max-over-CHANNEL normalized vectors, meaned over C
    so cosmse_map(p, t).mean() == Cosine_MSE()(p, t)["total"]. test_cosmse_map.py asserts that;
    if you touch Cosine_MSE, this must move with it or the two silently diverge.
    """
    b, c, h, w = pred.shape
    p = pred.permute(0, 2, 3, 1).reshape(-1, c)
    t = target.permute(0, 2, 3, 1).reshape(-1, c)

    cos = 1.0 - torch.nn.functional.cosine_similarity(p, t, dim=1)

    min_val = torch.min(t, dim=1, keepdim=True).values
    max_val = torch.max(t, dim=1, keepdim=True).values
    scale = max_val - min_val + 1e-6
    mse = (((p - min_val) / scale - (t - min_val) / scale) ** 2).mean(dim=1)

    return (cos + mse).view(b, h, w)
