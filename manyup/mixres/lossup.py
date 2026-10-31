"""LossUp: predict the per-pixel label loss of the model's crop-free pass, per season map (and
for the pooled prediction), from the model's own low-resolution prediction, optionally guided by
the high-resolution input (mAnyUp) -- the quantity the label oracles select by, made label-free.

Per map (a season's head map, or the pooled prediction):
    stats   the map's per-pixel log-probabilities, entropy and top-2 margin (64 x 64)
    tokens  the season token grid (the pooled map: the mean over used seasons), (D, gh, gw)
    guide   the high-resolution composite of the map's months (the pooled map: all months)
mode
    stats     head on the stats alone
    bilinear  + the tokens, bilinearly upsampled
    manyup    + the tokens upsampled by mAnyUp, guided by the composite
-> log(1 + loss) per pixel.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ANYUP_REPO = Path(__file__).resolve().parents[2] / "third_party" / "anyup"   # = exp.common.paths

MODES = ("stats", "bilinear", "manyup")


class LossUp(nn.Module):
    def __init__(self, dim: int, n_cls: int, guide_ch: int, mode: str = "manyup",
                 hidden: int = 64, qk_dim: int = 64, window_ratio: float = 0.5):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"LossUp mode {mode!r}: {MODES}")
        self.mode = mode
        c_in = n_cls + 2
        if mode != "stats":
            self.tok = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
            c_in += hidden
        if mode == "manyup":
            if str(ANYUP_REPO) not in sys.path:
                sys.path.insert(0, str(ANYUP_REPO))
            from manyup.mAnyUp import mAnyUp
            self.up = mAnyUp(input_dim=guide_ch, qk_dim=qk_dim, feat_dim=hidden,
                             window_ratio=window_ratio)
        self.head = nn.Sequential(nn.Conv2d(c_in, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(hidden, 1, 1))

    @staticmethod
    def stats(logits):
        lp = logits.float().log_softmax(1)
        p = lp.exp()
        ent = -(p * lp).sum(1, keepdim=True)
        top2 = p.topk(2, dim=1).values
        return torch.cat([lp, ent, top2[:, :1] - top2[:, 1:]], 1)

    def forward(self, logits, tokens=None, guide=None):
        """logits (N, n_cls, H, W), tokens (N, D, gh, gw), guide (N, C, H, W) -> (N, H, W)."""
        x = [self.stats(logits)]
        if self.mode != "stats":
            t = self.tok(tokens.float().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if self.mode == "bilinear":
                t = F.interpolate(t, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            else:
                t = self.up(guide.float(), t, output_size=logits.shape[-2:])
            x.append(t)
        return self.head(torch.cat(x, 1))[:, 0]
