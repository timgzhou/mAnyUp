"""Feature upsamplers on a fixed model: AnyUp (wimmerth/anyup, pretrained on RGB encoders) and
Upsample Anything (per-image test-time fit, upstream's code path in manyup.upsamplers.jbu), with
bilinear as the control.

    tokens (B, S, h, w, D)   the fixed model's final tokens, seasons or months
    upsample                 each used step alone, guided by that step's own RGB composite, to
                             image resolution -> (B, S, D, P, P), as both papers do
    readout                  per pixel: a = score(f) + log months; out = sum_s softmax_s(a) head(f)

The head is a new per-pixel linear map, both papers' protocol; the score is the fixed model's
readout score. Guidance: per-frame 2-98 percentile stretch, not AnyUp's usual min-max (PASTIS's
-10000 nodata frames would set the range). Steps with no present month are not upsampled.
Upsample Anything's kernels depend on the guidance alone, so its output is a convex combination
of the tokens, like bilinear's: it can take a bilinear run's head and fit at test time only.

Upsample Anything's fit is elementwise arithmetic the MAC tracer does not see: fit_macs.
"""
import torch
import torch.nn.functional as F
from torch import nn

from exp.bench.flop_trace import add_macs
from manyup.data.guidance import S2_RGB, GuidanceSpec, percentile_per_image
from manyup.mixres.baselines.base import BaselineModel, rgb_frames
from manyup.upsamplers import JBU, AnyUp

FIT_STEPS = 50              # upstream: updates 0 .. 50 of its loop
R_DISC = 197                # offsets within radius R_max = 8 (sigma starts at 16: all in reach)
# multiplications per (HR pixel, offset) of one kernel evaluation: rotation 4, spatial 4, range
# one per guide channel + 1; the backward pass about as many again
PAIR_MACS = 2 * (4 + 4 + 3 + 1)


def fit_macs(frames: int, H: int, D: int) -> float:
    """Upsample Anything on `frames` H x H frames: FIT_STEPS + 1 fit passes on the RGB guide
    (+ 3 output channels, each forward + backward), one render of D channels (one pass)."""
    px = frames * H * H * R_DISC
    return px * ((FIT_STEPS + 1) * (PAIR_MACS + 2 * 3) + PAIR_MACS // 2 + D)


class Bilinear(nn.Module):
    def upsample_frames(self, feats, guide, size):
        return F.interpolate(feats, size=size, mode="bilinear", align_corners=False)


def build_upsampler(name: str) -> nn.Module:
    if name == "bilinear":
        return Bilinear()
    if name == "anyup":
        return AnyUp(None)
    return JBU(GuidanceSpec({"S2": S2_RGB}, None, norm="percentile"), FIT_STEPS, "upa",
               range_sum=True)


class UpsampleModel(BaselineModel):
    new_modules = ("upsampler.", "pixel_head.")

    def __init__(self, task, cfg, pretrained: bool = True):
        super().__init__(task, cfg, pretrained)
        self.upsampler = build_upsampler(cfg.baseline.upsampler)
        self.pixel_head = nn.Linear(self.mix.encoder.embedding_size, task.num_outputs)

    def _inputs(self, ms, ss, season, present):
        ms, ss, season, present = super()._inputs(ms, ss, season, present)
        self._guide = rgb_frames(self.task, ss, self.cfg.lr_patch)
        return ms, ss, season, present

    def decode(self, lr_F, hr_F, plans, season, present, logw=None):
        if plans:
            raise ValueError("upsample baseline: no crops")
        B, S, h, w, D = lr_F.shape
        P = self._guide.shape[-1]
        months = (F.one_hot(season.clamp(min=0), S).float()
                  * (present & (season >= 0)).float()[..., None]).sum(1)         # (B, S)
        used = months > 0
        guide = percentile_per_image(self._guide[used])                          # (N, 3, P, P)
        feats = lr_F[used].permute(0, 3, 1, 2).float().contiguous()              # (N, D, h, w)
        if self.run_cfg.baseline.upsampler == "upa":
            add_macs(fit_macs(len(feats), P, D))
        up = self.upsampler.upsample_frames(feats, guide, (P, P)).permute(0, 2, 3, 1)
        a = self.readout._score(up).float() + months[used].log()[:, None, None]  # (N, P, P)
        full = a.new_full((B, S, P, P), float("-inf"))
        full[used] = a
        z = self.pixel_head(up.float())                                          # (N, P, P, C)
        wgt = full.softmax(1)[used]
        out = z.new_zeros(B, P, P, z.shape[-1])
        out.index_add_(0, used.nonzero()[:, 0], z * wgt[..., None])
        out = out.permute(0, 3, 1, 2)
        return out[..., self.lo:P - self.lo, self.lo:P - self.lo]
