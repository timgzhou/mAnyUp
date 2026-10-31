"""APT (Choudhury et al., ICLR 2026; rccchoudhury/apt), adaptive patches on every image of a
series: each month (or season composite) gets its own two-scale partition, and all its tokens
join one sequence per sample.

    entropy    upstream's: Shannon entropy (bits) of a coarse patch's grayscale in 512 bins of
               [0, 255]; bands stretched to [0, 255] by fixed per-dataset percentiles, the
               counterpart of an 8-bit image, so a flat tile stays flat (results/<task>/apt.json)
    partition  entropy < tau: one coarse token, else its 4 fine tokens. tau per coarse patch, the
               baseline.keep quantile of the training entropies (upstream hand-tunes one tau per
               model; ImageNet's 5.5 bits means nothing on 64 px reflectances)
    coarse     the patch resized and embedded (OlmoEarth's FlexiViT at the coarse patch), plus a
               zero-init map of its fine sub-patch embeddings, 2 x 2 conv-pooled (upstream's
               TokenizedZeroConvPatchAttn)
    positions  OlmoEarth's own sin-cos at the token's place, as a crop's
    dense      upstream repeats coarse tokens to the grid. Here every token writes its footprint
               through the fixed models' readout, its time attention included: the same dense
               output as repeating, with the model's own head
A month without every modality has no tokens (MixedResModel crops' rule).
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from manyup.encoders.olmoearth import _pad_space, _spatial_size, crop_patches
from manyup.inject import Packed
from manyup.mixres.baselines.base import BaselineModel, frame_side, rgb_frames
from manyup.mixres.model import Out
from manyup.mixres.plan import LOOKUP_PS, Plan
from manyup.mixres.selectors import Selection

BINS = 512                  # upstream's compute_patch_entropy_vectorized
LUMA = (0.2989, 0.5870, 0.1140)


def grayscale(rgb: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """(..., 3, H, W) + per-band stretch (3,) -> (..., H, W) in [0, 255]."""
    lo, hi = lo.to(rgb.device)[:, None, None], hi.to(rgb.device)[:, None, None]
    x = ((rgb - lo) / (hi - lo) * 255).clamp(0, 255)
    return (x * torch.tensor(LUMA, device=rgb.device)[:, None, None]).sum(-3)


def patch_entropy(gray: torch.Tensor, patch: int) -> torch.Tensor:
    """(N, H, W) in [0, 255] -> (N, H / patch, W / patch) entropy in bits."""
    N, H, W = gray.shape
    h, w = H // patch, W // patch
    bins = (gray / 255 * BINS).long().clamp(max=BINS - 1)
    bins = F.pixel_unshuffle(bins[:, None].float(), patch).long()            # (N, p^2, h, w)
    cell = torch.arange(N * h * w, device=gray.device).view(N, 1, h, w).expand_as(bins)
    hist = torch.bincount((cell * BINS + bins).flatten(), minlength=N * h * w * BINS)
    p = hist.view(N, h, w, BINS).float() / (patch * patch)
    return -torch.xlogy(p, p).sum(-1) / torch.log(torch.tensor(2.0))


def calibration(task_name: str) -> dict:
    return json.loads(Path(f"results/{task_name}/apt.json").read_text())


class AptSelector:
    """The partition as two crop groups. Reads pixels only."""
    scorer = None

    def __init__(self, task, cfg, model):
        c = calibration(task.name)
        self.task, self.patch, self.fine = task, cfg.model.lr_patch, cfg.select.crop_px
        self.months = cfg.baseline.tokens == "months"
        self.lo, self.hi = torch.tensor(c["lo"]), torch.tensor(c["hi"])
        key = f"{'months' if self.months else 'seasons'}_ps{self.patch}"
        keep = cfg.baseline.keep            # (1, 0: every patch coarse, fine)
        self.tau = (float("inf") if keep >= 1 else float("-inf") if keep <= 0
                    else c["tau"][key][f"{keep:g}"])
        self.frame_lo = model.lo

    def modules(self) -> list:
        return []

    def __call__(self, ctx) -> Selection:
        s = ctx.ms if self.months else ctx.ss
        valid = ctx.valid if self.months else ctx.used
        rgb = rgb_frames(self.task, s, self.patch)                       # (B, T, 3, P, P)
        B, T, _, P, _ = rgb.shape
        h = grayscale(rgb.flatten(0, 1), self.lo, self.hi)
        ent = patch_entropy(h, self.patch).view(B, T, P // self.patch, P // self.patch)
        coarse = (ent < self.tau) & valid[:, :, None, None]
        fine = (ent >= self.tau) & valid[:, :, None, None]
        fine = fine.repeat_interleave(2, 2).repeat_interleave(2, 3)
        plans = []
        for mask, p in ((coarse, self.patch), (fine, self.fine)):
            b, t, gy, gx = mask.nonzero(as_tuple=True)
            py, px = gy * p - self.frame_lo, gx * p - self.frame_lo      # tile pixels
            plans.append(Plan(b, t, (py // LOOKUP_PS).clamp(0, 15), (px // LOOKUP_PS).clamp(0, 15),
                              py, px, p))
        return Selection(plans, coarse)


class AptModel(BaselineModel):
    new_modules = ("patch_attn.", "zero_conv.")

    def __init__(self, task, cfg, pretrained: bool = True):
        super().__init__(task, cfg, pretrained)
        D = self.mix.encoder.embedding_size
        self.patch_attn = nn.Conv2d(D, D, 2, 2)
        self.zero_conv = nn.Linear(D, D)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        self.readout.no_base = True

    def _crops(self, src, p: Plan):
        """Plan p's windows of src on the padded frame (N, p x p single-frame samples)."""
        return crop_patches(src, p.b, p.t, p.py + self.lo, p.px + self.lo, p.patch, pixels=True)

    def _sub_embeddings(self, src, p: Plan):
        """Upstream's second path: each coarse patch's fine sub-patch embeddings, no encodings,
        conv-pooled -> (N, token slots, D)."""
        enc, f = self.mix.encoder, p.patch // 2
        sub = crop_patches(src, p.b.repeat_interleave(4), p.t.repeat_interleave(4),
                           (p.py + self.lo).repeat_interleave(4)
                           + torch.tensor([0, 0, f, f], device=p.b.device).repeat(len(p)),
                           (p.px + self.lo).repeat_interleave(4)
                           + torch.tensor([0, f, 0, f], device=p.b.device).repeat(len(p)),
                           f, pixels=True)
        tokens, masks, _ = enc.split_tokens_masks_and_dims(enc.patch_embeddings.forward(sub, f))
        x, _ = enc.collapse_and_combine_hwtc({**tokens, **masks})        # (4 N, slots, D)
        n, D = x.shape[1:]
        x = x.view(-1, 2, 2, n, D).permute(0, 3, 4, 1, 2).reshape(-1, D, 2, 2)
        return self.zero_conv(self.patch_attn(x)[..., 0, 0]).view(-1, n, D)

    def tokenize(self, src, p: Plan) -> Packed:
        tok = self.mix.embed(self._crops(src, p), p.patch, slot=p.t, py=p.py + self.lo,
                             px=p.px + self.lo)
        if p.patch == self.cfg.lr_patch:
            tok.x = tok.x + self._sub_embeddings(src, p).to(tok.x.dtype)[tok.keep]
        return tok

    def forward(self, ms, ss, season, present, plan, logw=None, dates=None, seasons=None,
                lr_enc=None) -> Out:
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        plans = [p for p in plan if len(p)]
        side = frame_side(ss, self.cfg.lr_patch)
        src = _pad_space(ss, side) if side != _spatial_size(ss) else ss
        B = season.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sets = [self.tokenize(src, p) for p in plans]
            seq = torch.cat([torch.repeat_interleave(p.b, t.seqlens) for p, t in zip(plans, sets)])
            x = torch.cat([self.mix._typed(t, self.mix.hr_type.dtype).to(sets[0].x.dtype)
                           for t in sets])
            order = torch.argsort(seq, stable=True)
            y = self.mix.blocks(x[order], torch.bincount(seq, minlength=B), 0,
                                len(self.mix.encoder.blocks))
            out = torch.empty_like(y)
            out[order] = y
            out = self.mix.encoder.norm(out)
            parts = torch.split(out, [len(t.x) for t in sets])
            hr_F = torch.cat([t.pooled(o).float()[:, 0, 0, 0] for t, o in zip(sets, parts)])
        g = side // self.cfg.lr_patch
        lr_F = hr_F.new_zeros(B, 1, g, g, hr_F.shape[-1])       # no season tokens (no_base)
        z = torch.zeros_like(season)
        return Out(self._decode(lr_F, hr_F, plans, z, present), lr_F, hr_F)
