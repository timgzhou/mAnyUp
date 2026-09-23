"""timAnyUp: temporal, adaptive feature upsampling with a queryable high-res side channel.

Turns two CHEAP feature maps into one EXPENSIVE one:

    F_lrhc  (T,C,h,w)   low-res, high-context    ps16 tile64, full time series
    F_hrlc  (T,C,H,W)   high-res, low-context    ps4  tile4, per-timestep (no cross-t attn)
    ------------------------------------------------------------------------------------
    F_hrhc  (T,C,H,W)   high-res, high-context   ps4  tile64  <- the TARGET

Three parts, only two of them new:

  1. mAnyUp (REUSED, unchanged) upsamples F_lrhc per timestep, guided by that timestep's
     image  ->  F_lrhc_up (T,C,H,W).
  2. query head (NEW) scores every (t,y,x) location for "how badly will the upsampler fail
     here". The top-k locations are looked up in F_hrlc and averaged into F_lrhc_up.
  3. transform head (NEW) maps F_hrlc into F_lrhc_up's feature space before that blend --
     the two arms are genuinely different spaces (measured relative gap ~1.28), so a raw
     average would mix incompatible representations.

The query head deliberately reads a BILINEAR upsample of F_lrhc, not mAnyUp's output, so the
mask does not depend on the upsampler: the HR lookups can be issued in parallel with (or
before) the upsample, and the head sees a stationary input distribution while mAnyUp is still
moving. `query_input="upsampled"` switches to the coupled variant for ablation.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .mAnyUp import mAnyUp


class QueryHead(nn.Module):
    """Score each (t, y, x) location: "how badly does the cheap upsample fail here?"

    Patch-wise `T*C -> T` linear: every location sees ALL its timesteps at once, so the head
    can notice that one timestep looks wrong while the others are fine -- the temporal
    signal a per-timestep `C -> 1` head is blind to. Deliberately NOT convolutional: at a 4x4
    F_lrhc grid a 3x3 kernel already spans half the tile, so spatial context would cost 9x
    the parameters for almost no new information.

    Outputs raw logits (B,T,H,W); the trainer applies sigmoid for the loss and ranks the raw
    scores for top-k (monotonic, so the sigmoid does not affect selection).
    """

    def __init__(self, feat_dim: int, num_frames: int):
        super().__init__()
        self.C, self.T = feat_dim, num_frames
        self.fc = nn.Linear(num_frames * feat_dim, num_frames)

    def forward(self, x):
        # x: (B,T,C,H,W) -> (B,H,W,T*C) -> (B,H,W,T) -> (B,T,H,W)
        B, T, C, H, W = x.shape
        assert (T, C) == (self.T, self.C), f"expected (T,C)=({self.T},{self.C}), got ({T},{C})"
        z = x.permute(0, 3, 4, 1, 2).reshape(B, H, W, T * C)
        return self.fc(z).permute(0, 3, 1, 2)


class TransformHead(nn.Module):
    """Map F_hrlc into F_lrhc_up's feature space. 2-layer pointwise (1x1) conv, C -> C -> C.

    Pointwise on purpose: this is a per-location change of basis between two feature spaces,
    not a spatial operation -- F_hrlc is already at the target resolution.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = None):
        super().__init__()
        hid = hidden_dim or feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hid, 1),
            nn.GELU(),
            nn.Conv2d(hid, feat_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class TimAnyUp(nn.Module):
    """The three stages plus a projector.

    The projector is a pointwise linear map (1x1 conv, C->C) applied to the FUSED map before
    the reconstruction loss. It is always on, not optional: the fused map is a reweighted
    average of low-res tokens plus a few looked-up F_hrlc vectors, so requiring it to EQUAL
    the target is ill-posed -- F_lrhc (ps16, high context) and F_hrhc (ps4) are different
    feature spaces. We only require that the fused map can LINEARLY PREDICT the target, which
    is also exactly what a downstream linear probe would ask of it. (train_manyup.py exposes
    the same idea as --proj_head; here it is unconditional.)
    """

    def __init__(self, input_dim=13, qk_dim=128, feat_dim=768, num_frames=12,
                 transform_depth=0, window_ratio=1.0, transform_hidden=None,
                 query_input="bilinear", **kwargs):
        super().__init__()
        if query_input not in ("bilinear", "upsampled"):
            raise ValueError(f"query_input must be bilinear|upsampled, got {query_input}")
        self.query_input = query_input
        self.feat_dim, self.num_frames = feat_dim, num_frames
        self.upsampler = mAnyUp(input_dim=input_dim, qk_dim=qk_dim, feat_dim=feat_dim,
                                transform_depth=transform_depth, window_ratio=window_ratio,
                                **kwargs)
        self.query_head = QueryHead(feat_dim, num_frames)
        self.transform_head = TransformHead(feat_dim, transform_hidden)
        # Pointwise linear projection into the target's feature space. See the class docstring:
        # always present, because we are reweighting low-res tokens and cannot expect the
        # result to land natively in the ps4 space.
        self.projector = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)

    # -- the three stages, exposed separately so the trainer can build its losses ----------
    def upsample(self, lrhc, guide, out_size):
        """F_lrhc (B,T,C,h,w) + guidance (B,T,G,gh,gw) -> F_lrhc_up (B,T,C,H,W).

        T is folded into the BATCH dim: the timesteps are upsampled independently, each guided
        by its own frame, in a single encoder call."""
        B, T, C, h, w = lrhc.shape
        feats = lrhc.reshape(B * T, C, h, w)
        img = guide.reshape(B * T, *guide.shape[2:])
        out = self.upsampler(img, feats, out_size)
        return out.reshape(B, T, C, *out_size)

    def query(self, lrhc, out_size, lrhc_up=None):
        """Score locations -> logits (B,T,H,W). Reads a bilinear upsample of F_lrhc by default
        (independent of the upsampler); `query_input="upsampled"` reads F_lrhc_up instead."""
        if self.query_input == "upsampled":
            if lrhc_up is None:
                raise ValueError("query_input='upsampled' needs lrhc_up")
            # DETACHED on purpose. The query head reads mAnyUp's output, so without this the
            # selection loss backpropagates INTO the upsampler and pushes it to make its own
            # output easy to predict error from -- an objective that competes with
            # reconstruction. Measured cost of the leak: upsample-only reconstruction 0.1629
            # vs 0.1542, and ~0.010 lower downstream mIoU at EVERY k including k=0 (where no
            # lookup happens at all, so only the damaged weights can explain it).
            # The head's INPUT is numerically unchanged; only the backward path is cut.
            # Same rule as the transform head's lrhc_up.detach() target in train_timanyup.
            x = lrhc_up.detach()
        else:
            B, T, C, h, w = lrhc.shape
            x = F.interpolate(lrhc.reshape(B * T, C, h, w).float(), size=out_size,
                              mode="bilinear", align_corners=False).reshape(B, T, C, *out_size)
        return self.query_head(x)

    def transform(self, hrlc):
        """F_hrlc (B,T,C,H,W) -> same shape, mapped into F_lrhc_up's space."""
        B, T, C, H, W = hrlc.shape
        return self.transform_head(hrlc.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)

    def project(self, fused):
        """Linear map (B,T,C,H,W) -> target feature space. Applied before the reconstruction
        loss ONLY -- the transform loss stays in F_lrhc_up space, so its anchor is unaffected."""
        B, T, C, H, W = fused.shape
        return self.projector(fused.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)


def topk_mask(scores, k):
    """Flat top-k over (T,H,W) per sample -> bool mask (B,T,H,W) with exactly k True each.

    Flat (not per-timestep) on purpose: k is a per-sample lookup BUDGET, and the model should
    be free to spend it unevenly across time -- a cloudy timestep deserves more lookups than a
    clear one.
    """
    B, T, H, W = scores.shape
    n = T * H * W
    k = min(k, n)
    flat = scores.reshape(B, n)
    idx = flat.topk(k, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.reshape(B, T, H, W)


def random_mask(scores, k, generator=None):
    """Control for topk_mask: k locations chosen uniformly at random, same shape/count.

    This is the control that matters. At k=512 of 3072 locations, injecting real HR features
    at 17% of positions helps even if chosen blindly, so beating bilinear proves nothing about
    the LEARNED mask -- only beating THIS does."""
    B, T, H, W = scores.shape
    n = T * H * W
    k = min(k, n)
    noise = torch.rand(B, n, device=scores.device, generator=generator)
    idx = noise.topk(k, dim=1).indices
    mask = torch.zeros(B, n, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, idx, True)
    return mask.reshape(B, T, H, W)


def blend(lrhc_up, hrlc_t, mask):
    """Channel-wise mean of the two arms at the selected locations, F_lrhc_up elsewhere.

    The mask is non-differentiable in its INDICES (topk), so no gradient flows to the query
    head from here -- it learns only through its own selection loss. Gradient does flow to the
    upsampler and the transform head through the blended VALUES.
    """
    m = mask.unsqueeze(2).to(lrhc_up.dtype)          # (B,T,1,H,W) broadcast over C
    return lrhc_up * (1 - m) + 0.5 * (lrhc_up + hrlc_t) * m


def rank_normalize(err):
    """Per-sample rank transform of a (B,T,H,W) error map -> [0,1], worst location = 1.

    The query head is supervised on RELATIVE ordering, not error magnitude: top-k only needs
    to know which locations are worst, and regressing raw cosmse would make the loss chase
    per-sample error scale (a nuisance that selection never uses).
    """
    B = err.shape[0]
    flat = err.reshape(B, -1).float()
    n = flat.shape[1]
    order = flat.argsort(dim=1)
    ranks = torch.empty_like(flat)
    ar = torch.arange(n, device=err.device, dtype=flat.dtype).expand(B, n)
    ranks.scatter_(1, order, ar)
    return (ranks / max(n - 1, 1)).reshape(err.shape)
