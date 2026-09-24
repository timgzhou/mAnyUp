"""Compare feature upsamplers on one PASTIS sample: raw OlmoEarth LR features -> UPA -> AnyUp.

Extracted from UPA-vs-ANYUP.ipynb. Three ways to get a 64x64xD feature map out of a coarse
(gH,gW,D) OlmoEarth feature map, all guided by the same S2 RGB image:

  1. raw       -- the low-res features, no upsampling (shown at native gHxgW)
  2. UPA       -- per-image test-time optimized joint bilateral upsampler. Fits an anisotropic
                  pixelwise JBU to reconstruct the HR RGB from its own bicubic downsample
                  (self-supervised, never sees the features), then applies the learned kernels
                  to the feature map.
  3. UPMA      -- "upsample many things": identical machinery to UPA, but guided by the full
                  multispectral S2 stack instead of RGB. NIR/SWIR carry crop-boundary structure
                  RGB misses. Bands are percentile-stretched per channel so none dominates the
                  L1 target, and the bilateral range term averages (not sums) over channels so
                  sigma_r keeps the same meaning at any band count.
  4. AnyUp     -- pretrained feature upsampler pulled from torch.hub (wimmerth/anyup), frozen.

Optionally a panel with the native per-pixel (patch size 1) features as a reference. The last
panel is the PASTIS ground-truth semantic map, so you can judge whether the structure each
upsampler invents actually lines up with the real parcel boundaries.

All feature panels are colored with ONE shared PCA basis, fit on the raw low-res map, so color
differences between panels are real and not per-panel recoloring.

Note the methods are fed differently normalized guidance: UPA gets the percentile-stretched
uint8 display RGB, UPMA the percentile-stretched multispectral stack, AnyUp min-max +
ImageNet-normalized float RGB (what its pretrained weights expect). That is a confound if you
read the figure as a pure architecture comparison. UPA vs UPMA *is* a clean ablation, though --
same kernel and optimizer, only the guidance band count differs.

    source env_setup/env_olmo.sh        # AnyUp pulls torch.hub; UPA/UPMA need cuda
    python exp/upsamplers/upa_anyup.py
    python exp/upsamplers/upa_anyup.py --img_idx 3 --guide_bands all
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")        # headless cluster: never try an interactive backend (it hangs)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.optim.lr_scheduler import LambdaLR
from exp.common.paths import FEATURES

RGB_BANDS = [3, 2, 1]        # B04/B03/B02 = R,G,B in the 13-band stack
# Surface-structure bands for UPMA guidance: B02-B08A + B11/B12. Drops B01 (aerosol, 60m),
# B09 (water vapour) and B10 (cirrus) -- atmospheric bands that carry no crop-parcel structure,
# and which a per-channel stretch would otherwise amplify to noise and give equal weight in
# the range term.
SURFACE_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
DISPLAY_PX = 128             # all feature panels rendered at this size (nearest, keeps blocks)

# PASTIS class names + colormap (same 20-class scheme as visualize_olmoearth_pastis.py).
CLASSES = [
    'background', 'meadow', 'soft_winter_wheat', 'corn', 'winter_barley',
    'winter_rapeseed', 'spring_barley', 'sunflower', 'grapevine', 'beet',
    'winter_triticale', 'winter_durum_wheat', 'fruits_vegetables_flowers',
    'potatoes', 'leguminous_fodder', 'soybeans', 'orchard', 'mixed_cereal',
    'sorghum', 'void_label',
]
CMAP = plt.get_cmap('tab20', 20)
IGNORE_INDEX = -1            # targets.pt stores ignore as -1; remapped to 19 (void) for display

USE_AMP = True
AMP_DTYPE = torch.float16

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TIME_POOLS = ("mean", "median")   # how a (T,...) stack is collapsed to a single frame


def time_pool(x: torch.Tensor, mode: str = "mean", dim: int = 0) -> torch.Tensor:
    """Collapse the time axis of a (T,...) stack to a single frame.

    Guidance for UPA/UPMA/AnyUp/mAnyUp is a single image, so the PASTIS time series has to be
    reduced first. "mean" is the historical default. "median" is the robust alternative: PASTIS
    S2 series carry undetected cloud, haze and shadow frames, and a mean drags those bright/dark
    outliers into the composite, softening exactly the parcel boundaries the guided upsamplers
    key on. A per-pixel median rejects them as long as they are the minority over T.

    Note this only changes the GUIDANCE composite -- the feature-side reduction stays whatever
    the head does, so a change here is attributable to guidance quality alone.
    """
    if mode == "mean":
        return x.mean(dim)
    if mode == "median":
        return x.median(dim).values
    raise ValueError(f"unknown time_pool {mode!r}, expected one of {TIME_POOLS}")


# ---------------------------------------------------------------------------
# UPA / UPMA: per-image optimized anisotropic joint bilateral upsampling
# ---------------------------------------------------------------------------
def percentile_stretch(x: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0) -> np.ndarray:
    """(C,H,W) -> per-channel percentile stretch into [0,1].

    Robust alternative to per-channel min-max: satellite bands routinely have a handful of
    specular / edge-artifact pixels that would otherwise set the whole range and crush the
    real dynamic range into a sliver. Matches what raw_rgb() already does for display, so the
    3-band and 13-band guidance paths differ only in band count.
    """
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, lo_p, axis=(1, 2), keepdims=True)
    hi = np.percentile(x, hi_p, axis=(1, 2), keepdims=True)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def _fit_jbu(hr, lr_modality, fit_steps: int):
    """Fit the pixelwise JBU to reconstruct `hr` from its own bicubic downsample, then apply
    the learned kernels to `lr_modality`. hr: [1,Cg,H,W] float in [0,1] on cuda."""
    H, W = hr.shape[-2:]
    Hl, Wl = lr_modality.shape[-2:]
    scale = int(H / Hl)
    lr = F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False)

    model = LearnablePixelwiseAnisoJBU_NoParent(Hl, Wl, scale=scale).cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    # NOTE: the decay schedule is laid out for a 5100-step budget but the loop breaks at
    # fit_steps (50 in the notebook), so the lr has only decayed to ~8e-2 by then.
    max_steps = 5100
    gamma = (1e-9 / 1e-1) ** (1.0 / max_steps)
    scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for step in range(max_steps + 1):
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            pred = model(lr, hr)
            loss = F.l1_loss(pred, hr)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        scheduler.step()

        if step == fit_steps:
            break

    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
        hr_feat = model(lr_modality, hr)
    return hr_feat


def UPA(HR_img, lr_modaliry, fit_steps: int = 50):
    """Upsample Anything: RGB guidance.

    HR_img: (H,W,3) uint8 array. lr_modaliry: [1,C,Hl,Wl] cuda. -> [1,C,H,W].
    """
    hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    return _fit_jbu(hr, lr_modaliry, fit_steps)


def UPMA(guide, lr_modality, fit_steps: int = 50):
    """Upsample Many things: full multispectral guidance.

    Identical machinery to UPA -- same kernel, same parametrization, same optimizer -- the
    only difference is that the guide carries every S2 band instead of just RGB, so both the
    reconstruction target and the bilateral range term see the full spectrum. The extra bands
    (NIR/SWIR especially) carry crop-boundary structure that RGB misses, which is exactly the
    signal the anisotropic kernels are trying to find.

    guide: (Cg,H,W) float array, ALREADY percentile-stretched to [0,1] (no /255 here -- that
    would re-divide an already-normalized guide and collapse the range term).
    lr_modality: [1,C,Hl,Wl] cuda. -> [1,C,H,W].
    """
    hr = torch.from_numpy(np.asarray(guide, dtype=np.float32)).unsqueeze(0).cuda()
    return _fit_jbu(hr, lr_modality, fit_steps)


@torch.no_grad()
def _build_offsets(R_max: int, device: torch.device):
    """Return flattened neighbor offsets within square radius R_max."""
    offs = torch.arange(-R_max, R_max + 1, device=device)
    dY, dX = torch.meshgrid(offs, offs, indexing='ij')
    return dY.reshape(-1), dX.reshape(-1)  # [K], [K]


def _tanh_bound_pi(raw: torch.Tensor):
    """Map R -> (-pi, pi) smoothly."""
    return math.pi * torch.tanh(raw)


def gather_lr_scalar_general(map_lr: torch.Tensor, Ui: torch.Tensor, Vi: torch.Tensor):
    """
    map_lr: [1,1,Hl,Wl] or [Hl,Wl] or [H,W]
    Ui, Vi: [Bn, Hh, Wh] integer indices
    return: [Bn, Hh, Wh] gathered values
    """
    Hl, Wl = map_lr.shape[-2:]
    flat = Hl * Wl
    idx = (Ui * Wl + Vi).reshape(-1)
    t = map_lr.view(flat)
    vals = t.index_select(0, idx)
    return vals.view(Ui.shape[0], Ui.shape[1], Ui.shape[2])


def gs_jbu_aniso_noparent(
    feat_lr: torch.Tensor,     # [1,C,Hl,Wl], float16/float32 (cuda)
    guide_hr: torch.Tensor,    # [1,Cg,Hh,Wh], float32 (cuda) -- Cg=3 for RGB, 13 for full S2
    scale: int,
    sigma_x_map: torch.Tensor, # [1,1,Hl,Wl], float32
    sigma_y_map: torch.Tensor, # [1,1,Hl,Wl], float32
    theta_map: torch.Tensor,   # [1,1,Hl,Wl], float32 (radians)
    sigma_r_map: torch.Tensor, # [1,1,Hl,Wl], float32
    R_max: int = 4,
    alpha_dyn: float = 2.0,
    C_chunk: int = 512,
    Nn_chunk: int = 81,
    center_mode: str = "nearest",  # "nearest" | "floor"
    use_autocast: bool = True,
):
    _, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape

    dev = feat_lr.device
    dtype_feat = feat_lr.dtype
    dtype_acc = torch.float32

    # HR grid
    y = torch.arange(Hh, device=dev, dtype=torch.float32)
    x = torch.arange(Wh, device=dev, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')  # [Hh,Wh]

    u = (Y + 0.5) / scale - 0.5
    v = (X + 0.5) / scale - 0.5

    if center_mode == "nearest":
        uc = torch.round(u).clamp(0, Hl - 1).to(torch.long)
        vc = torch.round(v).clamp(0, Wl - 1).to(torch.long)
    elif center_mode == "floor":
        uc = torch.floor(u).clamp(0, Hl - 1).to(torch.long)
        vc = torch.floor(v).clamp(0, Wl - 1).to(torch.long)
    else:
        raise ValueError("center_mode must be 'nearest' or 'floor'")

    # dynamic radius from (upsampled) sigma_e = max(sx,sy)
    sigma_eff = torch.maximum(sigma_x_map, sigma_y_map)  # [1,1,Hl,Wl]
    sigma_eff_hr = F.interpolate(sigma_eff, (Hh, Wh), mode='bilinear', align_corners=False)
    R_map = torch.ceil(alpha_dyn * sigma_eff_hr).clamp_(min=1, max=R_max).to(torch.int64)  # [1,1,Hh,Wh]

    dY_all, dX_all = _build_offsets(R_max, dev)
    K = dY_all.numel()

    num_s = torch.zeros(C, Hh, Wh, device=dev, dtype=dtype_acc)
    den_s = torch.zeros(   Hh, Wh, device=dev, dtype=dtype_acc)
    m     = torch.full((Hh, Wh), float("-inf"), device=dev, dtype=dtype_acc)

    guide32 = guide_hr.to(torch.float32, copy=False)
    sx_map32 = sigma_x_map.to(torch.float32, copy=False)
    sy_map32 = sigma_y_map.to(torch.float32, copy=False)
    th_map32 = theta_map.to(torch.float32, copy=False)
    sr_map32 = sigma_r_map.to(torch.float32, copy=False)

    flat = Hl * Wl
    feat_flat = feat_lr[0].permute(1, 2, 0).reshape(flat, C).contiguous()

    guide_lr = F.interpolate(guide32, size=(Hl, Wl), mode='bilinear', align_corners=False)  # [1,Cg,Hl,Wl]

    autocast_ctx = torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16)

    with autocast_ctx:
        # streaming log-sum-exp over neighbor chunks: the running max `m` lets us tile the
        # K offsets without overflowing exp(), and gives the same answer as one big pass.
        for n0 in range(0, K, Nn_chunk):
            n1 = min(n0 + Nn_chunk, K)
            dY = dY_all[n0:n1].view(-1, 1, 1)  # [Bn,1,1]
            dX = dX_all[n0:n1].view(-1, 1, 1)
            Bn = dY.shape[0]

            Ui = torch.clamp(uc.unsqueeze(0) + dY, 0, Hl - 1)
            Vi = torch.clamp(vc.unsqueeze(0) + dX, 0, Wl - 1)

            rad2 = (dY ** 2 + dX ** 2)
            mask = (rad2 <= (R_map ** 2)).squeeze(0).squeeze(0)  # [Bn,Hh,Wh]

            cy = (Ui.to(torch.float32) + 0.5) * scale - 0.5
            cx = (Vi.to(torch.float32) + 0.5) * scale - 0.5
            dx = X.unsqueeze(0) - cx
            dy = Y.unsqueeze(0) - cy

            # each contributing LR pixel brings its OWN (sx,sy,th,sr) -- the params are gathered
            # at the neighbor's coords, not the output pixel's.
            sx = gather_lr_scalar_general(sx_map32, Ui, Vi).clamp_min(1e-6)  # [Bn,Hh,Wh]
            sy = gather_lr_scalar_general(sy_map32, Ui, Vi).clamp_min(1e-6)
            th = gather_lr_scalar_general(th_map32, Ui, Vi)
            sr = gather_lr_scalar_general(sr_map32, Ui, Vi).clamp_min(1e-6)

            cos_t, sin_t = torch.cos(th), torch.sin(th)
            x_p = dx * cos_t + dy * sin_t
            y_p = -dx * sin_t + dy * cos_t
            log_ws = -(x_p ** 2) / (2 * sx ** 2 + 1e-8) - (y_p ** 2) / (2 * sy ** 2 + 1e-8)  # [Bn,Hh,Wh]

            # range term over ALL guide channels, MEAN not sum: keeps diff2 (and hence the
            # meaning of sigma_r) on the same scale whether the guide is 3-band RGB or 13-band
            # S2, so learned sigma_r is comparable across guidance configs.
            Cg = guide32.shape[1]
            diff2 = torch.zeros_like(log_ws)
            for cg in range(Cg):
                g_c = gather_lr_scalar_general(guide_lr[0, cg, ...], Ui, Vi)  # [Bn,Hh,Wh]
                diff2 = diff2 + (guide32[0, cg] - g_c) ** 2
            diff2 = diff2 / Cg
            log_wr = -diff2 / (2.0 * sr * sr + 1e-8)

            log_w = log_ws + log_wr
            log_w = torch.where(mask, log_w, torch.full_like(log_w, float("-inf")))

            m_chunk = torch.max(log_w, dim=0).values            # [Hh,Wh]
            valid = torch.isfinite(m_chunk)
            if not valid.any():
                continue

            m_new = m.clone()
            m_new[valid] = torch.maximum(m[valid], m_chunk[valid])

            delta = (m - m_new).clamp_max(0)
            scale_old = torch.ones_like(den_s)
            scale_old[valid] = torch.exp(delta[valid])
            den_s.mul_(scale_old)
            num_s.mul_(scale_old.unsqueeze(0))

            log_w_shift = log_w - m_new.unsqueeze(0)
            log_w_shift[:, ~valid] = float("-inf")
            s = torch.exp(log_w_shift)  # [Bn,Hh,Wh]

            den_s.add_(s.sum(0))

            idx_flat = (Ui * Wl + Vi).reshape(-1)
            for c0 in range(0, C, C_chunk):
                c1 = min(c0 + C_chunk, C)
                feat_sel = feat_flat.index_select(0, idx_flat)[:, c0:c1]  # [Bn*Hh*Wh, Cc]
                feat_sel = feat_sel.view(Bn, Hh, Wh, c1 - c0)             # [Bn,Hh,Wh,Cc]
                num_s[c0:c1].add_((feat_sel * s[..., None]).sum(dim=0).permute(2, 0, 1))

            m = m_new

    out_raw = (num_s / den_s.clamp_min(1e-8)).unsqueeze(0).to(dtype_feat)  # [1,C,Hh,Wh]
    fallback = F.interpolate(feat_lr, size=(Hh, Wh), mode='bilinear', align_corners=False)
    tiny = (den_s < 1e-6).unsqueeze(0).unsqueeze(0)
    out = torch.where(tiny, fallback, out_raw)
    return out


class LearnablePixelwiseAnisoJBU_NoParent(nn.Module):
    """One anisotropic Gaussian per LR pixel (4 params: sigma_x, sigma_y, theta, sigma_r).

    "NoParent" = no coarse-to-fine hierarchy; every LR pixel's params are independent. The
    params live on the LR grid and are gathered at the CONTRIBUTING pixel, so the HR output is
    a normalized blend of the LR pixels whose learned influence fields reach it.
    """

    def __init__(
        self,
        Hl: int,
        Wl: int,
        scale: int = 16,
        init_sigma: float = 16.0,
        init_sigma_r: float = 0.12,
        R_max: int = 8,
        alpha_dyn: float = 2.0,
        center_mode: str = "nearest",
        eval_C_chunk: int = 128,
        eval_Nn_chunk: int = 49,
        use_autocast: bool = True,
    ):
        super().__init__()
        self.scale = int(scale)
        self.R_max = int(R_max)
        self.alpha_dyn = float(alpha_dyn)
        self.center_mode = center_mode
        self.eval_C_chunk = int(eval_C_chunk)
        self.eval_Nn_chunk = int(eval_Nn_chunk)
        self.use_autocast = bool(use_autocast)

        self.sx_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma)),   dtype=torch.float32))
        self.sy_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma)),   dtype=torch.float32))
        self.th_raw = nn.Parameter(torch.zeros( (1, 1, Hl, Wl),                             dtype=torch.float32))
        self.sr_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma_r)), dtype=torch.float32))

    def forward(self, feat_lr: torch.Tensor, guide_hr: torch.Tensor):
        """
        feat_lr:  [1, C, Hl, Wl]  (same LR spatial grid that parameters live on)
        guide_hr: [1, Cg, Hh, Wh] (Hh = Hl * scale, Wh = Wl * scale; Cg = any channel count)
        return:   [1, C, Hh, Wh]
        """
        sigma_x = torch.exp(self.sx_raw)                 # [1,1,Hl,Wl] > 0
        sigma_y = torch.exp(self.sy_raw)                 # [1,1,Hl,Wl] > 0
        theta   = _tanh_bound_pi(self.th_raw)            # [1,1,Hl,Wl] in (-pi, pi)
        sigma_r = torch.exp(self.sr_raw)                 # [1,1,Hl,Wl] > 0

        C = int(feat_lr.shape[1])
        K = int((2 * self.R_max + 1) * (2 * self.R_max + 1))
        # training runs on a 3-channel RGB target, so it fits without tiling; eval on D=768
        # features needs the chunking to stay in VRAM (chunking does not change the result).
        if self.training:
            C_chunk = C
            Nn_chunk = K
        else:
            C_chunk = min(self.eval_C_chunk, C)
            Nn_chunk = min(self.eval_Nn_chunk, K)

        return gs_jbu_aniso_noparent(
            feat_lr, guide_hr, self.scale,
            sigma_x, sigma_y, theta, sigma_r,
            R_max=self.R_max, alpha_dyn=self.alpha_dyn,
            C_chunk=C_chunk, Nn_chunk=Nn_chunk,
            center_mode=self.center_mode,
            use_autocast=self.use_autocast,
        )


# ---------------------------------------------------------------------------
# display helpers (PCA / raw RGB), matching visualize_features.py conventions
# ---------------------------------------------------------------------------
def raw_rgb(s2_t: torch.Tensor) -> np.ndarray:
    """(13,H,W) one timestep -> (H,W,3) float in [0,1], percentile-stretched."""
    rgb = s2_t[RGB_BANDS].float().numpy().transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, 2, (0, 1)), np.percentile(rgb, 98, (0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def pca_rgb_shared(fit: torch.Tensor, apply_to):
    """Fit top-3 PCA + min-max on `fit` (H,W,D); apply that SAME basis to every map in
    `apply_to`. Shared basis => identical feature vectors get identical colors, so color
    differences across panels are real (not per-panel recoloring)."""
    Hf, Wf, D = fit.shape
    xf = fit.reshape(Hf * Wf, D).float().numpy()
    mean = xf.mean(0, keepdims=True)
    cov = ((xf - mean).T @ (xf - mean)) / max(Hf * Wf - 1, 1)
    _, evecs = np.linalg.eigh(cov)
    dirs = evecs[:, -3:][:, ::-1]
    proj_fit = (xf - mean) @ dirs
    lo, hi = proj_fit.min(0), proj_fit.max(0)
    out = []
    for m in apply_to:
        H, W, _ = m.shape
        x = m.reshape(H * W, D).float().numpy()
        proj = ((x - mean) @ dirs).reshape(H, W, 3)
        out.append(np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1))
    return out


def nearest_resize(img: np.ndarray, size: int = DISPLAY_PX) -> np.ndarray:
    H, W = img.shape[:2]
    yi = (np.arange(size) * H // size).clip(0, H - 1)
    xi = (np.arange(size) * W // size).clip(0, W - 1)
    return img[yi][:, xi]


def _norm_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """(3,H,W) -> per-channel min-max then ImageNet standardization (what AnyUp expects)."""
    flat = rgb.reshape(*rgb.shape[:-3], 3, -1)
    lo = flat.amin(-1).reshape(*rgb.shape[:-3], 3, 1, 1)
    hi = flat.amax(-1).reshape(*rgb.shape[:-3], 3, 1, 1)
    return ((rgb - lo) / (hi - lo + 1e-6) - _IMAGENET_MEAN) / _IMAGENET_STD


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # exp/upsamplers/upa_anyup.py -> repo root is parents[2]; .parent pointed at
    # exp/upsamplers/ and made data_splits resolve to a nonexistent path.
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2],
                   help="repo root (holds exp/ and data/)")
    p.add_argument("--features_root", type=Path,
                   default=FEATURES)
    p.add_argument("--lr_dir", default="oe_base_s2_ps4_tile64",
                   help="low-res feature dir to upsample (16x16x768 -> scale 4 up to 64x64)")
    p.add_argument("--ref_dir", default="oe_base_s2_ps1_tile32",
                   help="native per-pixel features for the optional reference panel")
    p.add_argument("--split", default="test")
    p.add_argument("--img_idx", type=int, default=1)
    p.add_argument("--fit_steps", type=int, default=50,
                   help="UPA/UPMA test-time optimization steps")
    p.add_argument("--guide_bands", default="surface",
                   help="UPMA guidance bands: 'surface' (B02-B08A,B11,B12), 'all' (13 bands), "
                        "or a comma-separated index list e.g. '3,2,1,7'")
    p.add_argument("--out", type=Path, default=Path("results/pastis/feature_viz/upa_anyup.png"))
    args = p.parse_args()

    repo = args.repo
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()   # before any olmoearth_pretrain.evals import

    data_splits = repo / "data" / "pastis_olmoearth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load the low-res OlmoEarth features + guidance for this one sample ----
    # .mean(0) collapses the stored TIME axis: (T,gH,gW,D) -> (gH,gW,D). The coarse gHxgW grid
    # comes from OlmoEarth's patch size, not from spatial pooling.
    lr = torch.load(args.features_root / args.lr_dir / f"pastis_r_{args.split}" /
                    f"{args.img_idx}.pt").float().mean(0)                       # (gH,gW,D)
    gH, gW, D = lr.shape
    s2 = torch.load(data_splits / f"pastis_r_{args.split}" / "s2_images" /
                    f"{args.img_idx}.pt")                                       # (T,13,64,64)
    rgb_disp = raw_rgb(s2.float().mean(0))                                      # (64,64,3) in [0,1]
    out_hw = (s2.shape[-2], s2.shape[-1])                                       # (64,64)
    print(f"[load] lr {tuple(lr.shape)}  s2 {tuple(s2.shape)}  scale={out_hw[0] // gH}")

    # ---- ground truth: one (N,64,64) int64 tensor for the whole split, indexed by img_idx ----
    targets = torch.load(data_splits / f"pastis_r_{args.split}" / "targets.pt")
    gt = targets[args.img_idx].numpy()                                          # (64,64), -1..18
    gt_disp = np.where(gt == IGNORE_INDEX, 19, gt)                              # ignore -> void_label
    print(f"[gt] classes present: "
          f"{[CLASSES[c] for c in np.unique(gt_disp).astype(int) if 0 <= c <= 19]}")

    # ============ method 1: raw low-res features (no upsampling) ============
    lr_hw = lr                                                                  # (gH,gW,D)

    # ============ method 2: UPA (per-image optimized JBU) ============
    # guidance here is the percentile-stretched DISPLAY rgb, round-tripped through uint8
    # (UPA divides by 255 internally) -- deliberately different from AnyUp's normalization.
    hr_img_255 = (rgb_disp * 255).astype(np.uint8)                              # HxWx3 uint8
    lr_feat_in = lr.permute(2, 0, 1).unsqueeze(0).to(device)                    # (1,D,gH,gW)
    upa_hr = UPA(hr_img_255, lr_feat_in, fit_steps=args.fit_steps)              # (1,D,64,64)
    upa_hw = upa_hr.squeeze(0).permute(1, 2, 0).float().cpu()                   # (64,64,D)
    print(f"[UPA] fit {args.fit_steps} steps -> {tuple(upa_hw.shape)}")

    # ============ method 3: UPMA (same JBU, full multispectral guidance) ============
    if args.guide_bands == "surface":
        guide_bands = SURFACE_BANDS
    elif args.guide_bands == "all":
        guide_bands = list(range(s2.shape[1]))
    else:
        guide_bands = [int(b) for b in args.guide_bands.split(",")]
    # time-mean first (same as the RGB path), then per-band percentile stretch to [0,1] so no
    # single band dominates the L1 reconstruction target or the range term.
    guide_ms = percentile_stretch(s2.float().mean(0)[guide_bands].numpy())      # (Cg,64,64)
    upma_hr = UPMA(guide_ms, lr_feat_in, fit_steps=args.fit_steps)              # (1,D,64,64)
    upma_hw = upma_hr.squeeze(0).permute(1, 2, 0).float().cpu()                 # (64,64,D)
    print(f"[UPMA] bands={guide_bands} ({len(guide_bands)}ch) fit {args.fit_steps} steps "
          f"-> {tuple(upma_hw.shape)}")

    # ============ method 3: pretrained AnyUp (torch.hub, frozen) ============
    s2_full = torch.load(data_splits / f"pastis_r_{args.split}" / "s2_images" /
                         f"{args.img_idx}.pt").float()
    rgb_guide = _norm_rgb(s2_full.mean(0)[RGB_BANDS])                           # (3,64,64)
    anyup = torch.hub.load("wimmerth/anyup", "anyup_multi_backbone",
                           use_natten=False, pretrained=True).to(device).eval()
    with torch.no_grad():
        anyup_hr = anyup(rgb_guide.unsqueeze(0).to(device),
                         lr.permute(2, 0, 1).unsqueeze(0).to(device),
                         output_size=out_hw)                                    # (1,D,64,64)
    anyup_hw = anyup_hr.squeeze(0).permute(1, 2, 0).float().cpu()               # (64,64,D)
    print(f"[AnyUp] -> {tuple(anyup_hw.shape)}")

    # ============ optional native reference (ps1 tile64) ============
    ref_hw = None
    ref_path = args.features_root / args.ref_dir / f"pastis_r_{args.split}" / f"{args.img_idx}.pt"
    if ref_path.exists():
        cand = torch.load(ref_path).float().mean(0)                             # (64,64,D)
        if cand.shape[-1] == D:
            ref_hw = cand
        else:
            print(f"ref D={cand.shape[-1]} != {D}; skipping reference panel")
    else:
        print(f"no reference at {ref_path}; skipping reference panel")

    # ---- color every feature panel with ONE shared PCA basis (fit on the low-res map) ----
    maps = [lr_hw, upa_hw, upma_hw, anyup_hw] + ([ref_hw] if ref_hw is not None else [])
    colored = pca_rgb_shared(lr_hw, maps)

    # panels carry a kind flag: "rgb" panels are plain images, "label" needs the categorical
    # colormap with fixed vmin/vmax so colors mean the same class across every figure.
    panels = [(rgb_disp, "raw RGB (mean T)", "rgb"),
              (nearest_resize(colored[0], DISPLAY_PX), f"OlmoEarth raw\n({gH}x{gW}x{D})", "rgb"),
              (nearest_resize(colored[1], DISPLAY_PX), f"UPA (RGB guide)\n-> 64x64x{D}", "rgb"),
              (nearest_resize(colored[2], DISPLAY_PX),
               f"UPMA ({len(guide_bands)}-band guide)\n-> 64x64x{D}", "rgb"),
              (nearest_resize(colored[3], DISPLAY_PX), f"AnyUp -> 64x64x{D}", "rgb")]
    if ref_hw is not None:
        panels.append((nearest_resize(colored[4], DISPLAY_PX),
                       f"{args.ref_dir}\n(native 64x64x{D})", "rgb"))
    # GT last, at native 64x64 -- nearest_resize would be lossless here but imshow's
    # interpolation="nearest" already keeps the label blocks crisp.
    panels.append((gt_disp, "ground truth\n(PASTIS semantic)", "label"))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.8))
    for ax, (img, ttl, kind) in zip(axes, panels):
        if kind == "label":
            ax.imshow(img, cmap=CMAP, vmin=0, vmax=19, interpolation="nearest")
        else:
            ax.imshow(img)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    present = [i for i in np.unique(gt_disp).astype(int) if 0 <= i <= 19]
    handles = [mpatches.Patch(color=CMAP(i), label=f"{i}: {CLASSES[i]}") for i in present]
    fig.legend(handles=handles, bbox_to_anchor=(1.005, 0.98), loc="upper left", fontsize=7)

    fig.suptitle(f"{args.lr_dir} -- {args.split} #{args.img_idx} | shared PCA basis "
                 f"(fit on raw features)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

# python -u exp/upsamplers/upa_anyup.py