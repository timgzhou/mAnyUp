"""JBU: per-image test-time-optimized anisotropic joint bilateral upsampling (UPA / UPMA).

For every (feature map, guidance image) pair, a pixelwise anisotropic JBU (4 parameters per
low-res pixel: sigma_x, sigma_y, theta, sigma_r) is fitted to reconstruct the guidance from
its own bicubic downsample -- it never sees the features -- and the fitted kernels are then
applied to the feature map. Outputs are normalized convex combinations of the input features
("input" space). UPA and UPMA are the same machinery with different guidance:

    JBU.upa()   natural-colour RGB          ("Upsample Anything")
    JBU.upma()  the 13-band S2 stack        ("Upsample Many things")

both percentile-stretched per band (2-98 %). The legacy UPA also round-tripped its guide
through uint8 (an artefact of its RGB-image API); that quantization is not reproduced.
Needs CUDA (the kernel runs under fp16 autocast).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from manyup.data.guidance import LEGACY_S2_13, S2_RGB, GuidanceSpec
from manyup.upsamplers.base import FrameUpsampler

USE_AMP = True
AMP_DTYPE = torch.float16


def fit_jbu(hr: torch.Tensor, lr_modality: torch.Tensor, fit_steps: int) -> torch.Tensor:
    """Fit the pixelwise JBU to reconstruct `hr` from its own bicubic downsample, then apply
    the learned kernels to `lr_modality`. hr: [1,Cg,H,W] float in [0,1]; lr_modality:
    [1,C,Hl,Wl]; both on the same CUDA device. -> [1,C,H,W]."""
    H, W = hr.shape[-2:]
    Hl, Wl = lr_modality.shape[-2:]
    scale = int(H / Hl)
    lr = F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False)

    model = LearnablePixelwiseAnisoJBU_NoParent(Hl, Wl, scale=scale).to(hr.device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    # NOTE: the decay schedule is laid out for a 5100-step budget but the loop breaks at
    # fit_steps (50 by default), so the lr has only decayed to ~8e-2 by then.
    max_steps = 5100
    gamma = (1e-9 / 1e-1) ** (1.0 / max_steps)
    scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    with torch.enable_grad():                    # the fit is an optimization even at eval
        for step in range(max_steps + 1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
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
    with torch.no_grad(), torch.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        return model(lr_modality, hr)


class JBU(FrameUpsampler):
    def __init__(self, guidance: GuidanceSpec, fit_steps: int = 50, name: str = "jbu",
                 time_matching: str | None = None):
        """guidance: any spec; UPA/UPMA use norm="percentile". time_matching defaults to
        pooled for a pooled spec, else nearest."""
        super().__init__()
        tm = "pooled" if guidance.time_pool else (time_matching or "nearest")
        self._declare(f"{name}[{guidance.key}]", guidance, tm, "input")
        self.fit_steps = fit_steps

    @classmethod
    def upa(cls, time_pool: str | None = "mean", modality: str = "S2", **kw) -> "JBU":
        return cls(GuidanceSpec({modality: S2_RGB}, time_pool, norm="percentile"),
                   name="upa", **kw)

    @classmethod
    def upma(cls, time_pool: str | None = "mean", bands: dict | None = None, **kw) -> "JBU":
        return cls(GuidanceSpec(bands or {"S2": LEGACY_S2_13}, time_pool, norm="percentile"),
                   name="upma", **kw)

    def upsample_frames(self, feats: torch.Tensor, guide: torch.Tensor,
                        size: tuple[int, int]) -> torch.Tensor:
        if tuple(guide.shape[-2:]) != tuple(size):
            raise ValueError(f"{self.key} outputs at the guidance resolution "
                             f"{tuple(guide.shape[-2:])}, not {tuple(size)}")
        # one fit per image: batching would fit one kernel field to a batch-mean guide
        return torch.cat([fit_jbu(guide[n:n + 1].float(), feats[n:n + 1], self.fit_steps)
                          for n in range(len(feats))]).float()


# ---------------------------------------------------------------------------------------
# the kernel (moved unchanged from exp/upsamplers/upa_anyup.py)
# ---------------------------------------------------------------------------------------
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

    autocast_ctx = torch.autocast("cuda", enabled=use_autocast, dtype=torch.float16)

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
