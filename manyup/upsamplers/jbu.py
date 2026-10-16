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

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from manyup.data.guidance import LEGACY_S2_13, S2_RGB, GuidanceSpec, spec_to_dict
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


class KernelCache:
    """Fitted kernel fields by sample id: <dir>/meta.json + <dir>/<split>/shard_<i>of<n>.pt,
    each {"ids": [...], "raw": (n, T, 4, Hl, Wl) float32} (T = kernel fields per sample:
    1 when pooled, else one per feature step). A fit sees only the guidance, never the
    features, so one cache serves every encoder, feature cache and probe on this grid."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self._rows: dict[str, torch.Tensor] = {}
        for f in sorted(self.dir.glob("*/shard_*.pt")):
            part = torch.load(f, weights_only=True)
            self._rows.update(zip(part["ids"], part["raw"]))

    @staticmethod
    def exists(directory) -> bool:
        return (Path(directory) / "meta.json").exists()

    def covers(self, ids) -> bool:
        return all(i in self._rows for i in ids)

    def get(self, ids) -> torch.Tensor:
        return torch.stack([self._rows[i] for i in ids])              # (B, T, 4, Hl, Wl)


class JBU(FrameUpsampler):
    def __init__(self, guidance: GuidanceSpec, fit_steps: int = 50, name: str = "jbu",
                 time_matching: str | None = None, fit_batch: int = 128,
                 kernels: KernelCache | None = None):
        """guidance: any spec; UPA/UPMA use norm="percentile". time_matching defaults to
        pooled for a pooled spec, else nearest. kernels: fitted kernel fields to use instead
        of fitting (samples the cache lacks are fitted, fit_batch images at a time)."""
        super().__init__()
        tm = "pooled" if guidance.time_pool else (time_matching or "nearest")
        self._declare(f"{name}[{guidance.key}]", guidance, tm, "input")
        self.fit_steps, self.fit_batch, self.kernels = fit_steps, fit_batch, kernels
        self._ids = None

    def cache_meta(self, grid: tuple[int, int]) -> dict:
        """What a kernel cache for this upsampler must record (and match on reuse)."""
        return {"key": self.key, "guidance": spec_to_dict(self.guidance),
                "time_matching": self.time_matching, "fit_steps": self.fit_steps,
                "grid": list(grid)}

    def forward(self, lr, guidance, target, **side):
        self._ids = list(lr.ids)                  # upsample_frames looks kernels up by id
        try:
            return super().forward(lr, guidance, target, **side)
        finally:
            self._ids = None

    def fit(self, guide: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Kernel fields (N, 4, Hl, Wl) for guidance frames (N, Cg, H, W)."""
        return torch.cat([fit_kernels(guide[i:i + self.fit_batch].float(), grid,
                                      self.fit_steps)
                          for i in range(0, len(guide), self.fit_batch)])

    @classmethod
    def upa(cls, time_pool: str | None = "mean", rgb: dict | None = None, **kw) -> "JBU":
        return cls(GuidanceSpec(rgb or {"S2": S2_RGB}, time_pool, norm="percentile"),
                   name="upa", **kw)

    @classmethod
    def upma(cls, time_pool: str | None = "mean", bands: dict | None = None, **kw) -> "JBU":
        return cls(GuidanceSpec(bands or {"S2": LEGACY_S2_13}, time_pool, norm="percentile"),
                   name="upma", **kw)

    def upsample_frames(self, feats: torch.Tensor, guide: torch.Tensor,
                        size: tuple[int, int]) -> torch.Tensor:
        H, W = guide.shape[-2:]
        if H % size[0] or W % size[1]:
            raise ValueError(f"{self.key}: output {tuple(size)} must divide the guidance "
                             f"resolution {(H, W)}")
        grid = tuple(feats.shape[-2:])
        if self.kernels is not None and self._ids is not None and self.kernels.covers(self._ids):
            if self.kernels.meta != self.cache_meta(grid):
                raise ValueError(f"{self.key}: kernel cache {self.kernels.dir} was fitted "
                                 f"for {self.kernels.meta}")
            raw = self.kernels.get(self._ids).flatten(0, 1).to(feats.device)
            if self._frames is not None:                  # only the upsampled steps
                raw = raw[self._frames.to(raw.device)]
            if len(raw) != len(feats):
                raise ValueError(f"{self.key}: cached kernels are for another time setup")
        else:
            raw = self.fit(guide, grid)            # every image its own kernel field
        A = torch.cat([kernel_operator(raw[i:i + 64], guide[i:i + 64], size)
                       for i in range(0, len(raw), 64)])                     # (N, h*w, P)
        out = torch.einsum("nkp,ndk->ndp", A, feats.float().flatten(2))
        return out.view(*out.shape[:2], *size)


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


# ---------------------------------------------------------------------------------------
# batched: many images fitted and applied at once. Every image has its own kernel field,
# its own loss term and (Adam being elementwise) its own update, so a batch of N fits is N
# independent fits; tests/test_upsamplers.py checks it against the per-image code above.
# ---------------------------------------------------------------------------------------
PARAMS = ("sx_raw", "sy_raw", "th_raw", "sr_raw")      # raw kernel parameters, in this order


def jbu_batched(feat_lr, guide_hr, scale, sigma_x, sigma_y, theta, sigma_r, R_max=4,
                alpha_dyn=2.0, C_chunk=512, Nn_chunk=81, center_mode="nearest",
                use_autocast=True):
    """gs_jbu_aniso_noparent for a batch: feat_lr (B, C, Hl, Wl), guide_hr (B, Cg, Hh, Wh),
    parameter maps (B, 1, Hl, Wl) -> (B, C, Hh, Wh). Same arithmetic, out-of-place."""
    B, C, Hl, Wl = feat_lr.shape
    _, Cg, Hh, Wh = guide_hr.shape
    dev, dtype_feat = feat_lr.device, feat_lr.dtype
    y = torch.arange(Hh, device=dev, dtype=torch.float32)
    x = torch.arange(Wh, device=dev, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    u, v = (Y + 0.5) / scale - 0.5, (X + 0.5) / scale - 0.5
    rnd = torch.round if center_mode == "nearest" else torch.floor
    uc = rnd(u).clamp(0, Hl - 1).to(torch.long)
    vc = rnd(v).clamp(0, Wl - 1).to(torch.long)

    sigma_eff_hr = F.interpolate(torch.maximum(sigma_x, sigma_y), (Hh, Wh), mode="bilinear",
                                 align_corners=False)
    R_map = torch.ceil(alpha_dyn * sigma_eff_hr).clamp(min=1, max=R_max).to(torch.int64)[:, 0]
    dY_all, dX_all = _build_offsets(R_max, dev)
    K = dY_all.numel()

    num_s = torch.zeros(B, C, Hh, Wh, device=dev)
    den_s = torch.zeros(B, Hh, Wh, device=dev)
    m = torch.full((B, Hh, Wh), float("-inf"), device=dev)
    guide32 = guide_hr.float()
    maps = [p.float().reshape(B, Hl * Wl) for p in (sigma_x, sigma_y, theta, sigma_r)]
    feat_flat = feat_lr.permute(0, 2, 3, 1).reshape(B, Hl * Wl, C)
    guide_lr = F.interpolate(guide32, size=(Hl, Wl), mode="bilinear",
                             align_corners=False).reshape(B, Cg, Hl * Wl)

    with torch.autocast("cuda", enabled=use_autocast, dtype=torch.float16):
        for n0 in range(0, K, Nn_chunk):
            n1 = min(n0 + Nn_chunk, K)
            dY, dX = dY_all[n0:n1].view(-1, 1, 1), dX_all[n0:n1].view(-1, 1, 1)
            Bn = dY.shape[0]
            Ui = torch.clamp(uc.unsqueeze(0) + dY, 0, Hl - 1)
            Vi = torch.clamp(vc.unsqueeze(0) + dX, 0, Wl - 1)
            idx = (Ui * Wl + Vi).reshape(-1)
            gather = lambda t: t[:, idx].view(B, Bn, Hh, Wh)            # noqa: E731
            mask = (dY ** 2 + dX ** 2).unsqueeze(0) <= (R_map ** 2).unsqueeze(1)
            dx = X.unsqueeze(0) - ((Vi.to(torch.float32) + 0.5) * scale - 0.5)
            dy = Y.unsqueeze(0) - ((Ui.to(torch.float32) + 0.5) * scale - 0.5)
            sx, sy, th, sr = (gather(t) for t in maps)
            sx, sy, sr = sx.clamp_min(1e-6), sy.clamp_min(1e-6), sr.clamp_min(1e-6)
            cos_t, sin_t = torch.cos(th), torch.sin(th)
            x_p = dx * cos_t + dy * sin_t
            y_p = -dx * sin_t + dy * cos_t
            log_ws = -(x_p ** 2) / (2 * sx ** 2 + 1e-8) - (y_p ** 2) / (2 * sy ** 2 + 1e-8)
            diff2 = torch.zeros_like(log_ws)
            for cg in range(Cg):
                diff2 = diff2 + (guide32[:, cg].unsqueeze(1) - gather(guide_lr[:, cg])) ** 2
            log_w = log_ws - (diff2 / Cg) / (2.0 * sr * sr + 1e-8)
            log_w = torch.where(mask, log_w, torch.full_like(log_w, float("-inf")))

            m_chunk = log_w.max(dim=1).values
            valid = torch.isfinite(m_chunk)
            if not valid.any():
                continue
            m_new = torch.where(valid, torch.maximum(m, m_chunk), m)
            scale_old = torch.where(valid, torch.exp((m - m_new).clamp_max(0)),
                                    torch.ones_like(den_s))
            den_s = den_s * scale_old
            num_s = num_s * scale_old.unsqueeze(1)
            shift = torch.where(valid.unsqueeze(1), log_w - m_new.unsqueeze(1),
                                torch.full_like(log_w, float("-inf")))
            s = torch.exp(shift)                                          # (B, Bn, Hh, Wh)
            den_s = den_s + s.sum(1)
            parts = []
            for c0 in range(0, C, C_chunk):
                c1 = min(c0 + C_chunk, C)
                sel = feat_flat[:, idx, c0:c1].view(B, Bn, Hh, Wh, c1 - c0)
                parts.append((sel * s.unsqueeze(-1)).sum(1).permute(0, 3, 1, 2))
            num_s = num_s + torch.cat(parts, 1)
            m = m_new

    out = (num_s / den_s.clamp_min(1e-8).unsqueeze(1)).to(dtype_feat)
    fallback = F.interpolate(feat_lr, size=(Hh, Wh), mode="bilinear", align_corners=False)
    return torch.where((den_s < 1e-6).unsqueeze(1), fallback, out)


class BatchedJBU(nn.Module):
    """B independent pixelwise JBUs (LearnablePixelwiseAnisoJBU_NoParent, same defaults)."""

    def __init__(self, B, Hl, Wl, scale, init_sigma=16.0, init_sigma_r=0.12, R_max=8,
                 alpha_dyn=2.0, eval_C_chunk=128, eval_Nn_chunk=49, raw=None):
        super().__init__()
        self.scale, self.R_max, self.alpha_dyn = int(scale), int(R_max), float(alpha_dyn)
        self.eval_C_chunk, self.eval_Nn_chunk = eval_C_chunk, eval_Nn_chunk
        init = (math.log(init_sigma), math.log(init_sigma), 0.0, math.log(init_sigma_r))
        for k, (name, v) in enumerate(zip(PARAMS, init)):
            t = torch.full((B, 1, Hl, Wl), v) if raw is None else raw[:, k:k + 1].float()
            setattr(self, name, nn.Parameter(t.clone()))

    def raw(self) -> torch.Tensor:
        """(B, 4, Hl, Wl): the fitted kernel field, what a kernel cache stores."""
        return torch.cat([getattr(self, n).detach() for n in PARAMS], 1)

    def forward(self, feat_lr, guide_hr):
        C = feat_lr.shape[1]
        K = (2 * self.R_max + 1) ** 2
        cc, nn_ = (C, K) if self.training else (min(self.eval_C_chunk, C),
                                                  min(self.eval_Nn_chunk, K))
        return jbu_batched(feat_lr, guide_hr, self.scale, torch.exp(self.sx_raw),
                           torch.exp(self.sy_raw), _tanh_bound_pi(self.th_raw),
                           torch.exp(self.sr_raw), R_max=self.R_max, alpha_dyn=self.alpha_dyn,
                           C_chunk=cc, Nn_chunk=nn_)


def fit_kernels(hr: torch.Tensor, grid: tuple[int, int], fit_steps: int) -> torch.Tensor:
    """Batched fit_jbu without the apply: hr (B, Cg, H, W) in [0, 1] -> raw kernel fields
    (B, 4, Hl, Wl). The loss is the SUM of per-image mean L1s, so each image's gradient is
    what its own fit would get."""
    H, W = hr.shape[-2:]
    Hl, Wl = grid
    scale = int(H / Hl)
    lr = F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False)
    model = BatchedJBU(len(hr), Hl, Wl, scale).to(hr.device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    gamma = (1e-9 / 1e-1) ** (1.0 / 5100)
    sched = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    with torch.enable_grad():
        for step in range(fit_steps + 1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                pred = model(lr, hr)
                loss = (pred.float() - hr).abs().mean(dim=(1, 2, 3)).sum()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model.raw()


def kernel_operator(raw: torch.Tensor, guide: torch.Tensor,
                    size: tuple[int, int]) -> torch.Tensor:
    """The upsampling as a matrix: with the kernel field fixed, the JBU output is a linear map
    of the features (weights depend only on guidance and kernels), so applying it once to the
    identity basis gives (N, Hl*Wl, P) with out = A^T feats. Area-pooled to `size` (P pixels);
    each output pixel's weights are non-negative and sum to 1. Much cheaper than applying the
    kernels to D-dim features: the expensive pass runs on Hl*Wl channels instead of D."""
    n, _, hl, wl = raw.shape
    eye = torch.eye(hl * wl, device=raw.device).view(1, hl * wl, hl, wl).expand(n, -1, -1, -1)
    out = apply_kernels(eye.contiguous(), guide, raw).float()
    if tuple(out.shape[-2:]) != tuple(size):
        out = F.adaptive_avg_pool2d(out, size)           # still a convex combination
    return out.flatten(2)


def apply_kernels(feats: torch.Tensor, guide: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """Fitted kernel fields (B, 4, Hl, Wl) applied to feats (B, C, Hl, Wl), guided by guide
    (B, Cg, H, W) -> (B, C, H, W)."""
    model = BatchedJBU(len(raw), *raw.shape[-2:], guide.shape[-2] // raw.shape[-2],
                       raw=raw).to(feats.device).eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        return model(feats, guide.float())                  # as fit_jbu's apply
