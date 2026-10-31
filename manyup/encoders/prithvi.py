"""Prithvi-EO-2.0 (IBM / NASA, arXiv:2412.02732) behind the encoder interface manyup.inject.MixedResOE
uses, so the mixed-resolution model runs it unchanged: same packing, same blocks API, same heads.

Prithvi is a plain MAE ViT, NOT a FlexiViT: one 16 x 16 Conv3d kernel (tubelet 1), 6 HLS bands,
fixed 3D sin-cos positions over (frame, row, col) in token units, pretrained on 4-frame series.
What makes it flexible here:

  patch size   PI-resize of the kernel (FlexiViT, Beyer et al. 2023) in the direction OlmoEarth
               resizes its INPUT (FlexiPatchEmbed: bicubic, antialiased): a p x p patch x gives
               <resize_{p -> 16}(x), w>, i.e. kernel w_p = R_p^T w with R_p the (256, p^2) resize
               matrix. Constant inputs keep their response (resizing preserves constants), so a
               ps8 or ps4 token has the norm of a ps16 one: no rescaling of the kernel is needed
               (a naively interpolated kernel would scale responses by (p / 16)^2).
  positions    the 3D sin-cos at continuous coordinates: a token's CENTRE in ps16 token units,
               (y + p / 2) / 16 - 1 / 2 (= the grid index at p = 16, the pretrained encoding), its
               time coordinate the season slot. Physical positions, not Prithvi's own bicubic
               rescaling of the 224 px grid onto smaller images.
  time / place the TL encodings (year + day of year, lat / lon) add with learned scales that
               pretraining drove to ~1e-6 (tiny-TL): time is kept (crops carry their month's date,
               the scale may grow while fine-tuning), location dropped.

Blocks are OlmoEarth Blocks carrying Prithvi's weights (timm's fused qkv split into q, k, v;
LayerNorm eps 1e-5, GELU, no layer scale, as timm builds Prithvi), so varlen FlashAttention and
MixedResOE's cross-attention paths work as for OlmoEarth. The cls token is dropped (dense task,
packed sequences); the head and readout are trained anyway.

Input: the OlmoEarth-packed sample (OlmoEarthInput, "sentinel2_l2a" only): its COMPUTED
normalization is an affine map per band, undone here, then Prithvi's own mean / std applied.
HLS B02, B03, B04, B05 (narrow NIR), B06, B07 (SWIR 1, 2) = Sentinel-2 B02, B03, B04, B8A, B11,
B12.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.attention import Block
from torch import nn

ROOT = Path(__file__).resolve().parents[2] / "cache" / "pretrained"
MODELS = {"prithvi_tiny": ("Prithvi-EO-2.0-tiny-TL", "Prithvi_EO_V2_tiny_TL.pt"),
          "prithvi_100m": ("Prithvi-EO-2.0-100M-TL", "Prithvi_EO_V2_100M_TL.pt")}
S2 = "sentinel2_l2a"
S2_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
BASE_PS = 16
_MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Prithvi's 1D sin-cos (get_1d_sincos_pos_embed_from_grid) at float positions -> (N, dim)."""
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, device=pos.device, dtype=torch.float64)
                            / (dim / 2.0))
    out = pos.reshape(-1, 1).double() * omega
    return torch.cat([out.sin(), out.cos()], 1).float()


def sincos_3d(t: torch.Tensor, y: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    """Prithvi's get_3d_sincos_pos_embed at continuous (t, y, x) -> (N, dim): [x | y | t] with
    6/16, 6/16, 4/16 of dim."""
    w = dim // 16 * 6
    return torch.cat([sincos_1d(x, w), sincos_1d(y, w), sincos_1d(t, dim - 2 * w)], 1)


@lru_cache(maxsize=None)
def resize_matrix(p: int) -> torch.Tensor:
    """(p^2, 256) M with M[i, j] = (resize_{p -> 16} e_i)_j, bicubic + antialias as
    OlmoEarth's FlexiPatchEmbed resizes its input; the kernel for p is w @ M^T."""
    basis = torch.eye(p * p, dtype=torch.float64).view(p * p, 1, p, p)
    if p == BASE_PS:
        return torch.eye(p * p, dtype=torch.float64)
    out = F.interpolate(basis, size=(BASE_PS, BASE_PS), mode="bicubic", antialias=True,
                        align_corners=False)
    return out.view(p * p, BASE_PS * BASE_PS)


class PrithviOE(nn.Module):
    """The encoder object MixedResOE wraps: blocks, norm, embedding_size, and mixres_embed (its
    tokenizer, in place of OlmoEarth's patch embedding + composite encodings)."""

    def __init__(self, name: str = "prithvi_tiny", pretrained: bool = True):
        super().__init__()
        folder, weights = MODELS[name]
        cfg = json.loads((ROOT / folder / "config.json").read_text())["pretrained_cfg"]
        D, depth, heads = cfg["embed_dim"], cfg["depth"], cfg["num_heads"]
        if cfg["patch_size"] != [1, BASE_PS, BASE_PS] or cfg["in_chans"] != len(S2_BANDS):
            raise ValueError(f"{folder}: patch {cfg['patch_size']}, {cfg['in_chans']} bands")
        self.embedding_size, self.name = D, name
        self.proj_w = nn.Parameter(torch.empty(D, len(S2_BANDS), BASE_PS * BASE_PS))
        self.proj_b = nn.Parameter(torch.zeros(D))
        nn.init.xavier_uniform_(self.proj_w.view(D, -1))
        self.time_scale = nn.Parameter(torch.full((1,), 0.1))
        self.blocks = nn.ModuleList(Block(D, heads, cfg["mlp_ratio"], qkv_bias=True,
                                          norm_layer=nn.LayerNorm, use_flash_attn=True)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(D)
        # OlmoEarth's computed normalization, undone: raw = (y - off) / scale, then Prithvi's
        m = Modality.get(S2)
        idx = [m.band_order.index(b) for b in S2_BANDS]
        norm = Normalizer(Strategy.COMPUTED)
        y0 = norm.normalize(m, np.zeros((1, len(m.band_order))))[0, idx]
        y1 = norm.normalize(m, np.full((1, len(m.band_order)), 1000.0))[0, idx]
        scale = (y1 - y0) / 1000
        mean, std = np.array(cfg["mean"]), np.array(cfg["std"])
        # x_prithvi = (y - off) / (scale std) - mean / std = a y + b
        self.register_buffer("band_idx", torch.tensor(idx), persistent=False)
        self.register_buffer("in_a", torch.tensor(1 / (scale * std), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("in_b", torch.tensor(-y0 / (scale * std) - mean / std,
                                                  dtype=torch.float32), persistent=False)
        if pretrained:
            self.load_prithvi(ROOT / folder / weights)

    def load_prithvi(self, path: Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        own = {"proj_w": enc["patch_embed.proj.weight"][:, :, 0].flatten(2),       # (D, 6, 256)
               "proj_b": enc["patch_embed.proj.bias"],
               "time_scale": enc["temporal_embed_enc.scale"],
               "norm.weight": enc["norm.weight"], "norm.bias": enc["norm.bias"]}
        for i in range(len(self.blocks)):
            p = f"blocks.{i}."
            q, k, v = enc[p + "attn.qkv.weight"].chunk(3)
            qb, kb, vb = enc[p + "attn.qkv.bias"].chunk(3)
            own.update({p + "attn.q.weight": q, p + "attn.k.weight": k, p + "attn.v.weight": v,
                        p + "attn.q.bias": qb, p + "attn.k.bias": kb, p + "attn.v.bias": vb})
            for n in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                      "attn.proj.weight", "attn.proj.bias", "mlp.fc1.weight", "mlp.fc1.bias",
                      "mlp.fc2.weight", "mlp.fc2.bias"):
                own[p + n] = enc[p + n]
        missing, unexpected = self.load_state_dict(own, strict=False)
        missing = [k for k in missing if not k.endswith(("band_idx", "in_a", "in_b"))]
        if missing or unexpected:
            raise ValueError(f"{path}: missing {missing[:4]}, unexpected {unexpected[:4]}")
        left = {k for k in enc if k.split(".")[0] not in ("blocks",)} - {
            "patch_embed.proj.weight", "patch_embed.proj.bias", "temporal_embed_enc.scale",
            "norm.weight", "norm.bias"}
        if left - {"cls_token", "pos_embed", "location_embed_enc.scale"}:
            raise ValueError(f"{path}: unused encoder weights {sorted(left)}")

    def kernel(self, p: int) -> torch.Tensor:
        """(D, 6, p^2) the PI-resized kernel for patch p."""
        if p == BASE_PS:
            return self.proj_w
        M = resize_matrix(p).to(self.proj_w.device, torch.float32)
        return self.proj_w.float() @ M.T

    def time_encoding(self, timestamps: torch.Tensor) -> torch.Tensor:
        """(..., 3) [day, month (0-11), year] -> (..., D): scale x [sincos(year) | sincos(doy)]
        (Prithvi's TemporalEncoder)."""
        start = torch.tensor(_MONTH_START, device=timestamps.device)
        doy = start[timestamps[..., 1]] + timestamps[..., 0]
        h = self.embedding_size // 2
        enc = torch.cat([sincos_1d(timestamps[..., 2].float(), h),
                         sincos_1d(doy.float(), self.embedding_size - h)], 1)
        return self.time_scale * enc.view(*timestamps.shape[:-1], -1)

    def mixres_embed(self, sample: MaskedOlmoEarthSample, patch_size: int,
                     slot: torch.Tensor | None = None, py: torch.Tensor | None = None,
                     px: torch.Tensor | None = None):
        """-> (tokens (S, N, D), mask (S, N), dims, masks) as OlmoEarth's
        collapse_and_combine_hwtc / split_tokens_masks_and_dims give them: one modality
        (sentinel2_l2a), one band set. Crops (slot set): T = 1, one patch each, at top-left
        pixel (py, px) and time slot `slot`."""
        x = getattr(sample, S2)                                          # (S, H, W, T, 12)
        if x is None:
            raise ValueError("Prithvi reads sentinel2_l2a only")
        mask = getattr(sample, MaskedOlmoEarthSample.get_masked_modality_name(S2))
        p, D = patch_size, self.embedding_size
        S, H, W, T, _ = x.shape
        if H % p or W % p:
            raise ValueError(f"{H} x {W} px is not a multiple of patch {p}")
        h, w = H // p, W // p
        x = x.index_select(-1, self.band_idx).float() * self.in_a + self.in_b
        x = x.view(S, h, p, w, p, T, -1).permute(0, 1, 3, 5, 6, 2, 4).reshape(S, h, w, T, -1, p * p)
        tok = torch.einsum("shwtcq,dcq->shwtd", x, self.kernel(p).float()) + self.proj_b.float()
        dev = tok.device
        c = lambda i: (i.float() + p / 2) / BASE_PS - 0.5                      # noqa: E731
        if slot is None:
            gy, gx = torch.meshgrid(torch.arange(h, device=dev) * p,
                                    torch.arange(w, device=dev) * p, indexing="ij")
            tt = torch.arange(T, device=dev)
            pos = sincos_3d(tt[:, None, None].expand(T, h, w).reshape(-1),
                            c(gy).expand(T, h, w).reshape(-1), c(gx).expand(T, h, w).reshape(-1),
                            D).view(T, h, w, D).permute(1, 2, 0, 3)            # (h, w, T, D)
            tok = tok + pos[None]
        else:
            if (h, w, T) != (1, 1, 1):
                raise ValueError(f"crops are one patch of one step, got {(h, w, T)}")
            pos = sincos_3d(slot.to(dev).float(), c(py.to(dev)), c(px.to(dev)), D)
            tok = tok + pos[:, None, None, None]
        tok = tok + self.time_encoding(sample.timestamps.to(dev))[:, None, None]
        tmask = mask[:, ::p, ::p, :, 0]                                  # (S, h, w, T)
        dims = {S2: (S, h, w, T, 1, D)}
        masks = {MaskedOlmoEarthSample.get_masked_modality_name(S2): tmask[..., None]}
        return tok.reshape(S, -1, D), tmask.reshape(S, -1), dims, masks
