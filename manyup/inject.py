"""Mixed-resolution OlmoEarth: inject high-res tokens into a low-res sequence mid-network.

    blocks [0, L)   the low-res sequence (e.g. seasons at ps16) on its own          -> lr_L
    select          anything that reads lr_L picks where high-res tokens are needed
    blocks [0, L)   each high-res crop (one ps4 patch of one date) on its own       -> hr_L
                    (a few tokens per crop: very cheap, and all crops in one call)
    blocks [L, n)   every sample's lr_L tokens + its crops' hr_L tokens, one sequence
    norm            OlmoEarth's final norm on every token

Both token sets run the SAME weights (OlmoEarth is a FlexiViT) and meet at the same depth.
Positions are OlmoEarth's own: absolute sin-cos in pixel units (gsd ratio = patch size), so a
crop at pixel (py, px) gets the encoding of that pixel whatever the patch size -- off the lr
grid by construction. A crop keeps its own month (month encoding) and takes the time SLOT
of the lr step it belongs to (e.g. its season), since slots index the lr sequence. A learned
type embedding (zero at init) marks injected tokens: the pretrained model never saw two patch
sizes in one sequence and has no other way to tell them apart.

`pos` (how a token's place is encoded; "corner" is the above):
    center   a crop takes the position of its CENTRE in the lr frame, py + (ps - lr_ps) / 2:
             OlmoEarth places a token at its top-left pixel, so a ps4 crop and the ps16 token
             around it would sit (16 - 4) / 2 = 6 px apart for the same centre; lr tokens keep
             their pretrained positions
    box      positions as "corner"; every spatial token (lr and crop) also gets
             W sincos(bottom-right pixel), W a zero-init linear map: with both corners a token
             carries its extent, hence its scale, and the pretrained model is unchanged at init

`doy`: OlmoEarth encodes a step's slot index and calendar month, never its day, so two dates
of one month differ only by a fractional slot (if the caller passes one). With doy every
multitemporal token (lr and crop) also gets W' fourier(day of year of its timestamp), W' a
zero-init linear map: harmonics 1 .. n / 2 of the year, so one day turns the top one by
~0.4 rad, and the pretrained model is unchanged at init.

`attn` (who attends to whom in blocks [L, n); "joint" is the above, one sequence per sample):
    oneway   the lr sequence runs on its own (= the crop-free model); every crop token attends
             to its own crop's tokens and to the keys / values of its sample's lr tokens at the
             same depth (a prefix: lr never reads crops, crops never read each other)
    overlap  oneway, lr keys limited to the tokens of the lr cell under the crop's centre
             (every lr step and band set of that cell)

`head_attn` (what a crop reads in blocks [0, L), before it joins the lr sequence; needs L > 0):
    alone    its own tokens only (the above)
    oneway   + the keys / values of its sample's lr tokens at the same depth, cached from the
             lr pass (head_blocks keeps them): context for the crop at the cost of a
             cross-attention, the lr tokens untouched until block L
    overlap  oneway, lr keys limited to the lr cell under the crop's centre

Everything runs through OlmoEarth's varlen FlashAttention: sequences are packed back to back
(cu_seqlens), so tokens never cross samples and padding never exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from olmoearth_pretrain.data.constants import BASE_GSD, Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue, TokensAndMasks
from olmoearth_pretrain.nn.encodings import get_1d_sincos_pos_encoding
from olmoearth_pretrain.nn.flexi_vit import PositionEncoding
from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths
from torch import nn

from manyup.encoders.olmoearth import pool_timestep


@dataclass
class Packed:
    """Valid tokens of S sequences, back to back."""
    x: torch.Tensor             # (N_valid, D)
    seqlens: torch.Tensor       # (S,) long
    keep: torch.Tensor          # (S, N) bool: which of each sequence's N token slots are valid
    dims: dict                  # modality -> token shape (S, h, w, [t,] bandsets, D)
    masks: dict                 # masked-modality name -> (S, h, w, [t,] bandsets) mask values
    yx: torch.Tensor | None = None  # crops: (S, 2) centre pixel (y, x) of each crop
    trace: list | None = None       # lr: its tokens entering each block of [0, L) (head_attn)
    typed: bool = False             # crops: hr_type already added (head_attn)

    def subset(self, sel: torch.Tensor) -> "Packed":
        """The sequences sel (S,) bool, their tokens included (crops: a part of the picks)."""
        tok = torch.repeat_interleave(sel, self.seqlens)
        return Packed(self.x[tok], self.seqlens[sel], self.keep[sel],
                      {m: (int(sel.sum()),) + tuple(d[1:]) for m, d in self.dims.items()},
                      {m: v[sel] for m, v in self.masks.items()},
                      None if self.yx is None else self.yx[sel], None, self.typed)

    def pooled(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """(S, T, h, w, D): per step, mean over bandsets then over present modalities
        (manyup.encoders.olmoearth.pool_timestep, i.e. OlmoEarthEncoder's output)."""
        x = self.x if x is None else x
        S, N = self.keep.shape
        full = x.new_zeros(S * N, x.shape[-1])
        full[self.keep.flatten()] = x
        full = full.view(S, N, -1)
        per, i = {}, 0
        for m, shape in self.dims.items():
            n = 1
            for s in shape[1:-1]:
                n *= s
            per[m] = full[:, i:i + n].reshape(*shape[:-1], -1)
            i += n
        tam = TokensAndMasks(**per, **self.masks)
        T = next(iter(per.values())).shape[3]
        return torch.stack([pool_timestep(tam, t) for t in range(T)], 1)


_MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def doy_encoding(timestamps: torch.Tensor, dim: int) -> torch.Tensor:
    """(..., 3) [day (1-31), month (0-11), year] -> (..., dim): sin and cos of harmonics
    1 .. dim / 2 of the day of year (365-day year; the year itself is not encoded)."""
    start = torch.tensor(_MONTH_START, device=timestamps.device)
    ang = 2 * torch.pi * (start[timestamps[..., 1]] + timestamps[..., 0] - 1).float() / 365
    ang = ang[..., None] * torch.arange(1, dim // 2 + 1, device=timestamps.device)
    return torch.cat([ang.sin(), ang.cos()], -1)


def sincos_2d(py: torch.Tensor, px: torch.Tensor, dim: int) -> torch.Tensor:
    """OlmoEarth's 2D spatial encoding at (possibly fractional) pixel positions -> (N, dim);
    equals get_2d_sincos_pos_encoding_with_resolution on a grid (x first, then y)."""
    return torch.cat([get_1d_sincos_pos_encoding(px.float(), dim // 2),
                      get_1d_sincos_pos_encoding(py.float(), dim // 2)], 1)


POS = ("corner", "center", "box")
ATTN = ("joint", "oneway", "overlap")
HEAD_ATTN = ("alone", "oneway", "overlap")
MAX_SEQS = 32768            # sequences per FlashAttention varlen call (see MixedResOE.blocks)


def _positions(seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed sequences -> per token (sequence index, position inside its sequence)."""
    seq = torch.repeat_interleave(torch.arange(len(seqlens), device=seqlens.device), seqlens)
    start = torch.cumsum(seqlens, 0) - seqlens
    return seq, torch.arange(len(seq), device=seqlens.device) - start[seq]


class MixedResOE(nn.Module):
    def __init__(self, encoder: nn.Module, inject_layer: int, pos: str = "corner",
                 lr_ps: int = 16, doy: bool = False, attn: str = "joint",
                 head_attn: str = "alone"):
        super().__init__()
        if pos not in POS:
            raise ValueError(f"pos {pos!r} not in {POS}")
        if attn not in ATTN:
            raise ValueError(f"attn {attn!r} not in {ATTN}")
        if head_attn not in HEAD_ATTN:
            raise ValueError(f"head_attn {head_attn!r} not in {HEAD_ATTN}")
        if head_attn != "alone" and inject_layer == 0:
            raise ValueError("head_attn acts in blocks [0, L): it needs inject_layer > 0")
        self.attn, self.head_attn = attn, head_attn
        if not 0 <= inject_layer <= len(encoder.blocks):
            raise ValueError(f"inject_layer {inject_layer} not in [0, {len(encoder.blocks)}]")
        if not encoder.blocks[0].attn.use_flash_attn:
            raise ValueError("MixedResOE packs sequences for varlen FlashAttention")
        self.encoder, self.L = encoder, inject_layer
        self.hr_type = nn.Parameter(torch.zeros(encoder.embedding_size))
        self.pos, self.lr_ps = pos, lr_ps
        if pos == "box":
            self.box = nn.Linear(encoder.composite_encodings.embedding_dim_per_embedding_type,
                                 encoder.embedding_size, bias=False)
            nn.init.zeros_(self.box.weight)
        if doy:
            self.doy = nn.Linear(encoder.composite_encodings.embedding_dim_per_embedding_type,
                                 encoder.embedding_size, bias=False)
            nn.init.zeros_(self.doy.weight)
        self.use_doy = doy

    # -- embedding ------------------------------------------------------------------------
    def embed(self, sample: MaskedOlmoEarthSample, patch_size: int, slot: torch.Tensor | None = None,
              py: torch.Tensor | None = None, px: torch.Tensor | None = None) -> Packed:
        """Patch embedding + OlmoEarth's encodings, packed. For crops (T = 1, one patch each),
        `slot` (S,) replaces time slot 0 and (py, px) (S,) top-left pixel positions replace
        (0, 0) (moved to the crop's centre in the lr frame under pos "center")."""
        enc = self.encoder
        x = enc.patch_embeddings.forward(sample, patch_size)
        tokens, masks, dims = enc.split_tokens_masks_and_dims(x)
        tokens = enc.composite_encodings.forward(tokens, sample.timestamps, patch_size, BASE_GSD)
        if slot is not None:
            n = enc.composite_encodings.embedding_dim_per_embedding_type
            dev = next(iter(tokens.values())).device
            zero = torch.zeros_like(py)
            yx = torch.stack([py, px], 1).float() + patch_size / 2             # true centre
            if self.pos == "center":
                py, px = (v.float() + (patch_size - self.lr_ps) / 2 for v in (py, px))
            d_time = (get_1d_sincos_pos_encoding(slot.float().to(dev), n)
                      - get_1d_sincos_pos_encoding(zero.float().to(dev), n))
            d_pos = sincos_2d(py.to(dev), px.to(dev), n) - sincos_2d(zero.to(dev), zero.to(dev), n)
            for m, t in tokens.items():
                if t.shape[1:4] != (1, 1, 1):
                    raise ValueError(f"crop tokens of {m} are {tuple(t.shape)}, want one patch")
                mod = Modality.get(m)
                delta = torch.zeros_like(t[:, 0, 0, 0, 0])                     # (S, D)
                if mod.is_multitemporal:
                    delta[:, n:2 * n] = d_time
                if mod.is_spatial:
                    delta[:, 3 * n:4 * n] = d_pos
                tokens[m] = t + delta[:, None, None, None, None].to(t.dtype)
        if self.pos == "box":
            n = enc.composite_encodings.embedding_dim_per_embedding_type
            for m, t in tokens.items():
                if not Modality.get(m).is_spatial:
                    continue
                h, w = t.shape[1:3]
                gy, gx = torch.meshgrid(torch.arange(1, h + 1, device=t.device) * patch_size,
                                        torch.arange(1, w + 1, device=t.device) * patch_size,
                                        indexing="ij")                         # bottom-right, px
                if slot is not None:                                           # crops: h = w = 1
                    gy, gx = gy + py.to(t.device)[:, None, None], gx + px.to(t.device)[:, None, None]
                br = sincos_2d(gy.reshape(-1), gx.reshape(-1), n).view(-1, h, w, n)
                add = self.box(br.to(self.box.weight.dtype))                   # (S or 1, h, w, D)
                tokens[m] = t + add.view(add.shape[:3] + (1,) * (t.ndim - 4) + add.shape[-1:]).to(t.dtype)
        if self.use_doy:
            n = enc.composite_encodings.embedding_dim_per_embedding_type
            for m, t in tokens.items():
                if not Modality.get(m).is_multitemporal:
                    continue
                add = self.doy(doy_encoding(sample.timestamps.to(t.device), n)
                               .to(self.doy.weight.dtype))                     # (S, T, D)
                lead = (1,) * (t.ndim - 4)                                     # h, w if spatial
                tokens[m] = t + add.view(add.shape[:1] + lead + add.shape[1:2] + (1,)
                                         + add.shape[-1:]).to(t.dtype)
        both = {**tokens, **masks}
        x, mask = enc.collapse_and_combine_hwtc(both)
        keep = mask == MaskValue.ONLINE_ENCODER.value
        return Packed(x[keep], keep.sum(1), keep, dims, masks, yx if slot is not None else None)

    # -- transformer ------------------------------------------------------------------------
    def blocks(self, x: torch.Tensor, seqlens: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        if lo >= hi:
            return x
        if len(seqlens) > MAX_SEQS:      # FlashAttention's varlen kernel fails ("invalid
            # configuration argument") past ~65k sequences in one call (GEOID series: 64 chips x
            # ~1000 one-token crops); sequences are independent, so chunks give the same result
            ends = torch.cumsum(seqlens, 0)
            out = []
            for i in range(0, len(seqlens), MAX_SEQS):
                a = int(ends[i - 1]) if i else 0
                b = int(ends[min(i + MAX_SEQS, len(seqlens)) - 1])
                out.append(self.blocks(x[a:b], seqlens[i:i + MAX_SEQS], lo, hi))
            return torch.cat(out)
        cu = get_cumulative_sequence_lengths(seqlens)
        mx = int(seqlens.max())
        for blk in self.encoder.blocks[lo:hi]:
            x = blk(x=x, cu_seqlens=cu, max_seqlen=mx, attn_mask=None, rope_positions=None)
        return x

    def head_blocks(self, p: Packed, lr: Packed | None = None,
                    owner: torch.Tensor | None = None) -> Packed:
        """blocks [0, L) on p's sequences. head_attn "oneway" / "overlap": crops (p.yx set) are
        given the lr Packed this method returned and their owner (S_crops,) = lr sequence
        index, and also read that lr sequence's keys / values."""
        if self.head_attn == "alone":
            return Packed(self.blocks(p.x, p.seqlens, 0, self.L), p.seqlens, p.keep, p.dims,
                          p.masks, p.yx)
        if p.yx is None:                                    # the lr pass: keep every depth
            trace, x = self._lr_trace(p.x, p.seqlens, 0, self.L)
            return Packed(x, p.seqlens, p.keep, p.dims, p.masks, None, trace)
        if lr is None or lr.trace is None or owner is None:
            raise ValueError(f"head_attn {self.head_attn}: crops need their lr pass and owner")
        if not len(owner):
            return Packed(p.x, p.seqlens, p.keep, p.dims, p.masks, p.yx, None, True)
        (x,) = self._crop_blocks(lr, lr.trace, [(p, owner)], 0, self.head_attn == "overlap")
        return Packed(x, p.seqlens, p.keep, p.dims, p.masks, p.yx, None, True)

    def _typed(self, p: Packed, dtype) -> torch.Tensor:
        return (p.x if p.typed else p.x + self.hr_type).to(dtype)

    def _lr_trace(self, x: torch.Tensor, seqlens: torch.Tensor, lo: int, hi: int
                  ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """blocks [lo, hi) on the lr sequences -> (their tokens entering each block, output)."""
        cu, mx, trace = get_cumulative_sequence_lengths(seqlens), int(seqlens.max()), []
        for blk in self.encoder.blocks[lo:hi]:
            trace.append(x)
            x = blk(x=x, cu_seqlens=cu, max_seqlen=mx, attn_mask=None, rope_positions=None)
        return trace, x

    def tail(self, lr: Packed, hr: Packed | None, owner: torch.Tensor | None
             ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """blocks [L, n) + norm on each lr sequence joined by the hr crops it owns
        (owner (N_crops,) = lr sequence index). -> (lr tokens (N_lr, D), hr tokens)."""
        if hr is None or owner is None or len(owner) == 0:
            return self.tail_multi(lr, [])[0], None
        lr_x, (hr_x,) = self.tail_multi(lr, [(hr, owner)])
        return lr_x, hr_x

    def tail_multi(self, lr: Packed, crop_sets: list[tuple[Packed, torch.Tensor]]
                   ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """tail with several crop sets (e.g. month crops, S2 date crops, S1 date crops), each
        (Packed, owner (N_set,)); every lr sequence is joined by the crops of every set it
        owns. -> (lr tokens (N_lr, D), [tokens of each set])."""
        S = len(lr.seqlens)
        crop_sets = [(p, o) for p, o in crop_sets if o is not None and len(o)]
        if not crop_sets:
            return self.encoder.norm(self.blocks(lr.x, lr.seqlens, self.L,
                                                 len(self.encoder.blocks))), []
        if self.attn != "joint":
            return self._tail_oneway(lr, crop_sets)
        dev = lr.x.device
        seq = torch.cat([torch.repeat_interleave(torch.arange(S, device=dev), lr.seqlens)]
                        + [torch.repeat_interleave(o.to(dev), p.seqlens) for p, o in crop_sets])
        x = torch.cat([lr.x] + [self._typed(p, lr.x.dtype) for p, _ in crop_sets])
        order = torch.argsort(seq, stable=True)          # each sample: its lr, then its crops
        y = self.blocks(x[order], torch.bincount(seq, minlength=S), self.L,
                        len(self.encoder.blocks))
        out = torch.empty_like(y)
        out[order] = y
        out = self.encoder.norm(out)
        sizes = [len(lr.x)] + [len(p.x) for p, _ in crop_sets]
        parts = list(torch.split(out, sizes))
        return parts[0], parts[1:]

    def _lr_cells(self, lr: Packed) -> torch.Tensor:
        """(N_lr,) the lr cell of every valid lr token, y * 4096 + x; -1 if not spatial."""
        cells = []
        for m, shape in lr.dims.items():
            n = math.prod(shape[1:-1])
            if Modality.get(m).is_spatial:
                h, w = shape[1:3]
                hw = torch.arange(n, device=lr.x.device) // (n // (h * w))
                cells.append(hw // w * 4096 + hw % w)
            else:
                cells.append(torch.full((n,), -1, device=lr.x.device))
        return torch.cat(cells).expand(len(lr.seqlens), -1)[lr.keep]

    def _tail_oneway(self, lr: Packed, crop_sets: list[tuple[Packed, torch.Tensor]]
                     ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """tail_multi for attn "oneway" / "overlap": the lr sequence on its own, the crops
        reading it."""
        trace, x_lr = self._lr_trace(lr.x, lr.seqlens, self.L, len(self.encoder.blocks))
        outs = self._crop_blocks(lr, trace, crop_sets, self.L, self.attn == "overlap")
        return self.encoder.norm(x_lr), [self.encoder.norm(o) for o in outs]

    def _crop_blocks(self, lr: Packed, trace: list[torch.Tensor],
                     crop_sets: list[tuple[Packed, torch.Tensor]], lo: int, overlap: bool
                     ) -> list[torch.Tensor]:
        """Blocks lo, lo + 1, .. (one per entry of trace = the lr tokens entering that block)
        on the crops, with the blocks' own weights: crop token i reads softmax over [lr keys
        of its sample (overlap: of its cell), keys of its own crop]; the lr tokens are only
        read. -> each set's tokens. Shape letters: S samples, Lm / Qm the most lr / crop
        tokens of a sample, C crops, bm the most tokens of a crop, nh heads."""
        S, dev = len(lr.seqlens), lr.x.device
        x_c = torch.cat([self._typed(p, lr.x.dtype) for p, _ in crop_sets])
        crop_len = torch.cat([p.seqlens for p, _ in crop_sets])                # (C,)
        owner = torch.cat([o.to(dev) for _, o in crop_sets])                   # (C,)
        t_crop, t_in = _positions(crop_len)                 # per crop token: crop, slot in it
        t_s = owner[t_crop]                                                    # its sample
        order = torch.argsort(t_s, stable=True)
        per = torch.bincount(t_s, minlength=S)
        t_q = torch.empty_like(t_s)
        t_q[order] = _positions(per)[1]                     # its slot among its sample's queries
        l_s, l_p = _positions(lr.seqlens)
        Lm, Qm, bm, C = int(lr.seqlens.max()), int(per.max()), int(crop_len.max()), len(crop_len)
        see = torch.zeros(S, Lm, dtype=torch.bool, device=dev)
        see[l_s, l_p] = True
        see = see[:, None].expand(S, Qm, Lm)                                   # lr keys that exist
        if overlap:
            lr_cell = torch.full((S, Lm), -2, device=dev)
            lr_cell[l_s, l_p] = self._lr_cells(lr)
            cyx = (torch.cat([p.yx for p, _ in crop_sets]).to(dev) // self.lr_ps).long()
            q_cell = torch.full((S, Qm), -3, device=dev)
            q_cell[t_s, t_q] = (cyx[:, 0] * 4096 + cyx[:, 1])[t_crop]
            see = see & ((lr_cell[:, None] == q_cell[:, :, None]) | (lr_cell[:, None] == -1))
        own = torch.arange(bm, device=dev)[None] < crop_len[t_crop, None]      # (Nq, bm)
        own_p = torch.ones(S, Qm, bm, dtype=torch.bool, device=dev)            # padding rows: 0s
        own_p[t_s, t_q] = own

        def padded(v, i, j, n, m):                          # (N, nh, d) -> (n, m, nh, d)
            out = v.new_zeros(n, m, *v.shape[1:])
            out[i, j] = v
            return out

        for blk, x_lr in zip(self.encoder.blocks[lo:], trace):
            a = blk.attn
            if a.attn_drop.p or PositionEncoding.is_rope(a.position_encoding):
                raise ValueError("one-way attention: no attention dropout, absolute positions")
            heads = lambda v: v.view(len(v), a.num_heads, -1)                  # noqa: E731
            n_lr, n_c = blk.norm1(x_lr), blk.norm1(x_c)
            k_lr = padded(a.k_norm(heads(a.k(n_lr))), l_s, l_p, S, Lm)         # (S, Lm, nh, d)
            v_lr = padded(heads(a.v(n_lr)), l_s, l_p, S, Lm)
            q = a.q_norm(heads(a.q(n_c)))                                      # (Nq, nh, d)
            k_c = padded(a.k_norm(heads(a.k(n_c))), t_crop, t_in, C, bm)       # (C, bm, nh, d)
            v_c = padded(heads(a.v(n_c)), t_crop, t_in, C, bm)
            z_lr = torch.einsum("sqhd,slhd->sqhl", padded(q, t_s, t_q, S, Qm), k_lr) * a.scale
            z_own = torch.einsum("nhd,nbhd->nhb", q, k_c[t_crop]) * a.scale
            z = torch.cat([z_lr.float().masked_fill(~see[:, :, None], float("-inf")),
                           padded(z_own.float(), t_s, t_q, S, Qm).masked_fill(
                               ~own_p[:, :, None], float("-inf"))], -1)
            w = z.softmax(-1).to(q.dtype)                                      # (S, Qm, nh, Lm + bm)
            out = (torch.einsum("sqhl,slhd->sqhd", w[..., :Lm], v_lr)[t_s, t_q]
                   + torch.einsum("nhb,nbhd->nhd", w[..., Lm:][t_s, t_q], v_c[t_crop]))
            x_c = x_c + blk.drop_path(blk.ls1(a.proj_drop(a.proj(out.reshape(len(out), -1)))))
            x_c = x_c + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(x_c))))
        return list(torch.split(x_c, [len(p.x) for p, _ in crop_sets]))


# -- selectors: where to inject ---------------------------------------------------------------
def _query_key_encoding(py, px, slot, month, dim: int = 32) -> torch.Tensor:
    """(..., 2 * dim // 2 + dim // 2 + 2) sin-cos of pixel position (dim), time slot (dim / 2)
    and calendar month (cyclic, 2): what a query or key needs to know about where and when
    it is. Positions in pixels, so ps4 queries and ps16 keys share one frame."""
    shape = py.shape
    pos = sincos_2d(py.reshape(-1), px.reshape(-1), dim)
    sl = get_1d_sincos_pos_encoding(slot.reshape(-1).float(), dim // 2)
    ang = 2 * torch.pi * month.reshape(-1).float() / 12
    mo = torch.stack([ang.sin(), ang.cos()], -1)
    return torch.cat([pos, sl, mo], -1).reshape(*shape, -1)


ENC_DIM = 32 + 16 + 2


class CrossAttnSelector(nn.Module):
    """Scores every (step, ps4 cell) by cross-attention from the step to the low-res sequence:

        query  (step t, cell)   [e_t, e_t - mean e over its season, encoding(position, slot,
                                month)]  -- e_t = the step's ps4 patch embedding
        keys   (season, token)  [lr_L token, encoding(position, slot, month)]
        h = q + Attn(q, K, V);  [h = h + SelfAttn over the steps of the same cell];
        h = h + MLP(h);  score = Linear(h)

    Nothing depends on the number of steps per season, so months and single dates alike.
    The optional self-attention across steps (`temporal`) lets a date be scored against the
    dates around it (abrupt change) and lets the scores of one cell coordinate. Inputs are
    detached: the selector learns from its own objective, never pushes on the backbone."""

    def __init__(self, dim: int, hidden: int = 64, heads: int = 4, temporal: bool = False):
        super().__init__()
        self.q_in = nn.Sequential(nn.LayerNorm(2 * dim + ENC_DIM), nn.Linear(2 * dim + ENC_DIM, hidden))
        self.kv_in = nn.Sequential(nn.LayerNorm(dim + ENC_DIM), nn.Linear(dim + ENC_DIM, hidden))
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.temporal = nn.MultiheadAttention(hidden, heads, batch_first=True) if temporal else None
        self.n_t = nn.LayerNorm(hidden) if temporal else None
        self.n_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, e, e_bar, lr, season, present, lr_valid, month_q, month_k):
        """e, e_bar (B, T, H, W, D); lr (B, G, h, w, D); season, present, month_q (B, T);
        lr_valid, month_k (B, G) -> scores (B, T, H, W)."""
        B, T, H, W, D = e.shape
        G, h, w = lr.shape[1:4]
        dev = e.device
        cy, cx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                                indexing="ij")
        r = 4 * H // h                                                        # key patch, px
        ky, kx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev),
                                indexing="ij")
        enc_q = _query_key_encoding(
            (4 * cy + 2).expand(B, T, H, W), (4 * cx + 2).expand(B, T, H, W),
            season.clamp(min=0)[:, :, None, None].expand(B, T, H, W),
            month_q[:, :, None, None].expand(B, T, H, W))
        enc_k = _query_key_encoding(
            (r * ky + r // 2).expand(B, G, h, w), (r * kx + r // 2).expand(B, G, h, w),
            torch.arange(G, device=dev)[None, :, None, None].expand(B, G, h, w),
            month_k[:, :, None, None].expand(B, G, h, w))
        q = self.q_in(torch.cat([e, e - e_bar, enc_q], -1).detach()).reshape(B, T * H * W, -1)
        kv = self.kv_in(torch.cat([lr, enc_k], -1).detach()).reshape(B, G * h * w, -1)
        pad = ~lr_valid[:, :, None, None].expand(B, G, h, w).reshape(B, -1)
        a, _ = self.cross(q, kv, kv, key_padding_mask=pad, need_weights=False)
        x = (q + a.nan_to_num(0.0)).reshape(B, T, H * W, -1)
        if self.temporal is not None:
            xt = self.n_t(x).permute(0, 2, 1, 3).reshape(B * H * W, T, -1)
            m = ~present[:, None, :].expand(B, H * W, T).reshape(B * H * W, T)
            m = m & ~m.all(1, keepdim=True)         # a cell with no step: attend to all (unused)
            a, _ = self.temporal(xt, xt, xt, key_padding_mask=m, need_weights=False)
            x = x + a.reshape(B, H * W, T, -1).permute(0, 2, 1, 3)
        x = x + self.mlp(self.n_mlp(x))
        return self.out(x)[..., 0].reshape(B, T, H, W)


# -- readout: irregular tokens -> a regular (step, cell) grid ---------------------------------
class EncCrossSelector(nn.Module):
    """The value predictor as attention, on tokens that carry OlmoEarth's OWN encodings (no
    one-hots, no extra sin-cos): a candidate (month t, ps4 cell) is its patch embedding with
    the encodings a crop of it would get (time slot, calendar month, position).

        q    = Linear([e, e - mean e over its season])
        x    = q + CrossAttn(q; keys = values = the tile's season tokens after blocks [0, L))
        x    = x + SelfAttn over the months of the same cell          (temporal)
        x    = x + SelfAttn over the cells of the same month          (spatial)
        x    = x + MLP(x);  score = Linear(x)

    Inputs are used as given (the caller detaches them)."""

    def __init__(self, dim: int, hidden: int = 256, heads: int = 4, temporal: bool = False,
                 spatial: bool = False, ff: int = 2, coarse: int = 0):
        """ff: feed-forward width / hidden. coarse: that many full self-attention layers
        (pre-norm, feed-forward = hidden) over the candidates' means per (month, low-res
        cell) -- T * h * w tokens, space and time together -- added back to every candidate
        of the cell (after the cross-attention)."""
        super().__init__()
        self.coarse = nn.ModuleList([nn.TransformerEncoderLayer(
            hidden, heads, hidden, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True) for _ in range(coarse)])
        self.q_in = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, hidden))
        self.kv_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.n_q = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.temporal = nn.MultiheadAttention(hidden, heads, batch_first=True) if temporal else None
        self.n_t = nn.LayerNorm(hidden) if temporal else None
        self.spatial = nn.MultiheadAttention(hidden, heads, batch_first=True) if spatial else None
        self.n_s = nn.LayerNorm(hidden) if spatial else None
        self.n_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, ff * hidden), nn.GELU(), nn.Linear(ff * hidden, hidden))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, e, e_bar, lr, present, lr_valid):
        """e, e_bar (B, T, H, W, D); lr (B, G, h, w, D); present (B, T); lr_valid (B, G)
        -> scores (B, T, H, W)."""
        B, T, H, W, D = e.shape
        G, h, w = lr.shape[1:4]
        q = self.q_in(torch.cat([e, e - e_bar], -1)).reshape(B, T * H * W, -1)
        kv = self.kv_in(lr).reshape(B, G * h * w, -1)
        pad = ~lr_valid[:, :, None, None].expand(B, G, h, w).reshape(B, -1)
        a, _ = self.cross(self.n_q(q), kv, kv, key_padding_mask=pad, need_weights=False)
        x = (q + a.nan_to_num(0.0)).reshape(B, T, H * W, -1)
        if len(self.coarse):
            r = H // h
            c = x.reshape(B, T, h, r, w, r, -1).mean((3, 5)).reshape(B, T * h * w, -1)
            m = ~present[:, :, None].expand(B, T, h * w).reshape(B, -1)
            m = m & ~m.all(1, keepdim=True)
            c0 = c
            for layer in self.coarse:
                c = layer(c, src_key_padding_mask=m)
            d = (c - c0).reshape(B, T, h, 1, w, 1, -1).expand(B, T, h, r, w, r, c.shape[-1])
            x = x + d.reshape(B, T, H * W, -1)
        if self.temporal is not None:
            xt = self.n_t(x).permute(0, 2, 1, 3).reshape(B * H * W, T, -1)
            m = ~present[:, None, :].expand(B, H * W, T).reshape(B * H * W, T)
            m = m & ~m.all(1, keepdim=True)         # a cell with no step: attend to all (unused)
            a, _ = self.temporal(xt, xt, xt, key_padding_mask=m, need_weights=False)
            x = x + a.reshape(B, H * W, T, -1).permute(0, 2, 1, 3)
        if self.spatial is not None:
            xs = self.n_s(x).reshape(B * T, H * W, -1)
            a, _ = self.spatial(xs, xs, xs, need_weights=False)
            x = x + a.reshape(B, T, H * W, -1)
        x = x + self.mlp(self.n_mlp(x))
        return self.out(x)[..., 0].reshape(B, T, H, W)


class GaussianSplatReadout(nn.Module):
    """Read the final tokens (coarse season tokens + sparse crop tokens) back onto the regular
    output grid of (fine step, ps4 cell), by kernel-weighted blending.

    Every token i carries a feature f_i (D), a place (y_i, x_i) in ps4-cell units, a time t_i
    in fine-step units, and a kernel it predicts from its own feature (one Linear, `_kernel`):

        sig_a_i, sig_b_i   spatial std along two perpendicular axes (ps4 cells)
        theta_i            rotation of those axes in the (y, x) plane -- space only; there is
                           no space-time rotation, and the time kernel is symmetric
        sig_t_i            temporal std (fine steps)
        log_w_i            the token's log-weight (its overall pull)

    The output at step t, cell p is a normalized blend of all tokens of that sample:

        a_i(t, p) = log_w_i                                         weight
                    - (u^2 / sig_a_i^2 + v^2 / sig_b_i^2) / 2       spatial Gaussian; (u, v) =
                                                                    p - pos_i rotated by theta_i
                    - (t - t_i)^2 / (2 sig_t_i^2)                   temporal Gaussian (`temporal`)
                    - gamma |phi(e_t(p)) - phi(e_i)|^2              guidance (`guided`)
        out(t, p) = sum_i softmax_i(a_i(t, p)) f_i                  = sum_i w_i f_i / sum_j w_j,
                                                                      w_i = exp(a_i)

    `temporal` False: no time term; step t may read only the season tokens of ITS season and
    the crops of ITS OWN step (the same tokens as the bilinear readout).
    `guided` (joint-bilateral): e_t(p) is the raw ps4 patch embedding of step t at cell p
    (OlmoEarth's patch-embed layer only, no transformer). A crop's guide is e at its own
    (step, cell); a season token's is e averaged over the present steps of its season and its
    4x4 footprint. phi = LayerNorm + Linear(D -> guide_dim), gamma = exp(log_gamma) >= 0, both
    learned. A token then pulls an output cell only as much as their raw appearance agrees,
    which is what can make the blend asymmetric in time (a post-flood step matches post-flood
    tokens, not pre-flood ones) and sharp across edges in space.
    Gate: a crop picked by a learned selector with score s has log sigmoid(s) added to its
    log_w (w multiplied by sigmoid(s)), so the selector gets a gradient; as sigmoid(s) -> 0 the
    crop fades out and the season token under it takes over.

    Init (params Linear zero, per-type bias) is close to bilinear + overwrite: season tokens
    broad (sig 2 cells = 8 px, log_w 0), crops tight (0.4 cells) and heavy (log_w 4).

    Cost: the weights a are materialized for every (sample, step, cell, token): (B, T, Q, K)
    floats, K = season tokens + the most crops any sample has. Tokens outside the window
    (`temporal` False) are masked, not skipped, so they cost memory too.

    Shape letters used below:
        B    samples                          D    feature dim
        T    fine steps (months, or dates)    NS   coarse season slots of the low-res sequence
        h, w low-res grid (4 x 4, ps16)        H, W output grid (16 x 16, ps4); r = H / h = 4
        Q    output cells per step = H * W    L    season tokens per sample = NS * h * w
        N    crops in the whole batch         M    most crops in any one sample
        K    token slots per sample = L + M   C    guide_dim
    """

    # kernel params in order: log sig_a, log sig_b, theta, log sig_t, log_w
    INIT = {"lr": [math.log(2.0), math.log(2.0), 0.0, 0.0, 0.0],
            "hr": [math.log(0.4), math.log(0.4), 0.0, math.log(0.5), 4.0]}

    def __init__(self, dim: int, temporal: bool = False, guided: bool = False, guide_dim: int = 16,
                 overlap: bool = False):
        """overlap: output cell p reads ONLY the tokens whose footprint contains it (the ps16
        season tokens covering p, the crops AT p) and there is no spatial Gaussian -- a
        time-only splat, no spatial-neighbour context (compare with the attention pools)."""
        super().__init__()
        if overlap and guided:
            raise ValueError("overlap is the time-only splat; no guidance")
        self.temporal, self.guided, self.overlap = temporal, guided, overlap
        self.params = nn.Linear(dim, 5)                   # feature -> kernel params (zero init)
        nn.init.zeros_(self.params.weight)
        nn.init.zeros_(self.params.bias)
        self.type_bias = nn.Parameter(torch.tensor([self.INIT["lr"], self.INIT["hr"]]))  # (2, 5)
        if guided:
            self.phi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, guide_dim))
            self.log_gamma = nn.Parameter(torch.tensor(-3.0))    # gamma = e^-3 ~ 0.05 at init

    def _kernel(self, f, kind: int):
        """f (..., D), kind 0 = season token, 1 = crop -> sig_a, sig_b, theta, sig_t, log_w,
        each (...)."""
        p = self.params(f.float()) + self.type_bias[kind]
        sig_a, sig_b = p[..., 0].clamp(-4, 4).exp(), p[..., 1].clamp(-4, 4).exp()
        return sig_a, sig_b, p[..., 2], p[..., 3].clamp(-4, 4).exp(), p[..., 4]

    def forward(self, lr, lr_valid, season, present, hr, hr_idx, hr_logw=None, e=None):
        """
        lr        (B, NS, h, w, D)   final season tokens (after the last block + norm)
        lr_valid  (B, NS) bool       season slot holds a real season (some present step in it)
        season    (B, T) long        season slot of each fine step; -1 = in no season
        present   (B, T) bool        fine step is real (not batch padding)
        hr        (N, D)             final crop tokens of the whole batch, grouped by sample
        hr_idx    4 x (N,) long      (sample b, step t, row y, col x) of each crop on (T, H, W)
        hr_logw   (N,) or None       extra log-weight per crop: log sigmoid(selector score)
        e         (B, T, H, W, D) or None   guidance (`guided`): raw ps4 patch embeddings
        -> (B, T, H, W, D)
        """
        B, NS, h, w, D = lr.shape
        T = season.shape[1]
        r = 4
        H, W = r * h, r * w
        Q, L = H * W, NS * h * w
        dev = lr.device
        steps = torch.arange(T, device=dev, dtype=torch.float)                  # (T,)
        real = (present & (season >= 0)).float()                                 # (B, T)

        # -- 1. season tokens, slots [0, L) in (season, row, col) order
        cell_y, cell_x = torch.meshgrid((torch.arange(h, device=dev) + 0.5) * r,  # (h, w) each:
                                        (torch.arange(w, device=dev) + 0.5) * r,  # ps16 centres
                                        indexing="ij")                           # in ps4 cells
        in_season = torch.nn.functional.one_hot(season.clamp(min=0), NS).float() * real[..., None]
        #   (B, T, NS): step t belongs to season s (real steps only)
        season_time = ((in_season * steps[None, :, None]).sum(1)
                       / in_season.sum(1).clamp(min=1))                          # (B, NS) mean step
        lr_feat = lr.reshape(B, L, D)
        lr_kern = self._kernel(lr_feat, 0)                                       # 5 x (B, L)

        # -- 2. crops: the batch's N crops go to slots [L, L + M) of their own sample
        b, t, y, x = hr_idx                                                      # (N,) each
        N = len(b)
        n_crops = torch.bincount(b, minlength=B)                                 # (B,)
        M = int(n_crops.max()) if N else 0
        K = L + M
        first = torch.cumsum(n_crops, 0) - n_crops                               # (B,) 1st crop of b
        col = L + torch.arange(N, device=dev) - first[b]                         # (N,) slot in [L, K)

        # -- 3. one (B, K) table of every token slot; slots past a sample's crops stay invalid
        feat = lr_feat.new_zeros(B, K, D)
        pos_y, pos_x, tok_time = (lr_feat.new_zeros(B, K) for _ in range(3))     # ps4 cells, steps
        valid = torch.zeros(B, K, dtype=torch.bool, device=dev)
        kern = [lr_feat.new_zeros(B, K) for _ in range(5)]                        # sig_a .. log_w
        owner = torch.full((B, K), -1, dtype=torch.long, device=dev)  # season (season token) or
        is_crop = torch.zeros(B, K, dtype=torch.bool, device=dev)     # step (crop) it belongs to
        is_crop[:, L:] = True
        feat[:, :L] = lr_feat
        pos_y[:, :L] = cell_y.reshape(1, 1, h * w).expand(B, NS, h * w).reshape(B, L)
        pos_x[:, :L] = cell_x.reshape(1, 1, h * w).expand(B, NS, h * w).reshape(B, L)
        tok_time[:, :L] = season_time[:, :, None].expand(B, NS, h * w).reshape(B, L)
        valid[:, :L] = lr_valid[:, :, None].expand(B, NS, h * w).reshape(B, L)
        for j in range(5):
            kern[j][:, :L] = lr_kern[j]
        owner[:, :L] = torch.arange(NS, device=dev)[None, :, None].expand(B, NS, h * w).reshape(B, L)
        if N:
            feat[b, col] = hr.to(feat.dtype)
            pos_y[b, col], pos_x[b, col] = y.float() + 0.5, x.float() + 0.5      # cell centres
            tok_time[b, col] = t.float()
            valid[b, col] = True
            hr_kern = self._kernel(hr, 1)                                        # 5 x (N,)
            for j in range(5):
                kern[j][b, col] = hr_kern[j]
            if hr_logw is not None:
                kern[4][b, col] = kern[4][b, col] + hr_logw.float()              # gate
            owner[b, col] = t
        sig_a, sig_b, theta, sig_t, log_w = kern                                 # (B, K) each

        # -- 4. log-weight a of every token at every output (step, cell): (B, T, Q, K)
        out_y, out_x = torch.meshgrid(torch.arange(H, device=dev) + 0.5,        # (H, W) output
                                      torch.arange(W, device=dev) + 0.5,        # cell centres
                                      indexing="ij")
        dy = out_y.reshape(1, 1, Q, 1) - pos_y[:, None, None, :]                 # (B, 1, Q, K)
        dx = out_x.reshape(1, 1, Q, 1) - pos_x[:, None, None, :]
        cos, sin = theta.cos()[:, None, None, :], theta.sin()[:, None, None, :]
        u, v = cos * dx + sin * dy, -sin * dx + cos * dy                         # rotated offset
        allowed = valid[:, None, None, :].expand(B, T, 1, K)                     # (B, T, 1, K)
        if self.overlap:
            # footprint test: a season token covers its 4x4 ps4 cells, a crop its own cell
            size = torch.where(is_crop, 1.0, float(r))[:, None, None, :]         # (B, 1, 1, K)
            covers = (((out_y.reshape(1, 1, Q, 1) // size) == (pos_y[:, None, None, :] // size))
                      & ((out_x.reshape(1, 1, Q, 1) // size) == (pos_x[:, None, None, :] // size)))
            a = log_w[:, None, None, :].expand(B, 1, Q, K).clone()               # (B, 1, Q, K)
            allowed = allowed & covers                                            # (B, 1, Q, K)
        else:
            a = (log_w[:, None, None, :] - 0.5 * (u / sig_a[:, None, None, :]) ** 2
                 - 0.5 * (v / sig_b[:, None, None, :]) ** 2)                     # (B, 1, Q, K)
        a = a.expand(B, T, Q, K)
        if self.temporal:
            dt = steps[None, :, None] - tok_time[:, None, :]                     # (B, T, K)
            a = a - (0.5 * (dt / sig_t[:, None, :]) ** 2)[:, :, None, :]
        else:                          # a step reads its own season's tokens and its own crops
            mine = torch.where(is_crop[:, None, :], owner[:, None, :] == steps.long()[None, :, None],
                               owner[:, None, :] == season[:, :, None])          # (B, T, K)
            allowed = allowed & mine[:, :, None, :]
        if self.guided and e is not None:
            guide_out = self.phi(e)                                              # (B, T, H, W, C)
            e_season = (torch.einsum("bts,bthwd->bshwd", in_season, e)           # (B, NS, H, W, D)
                        / in_season.sum(1).clamp(min=1)[:, :, None, None, None])
            e_season = torch.nn.functional.avg_pool2d(                           # (B, NS, h, w, D)
                e_season.flatten(0, 1).permute(0, 3, 1, 2), r).permute(0, 2, 3, 1)
            guide_tok = guide_out.new_zeros(B, K, guide_out.shape[-1])           # (B, K, C)
            guide_tok[:, :L] = self.phi(e_season.reshape(B, L, D))
            if N:
                guide_tok[b, col] = guide_out[b, t, y, x]
            guide_out = guide_out.reshape(B, T, Q, -1)                           # (B, T, Q, C)
            dist2 = ((guide_out ** 2).sum(-1, keepdim=True)                      # (B, T, Q, K)
                     + (guide_tok ** 2).sum(-1)[:, None, None, :]
                     - 2 * torch.einsum("btqc,bkc->btqk", guide_out, guide_tok))
            a = a - self.log_gamma.exp() * dist2.clamp(min=0)

        # -- 5. normalize over tokens (w_i / sum_j w_j) and blend the features
        a = a.masked_fill(~allowed, float("-inf"))
        wgt = torch.softmax(a, -1).nan_to_num(0.0)                               # (B, T, Q, K);
        out = torch.einsum("btqk,bkd->btqd", wgt, feat.float())                  # no token -> 0
        return out.reshape(B, T, H, W, D)


class CellAttnReadout(nn.Module):
    """Sparse readout for heads that pool over time (time_mean tasks): each output ps4 cell
    attention-pools the tokens NEAR it into one vector, skipping the dense (step, cell) grid.

    Cell p's candidate tokens (a different number per cell, padded per cell):
        season tokens   the ps16 cell containing p and its 8 neighbours, every used season
                        (9 * NS candidates; off-grid neighbours and unused seasons masked)
        crops           every crop whose ps4 cell is within `radius` cells of p (Chebyshev),
                        any step (radius 1: the 3x3 ps4 cells around p)

    Each token i predicts, from its feature (as GaussianSplatReadout):
        sig_a_i, sig_b_i, theta_i   anisotropic spatial Gaussian (ps4 cells)
        log_w_i                     log-weight (a selector's gate adds log sigmoid(score))
    and a content score per head, score_ih = Linear(LayerNorm(f_i))_h. Pooling, per head h:

        a_ih(p) = score_ih + log_w_i - (u^2 / sig_a_i^2 + v^2 / sig_b_i^2) / 2
        out_h(p) = sum_i softmax_i(a_ih(p)) f_i[head h's channel group]

    so the Gaussian says WHERE a token counts, the score says HOW MUCH it matters for the
    task (the job of the temporal attention pool), both in one softmax. There is no time
    kernel: the token features carry their own time; the pool picks among them.

    Init: scores zero; season tokens sig 2 cells, log_w log(3) (a season stands for ~3
    months), crops sig 0.4 cells, log_w 0 (one month) -- roughly the time mean of the bilinear
    + overwrite readout.

    Cost: (B, Q, Kc, D) gathered features, Kc = 9 * NS + the most crops near any one cell.

    Shape letters: B samples, NS season slots, h, w low-res grid (ps16), H, W output grid
    (ps4), r = H / h = 4, Q = H * W cells, N crops in the batch, D feature dim, nh heads.
    """

    INIT = {"lr": [math.log(2.0), math.log(2.0), 0.0, math.log(3.0)],
            "hr": [math.log(0.4), math.log(0.4), 0.0, 0.0]}

    def __init__(self, dim: int, heads: int = 4, radius: int = 1, overlap: bool = False):
        """overlap: cell p pools ONLY the tokens whose footprint contains it -- the ps16 season
        token covering p (every season) and the crops at p (any step) -- with no spatial
        Gaussian: logit = content score + log_w. No spatial-neighbour context."""
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.overlap = overlap
        self.heads, self.radius = heads, 0 if overlap else radius
        self.params = nn.Linear(dim, 4)                   # feature -> sig_a, sig_b, theta, log_w
        nn.init.zeros_(self.params.weight)
        nn.init.zeros_(self.params.bias)
        self.type_bias = nn.Parameter(torch.tensor([self.INIT["lr"], self.INIT["hr"]]))  # (2, 4)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, heads))
        nn.init.zeros_(self.score[1].weight)
        nn.init.zeros_(self.score[1].bias)

    def _token_terms(self, f, kind: int):
        """f (..., D) -> kernel (..., 4) [sig_a, sig_b, theta, log_w] and scores (..., nh)."""
        p = self.params(f.float()) + self.type_bias[kind]
        kern = torch.stack([p[..., 0].clamp(-4, 4).exp(), p[..., 1].clamp(-4, 4).exp(),
                            p[..., 2], p[..., 3]], -1)
        return kern, self.score(f.float())

    def forward(self, lr, lr_valid, hr, hr_idx, hr_logw=None):
        """
        lr        (B, NS, h, w, D)  final season tokens
        lr_valid  (B, NS) bool      season slot used by some present step
        hr        (N, D)            final crop tokens of the whole batch
        hr_idx    4 x (N,) long     (sample b, step t, row y, col x); t is unused (the pool is
                                    over tokens, not steps)
        hr_logw   (N,) or None      extra crop log-weight (log sigmoid of the selector score)
        -> (B, H, W, D) one pooled vector per ps4 cell
        """
        B, NS, h, w, D = lr.shape
        r = 4
        H, W = r * h, r * w
        Q, nh = H * W, self.heads
        dev = lr.device
        cell_y, cell_x = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                                        indexing="ij")
        cell_y, cell_x = cell_y.reshape(Q), cell_x.reshape(Q)                    # (Q,) each

        # -- season candidates: (Q, S) ps16 cells per output cell, then x NS seasons;
        #    S = 9 (the covering cell + its 8 neighbours) or 1 (overlap: the covering cell)
        off = torch.tensor([0] if self.overlap else [-1, 0, 1], device=dev)
        o = len(off)
        S = o * o
        ny = (cell_y // r)[:, None, None] + off[None, :, None]                   # (Q, o, 1)
        nx = (cell_x // r)[:, None, None] + off[None, None, :]                   # (Q, 1, o)
        ny, nx = ny.expand(Q, o, o).reshape(Q, S), nx.expand(Q, o, o).reshape(Q, S)
        on_grid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)                    # (Q, S)
        ny, nx = ny.clamp(0, h - 1), nx.clamp(0, w - 1)
        lr_kern, lr_score = self._token_terms(lr, 0)          # (B, NS, h, w, 4), (.., nh)
        s_feat = lr.float()[:, :, ny, nx]                                        # (B, NS, Q, S, D)
        s_kern = lr_kern[:, :, ny, nx]                                           # (B, NS, Q, S, 4)
        s_score = lr_score[:, :, ny, nx]                                         # (B, NS, Q, S, nh)
        s_pos = torch.stack([ny, nx], -1).float() * r + r / 2                    # (Q, S, 2) centres
        s_ok = on_grid[None, None] & lr_valid[:, :, None, None]                 # (B, NS, Q, S)
        # -> (B, Q, NS * S, ...)
        s_feat = s_feat.permute(0, 2, 1, 3, 4).reshape(B, Q, NS * S, D)
        s_kern = s_kern.permute(0, 2, 1, 3, 4).reshape(B, Q, NS * S, 4)
        s_score = s_score.permute(0, 2, 1, 3, 4).reshape(B, Q, NS * S, nh)
        s_pos = s_pos[None, :, None].expand(B, Q, NS, S, 2).reshape(B, Q, NS * S, 2)
        s_ok = s_ok.permute(0, 2, 1, 3).reshape(B, Q, NS * S)
        feats, kerns, scores, pos, ok = [s_feat], [s_kern], [s_score], [s_pos], [s_ok]

        # -- crop candidates: each crop listed at every cell within `radius`, padded per cell
        b, _, y, x = hr_idx
        N = len(b)
        if N:
            c_kern, c_score = self._token_terms(hr, 1)                           # (N, 4), (N, nh)
            if hr_logw is not None:
                c_kern = torch.cat([c_kern[:, :3], c_kern[:, 3:] + hr_logw.float()[:, None]], 1)
            R = self.radius
            d = torch.arange(-R, R + 1, device=dev)
            dy, dx = torch.meshgrid(d, d, indexing="ij")
            dy, dx = dy.reshape(-1), dx.reshape(-1)                               # (O,) offsets
            ty, tx = y[:, None] + dy[None], x[:, None] + dx[None]                # (N, O) target
            inside = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
            crop = torch.arange(N, device=dev)[:, None].expand_as(ty)[inside]    # (E,) entries
            cell = (b[:, None] * Q + ty * W + tx)[inside]                        # (E,) b*Q + q
            order = torch.argsort(cell, stable=True)
            crop, cell = crop[order], cell[order]
            count = torch.bincount(cell, minlength=B * Q)                        # (B*Q,)
            M = int(count.max())
            slot = torch.arange(len(cell), device=dev) - (torch.cumsum(count, 0) - count)[cell]
            idx = torch.full((B * Q, M), -1, dtype=torch.long, device=dev)       # crop per slot
            idx[cell, slot] = crop
            idx = idx.view(B, Q, M)
            c_ok = idx >= 0
            safe = idx.clamp(min=0)
            feats.append(hr.float()[safe])                                       # (B, Q, M, D)
            kerns.append(c_kern[safe])
            scores.append(c_score[safe])
            pos.append(torch.stack([y, x], -1).float()[safe] + 0.5)              # cell centres
            ok.append(c_ok)
        feat = torch.cat(feats, 2)                                               # (B, Q, Kc, D)
        kern = torch.cat(kerns, 2)                                               # (B, Q, Kc, 4)
        score = torch.cat(scores, 2)                                             # (B, Q, Kc, nh)
        pos = torch.cat(pos, 2)                                                  # (B, Q, Kc, 2)
        ok = torch.cat(ok, 2)                                                    # (B, Q, Kc)

        # -- pooling logits (B, Q, Kc, nh) and the per-head weighted sum
        centre = torch.stack([cell_y, cell_x], -1).float() + 0.5                 # (Q, 2)
        dy_, dx_ = (centre[None, :, None] - pos).unbind(-1)                      # (B, Q, Kc)
        sig_a, sig_b, theta, log_w = kern.unbind(-1)
        cos, sin = theta.cos(), theta.sin()
        u, v = cos * dx_ + sin * dy_, -sin * dx_ + cos * dy_
        a = log_w if self.overlap else (log_w - 0.5 * (u / sig_a) ** 2
                                        - 0.5 * (v / sig_b) ** 2)                # (B, Q, Kc)
        a = (a[..., None] + score).masked_fill(~ok[..., None], float("-inf"))   # (B, Q, Kc, nh)
        wgt = torch.softmax(a, 2).nan_to_num(0.0)
        out = torch.einsum("bqkn,bqknc->bqnc", wgt, feat.view(B, Q, -1, nh, D // nh))
        return out.reshape(B, H, W, D)


class TimeSetReadout(nn.Module):
    """Overlap-only, TIME-CONTINUOUS readout: each ps4 cell reads only the tokens overlapping
    it, weighted by where they sit in time relative to a query time -- so any time slice can be
    read (for change detection: the difference of two slices), not only a time mean.

    Cell p's candidates (padded per cell):
        low-res tokens  the ps16 token covering p, every used low-res slot (NS: seasons, or
                        months for a series), at time tau = its slot index
        crops           every crop AT p, at time tau = its step's slot + its fractional
                        position inside the slot (Collate's extra["frac"])
    Query time of step t: its slot + its fraction (the same units: slots).

    Each token i predicts from its feature (Linear, zero init + a per-type bias):
        log_w_i   log-weight;  sig_i   temporal std (slots)             (mode "gauss")
    and the weight of token i at query time t, softmax over the cell's candidates of
        gauss   a_i(t) = log_w_i - (t - tau_i)^2 / (2 sig_i^2)          (time-only splat)
        query   a_i(t) = log_w_i + q(t) . k_i / sqrt(dk),               (time-queried attention)
                q(t) = Linear(phi(t)),  k_i = Linear(f_i) + Linear(phi(tau_i)),
                phi = sin/cos of t at periods 0.5 .. 32 slots
    No spatial term. Init (gauss): low-res sig 0.5 slot, log_w 0; crops sig 0.25, log_w 2 (a
    crop dominates its own time, the slot tokens fill in between).

    weights(...) -> wgt (B, T, Q, K), feat (B, Q, K, D), and what the native readout needs;
    forward(...) -> (B, T, H, W, D) features at every step (any head: time mean, pair diff).

    Shape letters: B samples, NS low-res slots, h = w = 4 ps16 cells, H = W = 16 ps4 cells,
    Q = H * W, T steps, N crops, M the most crops at one cell, K = NS + M, D features.
    """

    PERIODS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    def __init__(self, dim: int, mode: str = "gauss", dk: int = 32):
        super().__init__()
        if mode not in ("gauss", "query"):
            raise ValueError(f"mode {mode!r} not in ('gauss', 'query')")
        self.mode, self.dk = mode, dk
        self.kern = nn.Linear(dim, 2)                     # feature -> [log sig, log_w]
        nn.init.zeros_(self.kern.weight)
        nn.init.zeros_(self.kern.bias)
        self.type_bias = nn.Parameter(torch.tensor([[math.log(0.5), 0.0],
                                                    [math.log(0.25), 2.0]]))
        if mode == "query":
            nf = 2 * len(self.PERIODS)
            self.q = nn.Linear(nf, dk)
            self.k_feat = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dk))
            self.k_time = nn.Linear(nf, dk)

    def _phi(self, t):
        w = 2 * math.pi / torch.tensor(self.PERIODS, device=t.device)
        a = t.float()[..., None] * w
        return torch.cat([a.sin(), a.cos()], -1)

    def weights(self, lr, lr_valid, season, present, frac, hr, hr_idx, hr_logw=None,
                t_query=None, q_valid=None):
        """
        lr        (B, NS, h, w, D)  final low-res tokens
        lr_valid  (B, NS) bool      slot used by some present step
        season    (B, T) long       slot of each step (-1: none);  present (B, T) bool
        frac      (B, T) float      each step's position inside its slot, in [-0.5, 0.5)
        hr        (N, D)            final crop tokens;  hr_idx 4 x (N,) (b, t, y, x) on the ps4 grid
        hr_logw   (N,) or None      crop gate log-weight
        t_query   (B, n) float or None   read at THESE times (slot units) instead of the T
                  steps' own times;  q_valid (B, n) bool: which of them are real
        -> wgt (B, T, Q, K) (0 where nothing is valid), feat (B, Q, K, D),
           crop_of (B, Q, M) long crop index per crop slot (-1 empty) or None
        """
        B, NS, h, w, D = lr.shape
        T = season.shape[1]
        r = 4
        H, W = h * r, w * r
        Q = H * W
        dev = lr.device
        cy, cx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                                indexing="ij")
        cy, cx = cy.reshape(Q), cx.reshape(Q)
        # -- low-res candidates: the covering ps16 token of every slot
        l_feat = lr.float()[:, :, cy // r, cx // r].transpose(1, 2)               # (B, Q, NS, D)
        l_kern = self.kern(lr.float()) + self.type_bias[0]                       # (B, NS, h, w, 2)
        l_kern = l_kern[:, :, cy // r, cx // r].transpose(1, 2)                  # (B, Q, NS, 2)
        l_tau = torch.arange(NS, device=dev).float()[None, None].expand(B, Q, NS)
        l_ok = lr_valid[:, None, :].expand(B, Q, NS)
        feats, kerns, taus, oks = [l_feat], [l_kern], [l_tau], [l_ok]
        crop_of = None
        b, t, y, x = hr_idx
        N = len(b)
        if N:
            c_kern = self.kern(hr.float()) + self.type_bias[1]                   # (N, 2)
            if hr_logw is not None:
                c_kern = torch.stack([c_kern[:, 0], c_kern[:, 1] + hr_logw.float()], -1)
            c_tau = season[b, t].clamp(min=0).float() + frac[b, t].float()      # (N,)
            cell = b * Q + y * W + x
            order = torch.argsort(cell, stable=True)
            cs = cell[order]
            count = torch.bincount(cs, minlength=B * Q)
            M = int(count.max())
            slot = torch.arange(N, device=dev) - (torch.cumsum(count, 0) - count)[cs]
            crop_of = torch.full((B * Q, M), -1, dtype=torch.long, device=dev)
            crop_of[cs, slot] = order
            crop_of = crop_of.view(B, Q, M)
            ok = crop_of >= 0
            safe = crop_of.clamp(min=0)
            feats.append(hr.float()[safe])                                       # (B, Q, M, D)
            kerns.append(c_kern[safe])
            taus.append(c_tau[safe])
            oks.append(ok)
        feat = torch.cat(feats, 2)                                               # (B, Q, K, D)
        kern = torch.cat(kerns, 2)                                               # (B, Q, K, 2)
        tau = torch.cat(taus, 2)                                                 # (B, Q, K)
        ok = torch.cat(oks, 2)
        log_w = kern[..., 1]
        t_q = season.clamp(min=0).float() + frac.float()                         # (B, T)
        q_ok = present & (season >= 0)
        if t_query is not None:
            t_q, q_ok = t_query.float(), q_valid
        if self.mode == "gauss":
            sig = kern[..., 0].clamp(-4, 4).exp()
            dt = t_q[:, :, None, None] - tau[:, None]                            # (B, T, Q, K)
            a = log_w[:, None] - 0.5 * (dt / sig[:, None]) ** 2
        else:
            q = self.q(self._phi(t_q))                                           # (B, T, dk)
            k = self.k_feat(feat) + self.k_time(self._phi(tau))                  # (B, Q, K, dk)
            a = log_w[:, None] + torch.einsum("btd,bqkd->btqk", q, k) / math.sqrt(self.dk)
        a = a.masked_fill(~ok[:, None], float("-inf"))
        a = a.masked_fill(~q_ok[:, :, None, None], float("-inf"))
        wgt = torch.softmax(a, -1).nan_to_num(0.0)                                # (B, T, Q, K)
        return wgt, feat, crop_of

    def forward(self, lr, lr_valid, season, present, frac, hr, hr_idx, hr_logw=None):
        """-> (B, T, H, W, D): every step's feature at every ps4 cell (a time slice each)."""
        B, NS, h, w, D = lr.shape
        wgt, feat, _ = self.weights(lr, lr_valid, season, present, frac, hr, hr_idx, hr_logw)
        out = torch.einsum("btqk,bqkd->btqd", wgt, feat)
        return out.reshape(B, season.shape[1], 4 * h, 4 * w, D)
