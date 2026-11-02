"""The decoder: every token through the pa2px head at its own patch size, pooled per pixel over
the tokens covering it (moved verbatim from exp.upsamplers.finetune_inject, --readout
native_set / native_mean)."""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

LR_PS, LOOKUP_PS = 16, 4


class NativeSetReadout(nn.Module):
    """--readout native_set: native_attn_logits as a per-cell SET, so any number of crops of
    any kind (month composites, single dates, several per cell) fit without a month grid --
    the date level's readout with overlapping tokens only.

    Cell p's candidates (padded per cell):
        season tokens   the ps16 token covering p, every used season   (NS)
        crops           every crop whose footprint covers p, any step or date, any level,
                        any patch size (ps4: 1 cell, ps8: 2 x 2 cells, ps16: 4 x 4)  (0 .. M)
    Each candidate i gives
        logits_i(px in p)   the native head: a season token its 16 x 16 block's 4 x 4 px in p,
                            a crop its 16 x 16 map area-downsampled to its own footprint
                            (ps4: 4 x 4 px, ps8: 8 x 8, ps16: kept at 16 x 16), then the
                            4 x 4 px of that footprint lying in p
        a_i                 score(f_i) + prior_i + gate_i
                            score = Linear(LayerNorm(f)) (scalar, zero init)
                            prior = log(present months of the season) for a season token (it
                                    stands for that many months, as in the month grid), 0 for
                                    a crop
                            gate  = log sigmoid(selector score) for a gated crop, else 0
    and out(px in p) = sum_i softmax_i(a_i) logits_i(px). No spatial term: the pool is over
    what overlaps p. At init (score 0) a cell with no crops is the native time mean exactly.

    Shape letters: B samples, NS season slots, h = w = 4 ps16 cells, H = W = 16 ps4 cells,
    Q = H * W, N crops in the batch, M the most crops at one cell, D features, C outputs,
    P = 64 px. forward(...) -> (B, C, P, P) logits.
    """

    def __init__(self, dim: int, lr_px: int = LR_PS, crop_px: int = LOOKUP_PS,
                 base_px: int | None = None, mean: bool = False, out_px: int | None = None,
                 months_prior: bool = True):
        """mean (--readout native_mean): no score -- NO parameters. Weights = the priors alone
        (a season token its present months, a crop 1: per month the mean of the tokens
        covering the pixel, then the mean over months), pooled in PROBABILITY space
        (classification: the weighted mean of each candidate's softmax; returns its log)."""
        super().__init__()
        self.lr_px, self.crop_px, self.mean = lr_px, crop_px, mean
        self.months_prior = months_prior    # False: no prior on the season tokens
        # base_px: px side of a SEASON token (--lr_patch 8: ps8 seasons). The head still writes
        # a lr_px x lr_px map; a smaller token's map is average-pooled to its footprint, the
        # same rule the crops follow.
        self.base_px = base_px or lr_px
        # out_px: season tokens that do not divide the tile (ps12 on 64 px) were embedded from
        # the tile padded to whole tokens (MixedResOE.embed); the map is cropped back, centred
        self.out_px = out_px
        self.no_base = False            # --no_seasons: the season tokens are not candidates
        self.season_drop = 0.0          # --season_drop p: training drops each (sample, season)
        self.score = None if mean else nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        if self.score is not None:
            nn.init.zeros_(self.score[1].weight)
            nn.init.zeros_(self.score[1].bias)

    def _score(self, f):
        return f.new_zeros(f.shape[:-1]) if self.score is None else self.score(f.float())[..., 0]

    def forward(self, linear: nn.Linear, lr_F, season, present, hr, hr_idx, hr_logw=None,
                hr_size=None, parts: bool = False, hr_px=None, hr_fp: int | None = None):
        """
        linear    the head, D -> C * lr_px^2
        lr_F      (B, NS, h, w, D)  final season tokens
        season    (B, T) long       season slot of each month (-1: none)
        present   (B, T) bool       month is real
        hr        (N, D)            final crop tokens (all levels)
        hr_idx    4 x (N,) long     (sample b, step t -- unused, row y, col x) on the ps4 grid:
                                    the top-left ps4 cell of the crop's footprint
        hr_logw   (N,) or None      crop gate log-weight
        hr_size   (N,) long or None footprint side in ps4 cells (1 ps4, 2 ps8, 4 ps16); None: 1
        hr_px     2 x (N,) long or None   top-left PIXEL of each crop when crops lie off the ps4
                                    grid (--jitter, --crop_px): a crop then covers part of
                                    several cells, and the pool runs per PIXEL over what covers
                                    it (wgt (B, Q, K, 16) in parts)
        hr_fp     the off-grid crops' footprint side in px, or (N,) long per crop (--level2
                  ps2 crops beside ps4 crops) (None: crop_px = 4; --crop_px 16:
                  one ps16 token read at its native 16 x 16 px)
        parts     also return (wgt (B, Q, K), candidate logits (B, Q, K, C, 16), the pooled
                  per-cell logits (B, Q, C, 16), the crop index of each crop slot (B, Q, M),
                  -1 empty (a ps8 crop fills 4 cells' slots), or None; the
                  first NS of K are the season tokens) -- for --joint's leave-one-out values
        """
        B, NS, h, w, D = lr_F.shape
        c, r = self.crop_px, self.lr_px // self.crop_px
        bp = self.base_px
        rs = bp // c                                         # ps4 cells per season-token side
        H, W = h * rs, w * rs
        Q, P = H * W, h * bp
        C = linear.out_features // self.lr_px ** 2
        dev = lr_F.device
        # -- season candidates: (B, NS, Q, C, c*c) logits of each season token over cell q
        s_log = linear(lr_F.float())
        if bp < self.lr_px:
            s_log = s_log.reshape(-1, C, self.lr_px, self.lr_px)
            s_log = (F.avg_pool2d(s_log, self.lr_px // bp) if self.lr_px % bp == 0
                     else F.adaptive_avg_pool2d(s_log, bp)).reshape(B, NS, h, w, C * bp * bp)
        s_log = rearrange(s_log, "b s gh gw (k i j) -> b s k (gh i) (gw j)",
                          k=C, i=bp, j=bp)                                     # (B, NS, C, P, P)
        s_log = rearrange(s_log, "b s k (y i) (x j) -> b (y x) s k (i j)", i=c, j=c)  # (B,Q,NS,C,c*c)
        n_s = (F.one_hot(season.clamp(min=0), NS).float()
               * (present & (season >= 0)).float()[..., None]).sum(1)          # (B, NS) months
        s_a = self._score(lr_F.float())
        if self.months_prior:
            s_a = s_a + n_s.clamp(min=1).log()[:, :, None, None]
        s_a = s_a.masked_fill((n_s == 0)[:, :, None, None], float("-inf"))     # unused seasons
        if self.no_base:                                                       # crops only
            s_a = torch.full_like(s_a, float("-inf"))
        elif self.training and self.season_drop:
            # whole season tokens out of the pool (every cell), crops kept; at least one used
            # season stays, so no cell is left without a candidate
            used = n_s > 0                                                     # (B, NS)
            drop = (torch.rand_like(n_s) < self.season_drop) & used
            keep = torch.multinomial(used.float() + 1e-6, 1)                   # (B, 1)
            rescue = (drop == used).all(1, keepdim=True)                       # all dropped
            drop = drop & ~(torch.zeros_like(drop).scatter(1, keep, True) & rescue)
            s_a = s_a.masked_fill(drop[:, :, None, None], float("-inf"))
        s_a = (s_a.repeat_interleave(rs, 2).repeat_interleave(rs, 3)          # (B, NS, H, W)
               .flatten(2).transpose(1, 2))                                    # (B, Q, NS)
        logs, acts = [s_log], [s_a]
        # -- crop candidates: one ENTRY per (crop, ps4 cell of its footprint), padded per
        #    cell to M
        b, _, y, x = hr_idx
        N = len(b)
        cover = None
        if N and hr_px is not None:        # off-grid ps4 crops: entries per overlapped cell
            # footprint side in px: one for all crops, or per crop (--level2 ps2 + ps4 crops)
            fps = (hr_fp if torch.is_tensor(hr_fp)
                   else torch.full((N,), hr_fp or c, dtype=torch.long, device=dev))
            c_a_tok = self._score(hr.float())
            if hr_logw is not None:
                c_a_tok = c_a_tok + hr_logw.float()
            py_all, px_all = hr_px
            ar_c = torch.arange(c, device=dev)
            e_log, e_cov, e_cell, e_crop = [], [], [], []
            for fp in fps.unique().tolist():
                ar_n = (fps == fp).nonzero(as_tuple=True)[0]                  # crops of size fp
                py, px, bg = py_all[ar_n], px_all[ar_n], b[ar_n]
                maps = linear(hr[ar_n].float()).view(-1, C, self.lr_px, self.lr_px)
                if fp < self.lr_px:
                    maps = (F.avg_pool2d(maps, self.lr_px // fp) if self.lr_px % fp == 0
                            else F.adaptive_avg_pool2d(maps, fp))
                ar_l = torch.arange(len(ar_n), device=dev)
                for da in range(fp // c + 1):
                    for db in range(fp // c + 1):
                        cy, cx = py // c + da, px // c + db
                        ui = cy[:, None] * c + ar_c[None] - py[:, None]        # (n, c) crop row
                        uj = cx[:, None] * c + ar_c[None] - px[:, None]
                        cov = (((ui >= 0) & (ui < fp))[:, :, None]
                               & ((uj >= 0) & (uj < fp))[:, None, :]
                               & ((cy < H) & (cx < W) & (cy >= 0) & (cx >= 0))[:, None, None])
                        keep = cov.flatten(1).any(1)
                        g = maps[ar_l[:, None, None], :, ui.clamp(0, fp - 1)[:, :, None],
                                 uj.clamp(0, fp - 1)[:, None, :]]                # (n, c, c, C)
                        e_log.append(g.permute(0, 3, 1, 2).flatten(2)[keep])    # (n, C, c*c)
                        e_cov.append(cov.flatten(1)[keep])
                        e_cell.append((bg * Q + cy * W + cx)[keep])  # (kept: inside the tile)
                        e_crop.append(ar_n[keep])
            crop, cov_e = torch.cat(e_log), torch.cat(e_cov)
            cell, ecrop = torch.cat(e_cell), torch.cat(e_crop)
            c_a = c_a_tok[ecrop]
            N = len(cell)
            order = torch.argsort(cell, stable=True)
            cell_s = cell[order]
            count = torch.bincount(cell_s, minlength=B * Q)
            M = int(count.max())
            slot = torch.arange(N, device=dev) - (torch.cumsum(count, 0) - count)[cell_s]
            idx = torch.full((B * Q, M), -1, dtype=torch.long, device=dev)
            idx[cell_s, slot] = order
            idx = idx.view(B, Q, M)
            ok = idx >= 0
            safe = idx.clamp(min=0)
            logs.append(crop[safe] * ok[..., None, None])
            acts.append(c_a[safe].masked_fill(~ok, float("-inf")))
            cover = torch.cat([torch.ones(B, Q, NS, c * c, dtype=torch.bool, device=dev),
                               cov_e[safe] & ok[..., None]], 2)                  # (B, Q, K, c*c)
        elif N:
            size = hr_size if hr_size is not None else torch.ones_like(b)
            maps = linear(hr.float()).view(N, C, self.lr_px, self.lr_px)       # (N, C, 16, 16)
            c_a_tok = self._score(hr.float())                                   # (N,)
            if hr_logw is not None:
                c_a_tok = c_a_tok + hr_logw.float()
            e_log, e_a, e_b, e_y, e_x, e_crop = [], [], [], [], [], []
            for sz in size.unique().tolist():                  # 1, 2, 4 cells per side
                sel = (size == sz).nonzero(as_tuple=True)[0]
                m = F.avg_pool2d(maps[sel], r // sz) if sz < r else maps[sel]   # (n, C, c*sz, c*sz)
                m = rearrange(m, "n k (i u) (j v) -> n (i j) k (u v)", u=c, v=c)  # (n, sz*sz, C, c*c)
                ii, jj = torch.meshgrid(torch.arange(sz, device=dev), torch.arange(sz, device=dev),
                                        indexing="ij")
                e_log.append(m.flatten(0, 1))
                e_a.append(c_a_tok[sel].repeat_interleave(sz * sz))
                e_b.append(b[sel].repeat_interleave(sz * sz))
                e_crop.append(sel.repeat_interleave(sz * sz))                   # entry -> crop
                e_y.append((y[sel][:, None] + ii.flatten()[None]).flatten())
                e_x.append((x[sel][:, None] + jj.flatten()[None]).flatten())
            crop, c_a = torch.cat(e_log), torch.cat(e_a)                       # (E, C, c*c), (E,)
            eb, ey, ex, ecrop = torch.cat(e_b), torch.cat(e_y), torch.cat(e_x), torch.cat(e_crop)
            N = len(eb)                                                         # entries
            cell = eb * Q + ey * W + ex                                         # (E,)
            order = torch.argsort(cell, stable=True)
            cell_s = cell[order]
            count = torch.bincount(cell_s, minlength=B * Q)
            M = int(count.max())
            slot = torch.arange(N, device=dev) - (torch.cumsum(count, 0) - count)[cell_s]
            idx = torch.full((B * Q, M), -1, dtype=torch.long, device=dev)
            idx[cell_s, slot] = order
            idx = idx.view(B, Q, M)
            ok = idx >= 0
            safe = idx.clamp(min=0)
            logs.append(crop[safe])                                             # (B, Q, M, C, c*c)
            acts.append(c_a[safe].masked_fill(~ok, float("-inf")))              # (B, Q, M)
        a = torch.cat(acts, 2)                                                  # (B, Q, K)
        cand = torch.cat(logs, 2)                                               # (B, Q, K, C, c*c)
        prob = self.mean and C > 1
        if prob:
            cand = cand.softmax(3)
        if cover is not None:               # per pixel, over the candidates covering it
            wgt = torch.softmax(a[..., None].masked_fill(~cover, float("-inf")), 2).nan_to_num(0.0)
            out = torch.einsum("bqkp,bqkcp->bqcp", wgt, cand)
        else:
            wgt = torch.softmax(a, 2).nan_to_num(0.0)
            out = torch.einsum("bqk,bqkcp->bqcp", wgt, cand)                    # (B, Q, C, c*c)
        if prob:                            # log of a distribution: cross_entropy reads it exactly
            out = out.clamp(min=1e-8).log()
        img = rearrange(out, "b (y x) k (i j) -> b k (y i) (x j)", y=H, x=W, i=c, j=c)
        if self.out_px is not None and img.shape[-1] > self.out_px:
            lo = (img.shape[-1] - self.out_px) // 2
            img = img[..., lo:lo + self.out_px, lo:lo + self.out_px]
        if parts:                                   # crop index (not entry) per crop slot
            cid = torch.where(idx >= 0, ecrop[idx.clamp(min=0)], -1) if N else None
            return img, (wgt, cand, out, cid)
        return img
