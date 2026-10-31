"""vid-TLDR (Choi et al., CVPR 2024; mlvlab/vid-TLDR, MIT), training-free token merging inside a
trained fixed monthly model: the months are the video's frames, one sequence over all of them.

get_objective_score and merging are upstream's (multi_modality/utils/vid_TLDR.py); merge_wavg
keeps its size weighting. What differs from upstream, all forced by the setting:
  - r is a share of the sequence's INPUT tokens (upstream's README: absolute counts of its 2352);
    sequences here differ in length (absent months), so equal-length samples run together.
  - upstream only needs its output tokens; a segmentation needs every location back, and
    background tokens merged with weight 0 must still read their destination's token, so each
    input token's cluster is tracked unweighted (`assign`, upstream's source_trace without the
    saliency) and the final tokens are copied back to the token grid.
  - tokens are ordered (t, row, col, band set), the video order upstream's a / b split runs on.
  - attention is dense: FlashAttention gives neither the attention map nor the log-size term.
"""
import math

import torch
import torch.nn.functional as F

from manyup.mixres.baselines.base import BaselineModel


def get_objective_score(score_attn):
    """(G, heads, N, N) attention -> (G, N, 1): upstream's saliency, sharper attention higher,
    background 0. (xlogy: float32 softmaxes reach exact zeros on long sequences.)"""
    score_attn = score_attn.mean(dim=1)
    scores = torch.xlogy(score_attn, score_attn).sum(dim=2).unsqueeze(-1)
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True).clamp_min(1e-12)
    score_mask = scores < scores.mean(dim=1, keepdim=True)
    scores = scores - scores.mean(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True).clamp_min(1e-12)
    scores[score_mask] = 0.0
    return scores


def merging(metric, r_merge, score_obj):
    """Upstream's bipartite soft matching. -> (merge function, remap (G, N) each token's index
    after the merge)."""
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        n, t1, t2 = a.shape[0], a.shape[-2], b.shape[-2]
        scores = (a @ b.transpose(-1, -2) + 1) / 2
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r_merge:, :]
        src_idx = edge_idx[..., :r_merge, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        unm_idx = unm_idx.sort(dim=1)[0]
        src_so = score_obj[..., ::2, :].gather(dim=-2, index=src_idx)

        keep = t1 - r_merge
        remap = torch.zeros(n, t1 + t2, dtype=torch.long, device=metric.device)
        ar = torch.arange(t1 + t2, device=metric.device)
        remap_a = remap[:, ::2].scatter(1, unm_idx[..., 0], torch.arange(keep, device=metric.device).expand(n, keep))
        remap_a = remap_a.scatter(1, src_idx[..., 0], keep + dst_idx[..., 0])
        remap[:, ::2] = remap_a
        remap[:, 1::2] = keep + ar[:t2]

    def merge(x, mode="sum", dtype=torch.float32):
        ori_dtype = x.dtype
        x = x.to(dtype=dtype)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        c = src.shape[-1]
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r_merge, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r_merge, c))
        src = src * src_so
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r_merge, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=-2).to(dtype=ori_dtype)

    return merge, remap


def merge_wavg(merge, x, size):
    size = torch.ones_like(x[..., 0, None]) if size is None else size
    size_max = size.amax(dim=-2, keepdim=True)
    x = merge(x * (size / size_max), mode="sum")
    size = merge(size, mode="sum")
    return x / (size / size_max), size


def attention(blk, x, size, want_map):
    """blk's attention on a dense (G, N, D), log size added to the scores (upstream's Attention)
    -> (output, the size-free attention map or None)."""
    a = blk.attn
    if a.attn_drop.p:
        raise ValueError("vid-TLDR: no attention dropout")
    G, N, D = x.shape
    h = blk.norm1(x)
    heads = lambda v: v.view(G, N, a.num_heads, -1).transpose(1, 2)          # noqa: E731
    q, k, v = a.q_norm(heads(a.q(h))), a.k_norm(heads(a.k(h))), heads(a.v(h))
    prior = None if size is None else size.log()[:, None, None, :, 0].to(q.dtype)
    if want_map:
        s = (q @ k.transpose(-2, -1)) * a.scale
        amap = s.float().softmax(-1)
        w = amap if prior is None else (s + prior).float().softmax(-1)
        out = w.to(v.dtype) @ v
    else:
        amap = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=prior)
    return a.proj_drop(a.proj(out.transpose(1, 2).reshape(G, N, -1))), amap


def video_order(lr) -> torch.Tensor:
    """(N_valid,) the sort key of each valid token: (t, row, col), band sets left in order."""
    keys = []
    for m, shape in lr.dims.items():
        if len(shape) != 6:
            raise ValueError(f"vid-TLDR: {m} is not (S, h, w, t, band sets, D)")
        h, w, t, b = shape[1:5]
        tt = torch.arange(t, device=lr.x.device)[None, None, :, None].expand(h, w, t, b)
        hh = torch.arange(h, device=lr.x.device)[:, None, None, None].expand(h, w, t, b)
        ww = torch.arange(w, device=lr.x.device)[None, :, None, None].expand(h, w, t, b)
        keys.append((tt * 4096 + hh) * 4096 + ww)
    return torch.cat([k.flatten() for k in keys]).expand(len(lr.seqlens), -1)[lr.keep]


class TldrModel(BaselineModel):
    def __init__(self, task, cfg, pretrained: bool = True):
        super().__init__(task, cfg, pretrained)
        self.shares = [float(v) for v in cfg.baseline.r.split(",") if float(v)]

    def _blocks(self, x):
        """Every block on (G, N, D), the same length; merges in the first len(shares) blocks,
        right after attention -> (G, N, D), each input token holding its cluster's final token."""
        G, N0, _ = x.shape
        size, assign = None, torch.arange(N0, device=x.device).expand(G, N0)
        for i, blk in enumerate(self.mix.encoder.blocks):
            n_merge = 0
            if i < len(self.shares):
                n_merge = min(math.floor(N0 * self.shares[i]), x.shape[1] // 2)
            y, amap = attention(blk, x, size, n_merge > 0)
            x = x + blk.drop_path(blk.ls1(y))
            if n_merge:
                merge, remap = merging(x.float(), n_merge, get_objective_score(amap))
                x, size = merge_wavg(merge, x, size)
                assign = remap.gather(1, assign)
            x = x + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(x))))
        return self.mix.encoder.norm(x).gather(1, assign[..., None].expand(-1, -1, x.shape[-1]))

    def encode(self, lr, crops, steps=None, lr_enc=None):
        if crops:
            raise ValueError("vid-TLDR: no crops")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            starts = torch.cumsum(lr.seqlens, 0) - lr.seqlens
            key = video_order(lr)
            out = None
            for n in lr.seqlens.unique().tolist():
                seqs = (lr.seqlens == n).nonzero()[:, 0]
                idx = starts[seqs, None] + torch.arange(n, device=lr.x.device)   # (G, n)
                idx = idx.gather(1, key[idx].argsort(dim=1, stable=True))
                y = self._blocks(lr.x[idx])
                if out is None:
                    out = y.new_empty(lr.x.shape)
                out[idx] = y
        return lr.pooled(out).float(), lr.x.new_zeros(0, lr.x.shape[-1]).float()
