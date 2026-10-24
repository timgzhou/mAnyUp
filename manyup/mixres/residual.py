"""The learned residual on the entropy window score (selectors.Residual) and its helpers (moved
verbatim from exp.upsamplers.finetune_inject, --res_mode / --res_arch)."""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class ResidualScorer(nn.Module):
    """--res_mode: a learned correction r (B, S, S) to the LR-only scorer's window entropy, per
    candidate window (stride = crop size). The last layer starts at zero: score = z(entropy) + r
    is the plain entropy selector at the start of training. --res_arch picks the inputs:
        conv        the frozen scorer's crop-free pass: its mean class probabilities, entropy
                    (mean of the pixels'), top-2 margin, the pixels' disagreement (entropy of the
                    mean - mean entropy) and its final tokens (mean over used seasons, projected
                    to 32) -> a 1x1 / 3x3 / 1x1 conv stack
        hr          the window's raw monthly pixels (the HR input the crops would encode): each
                    present month's C x P x P patch -> LayerNorm -> linear -> + window position
                    -> mean over present months -> MLP
        xattn       hr, each (window, month) patch embedding a query cross-attending to the
                    scorer's final season tokens (keys / values, + token position and season
                    embeddings; unused seasons masked) -> [query, attended] -> mean over present
                    months -> MLP
        xattn_conv  xattn's window vector as extra channels of conv's stack"""

    def __init__(self, dim: int, n_cls: int, hidden: int = 64, arch: str = "conv",
                 C: int = 0, P: int = 0, S: int = 0, gh: int = 0, NS: int = 0, heads: int = 4):
        super().__init__()
        self.arch, self.heads = arch, heads
        q_out = 0
        if arch != "conv":
            d = hidden
            self.pix = nn.Sequential(nn.LayerNorm(C * P * P), nn.Linear(C * P * P, d))
            self.qpos = nn.Parameter(torch.zeros(S * S, d))
            nn.init.trunc_normal_(self.qpos, std=0.02)
            q_out = d
            if arch in ("xattn", "xattn_conv"):
                self.kv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * d))
                self.kpos = nn.Parameter(torch.zeros(gh * gh, 2 * d))
                self.ksea = nn.Parameter(torch.zeros(NS, 2 * d))
                nn.init.trunc_normal_(self.kpos, std=0.02)
                nn.init.trunc_normal_(self.ksea, std=0.02)
                self.qn = nn.LayerNorm(d)
                self.o = nn.Linear(d, d)
                q_out = 2 * d
        if arch in ("conv", "xattn_conv"):
            self.tok = nn.Linear(dim, 32)
            self.net = nn.Sequential(nn.Conv2d(n_cls + 3 + 32 + (q_out if arch != "conv" else 0),
                                               hidden, 1), nn.GELU(),
                                     nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(hidden, 1, 1))
        else:
            self.net = nn.Sequential(nn.Linear(q_out, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def window_vec(self, pix, mon, lr_F, used):
        """pix (B, S, S, T, C*P*P) raw monthly window pixels, mon (B, T) present months, lr_F
        (B, NS, gh, gw, D) the scorer's final tokens, used (B, NS) -> (B, S, S, q_out)."""
        B, S, _, T, _ = pix.shape
        q = self.pix(pix.float()) + self.qpos.view(1, S, S, 1, -1)              # (B,S,S,T,d)
        if self.arch in ("xattn", "xattn_conv"):
            NS, gh, gw = lr_F.shape[1:4]
            kv = (self.kv(lr_F.float()) + self.kpos.view(1, 1, gh, gw, -1)
                  + self.ksea.view(1, NS, 1, 1, -1)).flatten(1, 3)              # (B, NS*gh*gw, 2d)
            k, v = kv.chunk(2, -1)
            h = self.heads
            qh = self.qn(q).flatten(1, 3).unflatten(-1, (h, -1)).transpose(1, 2)  # (B,h,L,dh)
            kh = k.unflatten(-1, (h, -1)).transpose(1, 2)
            vh = v.unflatten(-1, (h, -1)).transpose(1, 2)
            m = used[:, None, None, :, None, None].expand(B, 1, 1, NS, gh, gw).flatten(3)
            att = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=m)
            att = self.o(att.transpose(1, 2).flatten(2)).view(B, S, S, T, -1)
            q = torch.cat([q, att], -1)
        w = mon.float()[:, None, None, :, None]
        return (q * w).sum(3) / w.sum(3).clamp(min=1)                           # (B,S,S,q_out)

    def forward(self, feats, tok, pix=None, mon=None, lr_F=None, used=None):
        """feats (B, n_cls + 3, S, S), tok (B, S, S, D); the HR arms: + window_vec's inputs
        -> (B, S, S)."""
        if self.arch == "conv":
            t = self.tok(tok.float()).permute(0, 3, 1, 2)
            return self.net(torch.cat([feats.float(), t], 1))[:, 0]
        qv = self.window_vec(pix, mon, lr_F, used)
        if self.arch == "xattn_conv":
            t = self.tok(tok.float()).permute(0, 3, 1, 2)
            return self.net(torch.cat([feats.float(), t, qv.permute(0, 3, 1, 2)], 1))[:, 0]
        return self.net(qv)[..., 0]


def window_pixels(ms, P_):
    """The raw monthly pixels of every candidate window (stride = size = P_): (B, S, S, T,
    C*P_*P_), every float band of every modality, NaN -> 0."""
    xs = [v for k, v in ms.as_dict().items()
          if torch.is_tensor(v) and v.ndim == 5 and "mask" not in k and v.is_floating_point()]
    x = torch.cat(xs, -1).float().nan_to_num()                                  # (B,H,W,T,C)
    return rearrange(x, "b (s p) (r q) t c -> b s r t (c p q)", p=P_, q=P_)


def listnet_loss(pred, target, valid, beta=2.0):
    """ListNet (softmax cross-entropy) between (L, N) scores and z-scored targets over the valid
    entries of each list; lists with < 2 valid entries or constant targets are skipped."""
    t = target.masked_fill(~valid, 0.0)
    cnt = valid.sum(1, keepdim=True).clamp(min=1)
    mu = t.sum(1, keepdim=True) / cnt
    sd = (((t - mu) ** 2) * valid).sum(1, keepdim=True).div(cnt).sqrt()
    keep = (valid.sum(1) >= 2) & (sd[:, 0] > 1e-6)
    if not keep.any():
        return pred.sum() * 0
    z = ((t - mu) / sd.clamp(min=1e-6) * beta).masked_fill(~valid, float("-inf"))
    lp = pred.masked_fill(~valid, float("-inf")).log_softmax(1)
    return -(z.softmax(1) * lp.masked_fill(~valid, 0.0)).sum(1)[keep].mean()


class MonthQueries(nn.Module):
    """Per (window, month): the window's raw pixels -> LayerNorm -> linear (d) + window
    position; xattn also: those as queries cross-attending to the tile's season tokens (+ token
    position and season embeddings; unused seasons masked), [query, attended]. -> (B, S, S, T,
    d or 2d). The month path of ResidualScorer.window_vec before its mean over months (kept
    separate: ResidualScorer's parameter names must match the legacy checkpoints)."""

    def __init__(self, xattn: bool, dim: int, C: int, P: int, S: int, gh: int, NS: int,
                 d: int = 64, heads: int = 4):
        super().__init__()
        self.xattn, self.heads = xattn, heads
        self.pix = nn.Sequential(nn.LayerNorm(C * P * P), nn.Linear(C * P * P, d))
        self.qpos = nn.Parameter(torch.zeros(S * S, d))
        nn.init.trunc_normal_(self.qpos, std=0.02)
        self.out = 2 * d if xattn else d
        if xattn:
            self.kv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * d))
            self.kpos = nn.Parameter(torch.zeros(gh * gh, 2 * d))
            self.ksea = nn.Parameter(torch.zeros(NS, 2 * d))
            nn.init.trunc_normal_(self.kpos, std=0.02)
            nn.init.trunc_normal_(self.ksea, std=0.02)
            self.qn = nn.LayerNorm(d)
            self.o = nn.Linear(d, d)

    def forward(self, pix, lr_F, used):
        B, S, _, T, _ = pix.shape
        q = self.pix(pix.float()) + self.qpos.view(1, S, S, 1, -1)
        if not self.xattn:
            return q
        NS, gh, gw = lr_F.shape[1:4]
        kv = (self.kv(lr_F.float()) + self.kpos.view(1, 1, gh, gw, -1)
              + self.ksea.view(1, NS, 1, 1, -1)).flatten(1, 3)
        k, v = kv.chunk(2, -1)
        h = self.heads
        qh = self.qn(q).flatten(1, 3).unflatten(-1, (h, -1)).transpose(1, 2)
        kh, vh = (x.unflatten(-1, (h, -1)).transpose(1, 2) for x in (k, v))
        m = used[:, None, None, :, None, None].expand(B, 1, 1, NS, gh, gw).flatten(3)
        att = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=m)
        att = self.o(att.transpose(1, 2).flatten(2)).view(B, S, S, T, -1)
        return torch.cat([q, att], -1)


class SeasonResidual(nn.Module):
    """r per (season, location) for time selection (selectors.SeasonResidualScorer): the
    spatial ResidualScorer's input sets, per season instead of per window --
        conv        the season's own head map at the window (mean class probabilities, mean
                    entropy, top-2 margin, entropy of the mean - mean entropy) + its token
                    (projected to 32): 1x1 / 3x3 / 1x1 conv over the location grid
        hr          the raw pixels of the season's months at the window (MonthQueries, mean
                    over the season's present months) -> MLP
        xattn       hr's queries cross-attending to the tile's season tokens -> MLP
        xattn_conv  xattn's vector as extra channels of conv's stack
    The last layer starts at zero (r = 0: the plain entropy ranking)."""

    def __init__(self, dim: int, n_cls: int, arch: str = "conv", C: int = 0, P: int = 0,
                 S: int = 0, gh: int = 0, NS: int = 0, hidden: int = 64):
        super().__init__()
        self.arch = arch
        q_out = 0
        if arch != "conv":
            self.months = MonthQueries(arch in ("xattn", "xattn_conv"), dim, C, P, S, gh, NS)
            q_out = self.months.out
        if arch in ("conv", "xattn_conv"):
            self.tok = nn.Linear(dim, 32)
            self.net = nn.Sequential(nn.Conv2d(n_cls + 3 + 32 + (q_out if arch != "conv" else 0),
                                               hidden, 1), nn.GELU(),
                                     nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(hidden, 1, 1))
        else:
            self.net = nn.Sequential(nn.Linear(q_out, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats, lr_F, used, pix=None, season=None, valid=None):
        """feats (B, NS, n_cls + 3, S, S) per-season map features, lr_F (B, NS, gh, gw, D),
        used (B, NS); the pixel arms: pix (B, S, S, T, C*P*P), season (B, T), valid (B, T)
        -> r (B, NS, S, S)."""
        B, NS, _, S, _ = feats.shape
        qv = None
        if self.arch != "conv":
            q = self.months(pix, lr_F, used)                                # (B, S, S, T, q)
            w = (torch.nn.functional.one_hot(season.clamp(min=0), NS).bool()
                 & valid[..., None]).float()                                # (B, T, NS)
            qv = torch.einsum("bxytq,bts->bsxyq", q, w) / w.sum(1)[:, :, None, None, None].clamp(min=1)
        if self.arch in ("conv", "xattn_conv"):
            t = self.tok(lr_F.float())                                       # (B, NS, gh, gw, 32)
            t = F.interpolate(t.permute(0, 1, 4, 2, 3).flatten(0, 1), size=(S, S), mode="nearest")
            x = [feats.float().flatten(0, 1), t]
            if qv is not None:
                x.append(qv.flatten(0, 1).permute(0, 3, 1, 2))
            return self.net(torch.cat(x, 1))[:, 0].view(B, NS, S, S)
        return self.net(qv)[..., 0]
