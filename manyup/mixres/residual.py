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
