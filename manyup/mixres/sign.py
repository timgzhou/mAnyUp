"""The learned flip of the season-entropy ranking (selectors.EntropySign).

At most locations the season with the lowest prediction entropy is also the lowest-loss one, but
at a minority the ranking is inverted (figures/entropy_align/time_s8m4_s788.png), and a threshold
on the entropy level cannot find them (exp.mixres.time_flip_eval: the best threshold never
flips). SignNet predicts, per location, whether to take the highest-entropy seasons instead,
trained on the sign of that location's Spearman r between season entropy and season loss.
"""
import torch
import torch.nn.functional as F
from torch import nn


def season_spearman(E: torch.Tensor, L: torch.Tensor, used: torch.Tensor) -> torch.Tensor:
    """E, L (B, NS, S, S) per (season, location), used (B, NS) -> (B, S, S) Spearman r over each
    location's used seasons; nan with fewer than 3."""
    u = used[:, :, None, None].expand_as(E)

    def rank(x):
        r = x.masked_fill(~u, float("inf")).argsort(1).argsort(1).float()
        return r.masked_fill(~u, float("nan"))
    rE, rL = rank(E), rank(L)
    mE, mL = rE.nanmean(1, keepdim=True), rL.nanmean(1, keepdim=True)
    cov = ((rE - mE) * (rL - mL)).nansum(1)
    sd = (((rE - mE) ** 2).nansum(1) * ((rL - mL) ** 2).nansum(1)).sqrt()
    r = cov / sd.clamp(min=1e-9)
    return r.masked_fill(u.float().sum(1) < 3, float("nan"))


class SignNet(nn.Module):
    """Per location on the S x S grid of windows that tile the tile -> a flip logit.
    Inputs (all detached): the per-season window entropy z-scored over the location's used
    seasons, its mean, the used-season mask, each season's final token at the location
    (projected to 16), and the window's raw monthly pixels (LayerNorm -> 32, mean over present
    months). 1x1 -> 3x3 -> 1x1 conv; the last layer starts at bias `init_bias` (p(flip) ~ 0.05):
    the plain lowest-entropy selector at the start."""

    def __init__(self, dim: int, NS: int, pix: int, hidden: int = 64, init_bias: float = -3.0):
        super().__init__()
        self.NS = NS
        self.tok = nn.Linear(dim, 16)
        self.pix = nn.Sequential(nn.LayerNorm(pix), nn.Linear(pix, 32))
        c_in = NS + 1 + NS + 16 * NS + 32
        self.net = nn.Sequential(nn.Conv2d(c_in, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, E, used, lr_F, pix, mon):
        """E (B, NS, S, S) window entropy per season (finite), used (B, NS), lr_F (B, NS, gh, gw,
        D), pix (B, S, S, T, C*P*P), mon (B, T) present -> (B, S, S) flip logits."""
        B, NS, S, _ = E.shape
        u = used[:, :, None, None].float()
        n = u.sum(1, keepdim=True).clamp(min=1)
        mu = (E * u).sum(1, keepdim=True) / n
        sd = (((E - mu) ** 2) * u).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-4)
        ez = ((E - mu) / sd) * u
        t = self.tok(lr_F.float())                                           # (B,NS,gh,gw,16)
        t = t * used[:, :, None, None, None].float()
        t = F.interpolate(t.permute(0, 1, 4, 2, 3).flatten(1, 2), size=(S, S), mode="nearest")
        w = mon.float()[:, None, None, :, None]
        p = (self.pix(pix.float()) * w).sum(3) / w.sum(3).clamp(min=1)       # (B, S, S, 32)
        x = torch.cat([ez, mu, u.expand(B, NS, S, S), t, p.permute(0, 3, 1, 2)], 1)
        return self.net(x)[:, 0]
