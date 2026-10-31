"""Does per-season supervision make season entropy a season ranker? The readout (decoder.
NativeSetReadout) mixes every season token's head map in logit space and only the mixture is
trained, so a season's own map never has to be right alone. On a FROZEN co-trained model
(entropy + learned, one season; mixres_spacetime.sh), no backbone change:

  heads   one head per season slot (a copy of the shared head at the start), each trained with
          the cross-entropy of its season's own map (detached tokens; train split)
  T       a temperature per season slot, fitted by NLL on the val split, for the shared head
          and for the per-season heads

then on the test split, at the windows that tile the tile, per location over its used seasons:
spearman(entropy, loss) and the share of random's regret the lowest-entropy season saves
(regret = picked season's loss - the location's lowest), with the loss of
    oracle  the shared head's season map (what the time label oracle selects by)
    own     the scored head's own season map
next to the run's learned season scorer. -> rows of results/pastis/season_heads_probe.csv.

    python -u -m exp.mixres.season_heads_probe --setup s16m8 [--epochs 8]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import copy  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from exp.mixres.lossup_probe import RUNS, load  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.model import LR_PS  # noqa: E402
from manyup.mixres.selectors import (Context, pixel_entropy, pixel_label_loss,  # noqa: E402
                                     season_maps)
from manyup.mixres.sign import season_spearman  # noqa: E402


class SeasonHeads(nn.Module):
    """One copy of the shared head per season slot; maps as selectors.season_maps."""

    def __init__(self, model, NS):
        super().__init__()
        self.heads = nn.ModuleList(copy.deepcopy(model.head.linear) for _ in range(NS))
        self.heads.requires_grad_(True)             # the frozen model's head is not
        self.lr_patch = model.cfg.lr_patch

    def forward(self, lr_F):
        """(B, NS, gh, gw, D) -> (B, NS, C, 64, 64)."""
        B, NS, gh, gw, _ = lr_F.shape
        out = []
        for s, lin in enumerate(self.heads):
            m = lin(lr_F[:, s].float())
            C = m.shape[-1] // LR_PS ** 2
            m = m.view(B * gh * gw, C, LR_PS, LR_PS)
            if self.lr_patch < LR_PS:
                m = F.avg_pool2d(m, LR_PS // self.lr_patch)
            p = m.shape[-1]
            out.append(m.view(B, gh, gw, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, gh * p, gw * p))
        return torch.stack(out, 1)


def shared_maps(model, ctx):
    lr_F = ctx.out0.lr_F                        # the uniform pass outside autocast
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        s = season_maps(model, lr_F)
    return s.float().view(ctx.B, ctx.NS, *s.shape[1:])


def season_ce(maps, target, used):
    """maps (B, NS, C, H, W), used (B, NS) -> mean CE over used seasons' labelled pixels."""
    B, NS = maps.shape[:2]
    t = target.long()[:, None].expand(B, NS, *target.shape[-2:])
    ce = F.cross_entropy(maps.flatten(0, 1), t.flatten(0, 1), ignore_index=-1, reduction="none")
    m = (t.flatten(0, 1) != -1) & used.flatten()[:, None, None]
    return ce[m].mean()


def fit_temperature(model, heads, task, nw, bs, NS, batches=40):
    """-> log T (NS,) for the shared head and for the heads, by NLL on val."""
    logT = {k: torch.zeros(NS, device="cuda", requires_grad=True) for k in ("shared", "heads")}
    opt = torch.optim.Adam(list(logT.values()), lr=0.05)
    data = []
    for i, b in enumerate(loader(task, "val", bs, nw, False)):
        if i >= batches:
            break
        ctx = Context.from_batch(model, b)
        with torch.no_grad():
            sh = shared_maps(model, ctx)
            hd = heads(ctx.out0.lr_F)
        data.append((sh.half(), hd.half(), b.target.cuda(), ctx.used))
    for _ in range(30):
        for sh, hd, tgt, used in data:
            loss = sum(season_ce(m.float() / logT[k].exp()[None, :, None, None, None], tgt, used)
                       for k, m in (("shared", sh), ("heads", hd)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {k: v.detach() for k, v in logT.items()}


@torch.no_grad()
def evaluate(model, sel, heads, logT, task, nw, bs, limit=0):
    geo = sel.geo
    cell = geo.tiling().cuda()
    acc = {}
    for i, b in enumerate(loader(task, "test", bs, nw, False)):
        if limit and i >= limit:
            break
        ctx = Context.from_batch(model, b)
        B, NS = ctx.B, ctx.NS
        sh, hd = shared_maps(model, ctx), heads(ctx.out0.lr_F)
        tgt = b.target.cuda()

        def win(px):                                 # (B, NS, 64, 64) -> (B, NS, S, S)
            w = geo.window_mean(px.flatten(0, 1)).view(B, NS, 16, 16)
            return w[:, :, cell[:, None], cell[None, :]]

        def ent(m, T=None):
            m = m if T is None else m / T.exp()[None, :, None, None, None]
            return win(pixel_entropy(m.flatten(0, 1)).view(B, NS, 64, 64))

        def loss(m):
            t = tgt.repeat_interleave(NS, 0)
            return win(pixel_label_loss(m.flatten(0, 1), t).view(B, NS, 64, 64))
        L = {"oracle": loss(sh), "own_shared": loss(sh), "own_heads": loss(hd)}
        scores = {"shared": (ent(sh), "own_shared"), "shared_T": (ent(sh, logT["shared"]), "own_shared"),
                  "heads": (ent(hd), "own_heads"), "heads_T": (ent(hd, logT["heads"]), "own_heads")}
        r = sel.scorer(ctx, geo, ("season",)).season.float()[:, :, cell[:, None], cell[None, :]]
        scores["learned"] = (r, "own_shared")
        used = ctx.used
        u = used[:, :, None, None].expand(B, NS, len(cell), len(cell))
        ok2 = u.sum(1) >= 2
        for target_name in ("oracle", "own"):
            for name, (S, own) in scores.items():
                Lt = L["oracle"] if target_name == "oracle" else L[own]
                Lu = Lt.masked_fill(~u, float("nan"))
                lo = Lu.nan_to_num(float("inf")).amin(1)
                sp = season_spearman(S, Lt, used)
                pk = S.masked_fill(~u, float("inf")).argmin(1, keepdim=True)
                k = f"{target_name}/{name}"
                acc.setdefault(k + "/sp", []).append(sp[torch.isfinite(sp)].mean().item())
                acc.setdefault(k + "/reg", []).append((Lt.gather(1, pk)[:, 0] - lo)[ok2].sum().item())
                acc.setdefault(k + "/rand", []).append((Lu.nanmean(1) - lo)[ok2].sum().item())
        for name, (m, own) in (("shared", (sh, None)), ("heads", (hd, None))):
            acc.setdefault(f"ce/{name}", []).append(season_ce(m, tgt, used).item())
    out = {}
    for k, v in acc.items():
        if k.endswith("/sp") or k.startswith("ce/"):
            out[k] = sum(v) / len(v)
        elif k.endswith("/reg"):
            out[k[:-4] + "/regret_saved"] = 1 - sum(v) / sum(acc[k[:-4] + "/rand"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=tuple(RUNS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(0)
    run, cfg, task, model, sel = load(RUNS[a.setup], a.num_workers)
    b0 = next(iter(loader(task, "val", 4, a.num_workers, False)))
    NS = Context.from_batch(model, b0).NS
    heads = SeasonHeads(model, NS).cuda()
    with torch.no_grad():                       # the copies reproduce the shared head's maps
        c0 = Context.from_batch(model, b0)
        d = (heads(c0.out0.lr_F) - shared_maps(model, c0)).abs().max().item()
    print(f"heads vs shared head at init: max |diff| {d:.4f} (bf16 rounding only)", flush=True)
    opt = torch.optim.AdamW(heads.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        heads.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, b in enumerate(loader(task, "train", a.batch_size, a.num_workers, True)):
            if a.limit_batches and i >= a.limit_batches:
                break
            ctx = Context.from_batch(model, b)
            loss = season_ce(heads(ctx.out0.lr_F.detach()), b.target.cuda(), ctx.used)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        print(f"epoch {ep + 1}/{a.epochs} | season CE {tot / max(n, 1):.4f} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    heads.eval()
    logT = fit_temperature(model, heads, task, a.num_workers, a.batch_size, NS,
                           batches=a.limit_batches or 40)
    print("T shared", logT["shared"].exp().tolist(), "| T heads", logT["heads"].exp().tolist())
    res = evaluate(model, sel, heads, logT, task, a.num_workers, a.batch_size, a.limit_batches)
    for k in sorted(res):
        print(f"  {k:34s} {res[k]:.4f}")
    if not a.limit_batches:
        for k, v in res.items():
            _append(Path("results/pastis/season_heads_probe.csv"), {
                "job": os.environ.get("SLURM_JOB_ID", ""), "setup": a.setup, "run": run,
                "epochs": a.epochs, "metric": k, "value": round(v, 5),
                "T_shared": " ".join(f"{t:.3f}" for t in logT["shared"].exp().tolist()),
                "T_heads": " ".join(f"{t:.3f}" for t in logT["heads"].exp().tolist())})


if __name__ == "__main__":
    main()
