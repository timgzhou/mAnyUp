"""Is the readout's attention over seasons a label-free proxy for which season is right? The
readout (decoder.NativeSetReadout) pools each cell's season tokens with softmax weights
a = score(f) + log(present months), score = Linear(LayerNorm(f)) trained by the task loss alone.
On a FROZEN co-trained model (as exp.mixres.lossup_probe), test split, at the windows that tile
the tile (each window its season tokens' value, nearest), per location over its used seasons:
spearman(score, loss) and the share of random's regret saved by keeping the season it picks
(regret = picked season's loss - the location's lowest; loss = each season's own head map, the
time label oracle's criterion), for
    attn_score / attn_full   highest learned score / highest score + log months kept
    attn_score_low           the lowest learned score kept (the reverse reading)
    entropy                  lowest season entropy kept
    learned                  the run's learned season scorer
-> rows of results/pastis/attn_probe.csv.

    python -u -m exp.mixres.attn_probe --setup s16m8
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp.mixres.lossup_probe import RUNS, load  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import Context, Entropy, LabelLoss  # noqa: E402
from manyup.mixres.sign import season_spearman  # noqa: E402


@torch.no_grad()
def evaluate(model, sel, task, nw, bs, limit=0):
    geo = sel.geo
    cell = geo.tiling().cuda()
    acc = {}
    for i, b in enumerate(loader(task, "test", bs, nw, False)):
        if limit and i >= limit:
            break
        ctx = Context.from_batch(model, b)
        B, NS = ctx.B, ctx.NS
        c = cell
        L = LabelLoss()(ctx, geo, ("season",)).season.float()[:, :, c[:, None], c[None, :]]
        E = Entropy()(ctx, geo, ("season",)).season.float()[:, :, c[:, None], c[None, :]]
        R = sel.scorer(ctx, geo, ("season",)).season.float()[:, :, c[:, None], c[None, :]]
        lr_F = ctx.out0.lr_F.float()                                       # (B, NS, gh, gw, D)
        a = model.readout._score(lr_F)                                      # (B, NS, gh, gw)
        S = len(cell)
        a = F.interpolate(a, size=(S, S), mode="nearest")                   # (B, NS, S, S)
        months = (F.one_hot(ctx.season.clamp(min=0), NS).float()
                  * ctx.valid.float()[..., None]).sum(1)                    # (B, NS)
        full = a + months.clamp(min=1).log()[:, :, None, None]
        scores = {"attn_score": -a, "attn_full": -full, "attn_score_low": a, "entropy": E,
                  "learned": R}                                             # lower = keep
        used = ctx.used
        u = used[:, :, None, None].expand(B, NS, S, S)
        ok2 = u.sum(1) >= 2
        Lu = L.masked_fill(~u, float("nan"))
        lo = Lu.nan_to_num(float("inf")).amin(1)
        acc.setdefault("rand", []).append((Lu.nanmean(1) - lo)[ok2].sum().item())
        for name, s in scores.items():
            sp = season_spearman(s, L, used)
            pk = s.masked_fill(~u, float("inf")).argmin(1, keepdim=True)
            acc.setdefault(name + "/sp", []).append(sp[torch.isfinite(sp)].mean().item())
            acc.setdefault(name + "/reg", []).append((L.gather(1, pk)[:, 0] - lo)[ok2].sum().item())
        # how peaked the attention is: the mean max softmax weight over used seasons
        w = full.masked_fill(~u, float("-inf")).softmax(1).amax(1)
        acc.setdefault("attn_max_weight", []).append(w[ok2].mean().item())
    out = {"attn_max_weight": sum(acc["attn_max_weight"]) / len(acc["attn_max_weight"])}
    rr = sum(acc["rand"])
    for k, v in acc.items():
        if k.endswith("/sp"):
            out[k] = sum(v) / len(v)
        elif k.endswith("/reg"):
            out[k[:-4] + "/regret_saved"] = 1 - sum(v) / rr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=tuple(RUNS))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()
    run, cfg, task, model, sel = load(RUNS[a.setup], a.num_workers)
    res = evaluate(model, sel, task, a.num_workers, a.batch_size, a.limit_batches)
    for k in sorted(res):
        print(f"  {k:30s} {res[k]:.4f}")
    if not a.limit_batches:
        for k, v in res.items():
            _append(Path("results/pastis/attn_probe.csv"), {
                "job": os.environ.get("SLURM_JOB_ID", ""), "setup": a.setup, "run": run,
                "metric": k, "value": round(v, 5)})


if __name__ == "__main__":
    main()
