"""Do the quadrants of selectors.Quadrant mean what the policy assumes? On a FROZEN co-trained
model (as exp.mixres.lossup_probe), test split, per low-res cell: spatial uncertainty u_s
(pooled prediction entropy) and temporal uncertainty u_t (normalized entropy of the readout's
attention over seasons), split per sample at q_s / q_t (Quadrant's "sample" mode). Per quadrant:
    share          of the cells
    pooled_loss    mean label loss of the pooled prediction (high space should = more to fix)
    season_spread  mean (max - min) over the cell's used seasons of each season map's loss
                   (how much the choice of season matters)
    attn_saved     share of random's season regret the most-attended season saves (high time
                   should = attention less reliable)
-> rows of results/pastis/quadrant_probe.csv.

    python -u -m exp.mixres.quadrant_probe --setup s16m8 [--q_s 0.25 --q_t 0.5]
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
from manyup.mixres.config import QuadrantConfig  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import (Context, Geometry, Quadrant, pixel_label_loss,  # noqa: E402
                                     season_maps)

QUADS = ("hs_ht", "hs_lt", "ls_ht", "ls_lt")


@torch.no_grad()
def evaluate(model, task, nw, bs, q_s, q_t, limit=0):
    P = model.cfg.lr_patch
    quad = Quadrant(Geometry(P // 2), QuadrantConfig(q_s=q_s, q_t=q_t), P, model.device)
    acc = {k: {"n": 0, "loss": 0.0, "spread": 0.0, "reg": 0.0, "rand": 0.0} for k in QUADS}
    for i, b in enumerate(loader(task, "test", bs, nw, False)):
        if limit and i >= limit:
            break
        ctx = Context.from_batch(model, b)
        B, NS = ctx.B, ctx.NS
        u_s, u_t, top = quad.uncertainties(ctx)
        hs, ht = quad.high(u_s, q_s, None), quad.high(u_t, q_t, None)
        tgt = b.target.cuda()
        pl = F.avg_pool2d(pixel_label_loss(ctx.out0.logits.float(), tgt)[:, None], P)[:, 0]
        lr_F = ctx.out0.lr_F
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sm = season_maps(model, lr_F)
        sl = pixel_label_loss(sm.float(), tgt.repeat_interleave(NS, 0))
        sl = F.avg_pool2d(sl[:, None], P)[:, 0].view(B, NS, *pl.shape[-2:])        # (B,NS,gh,gw)
        u = ctx.used[:, :, None, None].expand_as(sl)
        Lu = sl.masked_fill(~u, float("nan"))
        lo = Lu.nan_to_num(float("inf")).amin(1)
        hi = Lu.nan_to_num(float("-inf")).amax(1)
        multi = u.sum(1) >= 2
        reg = sl.gather(1, top[:, None])[:, 0] - lo
        rnd = Lu.nanmean(1) - lo
        for k, m in zip(QUADS, (hs & ht, hs & ~ht, ~hs & ht, ~hs & ~ht)):
            a = acc[k]
            a["n"] += int(m.sum())
            a["loss"] += float(pl[m].sum())
            mm = m & multi
            a["spread"] += float((hi - lo)[mm].sum())
            a["reg"] += float(reg[mm].sum())
            a["rand"] += float(rnd[mm].sum())
            a["n_multi"] = a.get("n_multi", 0) + int(mm.sum())
    tot = sum(a["n"] for a in acc.values())
    out = {}
    for k, a in acc.items():
        out[k] = {"share": a["n"] / tot, "pooled_loss": a["loss"] / max(a["n"], 1),
                  "season_spread": a["spread"] / max(a["n_multi"], 1),
                  "attn_saved": 1 - a["reg"] / max(a["rand"], 1e-9)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=tuple(RUNS))
    ap.add_argument("--q_s", type=float, default=0.25)
    ap.add_argument("--q_t", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()
    run, cfg, task, model, sel = load(RUNS[a.setup], a.num_workers)
    res = evaluate(model, task, a.num_workers, a.batch_size, a.q_s, a.q_t, a.limit_batches)
    print(f"{'quadrant':10s} {'share':>7s} {'pooled loss':>12s} {'season spread':>14s} "
          f"{'attn saved':>11s}")
    for k, r in res.items():
        print(f"{k:10s} {r['share']:7.3f} {r['pooled_loss']:12.4f} {r['season_spread']:14.4f} "
              f"{r['attn_saved']:11.3f}")
        if not a.limit_batches:
            _append(Path("results/pastis/quadrant_probe.csv"), {
                "job": os.environ.get("SLURM_JOB_ID", ""), "setup": a.setup, "run": run,
                "q_s": a.q_s, "q_t": a.q_t, "quadrant": k, **{x: round(v, 5) for x, v in r.items()}})


if __name__ == "__main__":
    main()
