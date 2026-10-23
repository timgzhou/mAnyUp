"""Does a location's entropy level tell when its season-entropy ranking is inverted?

At each location the time arms crop the seasons whose own prediction has the LOWEST entropy.
Per location, the Spearman r over its seasons between season entropy and season loss is mostly
near +1 but has a second mode near -1 (figures/entropy_align/time_s8m4_s788.png: the high-loss
blob). Rule: flip the ranking -- crop the HIGHEST-entropy seasons -- where a location feature is
above a threshold:
    abs   the location's mean entropy over its used seasons
    pct   its percentile among the tile's locations
On a FROZEN oracle-trained time model (scripts/slurm/mixres_time_arms.sh, label-every_location):
  1. val: per feature decile, the mean r and the share of locations with r < 0 (a threshold can
     only work if r falls with the feature)
  2. val: the threshold whose picks match the oracle's picks best (a grid over quantiles)
  3. test mIoU with picks = oracle / lowest entropy / highest / flip(abs) / flip(pct) / random
-> printed, and results/pastis/time_flip_eval.csv.

    python -u -m exp.mixres.time_flip_eval --run mixres_pastis_ps8_label-every_location-one_n0_c4_warm_seed0_9d93609c
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import (H, NEG, Context, Entropy, EveryLocation,  # noqa: E402
                                     Geometry, LabelLoss, Scores)

QS = torch.linspace(0, 1, 21)


def location_scores(model, geo, batch):
    """-> per (b, season, location): entropy E and label loss L (B, NS, 16, 16; -inf off the
    candidates), used seasons (B, NS), candidate cells (16, 16), and the context."""
    ctx = Context.from_batch(model, batch)
    E = Entropy()(ctx, geo, ("season",)).season                 # as the selectors see them
    L = LabelLoss()(ctx, geo, ("season",)).season
    cand = geo.anchors(E.device)
    cand = torch.ones(H, H, dtype=torch.bool, device=E.device) if cand is None else cand
    return E, L, ctx.used, cand, ctx


def features(E, used, cand):
    """(B, 16, 16) mean entropy over used seasons, and its percentile among the tile's
    candidate locations."""
    u = used[:, :, None, None].float()
    m = (E.clamp(min=0) * u).sum(1) / u.sum(1).clamp(min=1)
    flat = m.masked_fill(~cand, float("nan")).flatten(1)
    rank = flat.nan_to_num(float("inf")).argsort(1).argsort(1).float()
    n = cand.sum().float()
    pct = (rank / (n - 1).clamp(min=1)).view_as(m)
    return m, pct


def spearman(E, L, used):
    """(B, 16, 16) Spearman r over each location's used seasons (nan with < 3)."""
    u = used[:, :, None, None].expand_as(E)

    def rank(x):
        x = x.masked_fill(~u, float("inf"))
        r = x.argsort(1).argsort(1).float()
        return r.masked_fill(~u, float("nan"))
    rE, rL = rank(E), rank(L)
    n = u.float().sum(1)
    mE, mL = rE.nanmean(1, keepdim=True), rL.nanmean(1, keepdim=True)
    cov = ((rE - mE) * (rL - mL)).nansum(1)
    sd = (((rE - mE) ** 2).nansum(1) * ((rL - mL) ** 2).nansum(1)).sqrt()
    r = cov / sd.clamp(min=1e-9)
    return r.masked_fill(n < 3, float("nan"))


def flipped(E, flip):
    """Season scores whose lowest = the pick: E where not flipped, -E where flipped."""
    s = torch.where(flip[:, None], -E, E)
    return s.masked_fill(~torch.isfinite(E), NEG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an oracle-trained every_location checkpoint")
    ap.add_argument("--val_batches", type=int, default=0, help="0: all")
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    ck = torch.load(f"checkpoints/{a.run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    task = TASKS[cfg.task]
    model = MixedResModel(task, cfg.model).cuda().eval()
    load_legacy(model, a.run)
    geo = Geometry(cfg.select.crop_px, cfg.select.loc_stride)
    keep = cfg.select.keep
    alloc = EveryLocation(keep, "low")
    bs = cfg.train.batch_size or task.batch_size[16]
    print(f"{a.run}: keep {keep}", flush=True)

    # 1 + 2: val features, r and the oracle's picks
    feats, rs, agree = {"abs": [], "pct": []}, [], []
    ths = None
    val = []
    with torch.no_grad():
        for i, b in enumerate(loader(task, "val", bs, a.num_workers, False)):
            if a.val_batches and i >= a.val_batches:
                break
            E, L, used, cand, ctx = location_scores(model, geo, b)
            m, pct = features(E, used, cand)
            r = spearman(E, L, used)
            ok = cand & torch.isfinite(r)
            feats["abs"].append(m[ok].cpu())
            feats["pct"].append(pct[ok].cpu())
            rs.append(r[ok].cpu())
            oracle = alloc(Scores(season=L), ctx, geo)
            val.append((E.cpu(), m.cpu(), pct.cpu(), oracle.cpu(), ctx.season.cpu(),
                        ctx.present.cpu(), ctx.valid.cpu(), used.cpu()))
    r_all = torch.cat(rs)
    print(f"val: {len(r_all)} locations with >= 3 seasons | mean r {r_all.mean():+.3f} | "
          f"r < 0: {(r_all < 0).float().mean():.1%}", flush=True)
    for name in ("abs", "pct"):
        f = torch.cat(feats[name])
        edges = f.quantile(QS[::2])
        print(f"  by {name} decile: " + " ".join(
            f"[{(r_all[(f >= lo) & (f <= hi)]).mean():+.2f} {(r_all[(f >= lo) & (f <= hi)] < 0).float().mean():.0%}]"
            for lo, hi in zip(edges[:-1], edges[1:])), flush=True)

    class Ctx:      # the parts of a Context the allocator reads
        def __init__(self, season, present, valid):
            self.season, self.present, self.valid = season, present, valid
            self.B, self.T = present.shape

    def match(th, name):
        """share of the oracle's picks matched by the flip rule at threshold th (val)."""
        hit = tot = 0
        for E, m, pct, oracle, season, present, valid, used in val:
            f = m if name == "abs" else pct
            pick = alloc(Scores(season=flipped(E, f > th)), Ctx(season, present, valid), geo)
            hit += int((pick & oracle).sum())
            tot += int(oracle.sum())
        return hit / max(tot, 1)

    best = {}
    for name in ("abs", "pct"):
        f = torch.cat(feats[name])
        cands = [float("inf")] + f.quantile(QS).tolist()[::-1] + [float("-inf")]
        res = [(match(th, name), th) for th in cands]
        acc, th = max(res)
        best[name] = th
        print(f"  {name}: oracle-pick match never flip {res[0][0]:.3f} | always {res[-1][0]:.3f}"
              f" | best {acc:.3f} at threshold {th:.4g} "
              f"(flips {(f > th).float().mean():.0%} of locations)", flush=True)

    # 3: test mIoU on the frozen model
    rules = ["oracle", "low", "high", "flip_abs", "flip_pct", "random"]
    out = Path(f"results/{cfg.task}/time_flip_eval.csv")
    for rule in rules:
        meter, crops, n = task.meter(), 0, 0
        torch.manual_seed(0)
        with torch.no_grad():
            for b in loader(task, "test", bs, a.num_workers, False):
                E, L, used, cand, ctx = location_scores(model, geo, b)
                m, pct = features(E, used, cand)
                sc = {"oracle": L, "low": E, "high": flipped(E, torch.ones_like(m, dtype=torch.bool)),
                      "flip_abs": flipped(E, m > best["abs"]), "flip_pct": flipped(E, pct > best["pct"]),
                      "random": torch.rand_like(E).masked_fill(~torch.isfinite(E), NEG)}[rule]
                mask = alloc(Scores(season=sc), ctx, geo)
                plan = geo.plan(mask)
                logits = model(b.ms, b.ss, b.season, b.present, plan, None, None,
                               ctx.seasons).logits
                meter.update(logits, b.target.cuda())
                crops, n = crops + len(plan), n + b.target.shape[0]
        res = meter.compute()
        print(f"TEST {rule:9s} {res['primary']:.4f} | {crops / n:.1f} crops/sample", flush=True)
        _append(out, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": a.run, "keep": keep,
                      "rule": rule, "threshold": best.get(rule.split("_")[-1], ""),
                      "crops_per_sample": round(crops / n, 1),
                      **{f"test_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
