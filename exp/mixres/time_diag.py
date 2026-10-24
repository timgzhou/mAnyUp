"""Why does random beat entropy at choosing seasons (time table, s8m4), although entropy ranks a
location's seasons well on average? Two diagnostics on the trained one-season time arms
(scripts/slurm/mixres_time_arms.sh, mixres_time_sign.sh), val split, no training:

1  WHERE IT MATTERS (the trained lowest-entropy model, its own uniform pass): per location with
   >= 3 seasons, the regret of a pick = its season's loss - the location's lowest season loss.
   Entropy's pick (lowest-entropy season) vs random's expected regret (mean over the seasons),
   by decile of the location's loss spread (max - min over its seasons: how much the choice
   matters). Share of random's regret that entropy saves = 1 - regret_entropy / regret_random.
2  NEIGHBOUR DIVERSITY (each arm's own trained model and selector): over 2 x 2 blocks of
   neighbouring candidate locations, the share whose four locations got identical months, and
   the distinct months the block covers (mean).

    python -u -m exp.mixres.time_diag --setup s8m4
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import Context, Entropy, LabelLoss, build_selector  # noqa: E402

ARMS = {"random": "season_random-every_location-one", "grid": "none-month_grid4",
        "entropy": "entropy-every_location-one", "entropy_sign": "entropy_sign-every_location-one",
        "oracle": "label-every_location-one"}


def run_name(ps, tag):
    for r in csv.DictReader(open("results/pastis/mixres.csv")):
        if r["run"].startswith(f"mixres_pastis_ps{ps}_{tag}_") and "-high" not in r["run"]:
            return r["run"]
    raise SystemExit(f"no run ps{ps} {tag}")


def load(run, bs, nw):
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    task = TASKS[cfg.task]
    model = MixedResModel(task, cfg.model).cuda().eval()
    load_legacy(model, run)
    b0 = next(iter(loader(task, "val", bs, nw, False)))
    sel = build_selector(cfg.select, model, (b0.ms, b0.ss))
    nets = nn.ModuleList(sel.modules())
    if len(nets):
        nets.load_state_dict(ck["selector"])
        nets.eval()
    return task, model, sel


@torch.no_grad()
def where_it_matters(task, model, sel, batches, bs, nw):
    """Regret (picked season's loss - the location's lowest) of the lowest-entropy pick, the
    highest-entropy pick (reversed) and a random pick (expected), binned two ways: by the
    location's loss spread (labels: how much the choice matters) and by its mean entropy
    (label-free: what a deployable reversal rule could bin by)."""
    geo = sel.geo
    cell = geo.tiling()
    spread, ment, reg_lo, reg_hi, reg_r = [], [], [], [], []
    for i, b in enumerate(loader(task, "val", bs, nw, False)):
        if i >= batches:
            break
        ctx = Context.from_batch(model, b)
        c = cell.to(ctx.present.device)
        E = Entropy()(ctx, geo, ("season",)).season[:, :, c[:, None], c[None, :]]
        L = LabelLoss()(ctx, geo, ("season",)).season[:, :, c[:, None], c[None, :]]
        u = ctx.used[:, :, None, None].expand_as(E)
        ok = u.sum(1) >= 3                                                  # (B, S, S)
        Lu = L.masked_fill(~u, float("nan"))
        lo, hi = Lu.nan_to_num(float("inf")).amin(1), Lu.nan_to_num(float("-inf")).amax(1)
        p_lo = E.masked_fill(~u, float("inf")).argmin(1, keepdim=True)
        p_hi = E.masked_fill(~u, float("-inf")).argmax(1, keepdim=True)
        reg_lo.append((L.gather(1, p_lo)[:, 0] - lo)[ok].cpu())
        reg_hi.append((L.gather(1, p_hi)[:, 0] - lo)[ok].cpu())
        reg_r.append((Lu.nanmean(1) - lo)[ok].cpu())
        spread.append((hi - lo)[ok].cpu())
        ment.append(E.masked_fill(~u, float("nan")).nanmean(1)[ok].cpu())
    s, me = torch.cat(spread), torch.cat(ment)
    lo_, hi_, r_ = torch.cat(reg_lo), torch.cat(reg_hi), torch.cat(reg_r)
    print(f"1  {len(s)} locations | mean regret: lowest-entropy {lo_.mean():.4f} | "
          f"highest-entropy {hi_.mean():.4f} | random {r_.mean():.4f}")
    for name, key in (("loss spread (labels)", s), ("mean entropy (label-free)", me)):
        edges = key.quantile(torch.linspace(0, 1, 11))
        print(f"   by {name} decile: regret lowest / highest / random | share of random regret")
        best_mix = 0.0
        for k in range(10):
            m = (key >= edges[k]) & (key <= edges[k + 1])
            a_, b_, c_ = lo_[m].mean(), hi_[m].mean(), r_[m].mean()
            best_mix += float(min(lo_[m].sum(), hi_[m].sum()))
            print(f"     {k + 1:2d}: {a_:.4f} / {b_:.4f} / {c_:.4f}"
                  f"{'  <- reversing helps' if b_ < a_ else ''} | {float(r_[m].sum() / r_.sum()):5.1%}")
        print(f"   best per-decile choice of lowest / highest ({name}): mean regret "
              f"{best_mix / len(s):.4f}", flush=True)


@torch.no_grad()
def diversity(task, model, sel, batches, bs, nw):
    geo = sel.geo
    cell = geo.tiling()
    S = len(cell)
    same = blocks = 0
    distinct = []
    torch.manual_seed(0)
    for i, b in enumerate(loader(task, "val", bs, nw, False)):
        if i >= batches:
            break
        ctx = Context.from_batch(model, b)
        m = sel(ctx).mask                                                   # (B, T, 16, 16)
        c = cell.to(m.device)
        m = m[:, :, c[:, None], c[None, :]]                                 # (B, T, S, S)
        blk = m[:, :, : S // 2 * 2, : S // 2 * 2].unflatten(2, (S // 2, 2)).unflatten(4, (S // 2, 2))
        blk = blk.permute(0, 2, 4, 1, 3, 5).flatten(4)                      # (B, by, bx, T, 4)
        has = blk.any(3).all(-1)                          # every location of the block has crops
        ident = (blk == blk[..., :1]).all(-1).all(-1) & has
        same += int(ident.sum())
        blocks += int(has.sum())
        distinct.append(blk.any(-1).sum(-1)[has].float().cpu())
    d = torch.cat(distinct)
    return same / max(blocks, 1), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=("s16m8", "s8m4"))
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--only_diversity", action="store_true")
    a = ap.parse_args()
    ps = 16 if a.setup == "s16m8" else 8
    print(f"== {a.setup} (one season per location)", flush=True)
    bs = 32
    if not a.only_diversity:
        task, model, sel = load(run_name(ps, ARMS["entropy"]), bs, a.num_workers)
        where_it_matters(task, model, sel, a.batches, bs, a.num_workers)
    print("2  neighbour diversity over 2 x 2 blocks of locations (each arm's own trained model):")
    for arm, tag in ARMS.items():
        task, model, sel = load(run_name(ps, tag), bs, a.num_workers)
        same, d = diversity(task, model, sel, a.batches, bs, a.num_workers)
        print(f"   {arm:13s} identical months in all 4: {same:6.1%} | distinct months per block "
              f"{d:.2f}", flush=True)


if __name__ == "__main__":
    main()
