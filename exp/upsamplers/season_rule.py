"""Learned windows, fixed season rule: does dropping the (unpredictable) per-window season
choice help? On the frozen better-half checkpoint with its linear rank head (arm A's head):
  1. the season prior: over the train split, how often each season index lands in a window's
     better (lower-loss) half under the oracle -> checkpoints/<ckpt>_season_prior.pt, (NS,) score
     = 1 - that share (low = usually picked), for --season_rule prior
  2. test with windows {learned (the head's location scores), oracle (label)} x season rule
     {score (the scores' better half: learned / oracle seasons), all, prior, random}
-> results/<task>/season_rule.csv (rows already there are skipped).

    python -m exp.upsamplers.season_rule --task pastis --setup s16m8 --n 8
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.learned_select import EPOCHS, SETUPS, build, collect, present_seasons, run_name  # noqa: E402

RULES = ("score", "all", "prior", "random")


@torch.no_grad()
def season_prior(m, dl, batches):
    """(NS,) share of (sample, oracle window) lists where each season index is in the better
    half, among lists where it is used; returned as 1 - share."""
    hit = tot = None
    for i, (*x, _) in enumerate(dl):
        if i >= batches:
            break
        rb = collect(m, x)
        sc = rb["sc"].float()                                              # (B, NS, H, W)
        used = present_seasons(rb)                                         # (B, NS)
        sc = sc.masked_fill(~used[:, :, None, None], float("-inf"))
        nused = used.sum(1)[:, None, None, None]
        half = (nused // 2).clamp(min=1)
        rank = sc.argsort(1, descending=True).argsort(1)                   # 0 = worst
        better = (rank >= nused - half) & (rank < nused) & torch.isfinite(sc)
        cand = torch.isfinite(rb["loc"].float())[:, None] & used[:, :, None, None]
        h, t = (better & cand).sum((0, 2, 3)).float(), cand.sum((0, 2, 3)).float()
        hit, tot = (h, t) if hit is None else (hit + h, tot + t)
    share = hit / tot.clamp(min=1)
    return 1 - share, share, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPOCHS))
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--prior_batches", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV, no save")
    a = ap.parse_args()
    task = TASKS[a.task]
    name = run_name(a.task, a.setup, a.n)
    m = build(a.task, a.setup, a.n)
    m.rank_head.load_state_dict(torch.load(f"checkpoints/{name}_rankhead.pt", map_location="cpu"))
    m.requires_grad_(False)
    prior_path = Path(f"checkpoints/{name}_season_prior.pt")
    if prior_path.exists() and not a.limit_batches:
        m.season_prior = torch.load(prior_path, map_location="cpu")
    else:
        dl = fi.loader(task, "train", a.batch_size, a.num_workers, True)
        m.season_prior, share, tot = season_prior(m, dl, a.limit_batches or a.prior_batches)
        print("better-half share per season index:", [round(float(v), 3) for v in share],
              "| lists:", [int(v) for v in tot], flush=True)
        if not a.limit_batches:
            torch.save(m.season_prior.cpu(), prior_path)
    dl = fi.loader(task, "test", a.batch_size, a.num_workers, False)
    out = Path(f"results/{a.task}/season_rule.csv")
    done = set()
    if out.exists() and not a.limit_batches:
        with open(out) as f:
            done = {(r["setup"], r["n"], r["windows"], r["rule"]) for r in csv.DictReader(f)}
    for windows in ("learned", "oracle"):
        for rule in RULES:
            if (a.setup, str(a.n), windows, rule) in done:
                continue
            # "score" reads the seasons from the same source as the windows: learned = arm A's
            # picks on the frozen model, oracle = the oracle itself
            m.score_src = ("learned" if windows == "learned" else "label")
            m.season_rule = rule
            torch.manual_seed(0)
            res, crops = fi.evaluate(m, task, dl, a.limit_batches)
            print(f"TEST {windows} windows, {rule} seasons: primary {res['primary']:.4f} | "
                  f"{crops:.0f} crops/sample", flush=True)
            if a.limit_batches:
                continue
            _append(out, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name,
                          "setup": a.setup, "n": a.n, "windows": windows, "rule": rule,
                          "crops_per_sample": round(crops, 1),
                          **{f"test_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
