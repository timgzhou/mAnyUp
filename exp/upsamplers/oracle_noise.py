"""How much ranking error can the better-half seasons oracle tolerate? Its own checkpoint tested
with its (season, window) picks made from label scores corrupted by noise, on the location
ranking, the season ranking, or both, and against no crops / random / grid crops at the same n.

Noise (Gaussian copula, per list): each list's scores -> normal quantiles of their ranks z,
then z' = rho z + sqrt(1 - rho^2) eps, eps ~ N(0, 1); corr(z, z') = rho, so rho = 1 is the oracle
and rho = 0 a uniformly random ranking. Lists are the same ones the oracle ranks:
    loc    a sample's candidate windows (which n windows)
    sea    a window's used seasons (which half is cropped)
    both   both, the same rho
Logged with each row: location recall, season agreement on shared windows and pick recall of the
noisy picks vs the clean oracle's (as select_diag logs the learned head), so a learned selector
can be placed on these curves. Controls on the same weights: none (no crops), season_random,
season_grid. -> results/<task>/oracle_noise.csv (rows already there are skipped).

    python -m exp.upsamplers.oracle_noise --task pastis --setup s16m8 --n 8
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
from exp.upsamplers.learned_select import EPOCHS, SETUPS, build, run_name  # noqa: E402

RHOS = (0.9, 0.75, 0.5, 0.25, 0.0)
CONTROLS = ("none", "season_random", "season_grid")


def gauss_ranks(s, dim):
    """Finite entries of s -> the normal quantiles of their ranks along dim (-inf kept)."""
    fin = torch.isfinite(s)
    x = s.masked_fill(~fin, float("-inf"))
    r = x.argsort(dim).argsort(dim).float()                        # -inf first: rank m.. last
    m = fin.sum(dim, keepdim=True).clamp(min=1).float()
    r = r - (s.shape[dim] - m)                                     # finite ranks 0 .. m-1
    q = ((r + 0.5) / m).clamp(1e-4, 1 - 1e-4)
    return torch.special.ndtri(q).masked_fill(~fin, float("-inf")), fin


def corrupt(s, dim, rho):
    if rho >= 1:
        return s
    z, fin = gauss_ranks(s, dim)
    z = rho * z + (1 - rho ** 2) ** 0.5 * torch.randn_like(z)
    return z.masked_fill(~fin, float("-inf"))


class Noisy:
    """Wraps the model's _season_pick: perturbs the oracle's scores, tallies overlap with the
    clean picks."""

    def __init__(self, m):
        self.m, self.orig = m, m._season_pick
        self.axis, self.rho, self.blank = None, 1.0, False
        self.reset()
        m._season_pick = self

    def reset(self):
        self.t = dict(loc_hit=0.0, loc_tot=0.0, sea_hit=0.0, sea_tot=0.0, all_hit=0.0, all_tot=0.0)

    def __call__(self, sc, season, present, B, H, n, lr, G, extra, loc=None):
        clean = self.orig(sc, season, present, B, H, n, lr, G, extra, loc)
        if self.blank:
            return torch.zeros_like(clean)
        if self.axis is None or not torch.is_tensor(loc):
            return clean
        sc_n = corrupt(sc.float(), 1, self.rho) if self.axis in ("sea", "both") else sc
        loc_n = (corrupt(loc.float().flatten(1), 1, self.rho).view_as(loc)
                 if self.axis in ("loc", "both") else loc)
        got = self.orig(sc_n, season, present, B, H, n, lr, G, extra, loc_n)
        rw, gw = clean.any(1), got.any(1)
        both = (rw & gw)[:, None]
        for k, v in (("loc_hit", rw & gw), ("loc_tot", rw), ("sea_hit", clean & got & both),
                     ("sea_tot", clean & both), ("all_hit", clean & got), ("all_tot", clean)):
            self.t[k] += float(v.sum())
        return got

    def recall(self):
        t = self.t
        if not t["loc_tot"]:
            return {}
        return {"location_recall": round(t["loc_hit"] / t["loc_tot"], 4),
                "season_agreement": round(t["sea_hit"] / max(t["sea_tot"], 1), 4),
                "pick_recall": round(t["all_hit"] / t["all_tot"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPOCHS))
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV")
    a = ap.parse_args()
    task = TASKS[a.task]
    name = run_name(a.task, a.setup, a.n)
    dl = fi.loader(task, "test", a.batch_size, a.num_workers, False)
    out = Path(f"results/{a.task}/oracle_noise.csv")
    done = set()
    if out.exists() and not a.limit_batches:
        with open(out) as f:
            done = {(r["run"], r["picks"], r["axis"], r["rho"], r["seed"]) for r in csv.DictReader(f)}
    m = build(a.task, a.setup, a.n)
    m.rank_head = None
    nz = Noisy(m)
    arms = ([("oracle", "-", 1.0)] + [("oracle", ax, r) for ax in ("loc", "sea", "both")
                                       for r in RHOS] + [(c, "-", 0.0) for c in CONTROLS])
    for picks, axis, rho in arms:
        key = (name, picks, axis, str(rho), str(a.seed))
        if key in done:
            continue
        m.sel_name = "oracle_bothalf" if picks in ("oracle", "none") else picks
        nz.blank = picks == "none"
        nz.axis, nz.rho = (None if axis == "-" else axis), rho
        nz.reset()
        torch.manual_seed(a.seed)
        res, crops = fi.evaluate(m, task, dl, a.limit_batches)
        rec = nz.recall()
        print(f"TEST {picks} {axis} rho={rho}: primary {res['primary']:.4f} | {crops:.0f} "
              f"crops/sample | {rec}", flush=True)
        if a.limit_batches:
            continue
        _append(out, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "setup": a.setup,
                      "n": a.n, "picks": picks, "axis": axis, "rho": rho, "seed": a.seed,
                      "location_recall": rec.get("location_recall", ""),
                      "season_agreement": rec.get("season_agreement", ""),
                      "pick_recall": rec.get("pick_recall", ""), "crops_per_sample": round(crops, 1),
                      **{f"test_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
