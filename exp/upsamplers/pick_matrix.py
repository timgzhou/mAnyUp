"""Does training with a selector's picks shape what the model can use? Every model trained at one
(task, setup, n) -- the better-half oracle, the fine-tuned learned-selector arms and season grid --
tested with every way of picking the crops:
    weights   oracle     oracle_bothalf checkpoint (trained with label picks)
              linear / conv / sattn / mattn / smattn   warm from it, fine-tuned with that frozen
                         rank head's picks (arm A / C / attention arms)
              grid       trained from scratch with season_grid picks
              linear_s1/s2, grid_s1/s2   their extra seeds (lowest n only)
    picks     oracle     label scores (better half), as at oracle training
              learned    the weights' own rank head (linear head for oracle / grid weights)
              season_grid, none (no crops)
The diagonal (oracle/oracle, linear/learned, grid/season_grid) reproduces each run's own test.
-> results/<task>/pick_matrix.csv; rows already there are skipped (chained reruns resume).

    python -m exp.upsamplers.pick_matrix --task pastis --setup s16m8 --n 8
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

ARCHS = ("linear", "conv", "sattn", "mattn", "smattn")
PICKS = ("oracle", "learned", "season_grid", "none")
LOWEST = {"s16m8": 8, "s8m4": 32}


def ckpt(task, setup, n, weights):
    """weights -> (checkpoint name, rank head arch)."""
    seed = ""
    if weights[-3:-1] == "_s":
        weights, seed = weights[:-3], "_seed" + weights[-1]
    base = run_name(task, setup, n)
    if weights == "oracle":
        return base, "linear"
    if weights == "grid":
        tag = "_cpx8_poscenter_lr" if setup == "s16m8" else "_poscenter_lrps8_lr"
        return (f"inject_fft_{task}_tiny_season_grid_L0_k{n / 256:.3g}_loc_native_set{tag}0.001"
                f"_ep{EPOCHS[task]}{seed}"), "linear"
    pre, post = SETUPS[setup][3], (f"_cpx8_st8_poscenter_rk{weights}_warm_lr" if setup == "s16m8"
                                   else f"_poscenter_rk{weights}_warm_lrps8_lr")
    return base.replace(pre, post) + seed, weights


def load(task, setup, n, weights):
    name, arch = ckpt(task, setup, n, weights)
    m = build(task, setup, n, arch)                   # oracle ckpt + an untrained head of arch
    sd = torch.load(f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected or any(not k.startswith("rank_head.") for k in missing):
        raise SystemExit(f"{name}: missing {missing[:5]} unexpected {unexpected[:5]}")
    if missing:                                        # oracle / grid: the frozen linear head
        m.rank_head.load_state_dict(torch.load(
            f"checkpoints/{run_name(task, setup, n)}_rankhead.pt", map_location="cpu"))
    m.requires_grad_(False)
    return m.eval(), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPOCHS))
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--redo", nargs="*", default=[], choices=PICKS,
                    help="recompute these picks even if logged (the reader keeps the last row)")
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV")
    a = ap.parse_args()
    task = TASKS[a.task]
    dl = fi.loader(task, "test", a.batch_size, a.num_workers, False)
    out = Path(f"results/{a.task}/pick_matrix.csv")
    done = set()
    if out.exists() and not a.limit_batches:
        with open(out) as f:
            done = {(r["setup"], r["n"], r["weights"], r["picks"]) for r in csv.DictReader(f)}
    weights = ("oracle",) + ARCHS + ("grid",)
    if a.n == LOWEST[a.setup]:
        weights += ("linear_s1", "linear_s2", "grid_s1", "grid_s2")
    for w in weights:
        todo = [p for p in PICKS if p in a.redo or (a.setup, str(a.n), w, p) not in done]
        if not todo:
            continue
        m, name = load(a.task, a.setup, a.n, w)
        for p in todo:
            m.sel_name = "season_grid" if p == "season_grid" else "oracle_bothalf"
            m.score_src = "learned" if p == "learned" else "label"
            pick, lattice = m._season_pick, (m.loc_px, m.loc_fine)
            if p == "season_grid":                     # season grid's own (default) lattice, not
                m.loc_px, m.loc_fine = None, False     # the oracle's --loc_stride windows
            if p == "none":                            # the oracle's pass, then an empty mask
                m._season_pick = lambda *x, **kw: torch.zeros_like(pick(*x, **kw))
            torch.manual_seed(0)
            res, crops = fi.evaluate(m, task, dl, a.limit_batches)
            m._season_pick, (m.loc_px, m.loc_fine) = pick, lattice
            print(f"TEST {w} weights, {p} picks: primary {res['primary']:.4f} | {crops:.0f} "
                  f"crops/sample", flush=True)
            if a.limit_batches:
                continue
            _append(out, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name,
                          "setup": a.setup, "n": a.n, "weights": w, "picks": p,
                          "crops_per_sample": round(crops, 1),
                          **{f"test_{k_}": round(v, 5) for k_, v in res.items()}})


if __name__ == "__main__":
    main()
