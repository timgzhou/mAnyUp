"""Is the better-half oracle's lead high-res DETAIL, or label information carried by WHICH
(season, window) pairs get crops? Same models and picks as exp.upsamplers.pick_matrix, but the
picked crops' pixels replaced (InjectFT.crop_content); the model is still told each crop's month
and position, so the readout sees the same set of candidates:
    real    the crop itself
    mean    right window and month, each band flattened to its mean over the crop (no detail)
    shift   right month, a random window of the same tile (wrong place)
    swap    right month and window, from the next sample in the batch with that month
Weights x picks: oracle x oracle, linear (arm A) x {oracle, learned}, grid x season_grid.
-> results/<task>/crop_content.csv; rows already there are skipped (chained reruns resume).

    python -m exp.upsamplers.crop_content_test --task pastis --setup s16m8 --n 8
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
from exp.upsamplers.learned_select import EPOCHS, SETUPS  # noqa: E402
from exp.upsamplers.pick_matrix import load  # noqa: E402

CONTENT = ("real", "mean", "shift", "swap")
ARMS = (("oracle", "oracle"), ("linear", "oracle"), ("linear", "learned"), ("grid", "season_grid"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPOCHS))
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV")
    a = ap.parse_args()
    task = TASKS[a.task]
    dl = fi.loader(task, "test", a.batch_size, a.num_workers, False)
    out = Path(f"results/{a.task}/crop_content.csv")
    done = set()
    if out.exists() and not a.limit_batches:
        with open(out) as f:
            done = {(r["setup"], r["n"], r["weights"], r["picks"], r["content"])
                    for r in csv.DictReader(f)}
    loaded = None
    for w, p in ARMS:
        todo = [c for c in CONTENT if (a.setup, str(a.n), w, p, c) not in done]
        if not todo:
            continue
        if loaded is None or loaded[0] != w:
            loaded = (w, *load(a.task, a.setup, a.n, w))
        m, name = loaded[1], loaded[2]
        m.sel_name = "season_grid" if p == "season_grid" else "oracle_bothalf"
        m.score_src = "learned" if p == "learned" else "label"
        lattice = (m.loc_px, m.loc_fine)
        if p == "season_grid":                  # season grid's own (default) lattice
            m.loc_px, m.loc_fine = None, False
        for c in todo:
            m.crop_content, m._swap_fallback = c, 0.0
            torch.manual_seed(0)
            res, crops = fi.evaluate(m, task, dl, a.limit_batches)
            print(f"TEST {w} weights, {p} picks, {c} content: primary {res['primary']:.4f} | "
                  f"{crops:.0f} crops/sample", flush=True)
            if a.limit_batches:
                continue
            _append(out, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name,
                          "setup": a.setup, "n": a.n, "weights": w, "picks": p, "content": c,
                          "crops_per_sample": round(crops, 1),
                          **{f"test_{k}": round(v, 5) for k, v in res.items()}})
        m.crop_content = "real"
        m.loc_px, m.loc_fine = lattice


if __name__ == "__main__":
    main()
