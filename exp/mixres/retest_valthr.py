"""Re-test finished global-mode Quadrant runs with the spatial threshold of the val split
(exp.mixres.train.val_threshold, the default at test time since 2026-10-26): the run's last
row of results/<task>/mixres.csv again, with the new test metrics, crops and GMACs and
test_threshold = val (a later row of the same run: what the plots read). The checkpoint is
left as it is (its selector keeps the training table).

    python -u -m exp.mixres.retest_valthr --runs <checkpoint name> ...
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

from exp.mixres.compute_breakdown import load  # noqa: E402
from exp.mixres.train import evaluate, measure_compute, val_threshold  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    for run in a.runs:
        cfg, task, model, sel = load(run, 8)
        path = Path(f"results/{cfg.task}/mixres.csv")
        with open(path, newline="") as f:
            base = [r for r in csv.DictReader(f) if r["run"] == run][-1]
        bs = cfg.train.batch_size or task.batch_size[16]
        dl = {s: loader(task, s, bs, 8, False) for s in ("val", "test")}
        if not val_threshold(model, sel, dl["val"]):
            raise SystemExit(f"{run}: not a global-mode Quadrant run")
        test, crops = evaluate(model, sel, task, dl["test"])
        cost = measure_compute(model, sel, dl["val"], cfg.train.compute_batches)
        row = {**base, "job": os.environ.get("SLURM_JOB_ID", ""),
               "test_crops_per_sample": round(crops, 1), **cost, "test_threshold": "val",
               **{f"test_{k}": round(v, 5) for k, v in test.items()}}
        print(f"{run}: test miou {base['test_miou']} -> {row['test_miou']} | crops "
              f"{base['test_crops_per_sample']} -> {row['test_crops_per_sample']} | GMACs "
              f"{base['gmacs']} -> {row['gmacs']}", flush=True)
        _append(path, row)


if __name__ == "__main__":
    main()
