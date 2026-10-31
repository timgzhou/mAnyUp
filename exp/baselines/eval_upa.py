"""Upsample Anything at test time: the head of the best-val bilinear run (exp.mixres.train) on the
same fixed model, valid because both outputs are convex combinations of the same tokens, kernels
fitted per test frame (~6 frames/s on an L40S, so the test set is split over jobs):

    python -u -m exp.baselines.eval_upa --task biomassters --patch 8 --tokens seasons --shard 0/4
    python -u -m exp.baselines.eval_upa --task biomassters --patch 8 --tokens seasons --merge 4
    -> a row of results/<task>/mixres.csv

--merge also traces the cost (GMACs include the fit, gmacs_analytic is that part).
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import functools  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.mixres.train import _append, flat, measure_compute, step  # noqa: E402
from manyup.mixres import load_legacy  # noqa: E402
from manyup.mixres.baselines import build_model, build_selector_for  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402


def best_bilinear(task, patch: int, tokens: str, seed: int) -> str:
    """The bilinear run (of the learning rates tried) with the best val metric."""
    d = pd.read_csv(f"results/{task.name}/mixres.csv", low_memory=False)
    d = d[d.run.astype(str).str.contains(
        rf"^mixres_{task.name}_ps{patch}_upsample-{tokens}-bilinear-lp_.*_seed{seed}_")]
    if not len(d):
        raise ValueError(f"no bilinear run for ps{patch} {tokens} seed {seed}: run it first")
    d = d.sort_values("val_primary", ascending=not task.meter().higher_is_better)
    return d.run.iloc[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--patch", type=int, required=True)
    ap.add_argument("--tokens", default="seasons")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th test batch, from the i-th")
    ap.add_argument("--merge", type=int, default=0, help="add n shards' meters, write the row")
    a = ap.parse_args()
    task = TASKS[a.task]
    head_run = best_bilinear(task, a.patch, a.tokens, a.seed)
    ck = torch.load(f"checkpoints/{head_run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    cfg.baseline.upsampler = "upa"
    cfg.train.head_only, cfg.train.epochs, cfg.train.freeze_model = False, 0, True
    cfg.train.init_ckpt = head_run
    name = cfg.name()
    part = lambda i, n: Path(f"checkpoints/{name}_shard{i}of{n}.pt")             # noqa: E731
    model = build_model(task, cfg).cuda().eval()
    load_legacy(model, head_run)
    selector = build_selector_for(task, cfg, model, None)
    dl = loader(task, "test", a.batch_size, a.num_workers, False)
    if a.merge:
        meter = functools.reduce(lambda x, y: x.merge(y), [
            torch.load(part(i, a.merge), weights_only=False) for i in range(a.merge)])
        test = meter.compute()
        with torch.no_grad():
            cost = measure_compute(model, selector, dl, 1)
        _append(Path(f"results/{a.task}/mixres.csv"), {
            "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, **flat(cfg.to_dict()),
            "batch_size": a.batch_size, "test_crops_per_sample": 0, **cost,
            **{f"test_{k}": round(v, 5) for k, v in test.items()}})
        print(f"TEST {name}: {test['primary']:.4f} | {cost['gmacs']:.2f} GMACs/sample", flush=True)
        return
    i, n = map(int, a.shard.split("/"))
    meter = task.meter()
    with torch.no_grad():
        for j, batch in enumerate(dl):
            if j % n != i:
                continue
            if a.limit_batches and j >= a.limit_batches * n:
                break
            logits = step(model, selector, batch)[0]
            meter.update(logits, batch.target.cuda())
            print(f"batch {j}/{len(dl)}", flush=True)
    torch.save(meter, part(i, n))


if __name__ == "__main__":
    main()
