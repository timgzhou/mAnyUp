"""Build a task's 64 px tile store (manyup/data/tile_store.py) from its raw data.

    python -u -m exp.common.build_tile_store --task biomassters --num_workers 16
    sbatch -J store_bio --gres=none --cpus-per-task=17 --mem=64G --time=6:00:00 \
        scripts/slurm/run.sh exp.common.build_tile_store --task biomassters --num_workers 16

Resumable (finished shards are kept). After building, every split is spot-checked against
the raw dataset read with the task's time setup (--check samples per split).
"""
import argparse
import random

import torch

from exp.common.paths import TILESTORE
from exp.common.tasks import SPLITS, TASKS
from manyup.data.tile_store import build_tile_store, store_name


def same(a, b, where: str) -> None:
    """Every field of two MTMMSamples equal (tensors exactly, dtype included)."""
    def eq(x, y, what):
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            ok = (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                  and x.dtype == y.dtype and x.shape == y.shape
                  and (torch.equal(x, y) or (x.is_floating_point()          # NaN == NaN
                                             and torch.equal(x.isnan(), y.isnan())
                                             and torch.equal(x.nan_to_num(), y.nan_to_num()))))
        else:
            ok = x == y
        if not ok:
            raise AssertionError(f"{where}: {what} differs")
    eq(a.id, b.id, "id")
    eq(a.split, b.split, "split")
    eq(a.meta, b.meta, "meta")
    eq(a.target_geometry, b.target_geometry, "target_geometry")
    eq(a.target, b.target, "target")
    eq(list(a.modalities), list(b.modalities), "modalities")
    for m, x in a.modalities.items():
        y = b.modalities[m]
        for f in ("data", "frame_valid", "pixel_valid"):
            if (getattr(x, f) is None) != (getattr(y, f) is None):
                raise AssertionError(f"{where}: {m}.{f} presence differs")
            if getattr(x, f) is not None:
                eq(getattr(x, f), getattr(y, f), f"{m}.{f}")
        for f in ("bands", "geometry", "history"):
            eq(getattr(x, f), getattr(y, f), f"{m}.{f}")
        eq(sorted(x.aux), sorted(y.aux), f"{m}.aux keys")
        for k in x.aux:
            eq(x.aux[k], y.aux[k], f"{m}.aux.{k}")
        if (x.time is None) != (y.time is None):
            raise AssertionError(f"{where}: {m}.time presence differs")
        if x.time is not None:
            eq(x.time.start, y.time.start, f"{m}.time.start")
            eq(x.time.end, y.time.end, f"{m}.time.end")
            eq(x.time.precision, y.time.precision, f"{m}.time.precision")
            eq(x.time.sources, y.time.sources, f"{m}.time.sources")


def check(task, split: str, n: int, seed: int = 0) -> None:
    store, raw = task.dataset(split), task.raw(split, task.policies())
    if type(store) is type(raw):
        raise SystemExit(f"{task.name}/{split}: no store found")
    if store.key != raw.key or len(store) != len(raw):
        raise AssertionError(f"{task.name}/{split}: key/length differ "
                             f"({store.key} {len(store)} vs {raw.key} {len(raw)})")
    for i in random.Random(seed).sample(range(len(raw)), min(n, len(raw))):
        same(store[i], raw[i], f"{task.name}/{split}/{i}")
    print(f"[{split}] {min(n, len(raw))} samples identical to the raw dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--check", type=int, default=200, help="samples per split to verify")
    a = ap.parse_args()
    task = TASKS[a.task]
    for split in a.splits.split(","):
        src = task.source(split)
        if store_name(src) != task.store_name:
            raise SystemExit(f"store name {store_name(src)} != task's {task.store_name}")
        build_tile_store(src, TILESTORE, task.store_dtypes, natives_per_shard=task.store_shard,
                         num_workers=a.num_workers)
        check(task, split, a.check)


if __name__ == "__main__":
    main()
