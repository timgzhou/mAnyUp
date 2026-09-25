"""Fit and cache UPA / UPMA kernel fields for a task (manyup/upsamplers/jbu.py, KernelCache).

A kernel field is fitted to the GUIDANCE only (reconstructing it from its own downsample), so
it is cached once per sample (and feature step, where the task keeps time) and then reused by
every LP, encoder and feature cache on the same token grid:

    FEATURES/jbu_kernels/<task>_<method>/meta.json
    FEATURES/jbu_kernels/<task>_<method>/<split>/shard_<i>of<n>.pt

Samples, their dates (for per-step guidance) and the token grid come from the task's ps16ts64
cache; any model size's cache gives the same ones.

    python -u -m exp.upsamplers.fit_jbu --task pastis --method upma --split train --shard 0/1
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader, Subset

from exp.common.tasks import SPLITS, TASKS
from exp.upsamplers.methods import build, kernel_dir
from exp.upsamplers.task_data import UpsamplerData


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--method", required=True, choices=("upa", "upma"))
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--shard", default="0/1", help="i/n: the i-th of n contiguous parts")
    ap.add_argument("--model_size", default="tiny", help="whose ps16ts64 cache lists samples")
    ap.add_argument("--batch_size", type=int, default=64, help="samples per loader batch")
    ap.add_argument("--fit_batch", type=int, default=128, help="images fitted at once")
    ap.add_argument("--fit_steps", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()

    task = TASKS[a.task]
    up = build(a.method, task, fit_steps=a.fit_steps)
    up.fit_batch = a.fit_batch
    d = UpsamplerData(task, a.split, a.model_size, ("lr",))
    up.check(d.ds)
    grid = tuple(d[0][1]["lr"].shape[1:3])
    out = kernel_dir(task, a.method)
    meta = up.cache_meta(grid)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        have = json.loads((out / "meta.json").read_text())
        if have != meta:
            raise SystemExit(f"{out} holds kernels fitted for {have}, not {meta}")
    else:
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    i, n = map(int, a.shard.split("/"))
    f = out / a.split / f"shard_{i}of{n}.pt"
    if f.exists():
        print(f"{f} exists; nothing to do")
        return
    N = len(d)
    part = Subset(d, range(N * i // n, N * (i + 1) // n))
    dl = DataLoader(part, batch_size=a.batch_size, num_workers=a.num_workers,
                    collate_fn=d.collate)
    print(f"{up.key} | {up.time_matching} | grid {grid} | {a.split} shard {a.shard}: "
          f"{len(part)} samples -> {f}", flush=True)
    ids, raws, t0 = [], [], time.time()
    for k, batch in enumerate(dl):
        g = up.build_guidance(batch.guidance, batch.feats["lr"]).cuda()
        B = len(batch.ids)
        g = g.reshape(-1, *g.shape[-3:])                   # (B * T', Cg, H, W)
        raws.append(up.fit(g, grid).reshape(B, -1, 4, *grid).cpu())
        ids += batch.ids
        if k % 20 == 0:
            done = len(ids)
            print(f"  {done}/{len(part)} samples, {(time.time() - t0) / done * 1000:.0f} ms "
                  "per sample", flush=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    torch.save({"ids": ids, "raw": torch.cat(raws)}, tmp)
    tmp.replace(f)
    print(f"wrote {f}: {len(ids)} samples in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
