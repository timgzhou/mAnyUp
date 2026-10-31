"""Measured inference memory and speed of a forward pass over real batches, the same protocol for
every arm (exp.mixres.bench_speed: the mixed-resolution runs and the fixed-resolution models).
The batches are read from the loader first (no data loading in either measurement); WARMUP
batches run untimed first (cudnn / allocator / autocast caches).

    peak_memory      the batches stay in host memory and each forward copies its own to the GPU,
                     as at deployment: one batch's inputs + the weights + the activations
        peak_mem_gb      torch.cuda.max_memory_allocated (weights included)
        act_mem_gb       the same minus what was allocated before (the weights)
    throughput       a few batches moved to the GPU beforehand and cycled through, so the host ->
                     GPU copy (unpinned, it dominated the legacy fixed arms' time) is not timed;
                     TIMED batches between two cuda synchronizations. Use a large batch: the
                     tiny model's small batches are bound by kernel launches and Python, not by
                     the GPU (batch 64: ~20% GPU busy for a ps16 model)
        samples_per_s    samples over the timed batches / their wall-clock time
        ms_per_sample    1000 / samples_per_s
"""
import csv
import dataclasses
import itertools
import time
from pathlib import Path

import torch

WARMUP = 3
TIMED = 8


def take(dl, n: int) -> list:
    """The loader's first n batches, in host memory."""
    out = []
    for b in dl:
        if len(out) >= n:
            break
        out.append(b)
    return out


def to_gpu(x):
    """Every tensor inside x (tensors, tuples / named tuples, lists, dicts, dataclasses) on the
    GPU, the structure kept."""
    if torch.is_tensor(x):
        return x.cuda(non_blocking=True)
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(to_gpu(v) for v in x))
    if isinstance(x, (tuple, list)):
        return type(x)(to_gpu(v) for v in x)
    if isinstance(x, dict):
        return {k: to_gpu(v) for k, v in x.items()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: to_gpu(getattr(x, f.name))
                                         for f in dataclasses.fields(x) if f.init})
    return x


@torch.no_grad()
def peak_memory(forward, batches: list) -> dict:
    """forward(batch) runs one host-memory batch -> peak_mem_gb, act_mem_gb over the batches after
    the warm-up ones."""
    if len(batches) <= WARMUP:
        raise ValueError(f"peak_memory: {len(batches)} batches, need more than {WARMUP}")
    for b in batches[:WARMUP]:
        forward(b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for b in batches[WARMUP:]:
        forward(b)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {"peak_mem_gb": round(peak / 2**30, 4), "act_mem_gb": round((peak - base) / 2**30, 4)}


@torch.no_grad()
def throughput(forward, batches: list, n_of) -> dict:
    """forward(batch) runs one batch, n_of(batch) its sample count: every batch once untimed, then
    TIMED batches cycling through them, all on the GPU beforehand -> samples_per_s, ..."""
    gpu = [to_gpu(b) for b in batches]
    try:
        for b in gpu:
            forward(b)
        torch.cuda.synchronize()
        n, t0 = 0, time.perf_counter()
        for b in itertools.islice(itertools.cycle(gpu), TIMED):
            forward(b)
            n += n_of(b)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finally:
        del gpu
        torch.cuda.empty_cache()
    return {"samples_per_s": round(n / dt, 2), "ms_per_sample": round(1000 * dt / n, 4),
            "bench_samples": n, "gpu": torch.cuda.get_device_name()}


def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
