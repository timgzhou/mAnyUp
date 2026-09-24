"""A materialized per-sample tensor (e.g. guidance), keyed by sample id.

    <dir>/meta.json         what produced it (dataset key, spec key, ...)
    <dir>/ids.txt           sample id of every idx
    <dir>/<idx>.pt          one tensor per sample

Built once from a function of the sample id and then only read; `meta` must match on reuse,
so a cache made from different data or a different spec is never silently picked up.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class _Build(Dataset):
    """Item k = a CHUNK of consecutive ids, so neighbouring samples (e.g. the tiles of one
    native patch) are built by the same worker and can share its loaded data."""

    def __init__(self, ids, fn, chunk):
        self.ids, self.fn, self.chunk = ids, fn, chunk

    def __len__(self):
        return -(-len(self.ids) // self.chunk)

    def __getitem__(self, k):
        lo = k * self.chunk
        return [(i, self.fn(self.ids[i])) for i in range(lo, min(lo + self.chunk, len(self.ids)))]


class TensorCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.ids = (self.dir / "ids.txt").read_text().split()
        self._pos = {sid: i for i, sid in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._pos

    def get(self, sample_id: str) -> torch.Tensor:
        return torch.load(self.dir / f"{self._pos[sample_id]}.pt")

    @classmethod
    def ensure(cls, directory: str | Path, ids: Iterable[str], fn: Callable[[str], torch.Tensor],
               meta: dict, num_workers: int = 8) -> "TensorCache":
        """Open the cache at `directory`, building it from fn(id) for every id if it is not
        complete. An existing cache whose meta differs is an error, never reused."""
        d, ids = Path(directory), list(ids)
        if (d / "meta.json").exists():
            have = json.loads((d / "meta.json").read_text())
            if have != meta:
                raise ValueError(f"{d} was built from {have}, not {meta}")
            if (d / "ids.txt").exists() and all((d / f"{i}.pt").exists() for i in range(len(ids))):
                cache = cls(d)
                if cache.ids != ids:
                    raise ValueError(f"{d}: cached ids differ from the requested ones")
                return cache
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        # fork: building is CPU-only, and spawn (which some libraries switch to) would re-import
        # the caller's __main__
        loader = DataLoader(_Build(ids, fn, chunk=16), batch_size=None, num_workers=num_workers,
                            multiprocessing_context="fork" if num_workers else None)
        for chunk in tqdm(loader, desc=f"build {d.name}", leave=False):
            for i, x in chunk:
                torch.save(x.clone(), d / f"{i}.pt")
        (d / "ids.txt").write_text("".join(s + "\n" for s in ids))    # written last: completeness
        return cls(d)
