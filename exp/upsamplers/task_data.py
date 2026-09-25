"""Feature-upsampler data for a task in exp/common/tasks.py: cached OlmoEarth features
(exp.olmoearth.extract, store per_timestep) joined by sample id with the task's own samples,
which the upsamplers read their guidance from.

Roles (cache configs, all tiny/base alike):
    lr      series ps16 tile64        the input every upsampler refines
    hr      per_timestep ps4 tile4    timAnyUp's high-res, low-context side feature
    target  series ps4 tile64         what mAnyUp / timAnyUp are trained to predict
Every cache of a task holds the same samples in the same order as task.dataset (checked).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from exp.common.paths import FEATURES
from exp.common.tasks import Task
from exp.olmoearth.extract import cache_name
from manyup.data import mtmm_collate
from manyup.data.feature_cache import FeatureCache
from manyup.data.features import FeatureSeries
from manyup.data.geometry import Geometry
from manyup.data.sample import MTMMBatch

ROLES = {"lr": ("series", 16, 64), "hr": ("per_timestep", 4, 4), "target": ("series", 4, 64)}
OUT_GRID = 16          # upsamplers output the ps4 token grid of a 64 px sample


def role_cache(task: Task, model_size: str, role: str) -> str:
    mode, ps, ts = ROLES[role]
    return cache_name(task.name, model_size, mode, ps, ts, "per_timestep")


@dataclass
class Batch:
    ids: list[str]
    feats: dict[str, FeatureSeries]       # role -> features
    guidance: MTMMBatch | None
    target: torch.Tensor                  # the task's target (classes long, or standardized)

    def to(self, device) -> "Batch":
        for f in self.feats.values():
            for k in ("tokens", "frame_valid", "t_start", "t_end"):
                setattr(f, k, getattr(f, k).to(device, non_blocking=True))
        self.target = self.target.to(device, non_blocking=True)
        return self


class UpsamplerData(Dataset):
    def __init__(self, task: Task, split: str, model_size: str, roles: tuple[str, ...],
                 guidance: bool = True, root=FEATURES):
        self.caches = {r: FeatureCache(root, role_cache(task, model_size, r), split)
                       for r in roles}
        first = self.caches[roles[0]]
        for r, c in self.caches.items():
            if c.meta["store"] != "per_timestep":
                raise ValueError(f"{c.name}: upsamplers need every timestep (store per_timestep)")
            if c.ids != first.ids:
                raise ValueError(f"{c.name}/{split}: samples differ from {first.name}")
        self.ids = first.ids
        self.ds = task.dataset(split) if guidance else None
        if self.ds is not None:
            if len(self.ds) != len(self.ids) or any(
                    self.ds.sample_id(i) != s for i, s in enumerate(self.ids)):
                raise ValueError(f"{task.name}/{split}: dataset samples differ from the caches")
            self.modalities = set(task.guidance) | set(task.rgb)
        tg = torch.load(first.split_dir / "target.pt")
        self.target = tg.long() if tg.dtype == torch.int16 else tg
        self.embed_dim = first.meta["embed_dim"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        tok = {r: torch.load(c.split_dir / f"{i}.pt") for r, c in self.caches.items()}
        s = None
        if self.ds is not None:
            s = self.ds[i]
            s.modalities = {m: md for m, md in s.modalities.items() if m in self.modalities}
        return i, tok, s

    def collate(self, items) -> Batch:
        idx = torch.tensor([i for i, _, _ in items])
        ids = [self.ids[i] for i in idx.tolist()]
        feats = {}
        for r, c in self.caches.items():
            tok = torch.stack([t[r] for _, t, _ in items])
            g = Geometry.unknown(tuple(tok.shape[2:4]))
            feats[r] = FeatureSeries(ids, tok, c.t_start[idx], c.t_end[idx], None,
                                     c.frame_valid[idx], [g] * len(ids),
                                     {"cache": c.name})
        guid = mtmm_collate([s for _, _, s in items]) if self.ds is not None else None
        return Batch(ids, feats, guid, self.target[idx])


def out_geometry(batch: Batch) -> list[Geometry]:
    return [Geometry.unknown((OUT_GRID, OUT_GRID))] * len(batch.ids)
