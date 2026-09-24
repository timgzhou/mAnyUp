"""Pack an MTMMBatch into what UTAE (third_party/utae) takes.

UTAE reads, per satellite, data (B, T, C, H, W) and dates (B, T) in days since a reference
day, and treats an all-zero frame as padding. Defaults are the official PASTIS loader's:
dates since 2018-09-01, channel mean/std from NORM_<sat>_patch.json averaged over the folds
of the split being loaded (so train, val and test each use their own folds' statistics).
Other datasets pass `norm` (per-modality channel mean/std) and `reference_day` (None = each
sample's first frame, for datasets whose samples span different years).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import torch

from manyup.data.datasets.pastis import DEFAULT_ROOT, FOLDS
from manyup.data.sample import MTMMBatch
from manyup.data.time import to_days

REFERENCE_DAY = to_days(dt.date(2018, 9, 1))


class UTAEInput:
    def __init__(self, sats: list[str], split: str, root: str | Path = DEFAULT_ROOT,
                 norm: dict | None = None, reference_day: int | None = REFERENCE_DAY):
        self.sats, self.reference_day = list(sats), reference_day
        self.norm = {}
        if norm is not None:
            for s in self.sats:
                mean, std = (torch.tensor(v, dtype=torch.float32) for v in norm[s])
                self.norm[s] = (mean.view(1, 1, -1, 1, 1), std.view(1, 1, -1, 1, 1))
            return
        for s in self.sats:
            vals = json.loads((Path(root) / f"NORM_{s}_patch.json").read_text())
            folds = FOLDS[split]
            # averaged in float64, then cast, exactly as the official loader does
            mean = torch.tensor([vals[f"Fold_{f}"]["mean"] for f in folds], dtype=torch.float64).mean(0).float()
            std = torch.tensor([vals[f"Fold_{f}"]["std"] for f in folds], dtype=torch.float64).mean(0).float()
            self.norm[s] = (mean.view(1, 1, -1, 1, 1), std.view(1, 1, -1, 1, 1))

    def __call__(self, batch: MTMMBatch) -> tuple[dict, dict]:
        data, dates = {}, {}
        for s in self.sats:
            mb = batch.modalities[s]
            mean, std = self.norm[s]
            x = (mb.data.float() - mean) / std
            invalid = ~mb.frame_valid
            x[invalid] = 0.0                       # UTAE's padding signal: an all-zero frame
            data[s] = x
            ref = (mb.t_start[:, :1] if self.reference_day is None else self.reference_day)
            dates[s] = torch.where(invalid, 0, mb.t_start - ref)
        return data, dates
