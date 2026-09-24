"""Pack an MTMMBatch from PastisDataset into what UTAE (third_party/utae) takes.

UTAE reads, per satellite, data (B, T, C, H, W) and dates (B, T) in days since 2018-09-01,
and treats an all-zero frame as padding. Normalization follows the official PASTIS loader:
channel mean/std from NORM_<sat>_patch.json, averaged over the folds of the split being
loaded (so train, val and test each use their own folds' statistics).
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
    def __init__(self, sats: list[str], split: str, root: str | Path = DEFAULT_ROOT):
        self.sats = list(sats)
        self.norm = {}
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
            dates[s] = torch.where(invalid, 0, mb.t_start - REFERENCE_DAY)
        return data, dates
