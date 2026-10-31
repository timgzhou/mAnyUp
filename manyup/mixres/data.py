"""Batches for manyup.mixres: each sample as monthly composites (what crops are cut from) and as
season composites (the uniform tokens), packed for OlmoEarth, plus each month's season slot,
which months count, the label, and optionally every raw acquisition (dates=True, for date crops).

The legacy exp.upsamplers.finetune_inject loader with its legacy-only options removed (year /
raw low-res steps, the self-teacher's subsets, change-detection event steps, series pairs);
exp.mixres.check_data checks the two give identical batches.
"""
from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data._utils.pin_memory import pin_memory as _pin

from exp.common.tasks import Task
from manyup.data import All, GroupSteps, mtmm_collate
from manyup.encoders.olmoearth import OlmoEarthInput
from manyup.seasonal import season_index


@dataclass
class Batch:
    """B samples, T month slots, NS season slots.
    ms, ss     monthly / season MaskedOlmoEarthSample
    season     (B, T) long   each month's season slot (-1: none)
    present    (B, T) bool   the month is real (every modality has a frame, its season has data)
    target     the task label
    frac       (B, T) float  each month's position inside its season, in [-0.5, 0.5)
    dates      {modality: (sample (T_m steps), valid (B, T_m), month (B, T_m; -1 outside the
               month grid), position in its season (B, T_m))} or None"""
    ms: object
    ss: object
    season: torch.Tensor
    present: torch.Tensor
    target: torch.Tensor
    frac: torch.Tensor
    dates: dict | None = None

    def pin_memory(self):
        return Batch(*(_pin(getattr(self, f)) for f in
                       ("ms", "ss", "season", "present", "target", "frac", "dates")))


def within(t_start, t_end, s_start, s_end, idx):
    """Fractional position of each step's midpoint inside the step it belongs to (idx, -1 =
    none), in [-0.5, 0.5): e.g. a month inside its season, a date inside its month.
    t_* (B, T), s_* (B, S), idx (B, T) -> (B, T) float (0 where idx < 0)."""
    i = idx.clamp(min=0)
    lo, hi = s_start.gather(1, i).double(), s_end.gather(1, i).double()
    mid = (t_start + t_end).double() / 2
    f = ((mid - lo) / (hi - lo + 1).clamp(min=1) - 0.5).float()
    return torch.where(idx >= 0, f, torch.zeros_like(f))


class Paired(Dataset):
    """(months, seasons[, raw]): the task's sample, the same with its months grouped into
    seasons, and (dates) every raw frame of each modality on its own dates."""

    def __init__(self, task: Task, split: str, dates: bool = False):
        lr = task.lr_policy()
        self.lr = GroupSteps(lr) if isinstance(lr, int) else lr
        self.dates = dates
        if dates:
            self.ds = task.open(split, {m: All() for m in task.olmoearth.modalities})
            self.policies = task.policies()
        else:
            self.ds = task.dataset(split)
        self.spec = self.ds.SPEC.modalities

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        raw = s
        if self.dates:
            s = replace(raw, modalities={m: self.policies[m](md, self.spec[m])
                                         for m, md in raw.modalities.items()})
        seasons = replace(s, modalities={m: self.lr(md, self.spec[m])
                                         for m, md in s.modalities.items()})
        return (s, seasons, raw) if self.dates else (s, seasons)


class Collate:
    def __init__(self, task: Task, dates: bool = False):
        m, imp = task.olmoearth.modalities, task.olmoearth.impute
        self.task, self.mods, self.ref = task, list(m), next(iter(m))
        self.month_in = OlmoEarthInput(m, imp, timestamp="start")
        self.season_in = OlmoEarthInput(m, imp, timestamp="mid")
        self.date_in = ({k: OlmoEarthInput({k: v}, {k: imp[k]} if k in imp else {},
                                           timestamp="mid") for k, v in m.items()}
                        if dates else None)

    def __call__(self, items) -> Batch:
        months = mtmm_collate([it[0] for it in items])
        seasons = mtmm_collate([it[1] for it in items])
        ms, _ = self.month_in.to_olmoearth(months)
        ss, s_valid = self.season_in.to_olmoearth(seasons)
        ref, sref = months.modalities[self.ref], seasons.modalities[self.ref]
        season = season_index(sref.t_start, sref.t_end, ref.t_start, ref.t_end)     # (B, T)
        every = torch.stack([months.modalities[m].frame_valid for m in self.mods]).all(0)
        present = every & (season >= 0) & s_valid.gather(1, season.clamp(min=0))
        frac = within(ref.t_start, ref.t_end, sref.t_start, sref.t_end, season)
        dates = None
        if self.date_in is not None:
            raw = mtmm_collate([it[2] for it in items])
            dates = {}
            for m, inp in self.date_in.items():
                dm = raw.modalities[m]
                d_sample, d_valid = inp.to_olmoearth(raw)
                month = season_index(ref.t_start, ref.t_end, dm.t_start, dm.t_end)
                d_season = torch.where(month >= 0, season.gather(1, month.clamp(min=0)),
                                       torch.full_like(month, -1))
                d_frac = within(dm.t_start, dm.t_end, sref.t_start, sref.t_end, d_season)
                dates[m] = (d_sample, d_valid & (month >= 0), month, d_frac)
        return Batch(ms, ss, season, present, self.task.target(months), frac, dates)


def loader(task: Task, split: str, batch_size: int, num_workers: int, shuffle: bool,
           dates: bool = False, epoch_samples: int = 0) -> DataLoader:
    """epoch_samples > 0 (shuffled splits): each pass draws that many samples without
    replacement (a fresh random subset per epoch)."""
    ds = Paired(task, split, dates)
    sampler = (RandomSampler(ds, num_samples=epoch_samples)
               if shuffle and 0 < epoch_samples < len(ds) else None)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler,
                      num_workers=num_workers, collate_fn=Collate(task, dates),
                      # dates: a batch holds ~1.4 GB of raw frames of changing length; pinned,
                      # the CUDA host allocator kept ~29 GB of blocks (legacy, 2026-10-04)
                      persistent_workers=num_workers > 0 and (shuffle or not dates),
                      prefetch_factor=(1 if dates else 2) if num_workers else None,
                      drop_last=shuffle, pin_memory=not dates)
