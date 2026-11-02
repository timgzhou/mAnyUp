"""The loader's temporal compositing as one batched GPU step, for exp.mixres.bench_speed's
throughput: every arm starts from the raw acquisitions, as a deployed model would (composites are
not stored anywhere), and pays for what it uses

    raw frames -> monthly means -> season means -> OlmoEarth bands + normalization

The same arithmetic as manyup.data.temporal (Pool, PoolYears, GroupSteps: means over the valid
pixels; a season is the mean of its monthly means) and OlmoEarthInput.to_olmoearth, checked
against the loader's batches by check(). Dates are not pixels: which month each frame falls in is
worked out on the host beforehand, and the timestamps and frame masks are kept from the loader's
batch.
"""
from dataclasses import dataclass

import numpy as np
import torch

from manyup.data import All, GroupSteps
from manyup.data.sample import mtmm_collate
from manyup.data.temporal import Pool
from manyup.encoders.olmoearth import OlmoEarthInput


@dataclass
class RawModality:
    """data (B, F, C, H, W) as stored, valid (B, F, H, W) bool, month (B, F) slot (-1: none)."""
    data: torch.Tensor
    valid: torch.Tensor
    month: torch.Tensor


@dataclass
class Raw:
    modalities: dict


def _mean(total, count):
    return torch.where(count > 0, total / count.clamp(min=1), torch.zeros_like(total))


class Composer:
    def __init__(self, task):
        self.task, self.policies = task, task.policies()
        self.spec = task.olmoearth
        self.size = task.lr_policy().size if isinstance(task.lr_policy(), GroupSteps) else 1
        if not isinstance(task.lr_policy(), GroupSteps):
            raise ValueError(f"{task.name}: seasons are not GroupSteps")
        self.inp = OlmoEarthInput(self.spec.modalities, self.spec.impute)
        self.norm = {}

    def _months_of(self, md, pol) -> tuple[torch.Tensor, int]:
        """Each frame's month slot, and the slot count."""
        if isinstance(pol, All):
            return torch.arange(md.num_frames), md.num_frames
        if not isinstance(pol, Pool):
            raise ValueError(f"unsupported policy {pol}")
        slot, months = torch.full((md.num_frames,), -1), pol._months(md)
        for j, (lo, hi) in enumerate(months):
            slot[(md.time.start >= lo) & (md.time.start <= hi)] = j
        return slot, len(months)

    def raw_batches(self, split: str, batch_size: int, n: int) -> list[Raw]:
        """The first n batches of the split (unshuffled, so the loader's batches: the last can
        be short, or missing), every frame."""
        ds = self.task.open(split, {m: All() for m in self.spec.modalities})
        out = []
        for k in range(n):
            items = [ds[i] for i in range(k * batch_size, min((k + 1) * batch_size, len(ds)))]
            if not items:
                break
            mods = mtmm_collate(items).modalities
            raw = {}
            for m, mb in mods.items():
                if m not in self.spec.modalities:
                    continue
                pol = self.policies[m]
                slots = [self._months_of(it.modalities[m], pol) for it in items]
                month = torch.full(mb.frame_valid.shape, -1)
                for i, (s, _) in enumerate(slots):
                    month[i, :len(s)] = s
                B, F, _, H, W = mb.data.shape
                v = mb.frame_valid.view(B, F, 1, 1).expand(B, F, H, W)
                if mb.pixel_valid is not None:
                    v = v & mb.pixel_valid
                raw[m] = RawModality(mb.data, v, month)
                self.norm.setdefault(m, self._affine(m, mb.bands))
                self.T = slots[0][1]
            out.append(Raw(raw))
        return out

    def _affine(self, m: str, bands) -> tuple:
        """The band selection and the Normalizer as shift + scale per band (read off by
        evaluating it at 0 and 1; check() confirms it is affine on the real data)."""
        om = self.inp.olmo[m]
        n = len(om.band_order)
        at = lambda v: self.inp.normalizer.normalize(om, np.full((1, 1, 1, 1, n), v, "f4"))  # noqa: E731
        lo, hi = at(0.0), at(1.0)
        return (self.inp._band_index(m, bands), torch.from_numpy(lo.reshape(n).astype("f4")),
                torch.from_numpy((hi - lo).reshape(n).astype("f4")))

    def months(self, raw: Raw) -> dict:
        """{modality: (x (B, T, C, H, W) float, valid (B, T, H, W))}."""
        out = {}
        for m, r in raw.modalities.items():
            x = r.data.float()
            if isinstance(self.policies[m], All):
                out[m] = (x, r.valid)
                continue
            B, F, C, H, W = x.shape
            hit = torch.zeros(B, self.T, F, device=x.device)        # (B, T, F): frame in month
            f = (r.month >= 0).nonzero(as_tuple=True)
            hit[f[0], r.month[f], f[1]] = 1
            mask = r.valid.float()
            total = hit @ (x * mask[:, :, None]).flatten(2)         # (B, T, C H W)
            count = hit @ mask.flatten(2)                           # (B, T, H W)
            xm = _mean(total.view(B, self.T, C, H, W), count.view(B, self.T, 1, H, W))
            out[m] = (xm, count.view(B, self.T, H, W) > 0)
        return out

    def seasons(self, months: dict) -> dict:
        out = {}
        for m, (x, v) in months.items():
            B, T, C, H, W = x.shape
            g = self.size
            mask = v.float()[:, :, None]
            total = (x * mask).view(B, T // g, g, C, H, W).sum(2)
            count = mask.view(B, T // g, g, 1, H, W).sum(2)
            out[m] = (_mean(total, count), count[:, :, 0] > 0)
        return out

    def pack(self, sample, comp: dict):
        """sample (the loader's) with its bands replaced by the composites."""
        new = {}
        for m, (x, _) in comp.items():
            idx, shift, scale = self.norm[m]
            y = x[:, :, idx].permute(0, 3, 4, 1, 2)                 # (B, H, W, T, C)
            new[self.inp.olmo[m].name] = y * scale.to(y.device) + shift.to(y.device)
        return sample._replace(**new)


@torch.no_grad()
def check(composer: Composer, batch, raw: Raw, tol: float = 2e-3):
    """Raises unless the composites equal the loader's batch -> the largest difference
    (normalized units)."""
    months = composer.months(raw)
    worst = 0.0
    for got, want in ((composer.pack(batch.ms, months), batch.ms),
                       (composer.pack(batch.ss, composer.seasons(months)), batch.ss)):
        for m in raw.modalities:
            k = composer.inp.olmo[m].name
            a, b = getattr(got, k).float(), getattr(want, k).float().to(getattr(got, k).device)
            if a.shape != b.shape:
                raise ValueError(f"{m}: shape {tuple(a.shape)} vs loader {tuple(b.shape)}")
            d = (a - b).abs()
            worst = max(worst, d.max().item())
    if worst > tol:
        raise ValueError(f"composites differ from the loader by up to {worst}")
    return worst
