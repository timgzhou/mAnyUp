"""Guidance images for feature upsamplers, built from MTMM samples by band NAME.

A GuidanceSpec lists, per modality, which bands form the guidance stack and in what order;
an entry "B01<-B02" means "a channel for B01, filled from B02" (imputation, explicit). Each
modality is normalized per image on its own scale (S1 dB and S2 reflectance would swamp each
other in a joint scale), then modalities are concatenated on the channel axis.

norm:
  "minmax"      per (frame, band) min-max to [0, 1] (mAnyUp, timAnyUp, AnyUp before ImageNet)
  "percentile"  per (frame, band) 2-98 percentile stretch, clipped to [0, 1] (UPA / UPMA)
time_pool:
  "mean" | "median"  one frame: the reduction over the frames that have data
  None               per timestep: (T, C, H, W), each frame normalized on its own
Pooling and normalization reproduce the legacy upsampler guidance exactly (checked in
tests/test_guidance.py against the legacy prep).

build_guidance works on one sample. build_guidance_batch works on an MTMMBatch and, for
per-timestep guidance, picks each modality's frames for given feature steps BY DATE
(MATCHINGS): modalities keep their own time axes, so S2 and S1 frames need not line up.

A feature step is only guided by frames of its OWN dates: guidance_present says, per step,
whether every guidance modality has a valid frame overlapping it. Upsamplers do not upsample
a step where it is False (a month with S1 but no S2, say): the matchings below would fill it
from a neighbouring date, i.e. guide one month with another month's image.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from manyup.data.sample import ModalityBatch, ModalityData, MTMMBatch

TIME_POOLS = ("mean", "median")

# The 13-channel S2 layout the legacy PASTIS prep stored and mAnyUp/timAnyUp were trained on
# (OlmoEarth's eval band order, with its imputations for bands PASTIS lacks).
LEGACY_S2_13 = ("B01<-B02", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                "B09<-B8A", "B10<-B11", "B11", "B12")
LEGACY_S1_2 = ("VV", "VH")
S2_RGB = ("B04", "B03", "B02")
# The guidance of each legacy guidance_mod (checkpoint args of train_manyup / train_timanyup).
LEGACY_GUIDANCE = {"s2": {"S2": LEGACY_S2_13}, "s1": {"S1A": LEGACY_S1_2},
                   "s2s1": {"S2": LEGACY_S2_13, "S1A": LEGACY_S1_2}}
NORMS = ("minmax", "percentile")
# How a feature step [t_start, t_end] picks a modality's guidance frame(s):
#   nearest       the valid frame whose midpoint is closest (ties: the earlier frame)
#   within_range  the mean of the valid frames overlapping the step; nearest if there is none
# Both fall back to OTHER dates when a step has no frame of its own; guidance_present marks
# those steps, and the upsamplers skip them.
MATCHINGS = ("nearest", "within_range")


@dataclass(frozen=True)
class GuidanceSpec:
    bands: dict[str, tuple[str, ...]]      # modality -> channel entries, in order
    time_pool: str | None                  # "mean" | "median" | None (per timestep)
    norm: str = "minmax"                   # one of NORMS

    def __post_init__(self):
        if self.time_pool is not None and self.time_pool not in TIME_POOLS:
            raise ValueError(f"time_pool {self.time_pool!r} not in {TIME_POOLS} or None")
        if self.norm not in NORMS:
            raise ValueError(f"norm {self.norm!r} not in {NORMS}")

    @property
    def channels(self) -> int:
        return sum(len(v) for v in self.bands.values())

    @property
    def key(self) -> str:
        mods = "+".join(f"{m}{len(b)}" for m, b in self.bands.items())
        return (f"{mods}-{self.time_pool or 'per_t'}"
                + ("" if self.norm == "minmax" else f"-{self.norm}"))


def spec_to_dict(spec: GuidanceSpec) -> dict:
    """JSON-able form (checkpoints, CSV provenance)."""
    return {"bands": {m: list(b) for m, b in spec.bands.items()}, "time_pool": spec.time_pool,
            "norm": spec.norm}


def spec_from_dict(d: dict) -> GuidanceSpec:
    return GuidanceSpec({m: tuple(b) for m, b in d["bands"].items()}, d["time_pool"],
                        d.get("norm", "minmax"))


def _source(entry: str) -> str:
    return entry.split("<-")[1] if "<-" in entry else entry


def minmax_per_image(x: torch.Tensor) -> torch.Tensor:
    """Per (frame, band) min-max over the spatial axes to [0, 1]; (C,H,W) or (T,C,H,W)."""
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


def percentile_per_image(x: torch.Tensor, lo: float = 0.02, hi: float = 0.98) -> torch.Tensor:
    """Per (frame, band) percentile stretch over the spatial axes, clipped to [0, 1]."""
    flat = x.float().flatten(-2).flatten(0, -2)                     # (frames * bands, H * W)
    qs = torch.tensor([lo, hi], device=x.device)
    step = max(1, (1 << 24) // flat.shape[1])                       # torch.quantile's size cap
    q = torch.cat([torch.quantile(c, qs, dim=-1) for c in flat.split(step)], dim=1)
    q = q.view(2, *x.shape[:-2], 1, 1)
    return ((x - q[0]) / (q[1] - q[0] + 1e-6)).clamp(0, 1)


def normalize(x: torch.Tensor, norm: str) -> torch.Tensor:
    return minmax_per_image(x) if norm == "minmax" else percentile_per_image(x)


def _pool(x: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    """(T, C, H, W) -> (C, H, W) over the valid frames; torch's median (lower middle)."""
    x = x[valid]
    if len(x) == 0:
        raise ValueError("no frame with data to build guidance from")
    return x.mean(0) if mode == "mean" else x.median(0).values


def select_bands(md: ModalityData | ModalityBatch, entries: tuple[str, ...]) -> torch.Tensor:
    """Raw float stack of these channel entries, in order (imputations filled):
    (T, C, H, W) for a sample, (B, T, C, H, W) for a batch."""
    idx = []
    for e in entries:
        src = _source(e)
        if src not in md.bands:
            raise ValueError(f"band {src} not in {md.bands}")
        idx.append(md.bands.index(src))
    return md.data.float()[..., idx, :, :]


def build_guidance(modalities: dict[str, ModalityData], spec: GuidanceSpec) -> torch.Tensor:
    """One sample's guidance: (C, H, W) if spec.time_pool else (T, C, H, W)."""
    parts = []
    for mod, entries in spec.bands.items():
        md = modalities[mod]
        x = select_bands(md, entries)                                 # (T, C, H, W)
        if spec.time_pool is None:
            if not md.frame_valid.all():
                raise ValueError(f"per-timestep guidance needs every {mod} frame to have data")
            parts.append(normalize(x, spec.norm))
        else:
            parts.append(normalize(_pool(x, md.frame_valid, spec.time_pool), spec.norm))
    return torch.cat(parts, dim=-3)


def _mid(t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
    return (t_start + t_end).double() / 2


def match_frames(mb: ModalityBatch, f_start: torch.Tensor, f_end: torch.Tensor,
                 how: str) -> torch.Tensor:
    """Weights (B, Tf, Tg) over a modality's frames for each feature step; rows sum to 1.

    A static modality (no time axis) gives its single frame to every step."""
    if how not in MATCHINGS:
        raise ValueError(f"matching {how!r} not in {MATCHINGS}")
    B, Tg = mb.frame_valid.shape
    Tf = f_start.shape[1]
    if mb.t_start is None:
        return torch.ones(B, Tf, 1)
    valid = mb.frame_valid[:, None, :]                                     # (B, 1, Tg)
    if not valid.any(-1).all():
        raise ValueError("a sample has no frame with data to build guidance from")
    d = (_mid(f_start, f_end)[:, :, None] - _mid(mb.t_start, mb.t_end)[:, None, :]).abs()
    nearest = d.masked_fill(~valid, float("inf")).argmin(-1)              # first on ties
    w = torch.nn.functional.one_hot(nearest, Tg).double()
    if how == "within_range":
        overlap = ((mb.t_start[:, None, :] <= f_end[:, :, None])
                   & (mb.t_end[:, None, :] >= f_start[:, :, None]) & valid).double()
        n = overlap.sum(-1, keepdim=True)
        w = torch.where(n > 0, overlap / n.clamp(min=1), w)
    return w


def guidance_present(batch: MTMMBatch, spec: GuidanceSpec, f_start: torch.Tensor,
                     f_end: torch.Tensor) -> torch.Tensor:
    """(B, Tf) bool: every guidance modality has a valid frame overlapping the feature step
    [f_start, f_end] (days). A static modality (no time axis) is always present."""
    ok = torch.ones(f_start.shape, dtype=torch.bool)
    f_start, f_end = f_start.cpu(), f_end.cpu()
    for mod in spec.bands:
        if mod not in batch.modalities:
            raise ValueError(f"guidance needs modality {mod!r}; batch has "
                             f"{sorted(batch.modalities)}")
        mb = batch.modalities[mod]
        if mb.t_start is None:
            continue
        overlap = ((mb.t_start[:, None, :] <= f_end[:, :, None])
                   & (mb.t_end[:, None, :] >= f_start[:, :, None])
                   & mb.frame_valid[:, None, :])
        ok &= overlap.any(-1)
    return ok


def build_guidance_batch(batch: MTMMBatch, spec: GuidanceSpec, *, match: str | None = None,
                         f_start: torch.Tensor | None = None,
                         f_end: torch.Tensor | None = None) -> torch.Tensor:
    """A batch's guidance: (B, C, H, W) if spec.time_pool, else (B, Tf, C, H, W) with one
    frame per feature step (f_start / f_end, (B, Tf) days), chosen by `match`. A step whose
    match is a single frame gets that frame exactly (it is indexed, not averaged)."""
    parts = []
    for mod, entries in spec.bands.items():
        if mod not in batch.modalities:
            raise ValueError(f"guidance needs modality {mod!r}; batch has "
                             f"{sorted(batch.modalities)}")
        mb = batch.modalities[mod]
        x = select_bands(mb, entries)                                     # (B, T, C, H, W)
        if spec.time_pool is not None:
            parts.append(torch.stack([normalize(_pool(x[b], mb.frame_valid[b], spec.time_pool),
                                                spec.norm) for b in range(len(x))]))
            continue
        if match is None or f_start is None or f_end is None:
            raise ValueError("per-timestep guidance needs match, f_start and f_end")
        w = match_frames(mb, f_start.cpu(), f_end.cpu(), match).to(x.device)
        one = (w > 0).sum(-1) == 1                                        # (B, Tf)
        picked = x[torch.arange(len(x))[:, None], w.argmax(-1)]            # (B, Tf, C, H, W)
        if not one.all():
            mixed = torch.einsum("bft,btchw->bfchw", w.to(x.dtype),
                                 x.masked_fill(~mb.frame_valid[:, :, None, None, None], 0))
            picked = torch.where(one[:, :, None, None, None], picked, mixed)
        parts.append(normalize(picked, spec.norm))
    return torch.cat(parts, dim=-3)
