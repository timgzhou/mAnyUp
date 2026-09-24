"""Guidance images for feature upsamplers, built from MTMM samples by band NAME.

A GuidanceSpec lists, per modality, which bands form the guidance stack and in what order;
an entry "B01<-B02" means "a channel for B01, filled from B02" (imputation, explicit). Each
modality is min-max normalized per image on its own scale (S1 dB and S2 reflectance would
swamp each other in a joint scale), then modalities are concatenated on the channel axis.

time_pool:
  "mean" | "median"  one frame: the reduction over the frames that have data
  None               per timestep: (T, C, H, W), each frame normalized on its own
Pooling and normalization reproduce the legacy upsampler guidance exactly (checked in
tests/test_guidance.py against the legacy prep).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from manyup.data.sample import ModalityData

TIME_POOLS = ("mean", "median")

# The 13-channel S2 layout the legacy PASTIS prep stored and mAnyUp/timAnyUp were trained on
# (OlmoEarth's eval band order, with its imputations for bands PASTIS lacks).
LEGACY_S2_13 = ("B01<-B02", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                "B09<-B8A", "B10<-B11", "B11", "B12")
LEGACY_S1_2 = ("VV", "VH")


@dataclass(frozen=True)
class GuidanceSpec:
    bands: dict[str, tuple[str, ...]]      # modality -> channel entries, in order
    time_pool: str | None                  # "mean" | "median" | None (per timestep)

    def __post_init__(self):
        if self.time_pool is not None and self.time_pool not in TIME_POOLS:
            raise ValueError(f"time_pool {self.time_pool!r} not in {TIME_POOLS} or None")

    @property
    def channels(self) -> int:
        return sum(len(v) for v in self.bands.values())

    @property
    def key(self) -> str:
        mods = "+".join(f"{m}{len(b)}" for m, b in self.bands.items())
        return f"{mods}-{self.time_pool or 'per_t'}"


def _source(entry: str) -> str:
    return entry.split("<-")[1] if "<-" in entry else entry


def minmax_per_image(x: torch.Tensor) -> torch.Tensor:
    """Per (frame, band) min-max over the spatial axes to [0, 1]; (C,H,W) or (T,C,H,W)."""
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


def _pool(x: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    """(T, C, H, W) -> (C, H, W) over the valid frames; torch's median (lower middle)."""
    x = x[valid]
    if len(x) == 0:
        raise ValueError("no frame with data to build guidance from")
    return x.mean(0) if mode == "mean" else x.median(0).values


def select_bands(md: ModalityData, entries: tuple[str, ...]) -> torch.Tensor:
    """Raw (T, C, H, W) float stack of these channel entries, in order (imputations filled)."""
    idx = []
    for e in entries:
        src = _source(e)
        if src not in md.bands:
            raise ValueError(f"band {src} not in {md.bands}")
        idx.append(md.bands.index(src))
    return md.data.float()[:, idx]


def build_guidance(modalities: dict[str, ModalityData], spec: GuidanceSpec) -> torch.Tensor:
    """One sample's guidance: (C, H, W) if spec.time_pool else (T, C, H, W)."""
    parts = []
    for mod, entries in spec.bands.items():
        md = modalities[mod]
        x = select_bands(md, entries)                                 # (T, C, H, W)
        if spec.time_pool is None:
            if not md.frame_valid.all():
                raise ValueError(f"per-timestep guidance needs every {mod} frame to have data")
            parts.append(minmax_per_image(x))
        else:
            parts.append(minmax_per_image(_pool(x, md.frame_valid, spec.time_pool)))
    return torch.cat(parts, dim=-3)
