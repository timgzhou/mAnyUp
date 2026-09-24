"""Datasets for training / checking the feature upsamplers, built on the data layer.

Features come from feature caches by sample id (exp.pastis.data.open_cache), guidance from a
materialized guidance cache that follows the LR cache's own time setup and sample size
(exp.pastis.data.PastisGuidance), so features and guidance always describe the same data.

PairedFeatureDataset (mAnyUp): (LR, HR) feature maps, both time-averaged over the steps that
have data, + pooled guidance.

TriFeatureDataset (timAnyUp) learns to turn two CHEAP feature maps into one EXPENSIVE one:
  F_lrhc  low-res, high-context   oe_base_s2_ps16_tile64        (T, 4, 4,C)  few tokens (ps16)
  F_hrlc  high-res, low-context   oe_base_s2_ps4_tile4_single   (T,16,16,C)  small tile, no
                                                                             cross-timestep attn
  F_hrhc  high-res, high-context  oe_base_s2_ps4_tile64         (T,16,16,C)  <- the TARGET
The time axis is KEPT (the upsampler runs per timestep, guidance is per timestep), so caches
with missing steps are refused.

DANGER: F_hrlc and F_hrhc have IDENTICAL shapes -- they differ only in the context they were
extracted with (`_single` = one encoder call per timestep). No shape assert can catch swapping
them, so the cfg NAMES are the only guard; check_arms() makes the expected relationship
explicit and is called by the trainers at startup.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from exp.common.paths import FEATURES
from exp.pastis.data import PastisGuidance, guidance_spec, open_cache
from exp.upsamplers.common import cfg_bits


def _setup(meta: dict) -> tuple[str, int]:
    """(time setup, sample size) of a cache; caches without them are legacy 64 px caches."""
    return meta.get("temporal", "legacy"), meta.get("image_size", 64)


def _common_ids(split: str, root: Path, cfgs: list[str], half: str):
    """Open the caches, check they describe the same data, and return (caches, ids): the ids in
    all of them, in the first cache's order, optionally one disjoint half (leak guard: the
    upsampler and the probe that reads it can be fitted on different samples)."""
    caches = [open_cache(root, c, split) for c in cfgs]
    for c in caches[1:]:
        if _setup(c.meta) != _setup(caches[0].meta):
            raise ValueError(f"{c.name} {_setup(c.meta)} and {caches[0].name} "
                             f"{_setup(caches[0].meta)} differ in (temporal, image_size); "
                             "features must come from the same data")
    ids = [i for i in caches[0].ids if all(i in c for c in caches[1:])]
    if half in ("first", "second"):
        mid = len(ids) // 2
        print(f"[{split}] id_half={half}: {mid if half == 'first' else len(ids) - mid} of {len(ids)} samples")
        ids = ids[:mid] if half == "first" else ids[mid:]
    if not ids:
        raise RuntimeError(f"no common {split} samples across {cfgs}")
    print(f"[{split}] " + " ".join(f"{c.name}={len(c)}" for c in caches) + f" -> {len(ids)} common")
    return caches, ids


def _time_mean(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(T, h, w, D) fp16 -> (D, h, w) fp32, mean over the steps with data."""
    f = tokens.float()
    f = f.mean(0) if valid.all() else f[valid].mean(0)
    return f.permute(2, 0, 1)


class PairedFeatureDataset(Dataset):
    """(lr (D,gh,gw), hr (D,GH,GW), guidance (C,H,W)) for mAnyUp training.

    Guidance pooling is baked into the learned weights (the guidance encoder adapts to the
    composite it sees), so a checkpoint must be evaluated with the same time_pool; the
    checkpoint name records it."""

    def __init__(self, features_root: Path, lr_cfg: str, hr_cfg: str, split: str,
                 time_pool: str = "mean", guidance_mod: str = "s2", half: str = "all",
                 guidance_root: Path = FEATURES):
        (self.lr, self.hr), self.ids = _common_ids(split, Path(features_root), [lr_cfg, hr_cfg], half)
        self.time_pool, self.guidance_mod = time_pool, guidance_mod
        self.guide = PastisGuidance(split, self.lr.meta, guidance_spec(
            "full", guidance_mod, time_pool)).materialize(self.ids, root=Path(guidance_root))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        a, b = self.lr.get(sid), self.hr.get(sid)
        return (_time_mean(a["tokens"], a["frame_valid"]), _time_mean(b["tokens"], b["frame_valid"]),
                self.guide.get(sid))


def check_arms(lrhc_cfg: str, hrlc_cfg: str, hrhc_cfg: str) -> None:
    """Fail loudly on an arm mix-up. Shapes cannot catch this (hrlc and hrhc match exactly), so
    validate what the NAMES claim: the low-res arm must have a bigger patch size than the two
    high-res arms, the high-res arms must agree on patch size, and the low-CONTEXT arm must be
    the `_single` one while the target must NOT be."""
    (lr_mods, lr_ps), (hl_mods, hl_ps), (hh_mods, hh_ps) = (
        cfg_bits(lrhc_cfg), cfg_bits(hrlc_cfg), cfg_bits(hrhc_cfg))
    if not (int(lr_ps) > int(hl_ps)):
        raise ValueError(f"F_lrhc must be COARSER than F_hrlc, got ps{lr_ps} vs ps{hl_ps} "
                         f"({lrhc_cfg} vs {hrlc_cfg})")
    if hl_ps != hh_ps:
        raise ValueError(f"F_hrlc and F_hrhc must share a patch size (the target grid), got "
                         f"ps{hl_ps} vs ps{hh_ps} ({hrlc_cfg} vs {hrhc_cfg})")
    # "_single" is a MIDDLE segment once a cache carries an _ft<tag> or _img<N> suffix
    # (oe_base_s2_ps4_tile4_single_ftp16ep64), so match the segment, not the end of the name.
    def _is_single(cfg: str) -> bool:
        return "_single" in cfg

    if not _is_single(hrlc_cfg):
        raise ValueError(f"F_hrlc must be a LOW-CONTEXT (_single) cache -- it is the cheap arm. "
                         f"Got {hrlc_cfg}, which was extracted as a time series.")
    if _is_single(hrhc_cfg):
        raise ValueError(f"F_hrhc is the high-context TARGET and must not be a _single cache. "
                         f"Got {hrhc_cfg}.")
    if not (lr_mods == hl_mods == hh_mods):
        raise ValueError(f"all three arms should share a modality, got {lr_mods}/{hl_mods}/{hh_mods}")


class TriFeatureDataset(Dataset):
    """(F_lrhc, F_hrlc, F_hrhc, guidance) for one sample: features (T,C,h,w) fp32 --
    channels-first per frame, so the trainer can fold T into the batch -- and per-timestep
    guidance (T,C,H,W), min-max normalized per (timestep, band)."""

    def __init__(self, features_root: Path, lrhc_cfg: str, hrlc_cfg: str, hrhc_cfg: str,
                 split: str, guidance_mod: str = "s2", half: str = "all",
                 guidance_root: Path = FEATURES):
        caches, self.ids = _common_ids(split, Path(features_root), [lrhc_cfg, hrlc_cfg, hrhc_cfg], half)
        self.lrhc, self.hrlc, self.hrhc = caches
        for c in caches:
            if not c.frame_valid.all():
                raise ValueError(f"{c.name}: timAnyUp keeps the time axis, and this cache has "
                                 "timesteps without data")
        self.guidance_mod = guidance_mod
        self.guide = PastisGuidance(split, self.lrhc.meta, guidance_spec(
            "full", guidance_mod, None)).materialize(self.ids, root=Path(guidance_root))

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _feat(cache, sid) -> torch.Tensor:
        # (T,h,w,C) fp16 -> (T,C,h,w) fp32
        return cache.get(sid)["tokens"].float().permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, i):
        sid = self.ids[i]
        return (self._feat(self.lrhc, sid), self._feat(self.hrlc, sid), self._feat(self.hrhc, sid),
                self.guide.get(sid))
