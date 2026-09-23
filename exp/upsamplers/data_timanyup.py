"""Dataset for timAnyUp: three co-located feature maps + per-timestep guidance imagery.

timAnyUp learns to turn two CHEAP feature maps into one EXPENSIVE one:

  F_lrhc  low-res, high-context   oe_base_s2_ps16_tile64        (T, 4, 4,C)  few tokens (ps16)
  F_hrlc  high-res, low-context   oe_base_s2_ps4_tile4_single   (T,16,16,C)  small tile, no
                                                                             cross-timestep attn
  F_hrhc  high-res, high-context  oe_base_s2_ps4_tile64         (T,16,16,C)  <- the TARGET

Unlike train_manyup, the time axis is KEPT (no mean-pool): the upsampler runs per timestep and
guidance is the per-timestep image stack, not a pooled composite.

DANGER: F_hrlc and F_hrhc have IDENTICAL shapes -- they differ only in the context they were
extracted with (`_single` = one encoder call per timestep, see exp/pastis/extract_features.py).
No shape assert can catch swapping them, so the cfg NAMES are the only guard; check_arms()
makes the expected relationship explicit and is called by the trainer at startup.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from exp.upsamplers.common import GUIDANCE_DIRS, cfg_bits, norm_guidance


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
    """Yields (F_lrhc, F_hrlc, F_hrhc, guidance) for one PASTIS sample.

    Features are stored (T,h,w,C) fp16 and returned (T,C,h,w) fp32 -- channels-first per frame,
    so the trainer can fold T into the batch dim and hand (B*T,C,h,w) straight to mAnyUp.
    Guidance is (T,C,64,64), min-max normalized per (timestep, band).
    """

    def __init__(self, lrhc_dir: Path, hrlc_dir: Path, hrhc_dir: Path, data_root: Path,
                 split: str, guidance_mod: str = "s2", half: str = "all"):
        self.lrhc_dir = lrhc_dir / f"pastis_r_{split}"
        self.hrlc_dir = hrlc_dir / f"pastis_r_{split}"
        self.hrhc_dir = hrhc_dir / f"pastis_r_{split}"
        self.guidance_mod = guidance_mod
        self.guide_dirs = [data_root / f"pastis_r_{split}" / f"{m}_images"
                           for m in GUIDANCE_DIRS[guidance_mod]]

        def indices(d: Path):
            return {int(p.stem) for p in d.glob("*.pt")}

        srcs = [self.lrhc_dir, self.hrlc_dir, self.hrhc_dir, self.guide_dirs[0]]
        counts = [len(indices(d)) for d in srcs]
        common = set.intersection(*(indices(d) for d in srcs))
        self.ids = sorted(common)

        # Optional disjoint half, same leak guard as train_manyup --id_half: the upsampler and
        # the LP probe that reads its output must not be fitted on the same samples.
        if half in ("first", "second"):
            mid = len(self.ids) // 2
            self.ids = self.ids[:mid] if half == "first" else self.ids[mid:]
            print(f"[{split}] id_half={half}: {len(self.ids)} of {len(common)} samples")
        if not self.ids:
            raise RuntimeError("no common samples across:\n  " + "\n  ".join(str(d) for d in srcs))
        print(f"[{split}] lrhc={counts[0]} hrlc={counts[1]} hrhc={counts[2]} "
              f"guide={counts[3]} -> {len(self.ids)} common samples")

    def __len__(self):
        return len(self.ids)

    def _feat(self, d: Path, idx: int) -> torch.Tensor:
        # (T,h,w,C) fp16 -> (T,C,h,w) fp32
        return torch.load(d / f"{idx}.pt").float().permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, i):
        idx = self.ids[i]
        lrhc = self._feat(self.lrhc_dir, idx)
        hrlc = self._feat(self.hrlc_dir, idx)
        hrhc = self._feat(self.hrhc_dir, idx)
        # Guidance kept PER TIMESTEP (T,C,64,64). Each modality is normalized on its own scale
        # before the concat -- S1 dB and S2 reflectance have very different dynamic ranges and a
        # joint min-max would swamp one of them (same rule as train_manyup).
        guide = torch.cat([norm_guidance(torch.load(d / f"{idx}.pt").float())
                           for d in self.guide_dirs], dim=1)
        return lrhc, hrlc, hrhc, guide
