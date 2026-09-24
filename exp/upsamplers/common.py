"""Shared plumbing for the feature-upsampler trainers (train_manyup, train_timanyup).

These started life inside train_manyup.py. They are here so the timAnyUp trainer can REUSE
them rather than fork them -- guidance normalization and the LR schedule in particular must
stay bit-identical across trainers, because checkpoints from either are evaluated by the same
lp_cached_features.py path and a silent drift in normalization would show up as an unexplained
metric gap rather than as an error.
"""
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from exp.common.paths import DATA, FEATURES

DATA_ROOT = DATA / "pastis_olmoearth"
FEATURES_ROOT = FEATURES

# Guidance band counts per modality (PASTIS prep: s2 = 13-band L2A, s1 = 2-band VV/VH;
# s2s1 stacks both -> 15 bands).
GUIDANCE_BANDS = {"s2": 13, "s1": 2, "s2s1": 15}
# Which per-sample image dirs a guidance modality reads, in channel order.
GUIDANCE_DIRS = {"s2": ["s2"], "s1": ["s1"], "s2s1": ["s2", "s1"]}


def guidance_mod_for(cfg: str) -> str:
    """Default guidance modality for a feature cfg name: MATCH the arm the features come
    from. Guiding S1 features with S2 imagery (or s2s1 features with S2 alone) is a
    cross-modal mismatch -- the guidance should carry the same modalities as the features."""
    m = re.match(r"oe_[a-z]+_([a-z0-9]+)_ps", cfg)
    mods = m.group(1) if m else "s2"
    return mods if mods in GUIDANCE_BANDS else "s2"


def cfg_bits(cfg: str):
    """(modalities, patch_size) parsed out of an oe_<size>_<mods>_ps<N>_tile<M>[_...] cfg name."""
    m = re.match(r"oe_[a-z]+_([a-z0-9]+)_ps(\d+)_", cfg)
    return (m.group(1), m.group(2)) if m else ("mods", "?")


def norm_guidance(s2: torch.Tensor) -> torch.Tensor:
    """Per-image, per-band min-max normalize the guidance to [0,1]. No ImageNet stats -- with
    13 bands and a from-scratch guidance encoder, a simple [0,1] scaling is the natural choice
    (the encoder learns its own band statistics).

    Accepts (C,H,W) or (T,C,H,W); normalizes each frame's bands independently, so a
    per-timestep guidance stack is scaled exactly as the pooled composite would be."""
    # min/max over the spatial axes only, so each (t, band) is scaled on its own -- identical
    # to the original per-band (C,H,W) formula, and framewise for a (T,C,H,W) stack.
    lo = s2.amin(dim=(-2, -1), keepdim=True)
    hi = s2.amax(dim=(-2, -1), keepdim=True)
    return (s2 - lo) / (hi - lo + 1e-6)


def stage_to_tmpdir(dirs: list[Path]) -> list[Path]:
    """Copy feature/data dirs into $SLURM_TMPDIR (node-local SSD) if set and they fit, returning
    the new paths. Cached features + images are large and moved to GPU every step, so reading
    them from fast local disk instead of Lustre is a big speedup. No-op (returns originals) if
    SLURM_TMPDIR is unset or the data doesn't fit in the available space."""
    tmp = os.environ.get("SLURM_TMPDIR")
    if not tmp:
        print("SLURM_TMPDIR unset -> reading features in place (no staging).")
        return dirs
    tmp = Path(tmp)

    def dir_bytes(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    total = sum(dir_bytes(d) for d in dirs)
    free = shutil.disk_usage(tmp).free
    if total > free * 0.95:
        print(f"staging: need {total/1e9:.1f} GB but only {free/1e9:.1f} GB free in {tmp} "
              f"-> reading in place.")
        return dirs

    staged = []
    for d in dirs:
        dst = tmp / d.name
        if dst.exists():
            print(f"staging: {dst} already present, reusing.")
        else:
            t0 = time.time()
            shutil.copytree(d, dst)
            print(f"staged {d} -> {dst} ({dir_bytes(d)/1e9:.1f} GB, {time.time()-t0:.0f}s)")
        staged.append(dst)
    return staged


def warmup_cosine(optimizer, total_steps, warmup_frac, lr, lr_min):
    """LambdaLR: linear warmup 0->1 over the first warmup_frac of steps, then cosine decay to
    lr_min/lr. Stepped per batch. total_steps = epochs * batches_per_epoch."""
    warmup_steps = max(1, int(total_steps * warmup_frac))
    floor = lr_min / lr if lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps                       # linear 0 -> 1
        # cosine 1 -> floor over the remaining steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pca_rgb_shared(fit_chw: torch.Tensor, maps_chw: list[torch.Tensor]) -> list[np.ndarray]:
    """Fit PCA (top-3 dirs + min-max) on `fit_chw` (C,H,W), apply the SAME basis to each map in
    `maps_chw` -> list of (H,W,3). Shared basis so the panels are color-comparable."""
    C = fit_chw.shape[0]
    xf = fit_chw.reshape(C, -1).T.float().cpu().numpy()      # (H*W, C)
    mean = xf.mean(0, keepdims=True)
    cov = ((xf - mean).T @ (xf - mean)) / max(xf.shape[0] - 1, 1)
    _, evecs = np.linalg.eigh(cov)
    dirs = evecs[:, -3:][:, ::-1]                            # (C,3)
    proj_fit = (xf - mean) @ dirs
    lo, hi = proj_fit.min(0), proj_fit.max(0)
    out = []
    for m in maps_chw:
        H, W = m.shape[-2:]
        x = m.reshape(C, -1).T.float().cpu().numpy()
        proj = ((x - mean) @ dirs).reshape(H, W, 3)
        out.append(np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1))
    return out


def raw_rgb(s2_chw: torch.Tensor) -> np.ndarray:
    """(C,64,64) -> (64,64,3) percentile-stretched display image.

    C depends on the guidance modality: 13 (S2) and 15 (S2+S1, S2 first) both index the true
    colour bands B04/B03/B02; 2 (S1 alone, VV/VH) has no colour bands, so we show VV/VH/VV."""
    c = s2_chw.shape[0]
    bands = [3, 2, 1] if c >= 13 else [0, 1, 0]
    rgb = s2_chw[bands].float().cpu().numpy().transpose(1, 2, 0)
    lo = np.percentile(rgb, 2, (0, 1)); hi = np.percentile(rgb, 98, (0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
