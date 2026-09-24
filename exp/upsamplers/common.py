"""Shared plumbing for the feature-upsampler trainers (train_manyup, train_timanyup): cfg-name
parsing, node-local staging, the LR schedule and viz helpers. Their inputs (features, guidance)
come from exp/upsamplers/data.py; guidance itself is built in manyup/data/guidance.py, so both
trainers and the probe that evaluates their checkpoints share one implementation.
"""
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from exp.common.paths import FEATURES
from exp.pastis.data import GUIDANCE_MODS, guidance_spec

FEATURES_ROOT = FEATURES

# Channels of the full guidance stack per guidance_mod (s2 13, s1 2, s2s1 15), from its spec.
GUIDANCE_BANDS = {m: guidance_spec("full", m, "mean").channels for m in GUIDANCE_MODS}


def guidance_mod_for(cfg: str) -> str:
    """Default guidance modality for a feature cfg name: MATCH the arm the features come
    from. Guiding S1 features with S2 imagery (or s2s1 features with S2 alone) is a
    cross-modal mismatch -- the guidance should carry the same modalities as the features."""
    m = re.match(r"oe_(?:v\d+_\d+_)?[a-z]+_([a-z0-9]+)_ps", cfg)
    mods = m.group(1) if m else "s2"
    return mods if mods in GUIDANCE_BANDS else "s2"


def cfg_bits(cfg: str):
    """(modalities, patch_size) parsed out of an oe_<size>_<mods>_ps<N>_tile<M>[_...] cfg name."""
    m = re.match(r"oe_(?:v\d+_\d+_)?[a-z]+_([a-z0-9]+)_ps(\d+)_", cfg)
    return (m.group(1), m.group(2)) if m else ("mods", "?")


def stage_caches(root: Path, rel_dirs: list[str]) -> Path:
    """Copy cache folders (feature caches, guidance caches) from `root` into $SLURM_TMPDIR at the
    SAME relative paths, and return the root to read them from: the staged one, or `root` if
    SLURM_TMPDIR is unset or they do not fit (then nothing is copied)."""
    tmp = os.environ.get("SLURM_TMPDIR")
    if not tmp:
        print("SLURM_TMPDIR unset -> reading caches in place (no staging).")
        return root
    srcs = [root / r for r in rel_dirs]
    total = sum(f.stat().st_size for d in srcs for f in d.rglob("*") if f.is_file())
    free = shutil.disk_usage(tmp).free
    if total > free * 0.95:
        print(f"staging: need {total/1e9:.1f} GB, {free/1e9:.1f} GB free -> reading in place.")
        return root
    for r, src in zip(rel_dirs, srcs):
        dst = Path(tmp) / r
        if not dst.exists():
            t0 = time.time()
            shutil.copytree(src, dst)
            print(f"staged {src} -> {dst} ({time.time() - t0:.0f}s)")
    return Path(tmp)


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
