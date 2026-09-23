"""Train a segmentation head on CACHED OlmoEarth features (no encoder).

Pairs with exp/pastis/extract_features.py: that script writes per-sample
features/<cfg>/pastis_r_<split>/<idx>.pt of shape (T, gH, gW, D). Here we load those and
fit a head directly, so there is no encoder forward pass -- head iteration is fast and
needs no GPU for the linear-probe heads.

Heads consume (B, T, gH, gW, D), mean over T, then probe (see the heads section):
  - lp_pa2pa_bu: 1x1 conv on tokens -> bilinear-upsample the predictions to label res.
  - lp_pa2px:    1x1 conv D->C*p^2 -> unfold sub-pixels (the live BackboneWithHead seg head).
The mIoU gap between them measures within-patch spatial structure vs. patch-level semantics.
Extra heads (lp_per_t, AnyUp-on-cache, ...) drop into the build_cached_head registry; AnyUp
variants would additionally load cached RGB guidance and are out of scope for v1.

Runs in the OlmoEarth venv (for segmentation_metrics); via salloc or even CPU:
    source env_setup/env_olmo.sh
    python -u -m exp.pastis.lp_cached_features --features oe_base_s2_ps4_tile64 --head_mode lp_pa2px
"""
import os
import sys

# Bootstrap before importing olmoearth_pretrain (only segmentation_metrics is needed; no
# model is ever loaded here). Mirrors exp/pastis/finetune_olmoearth.py.
from exp.common import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.metrics import segmentation_metrics

# AnyUp upsample+probe and the RGB-guidance loader are shared with the live finetune path
# (single source of truth). Importing the module is cheap; AnyUp (torch.hub) only loads when
# an AnyUpUpsampleProbe is actually constructed (i.e. only for the anyup heads).
from exp.pastis.finetune_olmoearth import AnyUpUpsampleProbe, _load_rgb_guidance

# Guidance time-pooling ("mean"|"median"), shared with the UPA/UPMA/AnyUp eval paths.
from exp.upsamplers.upa_anyup import time_pool, TIME_POOLS

# Cosine annealing decays the LR from args.lr to SCHEDULER_MIN_LR over args.epochs.
SCHEDULER_MIN_LR = 1e-6

NUM_CLASSES = 20            # PASTIS: 20 classes (class 19 = void -> ignore via -1 labels)
IGNORE_LABEL = -1
LABEL_SIZE = 64            # PASTIS label resolution

# Guidance mode each head needs from the dataset: "none" (LP heads), "mean" ((3,64,64)
# time-averaged RGB, plain anyup), or "temporal" ((T,3,64,64) per-timestep RGB, anyup_t1/t2).
HEAD_GUIDANCE = {
    "lp_pa2pa_bu": "none",
    "lp_pa2px": "none",
    "lp_pa2px_ens": "none",     # temporal ensemble of per-t pa2px probes (no guidance)
    "lp_bu_px2px": "none",      # bilinear-upsampled features + per-pixel probe (no guidance)
    "anyup": "mean",
    "anyup_t2": "temporal",
    "anyup_t1": "temporal",
    "anyup_t2_ens": "temporal",
    "anyup_t1_ens": "temporal",
    # mAnyUp: our trained upsampler with FULL 13-band S2 guidance (time-averaged), matching how
    # train_manyup.py fed it. "mean13" -> (13,64,64), distinct from anyup's 3-band "mean".
    "manyup": "mean13",
    "manyup_shared": "mean13",
    # timAnyUp needs PER-TIMESTEP 13-band guidance: it upsamples each frame independently,
    # guided by that frame's own image. "temporal13" -> (T,13,64,64).
    "timanyup": "temporal13",
    "timanyup_t": "temporal13",
}

# Whether the head collapses time via feats.mean(dim=1) as its first op. When True the dataset
# pre-reduces the cached (T,gH,gW,D) to (1,gH,gW,D) ONCE at preload, so we don't float-cast and
# ship the full T=12 tensor across PCIe every step only for the head to average it away (a 12x
# cut in RAM + host->device traffic; the head's mean over a singleton T is then a no-op). Only
# anyup_t1 needs the per-timestep features, so it opts out.
HEAD_REDUCES_TIME = {
    "lp_pa2pa_bu": True,
    "lp_pa2px": True,
    "lp_pa2px_ens": False,      # ensemble needs the full per-timestep feature map
    "lp_bu_px2px": True,
    "anyup": True,
    "anyup_t2": True,
    "anyup_t1": False,
    # t2_ens shares one time-pooled feature map (T comes from the 5-D per-t RGB, so the ensemble
    # still gets its T probes); t1_ens needs the real per-timestep features.
    "anyup_t2_ens": True,
    "anyup_t1_ens": False,
    "manyup": True,          # mAnyUp mean-pools T on the LR feats before upsampling
    "manyup_shared": True,
    # Both timAnyUp heads run the upsampler per timestep, so the dataset must keep the full
    # T. They differ in what happens AFTER: "timanyup" mean-pools the fused map before the
    # probe, "timanyup_t" probes each timestep and averages the logits.
    "timanyup": False,
    "timanyup_t": False,
}


S2_BANDS = 13   # full Sentinel-2 L2A stack used as mAnyUp guidance
# Which image dirs each guidance modality reads, in channel order (mirrors train_manyup).
GUIDANCE_DIRS = {"s2": ["s2"], "s1": ["s1"], "s2s1": ["s2", "s1"]}
# Band count each modality yields once concatenated (mirrors train_manyup.GUIDANCE_BANDS):
# s2 = 13-band L2A, s1 = 2-band VV/VH, s2s1 stacks both -> 15.
GUIDANCE_BANDS = {"s2": S2_BANDS, "s1": 2, "s2s1": S2_BANDS + 2}


def _load_s2_guidance(split: str, idx: int, tpool: str = "mean",
                      mod: str = "s2") -> torch.Tensor:
    """Full 13-band S2 guidance for mAnyUp: (T,13,64,64) -> pooled over T -> (13,64,64), per-band
    min-max normalized to [0,1]. Matches train_manyup (same pooling + _norm_guidance) so the
    frozen mAnyUp sees exactly the guidance distribution it trained on -- `tpool` MUST equal the
    checkpoint's train-time --time_pool or the encoder gets an unseen input distribution. Reads
    from exp.pastis.finetune_olmoearth's DATA_SPLITS (a module global the caller points at our
    data_splits)."""
    from exp.pastis import finetune_olmoearth as fmod
    def one(m: str) -> torch.Tensor:
        x = torch.load(Path(fmod.DATA_SPLITS) / f"pastis_r_{split}" / f"{m}_images" / f"{idx}.pt")
        x = time_pool(x.float(), tpool)                        # (C,64,64)
        flat = x.reshape(x.shape[0], -1)
        lo = flat.min(1).values.view(-1, 1, 1)
        hi = flat.max(1).values.view(-1, 1, 1)
        return (x - lo) / (hi - lo + 1e-6)

    # Mirrors train_manyup: normalize each modality on its OWN scale, then concat on the band
    # axis (s2 13 + s1 2 = 15 for the s2s1 arm).
    return torch.cat([one(m) for m in GUIDANCE_DIRS.get(mod, [mod])], dim=0)


# ----------------------------- data -----------------------------
class CachedFeatureDataset(torch.utils.data.Dataset):
    """Loads cached (T, gH, gW, D) features, the matching (64,64) label, and -- for AnyUp
    heads -- the RGB guidance image.

    Labels come from the original prep (targets.pt is a stacked (N,64,64) tensor); features
    are keyed by the same contiguous index used at extraction time.

    guidance: "none" -> rgb is an empty tensor (LP heads ignore it); "mean" -> (3,64,64)
    time-averaged RGB; "temporal" -> (T,3,64,64) per-timestep RGB. Guidance is built by
    exp.pastis.finetune_olmoearth._load_rgb_guidance, which reads s2_images from that module's
    DATA_SPLITS global -- main() points it at args.data_splits before loaders are built."""

    def __init__(self, features_dir: Path, data_splits: Path, split: str,
                 guidance: str = "none", max_ram_gb: float = 32.0,
                 reduce_time: bool = False, time_pool: str = "mean",
                 guidance_mod: str = "s2", id_half: str = "all"):
        self.guidance_mod = guidance_mod
        self.feat_dir = features_dir / f"pastis_r_{split}"
        self.labels = torch.load(data_splits / f"pastis_r_{split}" / "targets.pt")
        # Label resolution comes from the DATA, not a constant: a prepare_data.py
        # --image_size 128 prep yields (N,128,128) targets instead of (N,64,64), and the heads
        # upsample their logits to exactly this size.
        self.label_size = int(self.labels.shape[-1])
        self.split = split
        self.guidance = guidance
        # How the S2 series is collapsed into a single guidance frame ("mean"|"median"). Only
        # applies to the single-frame modes; "temporal" keeps every frame and ignores it.
        self.time_pool = time_pool
        # If the head mean-pools over time, collapse (T,gH,gW,D)->(1,gH,gW,D) once here so we
        # never float-cast / ship the full T tensor per step. Keeps a singleton T so heads that
        # do feats.mean(dim=1) / feats.shape[1] stay correct unchanged.
        self.reduce_time = reduce_time
        self.n = len(list(self.feat_dir.glob("*.pt")))
        if self.n != len(self.labels):
            raise ValueError(
                f"{split}: {self.n} feature files != {len(self.labels)} labels. "
                "Re-run extraction or check the features/<cfg> path.")

        # Speed path: torch.load-ing N small .pt files from Lustre EVERY epoch is the
        # bottleneck (the GPU sits idle waiting on disk). If the whole split fits in a RAM
        # budget, read it ONCE into a single fp16 tensor and index that instead -- epochs
        # then run at compute speed. Otherwise fall back to per-sample disk loading (still
        # works, just slower; pair with --num_workers). Decision is logged verbosely.
        # Optional disjoint half of the TRAIN split (see train_manyup --id_half): lets the LP
        # probe be fitted on samples the upsampler never saw, so a leaky upsampler cannot
        # flatter the probe. Applied to train only -- valid/test must stay whole to remain
        # comparable with every other run.
        self._ids = list(range(self.n))
        if id_half in ("first", "second") and split == "train":
            mid = self.n // 2
            self._ids = self._ids[:mid] if id_half == "first" else self._ids[mid:]
            print(f"[{split}] id_half={id_half}: {len(self._ids)} of {self.n} samples")
            self.n = len(self._ids)
            self.labels = self.labels[torch.tensor(self._ids)]

        self._feats = None          # set iff preloaded
        self._rgb = None
        self._maybe_preload(max_ram_gb)

    def _est_gb(self, per_sample_feat_elems: int) -> float:
        """Estimated RAM for the preloaded fp16 feature tensor (+ fp16 guidance if needed)."""
        feat_gb = self.n * per_sample_feat_elems * 2 / 1e9          # fp16 = 2 bytes
        rgb_gb = 0.0
        if self.guidance != "none":
            # mean: (3,64,64); temporal: (T,3,64,64) -- T read from a sample below
            rgb_gb = self._rgb_elems * self.n * 2 / 1e9
        return feat_gb + rgb_gb

    def _maybe_preload(self, max_ram_gb: float) -> None:
        if self.n == 0:
            return
        probe = torch.load(self.feat_dir / "0.pt")                 # (T, gH, gW, D), fp16 on disk
        T = probe.shape[0]
        # Shape actually stored per sample: (1,gH,gW,D) when the head averages over time.
        feat_shape = (1, *probe.shape[1:]) if self.reduce_time else tuple(probe.shape)
        feat_elems = int(torch.tensor(feat_shape).prod())
        # Guidance is at full image resolution, which equals the label resolution. Derive the
        # element count from the SAME shape the buffer is allocated with, so the RAM estimate
        # tracks mean13's 13/15 bands instead of assuming 3-channel RGB.
        px = self.label_size * self.label_size
        self._rgb_elems = int(torch.tensor(self._rgb_shape(T)[:-2]).prod()) * px
        est = self._est_gb(feat_elems)
        if est > max_ram_gb:
            print(f"[{self.split}] preload SKIPPED: est {est:.1f} GB > --max_ram_gb "
                  f"{max_ram_gb:.1f} GB. Falling back to per-sample disk loading "
                  f"({self.n} files/epoch); raise --max_ram_gb or --num_workers to speed up.")
            return
        print(f"[{self.split}] preloading {self.n} samples (~{est:.1f} GB fp16) into RAM "
              f"once{' [time-reduced]' if self.reduce_time else ''}; "
              f"epochs will run at compute speed...")
        # Keep features in fp16 to halve RAM; cast to float per-batch in __getitem__.
        self._feats = torch.empty((self.n, *feat_shape), dtype=torch.float16)
        rgb_buf = (torch.empty((self.n, *self._rgb_shape(T)), dtype=torch.float16)
                   if self.guidance != "none" else None)
        for i in tqdm(range(self.n), desc=f"preload {self.split}", leave=False):
            feat = torch.load(self.feat_dir / f"{self._ids[i]}.pt")   # (T,gH,gW,D) fp16
            # Mean over T in fp32 for accuracy, then store fp16; keep a singleton T dim.
            self._feats[i] = feat.float().mean(dim=0, keepdim=True).half() if self.reduce_time else feat
            if rgb_buf is not None:
                rgb_buf[i] = self._load_guidance(self._ids[i]).half()
        self._rgb = rgb_buf
        print(f"[{self.split}] preload done.")

    def _load_guidance(self, idx: int) -> torch.Tensor:
        """Guidance tensor for one sample, per self.guidance mode. mean13 -> full 13-band S2;
        mean/temporal -> 3-band RGB via the shared finetune loader."""
        if self.guidance == "temporal13":
            # timAnyUp guides each timestep with its OWN frame, so no time pooling here.
            # Normalization must match exp/upsamplers/common.norm_guidance exactly (per
            # (t,band) min-max over the spatial axes) or the frozen encoder sees an unseen
            # input scale -- so we import that function rather than re-deriving it.
            from exp.upsamplers.common import norm_guidance
            from exp.pastis import finetune_olmoearth as fmod
            mod = getattr(self, "guidance_mod", "s2")
            return torch.cat(
                [norm_guidance(torch.load(
                    Path(fmod.DATA_SPLITS) / f"pastis_r_{self.split}" / f"{m}_images"
                    / f"{idx}.pt").float())
                 for m in GUIDANCE_DIRS.get(mod, [mod])], dim=1)   # (T, C, H, W)
        if self.guidance == "mean13":
            # guidance_mod mirrors the mAnyUp checkpoint's train-time modality (s2 = 13-band,
            # s1 = 2-band); evaluating with a different one feeds an unseen distribution.
            return _load_s2_guidance(self.split, idx, tpool=self.time_pool,
                                     mod=getattr(self, "guidance_mod", "s2"))
        return _load_rgb_guidance(self.split, idx, temporal=self.guidance == "temporal",
                                  time_pool=self.time_pool)

    def _rgb_shape(self, T: int):
        L = self.label_size                      # guidance is at image == label resolution
        if self.guidance == "temporal13":
            return (T, GUIDANCE_BANDS[self.guidance_mod], L, L)
        if self.guidance == "mean13":
            # Channel count follows the checkpoint's guidance modality, not always 13 --
            # the s2s1 arm stacks S2+S1 into 15 bands.
            return (GUIDANCE_BANDS[self.guidance_mod], L, L)
        return (T, 3, L, L) if self.guidance == "temporal" else (3, L, L)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        label = self.labels[idx].long()                           # (64,64)
        if self._feats is not None:                                # RAM path
            feat = self._feats[idx].float()                        # (T, gH, gW, D)
            if self.guidance == "none":
                rgb = torch.empty(0)
            else:
                rgb = self._rgb[idx].float()
            return feat, label, rgb
        # disk fallback
        feat = torch.load(self.feat_dir / f"{self._ids[idx]}.pt").float()   # (T,gH,gW,D)
        if self.reduce_time:
            feat = feat.mean(dim=0, keepdim=True)                 # (1, gH, gW, D)
        if self.guidance == "none":
            rgb = torch.empty(0)
        else:
            rgb = self._load_guidance(self._ids[idx])
        return feat, label, rgb


class TimAnyUpFeatureDataset(CachedFeatureDataset):
    """CachedFeatureDataset + a SECOND cached feature map, for timAnyUp.

    timAnyUp consumes two arms: F_lrhc (the base `features_dir`, low-res/high-context) and
    F_hrlc (`hrlc_dir`, high-res/low-context). Everything else -- labels, guidance, preload,
    the RAM budget, id_half -- is inherited unchanged.

    A subclass rather than widening CachedFeatureDataset's return: every existing head and both
    eval loops unpack a fixed (feat, label, rgb) triple positionally, so changing that shape
    would touch all of them. Here the extra map rides along INSIDE the first element as a
    (lrhc, hrlc) tuple, which the collate handles natively and only _run_head unpacks -- so no
    existing path changes shape.

    DANGER: F_hrlc and F_hrhc are shape-identical and differ only in extraction context, so a
    wrong --hrlc_features is invisible to any shape check. The checkpoint records the cfg name
    it trained with and build_timanyup_loader() compares against it.
    """

    def __init__(self, features_dir, hrlc_dir, *args, **kwargs):
        self.hrlc_dir_root = Path(hrlc_dir)
        super().__init__(features_dir, *args, **kwargs)
        self.hrlc_dir = self.hrlc_dir_root / f"pastis_r_{self.split}"
        n_hrlc = len(list(self.hrlc_dir.glob("*.pt")))
        if n_hrlc < self.n:
            raise ValueError(f"{self.split}: F_hrlc has {n_hrlc} files but need {self.n} "
                             f"({self.hrlc_dir})")
        self._hrlc = None
        self._maybe_preload_hrlc(kwargs.get("max_ram_gb", 32.0))

    def _maybe_preload_hrlc(self, max_ram_gb: float) -> None:
        """Preload the second arm only if the FIRST one was preloaded -- mixing a RAM-resident
        arm with a disk-read arm would just move the bottleneck, not remove it."""
        if self._feats is None or self.n == 0:
            return
        probe = torch.load(self.hrlc_dir / f"{self._ids[0]}.pt")
        est = self.n * int(torch.tensor(probe.shape).prod()) * 2 / 1e9
        if est > max_ram_gb:
            print(f"[{self.split}] F_hrlc preload SKIPPED: est {est:.1f} GB > "
                  f"--max_ram_gb {max_ram_gb:.1f} GB -> per-sample disk loading for BOTH arms.")
            self._feats = None            # keep the two arms on the same path
            return
        print(f"[{self.split}] preloading F_hrlc ({est:.1f} GB fp16)...")
        self._hrlc = torch.empty((self.n, *probe.shape), dtype=torch.float16)
        for i in tqdm(range(self.n), desc=f"preload hrlc {self.split}", leave=False):
            self._hrlc[i] = torch.load(self.hrlc_dir / f"{self._ids[i]}.pt")
        print(f"[{self.split}] F_hrlc preload done.")

    def __getitem__(self, idx: int):
        feat, label, rgb = super().__getitem__(idx)
        if self._hrlc is not None:
            hrlc = self._hrlc[idx].float()
        else:
            hrlc = torch.load(self.hrlc_dir / f"{self._ids[idx]}.pt").float()
        return (feat, hrlc), label, rgb


# ----------------------------- heads -----------------------------
# Two linear-probe heads that probe WHAT a patch token encodes. The mIoU gap between them
# measures whether tokens carry within-patch spatial structure or only patch-level semantics
# -- a direct motivation for a learned, guidance-driven upsampler (e.g. AnyUp).
#
# Both first mean-pool over time -> (B, gH, gW, D).
#   lp_pa2pa_bu (patch->patch, then bilinear-upsample the PREDICTIONS):
#       1x1 conv D->C on the (gH,gW) token grid, then bilinear-upsample the C-channel logits
#       to the label resolution. Assumes a token says "this patch is class X"; pixel detail
#       comes purely from the spatial-smoothness inductive bias of bilinear upsampling.
#   lp_pa2px (patch->pixel):
#       1x1 conv D->C*patch_size^2, then rearrange the extra channels into sub-pixels
#       (b (c i j) gh gw) -> (b c (gh i) (gw j)). Assumes a token encodes the within-patch
#       spatial layout. This mirrors the live BackboneWithHead seg head exactly
#       (exp/pastis/finetune_olmoearth.py:313).
#
# lp_bu_px2px: bilinear-upsample FEATURES to label res, then a per-pixel 1x1 probe.
#
# On its OWN it is equivalent to lp_pa2pa_bu -- 1x1 conv and bilinear upsample are linear and
# act on disjoint axes (channel vs spatial), so they commute (verified: max|diff| ~2e-06). It
# earns its place as a SHARED DECODER: the same trained probe can be applied to a different
# feature map of the same width, e.g. a frozen upsampler's output. Then the two routes differ
# only in the feature map, not in the probe fitted to it, so any mIoU gap is attributable to
# the upsampling rather than to two separately-fitted probes. See CachedManyUpSharedProbe.


class LPPatchToPatchBU(nn.Module):
    """lp_pa2pa_bu: 1x1 conv on tokens -> bilinear-upsample the logits to label res."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE):
        super().__init__()
        self.probe = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.label_size = label_size

    def features(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:
        """Per-pixel feature map at label res, for KNN: mean-T token grid bilinear-upsampled
        (D,gH,gW)->(D,label,label). The probe-free counterpart of forward()."""
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B,D,gH,gW)
        if x.shape[-2:] != (self.label_size, self.label_size):
            x = F.interpolate(x, size=(self.label_size, self.label_size),
                              mode="bilinear", align_corners=True)
        return x                                                # (B,D,label,label)

    def forward(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:   # (B,T,gH,gW,D)
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
        logits = self.probe(x)                                  # (B, C, gH, gW)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


class LPBilinearToPixel(nn.Module):
    """lp_bu_px2px: bilinear-upsample the token grid to label res, then a 1x1 per-pixel probe.

    The probe is Conv2d(D -> C) applied at FULL label resolution, so it is shape-compatible with
    any (D, label, label) feature map -- which is what lets CachedManyUpSharedProbe reuse it on
    a frozen upsampler's output without refitting."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE):
        super().__init__()
        self.probe = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.label_size = label_size

    def features(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B,D,gH,gW)
        if x.shape[-2:] != (self.label_size, self.label_size):
            x = F.interpolate(x, size=(self.label_size, self.label_size),
                              mode="bilinear", align_corners=True)
        return x                                                # (B,D,label,label)

    def forward(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:
        return self.probe(self.features(feats))                 # (B,C,label,label)


class LPPatchToPixel(nn.Module):
    """lp_pa2px: 1x1 conv D->C*patch_size^2, then unfold the extra channels into sub-pixels.

    ensemble=False (default): mean-pool over T, then ONE probe (the original lp_pa2px).
    ensemble=True (lp_pa2px_ens): keep the full T feature map, fit an INDEPENDENT probe per
    timestep, and average the T per-pixel logits (pre-softmax) -- a temporal ensemble, mirroring
    the anyup *_ens variants but on the raw sub-pixel LP head. The per-t probes are created
    lazily on the first forward (T is a runtime dim); pass reduce_time=False for this head so the
    dataset keeps all T timesteps."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE, ensemble: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.label_size = label_size
        self.ensemble = ensemble
        out_ch = num_classes * patch_size * patch_size
        if ensemble:
            self.probe = None            # T independent probes, built lazily (T known at forward)
            self._out_ch = out_ch
        else:
            self.probe = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

    def _init_probes(self, T: int, device) -> None:
        self.probe = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, self._out_ch, 1) for _ in range(T)]).to(device)

    def _unfold(self, logits):
        """(B, C*p*p, gH, gW) -> (B, C, label, label) via sub-pixel unfold + resize if needed."""
        p = self.patch_size
        logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                           c=self.num_classes, i=p, j=p)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits

    def features(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:
        """Per-pixel features at label res for KNN. pa2px's sub-pixel unfold is a PROBE-space
        trick (it needs the class dim), so for feature-space KNN we fall back to the same
        bilinear-upsampled token grid as pa2pa_bu -- KNN on raw features doesn't use the p^2
        sub-pixel channels."""
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B,D,gH,gW)
        if x.shape[-2:] != (self.label_size, self.label_size):
            x = F.interpolate(x, size=(self.label_size, self.label_size),
                              mode="bilinear", align_corners=True)
        return x

    def forward(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:   # (B,T,gH,gW,D)
        if not self.ensemble:
            x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
            return self._unfold(self.probe(x))

        # Ensemble: per-timestep probe on that timestep's feature map, average the logits.
        T = feats.shape[1]
        if self.probe is None:
            self._init_probes(T, feats.device)
        acc = None
        for t in range(T):
            x_t = feats[:, t].permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
            logit_t = self._unfold(self.probe[t](x_t))          # (B, C, label, label)
            acc = logit_t if acc is None else acc + logit_t
        return acc / T                                          # mean of per-t per-pixel logits


# ---- AnyUp heads on cached features ----
# Same upsample+probe logic as the live finetune AnyUp heads (via the shared
# AnyUpUpsampleProbe), but features come from the cache (T,gH,gW,D) instead of the encoder.
# This is exactly why the cache keeps per-timestep features: the three variants differ only
# in how they feed the cached (T,gH,gW,D) and the RGB guidance into AnyUpUpsampleProbe:
#   anyup    : mean over T -> single (B,D,gH,gW); single mean RGB (B,3,64,64).      [guidance=mean]
#   anyup_t2 : mean over T -> single (B,D,gH,gW); per-timestep RGB (B,T,3,64,64).   [guidance=temporal]
#   anyup_t1 : per-timestep features (list of T (B,D,gH,gW)); per-timestep RGB.     [guidance=temporal]

class CachedAnyUp(nn.Module):
    """anyup: cached features mean-pooled over T, single mean RGB guidance.

    ensemble=False (default): the T AnyUp-upsampled maps are mean-pooled, then ONE probe.
    ensemble=True: each timestep's upsampled map gets its OWN probe and the T logits are
    averaged pre-softmax (a temporal ensemble). anyup (single-timestep) ignores the flag."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE, ensemble: bool = False):
        super().__init__()
        self.up = AnyUpUpsampleProbe(num_classes, ensemble=ensemble)
        self.label_size = label_size

    def _feats_2d(self, feats):                          # (B,T,gH,gW,D) -> (B,D,gH,gW)
        return feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()

    def _upsampled(self, feats_per_t, rgb) -> torch.Tensor:
        """Run AnyUp's per-timestep upsampling and mean-pool the T maps -> (B,D,64,64), WITHOUT
        the probe. Mirrors AnyUpUpsampleProbe.forward's non-ensemble accumulation so KNN reads
        exactly the features the LP probe would see. `feats_per_t` is a single (B,D,gH,gW) tensor
        (reused for all t) or a list of T of them; rgb is (B,3,64,64) or (B,T,3,64,64)."""
        out = (self.label_size, self.label_size)
        per_t_feats = isinstance(feats_per_t, (list, tuple))
        per_t_rgb = rgb.dim() == 5
        T = len(feats_per_t) if per_t_feats else (rgb.shape[1] if per_t_rgb else 1)
        shared_f = None if per_t_feats else feats_per_t.float()
        acc = None
        for t in range(T):
            f = feats_per_t[t].float() if per_t_feats else shared_f
            g = (rgb[:, t] if per_t_rgb else rgb).float()
            hr_t = self.up.anyup(g, f, output_size=out)      # (B,D,64,64)
            acc = hr_t if acc is None else acc + hr_t
        return acc / T

    def _feed(self, feats, rgb):
        """Per-subclass (feats_per_t, rgb) feed. Base: single time-pooled map."""
        return self._feats_2d(feats), rgb

    @torch.no_grad()
    def features(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Frozen AnyUp-upsampled per-pixel feature map (B,D,64,64) for KNN (pre-probe)."""
        f, g = self._feed(feats, rgb)
        return self._upsampled(f, g)

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        out = (self.label_size, self.label_size)
        return self.up(self._feats_2d(feats), rgb, out_size=out)


class CachedAnyUpT2(CachedAnyUp):
    """anyup_t2: shared time-pooled features, per-timestep RGB guidance (rgb is (B,T,3,64,64))."""
    # forward identical to CachedAnyUp: AnyUpUpsampleProbe loops T off the 5-D rgb, reusing
    # the single feature map for every timestep.


class CachedAnyUpT1(CachedAnyUp):
    """anyup_t1: per-timestep features AND per-timestep RGB (heaviest)."""

    def _feats_per_t(self, feats):
        # list of T (B,D,gH,gW), one per cached timestep
        return [feats[:, t].permute(0, 3, 1, 2).contiguous() for t in range(feats.shape[1])]

    def _feed(self, feats, rgb):                          # per-t features + per-t rgb
        return self._feats_per_t(feats), rgb

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        out = (self.label_size, self.label_size)
        return self.up(self._feats_per_t(feats), rgb, out_size=out)


class CachedManyUp(nn.Module):
    """mAnyUp head: FROZEN trained upsampler (+ optional projector) with a trainable LP probe.

    Pipeline: cached LR feats (mean-T) -> [frozen mAnyUp upsample to 64x64] -> [frozen projector
    if use_proj] -> trainable 1x1 probe -> bilinear to label size. Only the probe trains -- this
    is linear-probing on top of frozen mAnyUp-upsampled features. Guidance is the 13-band S2
    ('mean13'). The checkpoint (from train_manyup.py) carries model + optional proj_head weights,
    input_dim (13), and qk_dim."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 ckpt_path: str, use_proj: bool = True, label_size: int = LABEL_SIZE,
                 time_pool: str = "mean", native_out: bool = False):
        super().__init__()
        import sys
        sys.path.insert(0, "/scratch/timz/rs-change-detection/third_party/anyup")
        from anyup.model import AnyUp

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Rebuild the SAME architecture the checkpoint was trained with. "manyup" adds a
        # nonlinear post-attention transform, so its state_dict does not fit stock AnyUp.
        # Checkpoints predating the flag have no "arch" key and are stock AnyUp.
        arch = ck.get("arch", "anyup")
        if arch == "manyup":
            from anyup.mAnyUp import mAnyUp as AnyUp   # noqa: F811
        # The guidance encoder adapted to whatever composite it trained on, so evaluating with a
        # different one silently feeds it an unseen input distribution. Checkpoints written before
        # --time_pool existed have no entry and are mean by construction.
        trained_tp = ck.get("args", {}).get("time_pool", "mean")
        if trained_tp != time_pool:
            raise ValueError(
                f"guidance time_pool mismatch: {Path(ckpt_path).name} was TRAINED with "
                f"'{trained_tp}' but eval requested '{time_pool}'. Pass --time_pool {trained_tp}, "
                f"or retrain with train_manyup.py --time_pool {time_pool}.")
        up_kw = dict(input_dim=ck.get("input_dim", S2_BANDS), qk_dim=ck.get("qk_dim", 128),
                     # Width of the cross-attention window. It changes WHICH LR tokens each
                     # query may attend to, so evaluating a wide-window checkpoint at AnyUp's
                     # narrow 0.1 default would silently feed it an unseen attention pattern.
                     # Checkpoints predating the flag were trained at that default.
                     window_ratio=ck.get("window_ratio", 0.1))
        if arch == "manyup":
            up_kw["feat_dim"] = ck.get("feat_dim", embed_dim)
            # Depth of the post-attention transform. It changes the module list the weights
            # were saved from, so a depth mismatch is a load_state_dict error, not a silent
            # accuracy drop. Checkpoints predating the flag have one block by construction.
            up_kw["transform_depth"] = ck.get("transform_depth", 1)
        self.up = AnyUp(**up_kw)
        self.up.load_state_dict(ck["model"])
        self.proj = None
        if use_proj and ck.get("proj_head") is not None:
            self.proj = nn.Conv2d(embed_dim, embed_dim, 1)
            self.proj.load_state_dict(ck["proj_head"])
        # Freeze the whole upsampling pipeline; only the probe below is trained.
        for m in (self.up, self.proj):
            if m is not None:
                for prm in m.parameters():
                    prm.requires_grad = False
        self.probe = nn.Conv2d(embed_dim, num_classes, 1)     # the ONLY trainable module (LP)
        self.num_classes = num_classes
        self.label_size = label_size
        self.ckpt_path = ckpt_path

        # Output size the upsampler actually RUNS at. By default we ask for the label
        # resolution in one hop, which for a ps16 -> ps4 checkpoint means a 4x4 -> 64x64
        # (16x) upsample even though it was TRAINED for 4x4 -> 16x16 (4x). AnyUp accepts any
        # out_size, but the guidance attention was fitted at the training scale, so the
        # default runs the upsampler out of distribution.
        #
        # native_out=True runs the upsampler at exactly the grid it was trained to produce
        # (parsed from the checkpoint's own hr_cfg) and then reaches label resolution with a
        # pa2px probe -- Conv2d(D -> C*q^2) unfolded into q x q sub-pixels per token, where q
        # is the target config's patch size. So a ps16 -> ps4 checkpoint upsamples 4x4 -> 16x16
        # in-distribution and each ps4 token predicts its own 4x4 pixel block, exactly like
        # LPPatchToPixel on real ps4 features -- no bilinear anywhere. The probe is therefore
        # shape-compatible with an lp_pa2px probe trained on the real hr_cfg features.
        self.native_size = None
        self.sub_patch = 1
        if native_out:
            hr_cfg = ck.get("args", {}).get("hr_cfg", "")
            m = re.search(r"_ps(\d+)_", hr_cfg)
            if not m:
                raise ValueError(
                    f"--manyup_native_out needs the trained target grid, but checkpoint "
                    f"{Path(ckpt_path).name} has no parseable hr_cfg (got {hr_cfg!r}).")
            img = 128 if "_img128" in hr_cfg else 64
            self.sub_patch = int(m.group(1))          # target cfg's patch size (e.g. 4)
            self.native_size = img // self.sub_patch  # its token grid (e.g. 16)
            if self.native_size > label_size:
                raise ValueError(
                    f"trained target grid {self.native_size} exceeds label size {label_size}")
            # pa2px probe: one token -> its own sub_patch x sub_patch block of pixels.
            self.probe = nn.Conv2d(embed_dim, num_classes * self.sub_patch ** 2, 1)

    def _feats_2d(self, feats):                               # (B,T,gH,gW,D) -> (B,D,gH,gW)
        return feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def features(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Frozen mAnyUp-upsampled (+projected) feature map, (B,D,S,S).

        S is label_size by default, or the checkpoint's trained target grid under
        native_out. Shared by forward()'s LP probe and by KNN eval -- both read the SAME
        frozen features."""
        s = self.native_size or self.label_size
        # Guidance goes in at FULL image resolution regardless of the output grid, because that
        # is exactly what train_manyup does: it always passes the (C,64,64) guidance and asks
        # for a (GH,GW) output, letting AnyUp pool the queries down internally
        # (adaptive_avg_pool2d on the query encoder). Pre-resizing the guidance to the output
        # grid instead -- as this used to do under native_out -- hands the frozen guidance
        # encoder an input resolution it never saw in training; measured on 16 s2s1 samples it
        # cost ~1% HR reconstruction loss (0.09401 -> 0.09509) for no benefit.
        hr = self.up(rgb, self._feats_2d(feats), (s, s))
        if self.proj is not None:
            hr = self.proj(hr)
        return hr                                            # (B,D,S,S)

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        logits = self.probe(self.features(feats, rgb))
        if self.native_size is not None:
            # (B, C*q*q, S, S) -> (B, C, S*q, S*q) == label resolution by construction.
            q = self.sub_patch
            logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                               c=self.num_classes, i=q, j=q)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=False)
        return logits                                        # (B,C,label,label)


class CachedManyUpSharedProbe(CachedManyUp):
    """manyup_shared: frozen mAnyUp -> bilinear to label res -> a FROZEN px2px probe trained on
    bilinear-upsampled LR features (lp_bu_px2px).

    Nothing is trained here. The point is to isolate the upsampler: the LR route
    (bilinear -> probe) and this route (mAnyUp -> bilinear -> the SAME probe) share one decoder,
    so the mIoU difference is caused by the feature map alone, not by two probes that happened
    to fit differently. That control is only meaningful when the upsampler's output lives in the
    same feature space as its input -- true for a transform_depth=0, no-projector checkpoint,
    whose output is a pure attention-weighted sum of the LR features.

    Mismatched checkpoints are rejected rather than silently producing a meaningless number.
    """

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int, ckpt_path: str,
                 shared_probe: str, use_proj: bool = True, label_size: int = LABEL_SIZE,
                 time_pool: str = "mean", native_out: bool = False,
                 shared_probe_side: str = "lr", allow_mismatch: bool = False):
        super().__init__(embed_dim, num_classes, patch_size, ckpt_path, use_proj=use_proj,
                         label_size=label_size, time_pool=time_pool, native_out=native_out)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Which feature space does the borrowed probe live in?
        #   LR-side probe (fitted on bilinear-upsampled LR feats): only valid when the
        #     upsampler leaves the LR space untouched -- transform_depth=0, no projector.
        #   HR-side probe (fitted on the REAL target-resolution feats): valid for ANY
        #     architecture, because that is the space every upsampler is trained to predict.
        #     This is the control to use for transform_depth>0.
        # We cannot tell them apart from the weights, so trust the caller's --shared_probe_side
        # and only enforce the LR-side precondition.
        lr_side_invalid = (ck.get("transform_depth", 1) != 0
                           or ck.get("proj_head") is not None)
        if shared_probe_side == "lr" and lr_side_invalid and allow_mismatch:
            # Deliberate control: how much does the nonlinear transform move the output OUT of
            # the LR feature space? Measured by borrowing the LR probe anyway. Expected to be
            # worse -- that degradation IS the measurement -- so it must never be read as this
            # upsampler's honest score.
            print(f"WARNING: LR-side probe on transform_depth="
                  f"{ck.get('transform_depth', 1)} / proj={ck.get('proj_head') is not None}. "
                  f"The probe was fitted in a different feature space; this is a CONTROL, "
                  f"not a fair score for this checkpoint.")
        elif shared_probe_side == "lr" and lr_side_invalid:
            raise ValueError(
                f"an LR-side shared probe assumes the upsampler output stays in the LR feature "
                f"space, which holds only for transform_depth=0 with no projector. "
                f"{Path(ckpt_path).name} has transform_depth="
                f"{ck.get('transform_depth', 1)}, proj_head="
                f"{ck.get('proj_head') is not None}. Pass a probe trained on the HR features "
                f"and --shared_probe_side hr instead.")
        sd = torch.load(shared_probe, map_location="cpu", weights_only=False)["head_state"]
        w, b = sd["probe.weight"], sd["probe.bias"]
        if w.shape[0] != num_classes or w.shape[1] != embed_dim:
            raise ValueError(f"shared probe is Conv2d({w.shape[1]}->{w.shape[0]}), expected "
                             f"Conv2d({embed_dim}->{num_classes}); is it an lp_bu_px2px head?")
        self.probe = nn.Conv2d(embed_dim, num_classes, 1)
        self.probe.load_state_dict({"weight": w, "bias": b})
        for prm in self.probe.parameters():        # frozen: this head trains nothing
            prm.requires_grad = False
        self.shared_probe_path = shared_probe
        self.shared_probe_side = shared_probe_side

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        # features() gives (B,D,S,S) at the upsampler's output grid; bilinear to label res so
        # the px2px probe sees exactly the resolution it was trained at.
        hr = self.features(feats, rgb)
        if hr.shape[-2:] != (self.label_size, self.label_size):
            hr = F.interpolate(hr, size=(self.label_size, self.label_size),
                               mode="bilinear", align_corners=True)
        return self.probe(hr)                      # (B,C,label,label)


class CachedTimAnyUp(nn.Module):
    """timAnyUp head: FROZEN trained timAnyUp + a trainable LP probe.

    Pipeline per sample:
        (F_lrhc, F_hrlc), per-timestep guidance
          -> [frozen] upsample each timestep, query top-k, blend, project
          -> F_fused (B,T,D,S,S)
          -> pool_time="mean": mean over T -> ONE shared probe        (head_mode=timanyup)
             pool_time="probe": an INDEPENDENT probe per timestep,
                                average the T logits                  (head_mode=timanyup_t)
          -> label resolution

    The two modes answer different questions. "mean" asks what the fused map carries once time
    is averaged away -- comparable with every other time-pooled LP row. "probe" is a temporal
    ensemble: each timestep gets its own probe, so a frame that is informative can be weighted
    differently from a cloudy one. Only the probes train.

    The per-timestep probes MUST be independent. With one shared 1x1 probe the two modes are
    mathematically identical -- a 1x1 conv is linear, so mean-then-probe == probe-then-mean
    (verified: max |diff| 2e-07) -- and "timanyup_t" would silently be a no-op returning the
    same numbers as "timanyup". This mirrors lp_pa2px_ens / anyup_t*_ens, which use per-timestep
    probes for the same reason.
    """

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 ckpt_path: str, pool_time: str = "mean", k: int = None,
                 label_size: int = LABEL_SIZE, native_out: bool = False):
        super().__init__()
        import sys
        sys.path.insert(0, "/scratch/timz/rs-change-detection/third_party/anyup")
        from anyup.timAnyUp import TimAnyUp, topk_mask, blend

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cargs = ck.get("args", {})
        self.model = TimAnyUp(
            input_dim=ck.get("input_dim", S2_BANDS), qk_dim=cargs.get("qk_dim", 128),
            feat_dim=ck.get("feat_dim", embed_dim), num_frames=ck.get("num_frames", 12),
            transform_depth=cargs.get("transform_depth", 0),
            window_ratio=cargs.get("window_ratio", 1.0),
            query_input=cargs.get("query_input", "bilinear"))
        self.model.load_state_dict(ck["model"])
        for prm in self.model.parameters():
            prm.requires_grad = False
        self._topk, self._blend = topk_mask, blend
        # k defaults to the checkpoint's TRAINED budget: the query head was fitted to rank
        # locations under that budget, so evaluating at a different k is a real change (allowed,
        # but it must be deliberate, hence the print).
        self.k = k if k is not None else cargs.get("k", 512)
        if self.k != cargs.get("k", 512):
            print(f"NOTE: evaluating at k={self.k} but checkpoint trained with "
                  f"k={cargs.get('k')}")
        if pool_time not in ("mean", "probe"):
            raise ValueError(f"pool_time must be mean|probe, got {pool_time}")
        self.pool_time = pool_time
        self.num_classes = num_classes
        self.label_size = label_size
        self.ckpt_path = ckpt_path
        # The output grid is FORCED to the trained target grid -- unlike mAnyUp, this is not
        # optional. The blend fuses the upsampled map with F_hrlc at top-k LOCATIONS, so the two
        # must live on the same grid; upsampling to label resolution instead makes the blend a
        # shape error (and, if it were resized, would silently break the index correspondence).
        # Parsed from the checkpoint's own target cfg.
        hr_cfg = cargs.get("hrhc_cfg", "")
        m = re.search(r"_ps(\d+)_", hr_cfg)
        if not m:
            raise ValueError(f"checkpoint {Path(ckpt_path).name} has no parseable hrhc_cfg "
                             f"(got {hr_cfg!r}); cannot determine the fused grid.")
        img = 128 if "_img128" in hr_cfg else 64
        self.sub_patch = int(m.group(1))          # target cfg patch size (e.g. 4)
        self.native_size = img // self.sub_patch  # its token grid (e.g. 16)
        # native_out picks how we get from that grid to label resolution: a pa2px probe
        # (each token predicts its own q x q pixel block, no interpolation) or bilinear.
        self.native_out = native_out
        out_ch = num_classes * (self.sub_patch ** 2 if native_out else 1)
        self.num_frames = ck.get("num_frames", 12)
        if pool_time == "probe":
            # One INDEPENDENT probe per timestep (see the class docstring: a single shared
            # linear probe would make this mode identical to "mean").
            self.probe = nn.ModuleList([nn.Conv2d(embed_dim, out_ch, 1)
                                        for _ in range(self.num_frames)])
        else:
            self.probe = nn.Conv2d(embed_dim, out_ch, 1)

    def features(self, feats, rgb) -> torch.Tensor:
        """Fused+projected map. feats is the (F_lrhc, F_hrlc) pair from
        TimAnyUpFeatureDataset; returns (B,T,D,S,S) -- time is NOT collapsed here, so both
        pooling modes read the same frozen features."""
        lrhc, hrlc = feats
        # cached features are (B,T,gH,gW,D); the model wants (B,T,D,gH,gW)
        lrhc = lrhc.permute(0, 1, 4, 2, 3).contiguous()
        hrlc = hrlc.permute(0, 1, 4, 2, 3).contiguous()
        s = self.native_size or self.label_size
        # The upsampler and query head are ALWAYS frozen, so their half of the graph is never
        # needed. The transform head is optionally trainable (--timanyup_train_transform), and
        # a blanket @torch.no_grad() here would silently leave it with no graph to backprop
        # through -- "element 0 of tensors does not require grad". So: no_grad for the frozen
        # part, normal autograd for the transform when it is being refitted.
        train_tf = any(prm.requires_grad for prm in self.model.transform_head.parameters())
        with torch.no_grad():
            up = self.model.upsample(lrhc, rgb, (s, s))
            scores = self.model.query(lrhc, (s, s), lrhc_up=up)
            mask = self._topk(scores.float(), self.k)
        with torch.set_grad_enabled(train_tf):
            hrlc_t = self.model.transform(hrlc)
            fused = self._blend(up, hrlc_t, mask)
            return self.model.project(fused)                          # (B,T,D,s,s)

    def _to_label(self, logits):
        if self.native_out:
            q = self.sub_patch
            logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                               c=self.num_classes, i=q, j=q)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits

    def forward(self, feats, rgb) -> torch.Tensor:
        fused = self.features(feats, rgb)                            # (B,T,D,s,s)
        if self.pool_time == "mean":
            return self._to_label(self.probe(fused.mean(dim=1)))
        # Temporal ensemble: each timestep through its OWN probe, then average the LOGITS.
        T = fused.shape[1]
        if T != len(self.probe):
            raise ValueError(f"timanyup_t has {len(self.probe)} per-timestep probes but the "
                             f"batch has T={T} (checkpoint num_frames mismatch)")
        return torch.stack([self._to_label(self.probe[t](fused[:, t]))
                            for t in range(T)], dim=1).mean(dim=1)


def build_cached_head(name: str, embed_dim: int, num_classes: int, patch_size: int,
                      manyup_ckpt: str = None, manyup_use_proj: bool = True,
                      manyup_native_out: bool = False, manyup_shared_probe: str = None,
                      manyup_shared_probe_side: str = "lr",
                      manyup_allow_probe_mismatch: bool = False,
                      timanyup_ckpt: str = None, timanyup_k: int = None,
                      time_pool: str = "mean", label_size: int = LABEL_SIZE) -> nn.Module:
    # (class, extra kwargs). The _ens variants reuse the same wrapper but give each timestep its
    # own probe and average the per-timestep logits instead of mean-pooling features (see
    # AnyUpUpsampleProbe.ensemble). t1_ens = per-timestep features; t2_ens = shared time-pooled
    # features -- both with per-timestep hr probes.
    HEADS = {
        "lp_pa2pa_bu": (LPPatchToPatchBU, {}),
        "lp_pa2px": (LPPatchToPixel, {}),
        "lp_pa2px_ens": (LPPatchToPixel, {"ensemble": True}),
        "lp_bu_px2px": (LPBilinearToPixel, {}),
        "anyup": (CachedAnyUp, {}),
        "anyup_t2": (CachedAnyUpT2, {}),
        "anyup_t1": (CachedAnyUpT1, {}),
        "anyup_t2_ens": (CachedAnyUpT2, {"ensemble": True}),
        "anyup_t1_ens": (CachedAnyUpT1, {"ensemble": True}),
    }
    if name == "manyup_shared":
        if not (manyup_ckpt and manyup_shared_probe):
            raise ValueError("head_mode=manyup_shared requires --manyup_ckpt and "
                             "--manyup_shared_probe (an lp_bu_px2px head)")
        return CachedManyUpSharedProbe(embed_dim, num_classes, patch_size,
                                       ckpt_path=manyup_ckpt, shared_probe=manyup_shared_probe,
                                       use_proj=manyup_use_proj, time_pool=time_pool,
                                       label_size=label_size, native_out=manyup_native_out,
                                       shared_probe_side=manyup_shared_probe_side,
                                       allow_mismatch=manyup_allow_probe_mismatch)
    if name in ("timanyup", "timanyup_t"):
        if not timanyup_ckpt:
            raise ValueError(f"head_mode={name} requires --timanyup_ckpt")
        return CachedTimAnyUp(embed_dim, num_classes, patch_size, ckpt_path=timanyup_ckpt,
                              pool_time="mean" if name == "timanyup" else "probe",
                              k=timanyup_k, label_size=label_size,
                              native_out=manyup_native_out)
    if name == "manyup":
        if not manyup_ckpt:
            raise ValueError("head_mode=manyup requires a checkpoint (--manyup discovery or --manyup_ckpt)")
        return CachedManyUp(embed_dim, num_classes, patch_size,
                            ckpt_path=manyup_ckpt, use_proj=manyup_use_proj,
                            time_pool=time_pool, label_size=label_size,
                            native_out=manyup_native_out)
    if name not in HEADS:
        raise ValueError(f"head_mode={name!r} not in {list(HEADS)} (cached-feature heads)")
    cls, kwargs = HEADS[name]
    return cls(embed_dim, num_classes, patch_size, label_size=label_size, **kwargs)


# ----------------------------- train / eval -----------------------------
def _run_head(head, feats, rgb, device):
    """Forward a batch through any cached head. rgb is an empty tensor for LP heads.

    feats is normally a tensor; TimAnyUpFeatureDataset yields a (F_lrhc, F_hrlc) pair instead,
    which default collate turns into a list of two tensors. Unpacking it here keeps every
    single-arm head's signature untouched."""
    rgb = None if rgb.numel() == 0 else rgb.to(device)
    if isinstance(feats, (list, tuple)):
        return head(tuple(f.to(device) for f in feats), rgb)
    return head(feats.to(device), rgb)


@torch.no_grad()
def evaluate(head, loader, device):
    head.eval()
    preds, labels = [], []
    for feats, label, rgb in loader:
        logits = _run_head(head, feats, rgb, device)
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(label)
    return segmentation_metrics(torch.cat(preds), torch.cat(labels),
                                num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL)


# ----------------------------- KNN eval -----------------------------
@torch.no_grad()
def _collect_pixels(head, loader, device, max_pixels=None, seed=0):
    """Run head.features() over a loader, flatten to per-pixel (N,D) features + (N,) labels,
    dropping ignore-label pixels. If max_pixels is set, randomly subsample to that many (the KNN
    reference set is bounded this way). L2-normalizes features so dot product == cosine sim."""
    feats_all, labels_all = [], []
    for feats, label, rgb in loader:
        rgb_d = None if rgb.numel() == 0 else rgb.to(device)
        f = head.features(feats.to(device), rgb_d)              # (B,D,H,W)
        B, D, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, D)                # (B*H*W, D)
        lab = label.reshape(-1)                                 # (B*H*W,)
        keep = lab != IGNORE_LABEL
        feats_all.append(F.normalize(f[keep].float(), dim=1).cpu())
        labels_all.append(lab[keep])
    X = torch.cat(feats_all); y = torch.cat(labels_all)
    if max_pixels is not None and X.shape[0] > max_pixels:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:max_pixels]
        X, y = X[idx], y[idx]
    return X, y


@torch.no_grad()
def knn_evaluate(head, train_loader, test_loader, device, k=20,
                 ref_pixels=2_000_000, query_chunk=8192, seed=0):
    """Non-parametric KNN segmentation on frozen head.features(). Builds an L2-normalized
    reference set from (subsampled) TRAIN pixels, then for each TEST pixel takes a majority vote
    over its k nearest reference features (cosine similarity). No training. Returns the same
    segmentation_metrics dict as evaluate() for apples-to-apples comparison with LP."""
    head.eval()
    print(f"KNN: building reference from train (<= {ref_pixels} pixels)...")
    Xr, yr = _collect_pixels(head, train_loader, device, max_pixels=ref_pixels, seed=seed)
    Xr = Xr.to(device); yr = yr.to(device)
    print(f"KNN: reference {Xr.shape[0]} pixels x {Xr.shape[1]}-d; k={k}. Scoring test...")

    preds, labels = [], []
    for feats, label, rgb in test_loader:
        rgb_d = None if rgb.numel() == 0 else rgb.to(device)
        f = head.features(feats.to(device), rgb_d)             # (B,D,H,W)
        B, D, H, W = f.shape
        q = F.normalize(f.permute(0, 2, 3, 1).reshape(-1, D).float(), dim=1)  # (Bq,D)
        out = torch.empty(q.shape[0], dtype=torch.long)
        # Chunk queries so the (chunk x ref) similarity matrix fits in VRAM.
        for s in range(0, q.shape[0], query_chunk):
            qc = q[s:s + query_chunk].to(device)               # (c,D)
            sim = qc @ Xr.T                                     # (c, Nref) cosine
            nn_idx = sim.topk(k, dim=1).indices                # (c, k)
            votes = yr[nn_idx]                                 # (c, k) labels
            # majority vote per row via bincount over class ids
            maj = torch.stack([torch.bincount(v, minlength=NUM_CLASSES).argmax() for v in votes])
            out[s:s + query_chunk] = maj.cpu()
        preds.append(out.reshape(B, H, W))
        labels.append(label)
    return segmentation_metrics(torch.cat(preds), torch.cat(labels),
                                num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL)


RESULT_COLUMNS = [
    "timestamp", "features", "head_mode", "eval_kind", "manyup_ckpt", "manyup_use_proj",
    "manyup_native_out",
    # timAnyUp provenance: WHICH upsampler, and the lookup budget k it was evaluated at.
    # k is the method's central knob -- without it every timanyup row in the CSV looks
    # identical, and a k-sweep would be unreadable.
    # hrlc_features is NOT redundant with `features`: that column records only F_lrhc (the
    # arm being upsampled). F_hrlc is a second, independent input, and it is shape-identical
    # to the target cache -- so which one a run used cannot be inferred from any other column.
    # query_input is the SELECTOR variant (bilinear = mask from cheap inputs only and so
    # computable in parallel; upsampled = mask reads mAnyUp's output). Two rows differing only
    # by it are different methods, not different seeds.
    "timanyup_ckpt", "timanyup_hrlc", "timanyup_k", "timanyup_query_input",
    # How the transform was handled: frozen (stale under a new backbone), refit jointly, or
    # refit in a first stage before the probe. Rows differing only by this are different
    # methods, so the CSV must say which.
    "timanyup_train_transform", "timanyup_transform_epochs",
    "time_pool", "epochs", "lr", "knn_k", "batch_size", "seed",
    "test_miou", "test_overall_acc", "avg_epoch_sec",
]


def append_result(csv_path: Path, row: dict) -> None:
    """Append one run's args + test metrics to csv_path, writing the header if the file is new.
    If an EXISTING file has an older header (fewer columns, e.g. from before manyup_* were added),
    honor that file's header so rows stay column-aligned; keys not in it are dropped and missing
    ones filled blank (extrasaction='ignore', restval='')."""
    fieldnames = RESULT_COLUMNS
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        header = rows[0] if rows else None
        if header:
            missing = [c for c in RESULT_COLUMNS if c not in header]
            if missing:
                # MIGRATE rather than silently dropping the new columns: appending them to the
                # header and blank-filling the existing rows keeps history intact while making
                # the new fields recordable. (Dropping them instead makes runs that differ only
                # by a new flag indistinguishable in the results file.)
                fieldnames = header + missing
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(fieldnames)
                    for r in rows[1:]:
                        w.writerow(r + [""] * len(missing))
                print(f"results csv: added column(s) {missing} to {csv_path}")
            else:
                fieldnames = header      # match the file already on disk
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="LP on cached OlmoEarth features.")
    p.add_argument("--features", required=True,
                   help="extraction config folder name under --out_root, e.g. oe_base_s2s1_ps4_tile64")
    p.add_argument("--out_root", default="~/projects/aip-gpleiss/timz/features")
    p.add_argument("--data_splits", default="data/pastis_olmoearth")
    p.add_argument("--head_mode", default="lp_pa2px",
                   choices=list(HEAD_GUIDANCE))
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for the DISK-FALLBACK path; ignored when a split "
                        "is preloaded into RAM (then workers=0, since indexing RAM is instant "
                        "and forking would copy the big tensor into every worker).")
    p.add_argument("--max_ram_gb", type=float, default=32.0,
                   help="per-split RAM budget for preloading features into memory; splits "
                        "estimated above this fall back to per-sample disk loading.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results_csv", default="results/pastis/lp_olmoearth_pastis.csv",
                   help="append run args + test metrics to this CSV (created with a header "
                        "if absent).")
    # --- mAnyUp options (head_mode=manyup) ---
    p.add_argument("--time_pool", default="mean", choices=list(TIME_POOLS),
                   help="how the S2 series is collapsed into the guidance image (mean|median). "
                        "Ignored by lp_* heads (no guidance) and by the per-timestep guidance "
                        "modes 'temporal'/'temporal13' (anyup_t*/timanyup*, which guide each "
                        "frame with its own image and so pool nothing -- the CSV records it "
                        "blank for those). For --manyup this MUST match the checkpoint's "
                        "train-time --time_pool, else the frozen upsampler sees an input "
                        "distribution it never trained on.")
    p.add_argument("--manyup", action="store_true",
                   help="head_mode=manyup + auto-discover all mAnyUp checkpoints trained to "
                        "upsample --features (the LR config) and LP over each (one CSV row per).")
    p.add_argument("--manyup_ckpt", default=None,
                   help="explicit mAnyUp checkpoint to LP (instead of auto-discovery)")
    # --- timAnyUp options (head_mode=timanyup / timanyup_t) ---
    p.add_argument("--timanyup_ckpt", default=None,
                   help="timAnyUp checkpoint to LP (from train_timanyup.py). --features is the "
                        "F_lrhc arm; --hrlc_features is the queried F_hrlc arm.")
    p.add_argument("--hrlc_features", default=None,
                   help="high-res LOW-context feature cfg for timAnyUp (e.g. "
                        "oe_base_s2_ps4_tile4_single). Defaults to the checkpoint's own "
                        "hrlc_cfg, which is also what it is checked against.")
    p.add_argument("--timanyup_train_transform", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="unfreeze the timAnyUp TRANSFORM head during LP. Needed whenever the "
                        "features come from a different backbone than the checkpoint was "
                        "trained on (e.g. fine-tuned caches): the transform aligns F_hrlc to "
                        "F_lrhc_up space, and that alignment does not survive an encoder "
                        "change. The upsampler and query head stay frozen.")
    p.add_argument("--timanyup_transform_epochs", type=int, default=0,
                   help="two-stage schedule: train ONLY the transform for this many epochs "
                        "(probe frozen), then only the probe. 0 = fit both jointly the whole "
                        "run. Staging stops the probe from chasing a still-moving transform.")
    p.add_argument("--timanyup_k", type=int, default=None,
                   help="lookup budget at eval; defaults to the checkpoint's trained k")
    p.add_argument("--manyup_root", default="checkpoints/manyup",
                   help="root scanned for <features>__to__*/*.pth mAnyUp checkpoints")
    p.add_argument("--manyup_native_out", action="store_true",
                   help="run the frozen mAnyUp at the grid it was TRAINED to produce (parsed "
                        "from the checkpoint's hr_cfg, e.g. 16x16 for a _to_..._ps4_ ckpt), then "
                        "reach label size with a pa2px probe -- Conv2d(D -> C*q^2) unfolded into "
                        "q x q sub-pixels per token (q = the target cfg's patch size). Keeps the "
                        "upsampler in-distribution AND matches lp_pa2px on the real hr_cfg "
                        "features; no bilinear. Default (off) asks mAnyUp for full label "
                        "resolution in one hop with a per-pixel 1x1 probe.")
    p.add_argument("--manyup_shared_probe", default=None,
                   help="head_mode=manyup_shared: an lp_bu_px2px head (--save_head) to reuse "
                        "FROZEN on the upsampled features, so the LR and mAnyUp routes share "
                        "one decoder and differ only in the feature map.")
    p.add_argument("--manyup_shared_probe_side", default="lr", choices=("lr", "hr"),
                   help="which features --manyup_shared_probe was fitted on. 'lr' (default) "
                        "is the bilinear-upsampled LR route and requires transform_depth=0; "
                        "'hr' is a probe trained on the REAL target-resolution features and is "
                        "valid for any upsampler architecture.")
    p.add_argument("--manyup_allow_probe_mismatch", action="store_true",
                   help="permit an LR-side shared probe on an upsampler whose output leaves the "
                        "LR space (transform_depth>0 / projector). Only for the deliberate "
                        "control measuring that mismatch -- not a fair score.")
    p.add_argument("--id_half", default="all", choices=("all", "first", "second"),
                   help="fit the probe on only one half of the TRAIN split, for pairing with a "
                        "train_manyup --id_half upsampler trained on the other half")
    p.add_argument("--save_head", nargs="?", const="checkpoints/lp_heads", default=None,
                   help="save the best-val trained head (probe weights only) to this path or "
                        "DIRECTORY, so viz/eval can reuse it instead of retraining a probe.")
    p.add_argument("--viz", nargs="?", const="results/pastis/viz", default=None,
                   help="after training, write a 2x4 figure comparing the LR / mAnyUp / HR "
                        "routes on one test sample: features (shared PCA) on top, segmentation "
                        "predictions below. Optional value is the output DIRECTORY.")
    p.add_argument("--viz_sample", type=int, default=0,
                   help="index of the TEST sample to visualize (--viz)")
    p.add_argument("--viz_ref_epochs", type=int, default=8,
                   help="epochs for the LR/HR reference pa2px probes drawn in the --viz figure")
    p.add_argument("--manyup_use_proj", action=argparse.BooleanOptionalAction, default=True,
                   help="include the trained projector in the frozen mAnyUp pipeline "
                        "(--no-manyup_use_proj to probe the raw upsampled ps4-space features)")
    # --- KNN eval (instead of LP): non-parametric, no training ---
    p.add_argument("--knn", action="store_true",
                   help="evaluate features by KNN vote (no probe training) instead of LP. Works "
                        "for lp_pa2pa_bu/lp_pa2px (raw features) and manyup (upsampled features).")
    p.add_argument("--knn_k", type=int, default=20, help="neighbors per KNN query")
    p.add_argument("--knn_ref_pixels", type=int, default=2_000_000,
                   help="max train pixels in the KNN reference set (subsampled)")
    args = p.parse_args()

    # --manyup / --manyup_ckpt implies head_mode=manyup (convenience so you don't pass both).
    if args.manyup or args.manyup_ckpt:
        # --manyup_shared_probe selects the shared-decoder variant of the same pipeline.
        args.head_mode = "manyup_shared" if args.manyup_shared_probe else "manyup"

    torch.manual_seed(args.seed)
    # AnyUp runs its (attention-heavy) upsample in fp32; TF32 lets the L40s tensor cores do
    # those matmuls ~2x faster at negligible precision cost. Frozen AnyUp + tiny probe means
    # the slight TF32 rounding is immaterial to results. Big win for the anyup* heads, which
    # call AnyUp T times per sample.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = Path(args.out_root).expanduser() / args.features   # expanduser: default uses ~
    data_splits = Path(args.data_splits)

    meta_path = feat_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing; run exp/pastis/extract_features.py first.")
    meta = json.loads(meta_path.read_text())
    embed_dim = meta["embed_dim"]
    patch_size = meta["patch_size"]
    guidance = HEAD_GUIDANCE[args.head_mode]
    print(f"Features: {args.features} | shape {meta['feature_shape']} | "
          f"patch_size {patch_size} | head {args.head_mode} | guidance {guidance}"
          + (f" | time_pool {args.time_pool}" if guidance != "none" else ""))

    # _load_rgb_guidance reads s2_images from finetune_olmoearth_pastis.DATA_SPLITS (a module
    # global). Point it at our data_splits so AnyUp guidance comes from the right place.
    if guidance != "none":
        from exp.pastis import finetune_olmoearth as fmod
        fmod.DATA_SPLITS = args.data_splits

    reduce_time = HEAD_REDUCES_TIME[args.head_mode]
    # Guidance modality must match what the mAnyUp checkpoint TRAINED on, else the frozen
    # guidance encoder gets an unseen band count/distribution. The ckpt records it (older
    # ones predate the flag and were all S2-guided).
    # Loaders are built once and shared across every mAnyUp ckpt in this invocation, so read
    # the modality from whichever ckpt is being run (explicit --manyup_ckpt, else the first
    # discovered one). Mixed-guidance discovery sets would need per-ckpt loaders; they do not
    # occur in practice (a discovery set is one LR cfg, hence one arm).
    guidance_mod = "s2"
    _mu_ckpt = args.manyup_ckpt
    if args.head_mode in ("manyup", "manyup_shared") and not _mu_ckpt:
        _found = _discover_manyup_ckpts(Path(args.manyup_root), args.features)
        _mu_ckpt = str(_found[0]) if _found else None
    if _mu_ckpt:
        _ck = torch.load(_mu_ckpt, map_location="cpu", weights_only=False)
        guidance_mod = _ck.get("args", {}).get("guidance_mod") or "s2"
        print(f"mAnyUp guidance modality (from ckpt): {guidance_mod}")
        del _ck

    # timAnyUp: resolve the SECOND feature arm and verify both arms match the checkpoint.
    # F_hrlc and F_hrhc are shape-identical (they differ only in extraction context), so a
    # wrong arm is invisible to any shape check -- the cfg names are the only guard.
    _hrlc_dir = None
    if args.head_mode in ("timanyup", "timanyup_t"):
        if not args.timanyup_ckpt:
            raise ValueError(f"--head_mode {args.head_mode} requires --timanyup_ckpt")
        _tck = torch.load(args.timanyup_ckpt, map_location="cpu", weights_only=False)
        _ta = _tck.get("args", {})
        guidance_mod = _ta.get("guidance_mod") or "s2"
        trained_lrhc, trained_hrlc = _ta.get("lrhc_cfg"), _ta.get("hrlc_cfg")
        # A FINE-TUNED cache is deliberately a different arm than the checkpoint trained on
        # (same config, different backbone weights: oe_..._ps16_tile64_ftp16ep64 vs
        # oe_..._ps16_tile64). That is the whole point of the transfer experiment, so allow it
        # when the base config matches after stripping the _ft<tag> suffix -- but ONLY then,
        # so a genuinely wrong arm is still caught.
        _strip_ft = lambda c: re.sub(r"_ft[A-Za-z0-9]+$", "", c or "")
        _ft_variant = (_strip_ft(args.features) == _strip_ft(trained_lrhc or "")
                       and args.features != trained_lrhc)
        if _ft_variant:
            print(f"NOTE: --features {args.features} is a FINE-TUNED variant of the "
                  f"checkpoint's {trained_lrhc}. The upsampler/query head see features from a "
                  f"backbone they were not trained on; pass --timanyup_train_transform to "
                  f"refit the cross-arm alignment.")
        if trained_lrhc and args.features != trained_lrhc and not _ft_variant:
            raise ValueError(
                f"--features {args.features!r} is not the arm this checkpoint upsamples "
                f"(trained on {trained_lrhc!r}). The frozen upsampler would see features "
                f"from a different extraction config.")
        hrlc_cfg = args.hrlc_features or trained_hrlc
        if not hrlc_cfg:
            raise ValueError("--hrlc_features is required (checkpoint records no hrlc_cfg)")
        if (trained_hrlc and hrlc_cfg != trained_hrlc
                and _strip_ft(hrlc_cfg) != _strip_ft(trained_hrlc)):
            raise ValueError(
                f"--hrlc_features {hrlc_cfg!r} != checkpoint's {trained_hrlc!r}. These arms are "
                f"shape-identical, so this mismatch cannot be caught later -- pass the trained "
                f"arm, or omit --hrlc_features to use it automatically.")
        _hrlc_dir = Path(args.out_root).expanduser() / hrlc_cfg
        # Stash the resolved arm so the results CSV records what was actually read.
        args._resolved_hrlc_cfg = hrlc_cfg
        # Read from the checkpoint (top-level key on new ones, args on older ones) rather
        # than from a flag: the LP has no say in it, the trained model does.
        args._resolved_query_input = _tck.get("query_input") or _ta.get("query_input", "bilinear")
        print(f"timAnyUp: F_lrhc={args.features}  F_hrlc={hrlc_cfg}  "
              f"guidance={guidance_mod}  k={args.timanyup_k or _ta.get('k')}")
        del _tck

    def loader(split, shuffle):
        common = dict(guidance=guidance, max_ram_gb=args.max_ram_gb, reduce_time=reduce_time,
                      time_pool=args.time_pool, guidance_mod=guidance_mod,
                      id_half=args.id_half)
        if args.head_mode in ("timanyup", "timanyup_t"):
            ds = TimAnyUpFeatureDataset(feat_dir, _hrlc_dir, data_splits, split, **common)
        else:
            ds = CachedFeatureDataset(feat_dir, data_splits, split, **common)
        preloaded = ds._feats is not None
        # Preloaded: index RAM in-process (workers would duplicate the tensor). Disk fallback:
        # use workers + pin_memory + persistent_workers to overlap reads with GPU compute.
        workers = 0 if preloaded else args.num_workers
        use_cuda = device.type == "cuda"
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=workers,
            pin_memory=use_cuda,
            persistent_workers=workers > 0,
        )

    train_loader = loader("train", True)
    val_loader = loader("valid", False)
    test_loader = loader("test", False)

    # --- mAnyUp: discover the checkpoints to LP over. --manyup scans for models trained to
    # upsample THIS --features (the LR config): checkpoints/<features>__to__*/*.pth, latest epoch
    # per LR->HR pair. Each becomes its own LP run + CSV row. --manyup_ckpt runs a single explicit
    # one. For non-manyup heads this is a single [None] -> one normal run.
    if args.head_mode in ("manyup", "manyup_shared"):
        if args.manyup_ckpt:
            ckpts = [Path(args.manyup_ckpt)]
        else:
            ckpts = _discover_manyup_ckpts(Path(args.manyup_root), args.features)
            if not ckpts:
                raise FileNotFoundError(
                    f"no mAnyUp checkpoints for LR={args.features} under {args.manyup_root} "
                    f"(expected {args.features}__to__*/*.pth). Train one with train_manyup.sh.")
            print(f"mAnyUp: {len(ckpts)} checkpoint(s) to LP over:")
            for c in ckpts:
                print(f"  {c}")
    else:
        ckpts = [None]

    for ckpt in ckpts:
        run_one(args, device, embed_dim, patch_size, train_loader, val_loader, test_loader,
                manyup_ckpt=str(ckpt) if ckpt is not None else None)


def _discover_manyup_ckpts(manyup_root: Path, lr_features: str) -> list:
    """Find the latest-epoch checkpoint for each mAnyUp model that upsamples `lr_features`.
    Layout (from train_manyup.sh): <manyup_root>/<lr>__to__<hr>/manyup_..._ep<N>.pth."""
    pairs = sorted(manyup_root.glob(f"{lr_features}__to__*"))
    latest = []
    for d in pairs:
        cks = list(d.glob("*.pth"))
        if not cks:
            continue
        # pick highest ep<N> (fall back to mtime if names don't parse)
        def epoch_of(p: Path) -> int:
            stem = p.stem
            return int(stem.split("_ep")[-1]) if "_ep" in stem else -1
        latest.append(max(cks, key=lambda p: (epoch_of(p), p.stat().st_mtime)))
    return latest


def run_one(args, device, embed_dim, patch_size, train_loader, val_loader, test_loader,
            manyup_ckpt=None) -> None:
    """One LP training run (build head -> train -> eval -> log). Loaders are shared across
    mAnyUp checkpoints (same LR features + guidance), so only the head differs per run."""
    tag = f"{'KNN' if args.knn else 'LP'} run"
    proj_used = None                      # resolved below only for mAnyUp runs
    if manyup_ckpt:
        # A checkpoint trained with --no-proj_head has no projector to use, so the FLAG alone
        # mislabels the run. Report the conjunction -- what the head actually applies.
        _ckp = torch.load(manyup_ckpt, map_location="cpu", weights_only=False)
        proj_used = bool(args.manyup_use_proj and _ckp.get("proj_head") is not None)
        del _ckp
        print(f"\n===== mAnyUp {tag}: {manyup_ckpt} (proj={proj_used}) =====")
    # Label size from the actual targets (64 for the default prep, 128 for an --image_size 128
    # one) so the heads upsample their logits to whatever this dataset really uses.
    label_size = getattr(train_loader.dataset, "label_size", LABEL_SIZE)
    head = build_cached_head(args.head_mode, embed_dim, NUM_CLASSES, patch_size,
                             manyup_ckpt=manyup_ckpt,
                             manyup_use_proj=args.manyup_use_proj,
                             manyup_native_out=args.manyup_native_out,
                             manyup_shared_probe=args.manyup_shared_probe,
                             manyup_shared_probe_side=args.manyup_shared_probe_side,
                             manyup_allow_probe_mismatch=args.manyup_allow_probe_mismatch,
                             timanyup_ckpt=args.timanyup_ckpt, timanyup_k=args.timanyup_k,
                             time_pool=args.time_pool,
                             label_size=label_size).to(device)
    if getattr(head, "native_size", None):
        q = head.sub_patch
        print(f"mAnyUp native_out: upsampling to {head.native_size}x{head.native_size} "
              f"(its trained target grid), then pa2px probe D->C*{q}^2 unfolded "
              f"{q}x{q} per token -> {head.native_size * q}x{head.native_size * q}")

    # KNN: non-parametric, no training. Extract frozen features, vote over train neighbors, log.
    if args.knn:
        if not hasattr(head, "features"):
            raise ValueError(f"head_mode={args.head_mode!r} has no features() for KNN "
                             f"(supported: lp_pa2pa_bu, lp_pa2px, anyup*, manyup)")
        t0 = time.perf_counter()
        test = knn_evaluate(head, train_loader, test_loader, device,
                            k=args.knn_k, ref_pixels=args.knn_ref_pixels, seed=args.seed)
        elapsed = time.perf_counter() - t0
        print(f"KNN TEST {test.metrics}  ({elapsed:.1f}s)")
        _log_result(args, manyup_ckpt, test, avg_epoch_time=elapsed, eval_kind="knn",
                    proj_used=proj_used, timanyup_k=getattr(head, "k", None))
        return

    # AnyUp heads lazily create their real probe (Conv2d(embed_dim, C)) on the FIRST forward
    # (AnyUpUpsampleProbe starts with a 1x1x1 placeholder). Run a no-grad dry pass now so the
    # real probe exists before AdamW captures head.parameters() -- otherwise the only trainable
    # module is never optimized and the AnyUp heads don't learn. Mirrors the live finetune path
    # (exp/pastis/finetune_olmoearth.py dry pass before the optimizer). LP heads build their probe
    # in __init__, so this pass is a harmless no-op for them.
    with torch.no_grad():
        feats0, _label0, rgb0 = next(iter(train_loader))
        _run_head(head, feats0, rgb0, device)
    head = head.to(device)  # re-move in case _init_probe created the probe on a fresh device

    # manyup_shared trains NOTHING (frozen upsampler + frozen borrowed probe): there is no
    # probe to fit, so skip straight to the test evaluation instead of running empty epochs
    # through an optimizer with no parameters.
    if not any(prm.requires_grad for prm in head.parameters()):
        test = evaluate(head, test_loader, device)
        print(f"\n(no trainable parameters -- evaluation only)")
        print(f"TEST {test.metrics}")
        _log_result(args, manyup_ckpt, test, avg_epoch_time=float("nan"), eval_kind="lp",
                    proj_used=proj_used, timanyup_k=getattr(head, "k", None))
        if args.viz:
            _write_viz(args, head, manyup_ckpt, train_loader, test_loader, device, patch_size)
        return

    # timAnyUp on FINE-TUNED features: the frozen transform head was fitted to map F_hrlc
    # into F_lrhc_up space for ONE backbone. Under a different (fine-tuned) encoder both
    # endpoints move, so the stored mapping is stale and must be refitted. The upsampler and
    # query head are left frozen -- they consume guidance and the LR map, not F_hrlc, so they
    # transfer; only the cross-arm alignment does not.
    tf_params = []
    if args.timanyup_train_transform and hasattr(head, "model"):
        for prm in head.model.transform_head.parameters():
            prm.requires_grad = True
        tf_params = list(head.model.transform_head.parameters())
        print(f"timAnyUp: refitting transform head ({sum(p.numel() for p in tf_params):,} "
              f"params) for {args.timanyup_transform_epochs} epoch(s) "
              f"{'BEFORE the probe (two-stage)' if args.timanyup_transform_epochs > 0 else 'jointly with the probe'}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=SCHEDULER_MIN_LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_val_miou = float("-inf")

    # Two-stage schedule: for the first N epochs train ONLY the transform (probe frozen), then
    # only the probe (transform frozen). Fitting them jointly from scratch lets the probe chase
    # a transform that is still moving; staging gives the probe a settled feature space. N=0
    # means the usual single-stage joint fit.
    probe_params = [prm for n, prm in head.named_parameters()
                    if prm.requires_grad and not n.startswith("model.transform_head")]

    def _set_stage(ep):
        if not tf_params or args.timanyup_transform_epochs <= 0:
            return
        transform_stage = ep < args.timanyup_transform_epochs
        for prm in tf_params:
            prm.requires_grad = transform_stage
        for prm in probe_params:
            prm.requires_grad = not transform_stage

    epoch_times = []   # wall time (train + val) per epoch, for the average below
    for epoch in range(args.epochs):
        _set_stage(epoch)
        epoch_start = time.perf_counter()
        head.train()
        last_loss = float("nan")
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for feats, label, rgb in pbar:
            logits = _run_head(head, feats, rgb, device)
            loss = loss_fn(logits, label.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            pbar.set_postfix(loss=f"{last_loss:.4f}")

        val = evaluate(head, val_loader, device)
        scheduler.step()
        epoch_times.append(time.perf_counter() - epoch_start)
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {last_loss:.4f} | "
              f"val miou {val.primary:.4f} | {epoch_times[-1]:.1f}s | {val.metrics}")
        if val.primary > best_val_miou:
            best_val_miou = val.primary
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else float("nan")
    head.load_state_dict(best_state)
    test = evaluate(head, test_loader, device)
    print(f"\nBEST val miou {best_val_miou:.4f}")
    print(f"TEST {test.metrics}")
    print(f"Avg epoch time: {avg_epoch_time:.1f}s over {len(epoch_times)} epochs")

    _log_result(args, manyup_ckpt, test, avg_epoch_time, eval_kind="lp",
                proj_used=proj_used, timanyup_k=getattr(head, "k", None))

    if args.save_head:
        # Persist the BEST-val head so downstream tools (viz, further eval) read the probe this
        # run actually scored, instead of retraining their own and reporting a different number.
        # Only trainable params are saved: the frozen upsampler already lives in manyup_ckpt and
        # AnyUp's weights come from torch.hub, so re-saving them would bloat the file for nothing.
        out = Path(args.save_head)
        if out.is_dir() or not out.suffix:
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(manyup_ckpt).stem if manyup_ckpt else f"{args.features}_{args.head_mode}"
            out = out / f"lphead_{stem}.pth"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        trainable = {k for k, prm in head.named_parameters() if prm.requires_grad}
        torch.save({"head_state": {k: v for k, v in best_state.items()
                                   if k in trainable or k.rsplit(".", 1)[0] + ".weight" in trainable},
                    "head_mode": args.head_mode, "features": args.features,
                    "patch_size": patch_size, "embed_dim": embed_dim,
                    "label_size": test_loader.dataset.label_size,
                    "manyup_ckpt": manyup_ckpt, "manyup_native_out": args.manyup_native_out,
                    "manyup_use_proj": proj_used, "time_pool": args.time_pool,
                    "epochs": args.epochs, "best_val_miou": best_val_miou,
                    "test_miou": test.primary}, out)
        print(f"saved LP head -> {out}")

    if args.viz:
        _write_viz(args, head, manyup_ckpt, train_loader, test_loader, device, patch_size)


def _write_viz(args, head, manyup_ckpt, train_loader, test_loader, device,
               lr_patch: int) -> None:
    """Render the LR / mAnyUp / HR comparison figure for one test sample.

    The HR ("oracle") column needs the real fine-patch features, which this run never loads --
    they are a DIFFERENT cached config, named by the checkpoint's own hr_cfg. Build them here
    when they exist on disk; without them the figure degrades to LR vs mAnyUp rather than
    failing, since the oracle is a nice-to-have and the run has already produced its number.
    """
    from exp.pastis.viz_manyup_lp import save_lp_viz

    out_dir = Path(args.viz)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(manyup_ckpt).stem if manyup_ckpt else f"{args.features}_{args.head_mode}"
    out_path = out_dir / f"lpviz_{tag}_s{args.viz_sample}.png"

    hr_ds = hr_train_ds = None
    hr_patch = 4
    if manyup_ckpt:
        ck = torch.load(manyup_ckpt, map_location="cpu", weights_only=False)
        hr_cfg = ck.get("args", {}).get("hr_cfg", "")
        m = re.search(r"_ps(\d+)_", hr_cfg)
        hr_dir = Path(args.out_root) / hr_cfg if hr_cfg else None
        if m and hr_dir and hr_dir.exists():
            hr_patch = int(m.group(1))
            mk = lambda sp: CachedFeatureDataset(          # noqa: E731
                hr_dir, Path(args.data_splits), sp, guidance="none",
                max_ram_gb=args.max_ram_gb, reduce_time=True, time_pool=args.time_pool)
            hr_ds, hr_train_ds = mk("test"), mk("train")
        else:
            print(f"viz: no HR features at {hr_dir} -- drawing LR vs mAnyUp only")

    save_lp_viz(head, test_loader.dataset, hr_ds, device, out_path,
                sample_idx=args.viz_sample, run_tag=tag,
                num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL,
                label_size=test_loader.dataset.label_size,
                hr_patch=hr_patch, lr_patch=lr_patch,
                ref_epochs=args.viz_ref_epochs,
                train_ds=train_loader.dataset, hr_train_ds=hr_train_ds)


def _log_result(args, manyup_ckpt, test, avg_epoch_time, eval_kind="lp",
                proj_used=None, timanyup_k=None) -> None:
    """Append one run's test metrics + provenance to the results CSV. Shared by the LP and KNN
    paths. eval_kind distinguishes them; epochs/lr are blanked for KNN (not applicable)."""
    # Keep the CSV readable: ints/strings as-is, metrics+time as 2-decimal floats. lr is the
    # one value .2f would mangle (1e-3 -> 0.00), so log it with %g (compact, full precision).
    append_result(Path(args.results_csv), {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "features": args.features,
        "head_mode": args.head_mode,
        "eval_kind": eval_kind,
        # mAnyUp provenance: which upsampler checkpoint (+ whether its projector was used) so
        # looped runs are distinguishable in the CSV. Empty for non-manyup heads.
        "manyup_ckpt": Path(manyup_ckpt).name if manyup_ckpt else "",
        # What the head ACTUALLY applied: a --no-proj_head checkpoint has no projector, so the
        # flag alone would record proj=True for a run that used none. Falls back to the flag
        # when the caller did not resolve it.
        "manyup_use_proj": ((args.manyup_use_proj if proj_used is None else proj_used)
                            if manyup_ckpt else ""),
        "manyup_native_out": (args.manyup_native_out if manyup_ckpt else ""),
        # Blank for non-timAnyUp heads. k is the budget the head ACTUALLY used (the head
        # falls back to the checkpoint's trained k when --timanyup_k is unset), not the raw
        # flag -- recording the flag would leave the default runs blank.
        "timanyup_ckpt": (Path(args.timanyup_ckpt).name if args.timanyup_ckpt else ""),
        # The RESOLVED arm, not the raw --hrlc_features flag: it defaults to the checkpoint's
        # own hrlc_cfg when the flag is omitted, which is the common case, so logging the flag
        # would leave it blank for exactly the runs that matter.
        "timanyup_hrlc": getattr(args, "_resolved_hrlc_cfg", ""),
        "timanyup_k": (timanyup_k if (args.timanyup_ckpt and timanyup_k is not None) else ""),
        "timanyup_query_input": getattr(args, "_resolved_query_input", ""),
        "timanyup_train_transform": (args.timanyup_train_transform if args.timanyup_ckpt else ""),
        "timanyup_transform_epochs": (args.timanyup_transform_epochs if args.timanyup_ckpt else ""),
        # Blank whenever the head never pools time: guidance="none" (lp_*) builds no guidance
        # image at all, and the "temporal"/"temporal13" modes keep EVERY frame (each timestep
        # is guided by its own image), so _load_guidance returns before time_pool is read.
        # Recording argparse's default there would claim a pooling that never happened -- and
        # since time_pool IS identity-bearing for mAnyUp (it is in the checkpoint name and
        # enforced at load), a reader would reasonably believe it.
        "time_pool": ("" if HEAD_GUIDANCE[args.head_mode] in ("none", "temporal", "temporal13")
                      else args.time_pool),
        "epochs": ("" if eval_kind == "knn" else int(args.epochs)),
        "lr": ("" if eval_kind == "knn" else f"{args.lr:g}"),
        "knn_k": (args.knn_k if eval_kind == "knn" else ""),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "test_miou": f"{test.metrics['miou']:.2f}",
        "test_overall_acc": f"{test.metrics['overall_acc']:.2f}",
        "avg_epoch_sec": f"{avg_epoch_time:.2f}",
    })
    print(f"Appended {eval_kind} result to {args.results_csv}")


if __name__ == "__main__":
    main()
