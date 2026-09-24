"""Train mAnyUp: a guided feature upsampler for OlmoEarth (multimodal guidance).

Adapts wimmerth/anyup to our remote-sensing setting. AnyUp upsamples a low-res feature map to
a high-res one, guided by a co-located RGB image. We keep AnyUp's model + Cosine_MSE loss, but
replace its self-supervised crop trick with REAL low-res / high-res feature pairs from our cache:

  input  (LR feats): oe_base_s2_ps4_tile64   -> (T, 16, 16, 768)   cheap, coarse
  target (HR feats): oe_base_s2_ps1_tile1    -> (T, 64, 64, 768)   expensive, per-pixel  (swappable)
  guidance:          full Sentinel-2 image   -> (13, 64, 64)       high-res, multi-band

For v1 we mean-pool over T on both features and guidance -> a single (D, gH, gW) map each, matching
the plain 'anyup' head in lp_on_cached_features. The only architectural change vs AnyUp is the
guidance input channel count: 13 (S2 bands) instead of 3 (RGB). Trained from scratch (13!=3 makes
the pretrained RGB weights unusable anyway).

Losses (subset of AnyUp's three):
  anyup_hr   -- upsample LR -> compare to HR target. The objective.
  anyup_down -- area-downsample the prediction back to the LR grid, match the LR input feats.
  (anyup_reg is intentionally omitted for now.)

    source env_setup/env_olmo.sh
    python train_manyup.py --help
    # typical (from a GPU node):
    python train_manyup.py --epochs 20 --batch_size 8 --stage_to_tmpdir

The AnyUp repo is imported from its clone; set --anyup_repo if it moved.
"""
import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")        # headless cluster
import matplotlib.pyplot as plt

# Guidance time-pooling is shared with the UPA/UPMA/AnyUp paths (single source of truth).
from exp.upsamplers.upa_anyup import time_pool, TIME_POOLS
# Plumbing shared with train_timanyup.py -- guidance normalization and the LR schedule in
# particular MUST stay identical across trainers (see exp/upsamplers/common.py).
from exp.upsamplers.common import (
    DATA_ROOT, FEATURES_ROOT, GUIDANCE_BANDS, GUIDANCE_DIRS, guidance_mod_for,
    cfg_bits as _cfg_bits_shared, norm_guidance as _norm_guidance,
    stage_to_tmpdir, warmup_cosine as _warmup_cosine,
    pca_rgb_shared as _pca_rgb_shared, raw_rgb as _raw_rgb,
)
from exp.common.paths import ANYUP_REPO

# ----- import stock AnyUp from the vendored repo; mAnyUp + loss live in ./manyup -----
DEFAULT_ANYUP_REPO = str(ANYUP_REPO)


def _import_anyup(repo: str, arch: str = "anyup"):
    """Return (model_cls, Cosine_MSE) for the requested architecture.

    "anyup"  -- stock AnyUp: output is attn @ v, a convex combination of the LR feature
                vectors. A LINEAR probe on that is a linear function of the LR features, so it
                cannot beat a linear probe on the LR features directly (measured: 0.373 vs
                0.375 raw ps16).
    "manyup" -- adds a nonlinear ResBlock AFTER the cross-attention, so the upsampled map is no
                longer a linear function of the LR features and a linear probe can, in
                principle, read something new out of it."""
    sys.path.insert(0, repo)
    from manyup.loss import Cosine_MSE      # noqa: E402
    if arch == "manyup":
        from manyup.mAnyUp import mAnyUp    # noqa: E402
        return mAnyUp, Cosine_MSE
    from anyup.model import AnyUp          # noqa: E402
    return AnyUp, Cosine_MSE


# --------------------------------------------------------------------------------------------- #
# Dataset: pair (LR feats, HR feats, S2 guidance) per sample, skipping any index missing in any
# of the three sources (robust to partially-extracted feature dirs and to swapping the HR target).
# --------------------------------------------------------------------------------------------- #
class PairedFeatureDataset(Dataset):
    def __init__(self, lr_dir: Path, hr_dir: Path, s2_dir: Path, split: str,
                 time_pool: str = "mean", guidance_mod: str = "s2", half: str = "all"):
        self.lr_dir = lr_dir / f"pastis_r_{split}"
        self.hr_dir = hr_dir / f"pastis_r_{split}"
        # Guidance imagery. Which modality guides the upsampling is a CHOICE, not a
        # constant: guiding S1 features with S2 imagery is a cross-modal mismatch, so this
        # defaults to the arm the features come from (see --guidance_mod).
        self.guidance_mod = guidance_mod
        self.guide_dirs = [s2_dir / f"pastis_r_{split}" / f"{m}_images"
                           for m in GUIDANCE_DIRS[guidance_mod]]
        self.s2_dir = self.guide_dirs[0]   # index/count checks use the first
        # Guidance pooling is baked into the learned weights: the guidance encoder adapts to
        # whatever composite it sees here, so a checkpoint trained with mean guidance must be
        # evaluated with mean guidance. lp_on_cached_features._load_s2_guidance mirrors this,
        # and the ckpt name records it so the two cannot silently drift apart.
        self.time_pool = time_pool

        def indices(d: Path):
            return {int(p.stem) for p in d.glob("*.pt")}

        # Train only on indices present in ALL three sources.
        common = indices(self.lr_dir) & indices(self.hr_dir) & indices(self.s2_dir)
        self.ids = sorted(common)
        # Optional disjoint half. The upsampler and the LP probe that reads its output are
        # otherwise fitted on the SAME samples, so probe training sees features the upsampler
        # has already fit -- a leak that flatters mAnyUp relative to a baseline probe trained
        # on raw features. half="first"/"second" splits the ids so the two stages can be
        # trained on disjoint data (the probe takes the other half via --id_half).
        if half in ("first", "second"):
            mid = len(self.ids) // 2
            self.ids = self.ids[:mid] if half == "first" else self.ids[mid:]
            print(f"[{split}] id_half={half}: {len(self.ids)} of {len(common)} samples")
        if not self.ids:
            raise RuntimeError(f"no common {split} samples across\n  {self.lr_dir}\n  {self.hr_dir}"
                               f"\n  {self.s2_dir}")
        n_lr, n_hr, n_s2 = len(indices(self.lr_dir)), len(indices(self.hr_dir)), len(indices(self.s2_dir))
        print(f"[{split}] LR={n_lr} HR={n_hr} S2={n_s2} -> {len(self.ids)} common samples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        # features: (T, gH, gW, D) fp16 -> mean over T -> (D, gH, gW) fp32
        lr = torch.load(self.lr_dir / f"{idx}.pt").float().mean(0).permute(2, 0, 1)   # (D, gh, gw)
        hr = torch.load(self.hr_dir / f"{idx}.pt").float().mean(0).permute(2, 0, 1)   # (D, GH, GW)
        # guidance: each (T, C, 64, 64) fp32 -> pooled over T -> (C, 64, 64), normalized, then
        # concatenated on the band axis (s2 13 + s1 2 = 15 for the s2s1 arm). Normalizing BEFORE
        # the concat keeps each modality on its own [0,1] scale -- S1 dB and S2 reflectance have
        # very different dynamic ranges and a joint min-max would swamp one of them.
        s2 = torch.cat([_norm_guidance(time_pool(torch.load(d / f"{idx}.pt").float(),
                                                 self.time_pool))
                        for d in self.guide_dirs], dim=0)
        return lr, hr, s2


# --------------------------------------------------------------------------------------------- #
@torch.no_grad()
def save_epoch_viz(model, proj_head, sample, GH, GW, device, out_path, epoch,
                   run_tag: str = "") -> None:
    """4-panel viz of one held-out TEST sample: raw RGB | LR feats | mAnyUp output | HR target.
    Feature panels share one PCA basis (fit on HR) so they're directly comparable. mAnyUp panel
    is the projected output (proj_head applied) -- i.e. what the HR loss actually sees."""
    lr, hr, s2 = sample
    lr_b = lr.unsqueeze(0).to(device); s2_b = s2.unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        pred = model(s2_b, lr_b, (GH, GW))
        if proj_head is not None:
            pred = proj_head(pred)
    if was_training:
        model.train()
    pred = pred.squeeze(0).float()                           # (D,GH,GW)

    # Shared PCA basis fit on the HR target; lr upsampled (nearest) only for display sizing.
    lr_disp = F.interpolate(lr.unsqueeze(0).float(), size=(GH, GW), mode="nearest").squeeze(0)
    lr_rgb, mu_rgb, hr_rgb = _pca_rgb_shared(hr, [lr_disp, pred.cpu(), hr])

    fig, axes = plt.subplots(1, 4, figsize=(4 * 2.6, 2.8))
    panels = [_raw_rgb(s2), lr_rgb, mu_rgb, hr_rgb]
    titles = ["raw S2 RGB", f"LR feats ({lr.shape[-2]}x{lr.shape[-1]})",
              "mAnyUp (proj)", f"HR target ({GH}x{GW})"]
    for ax, p, t in zip(axes, panels, titles):
        ax.imshow(p); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    head = f"{run_tag}  --  " if run_tag else ""
    fig.suptitle(f"{head}epoch {epoch} -- test sample (shared PCA on HR)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved viz {out_path}")


@torch.no_grad()
def _bilinear_baseline(loader, criterion, GH, GW, device, n_batches=20) -> float:
    """Mean HR Cosine_MSE loss when the LR feats are upsampled by plain bilinear interpolation
    (no guidance, no model). This is the bar the learned upsampler must clear. Averaged over the
    first n_batches for a stable, cheap estimate."""
    tot, cnt = 0.0, 0
    for bi, (lr, hr, _s2) in enumerate(loader):
        lr, hr = lr.to(device), hr.to(device)
        up = F.interpolate(lr.float(), size=(GH, GW), mode="bilinear", align_corners=False)
        tot += criterion(up, hr)["total"].item()
        cnt += 1
        if bi + 1 >= n_batches:
            break
    return tot / max(cnt, 1)


# --------------------------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr_cfg", default="oe_base_s2_ps4_tile64",
                   help="low-res INPUT feature dir (cheap, coarse)")
    p.add_argument("--hr_cfg", default="oe_base_s2_ps1_tile64",
                   help="high-res TARGET feature dir (swap to oe_base_s2_ps1_tile64 when extracted)")
    p.add_argument("--split", default="train")
    p.add_argument("--id_half", default="all", choices=("all", "first", "second"),
                   help="train on only one half of the split's ids, so the upsampler and the "
                        "LP probe that consumes it can use disjoint samples")
    p.add_argument("--features_root", default=str(FEATURES_ROOT))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--anyup_repo", default=DEFAULT_ANYUP_REPO)
    p.add_argument("--arch", default="anyup", choices=("anyup", "manyup"),
                   help="anyup = stock (output is linear in the LR features); manyup = adds a "
                        "nonlinear ResBlock after the cross-attention so a linear probe can "
                        "extract information the LR features do not already carry linearly.")
    p.add_argument("--feat_dim", type=int, default=768,
                   help="encoder feature dim the manyup transform operates on (base = 768). "
                        "Ignored by --arch anyup.")
    p.add_argument("--transform_depth", type=int, default=0,
                   help="number of resblocks for feature transform after anyup")
    p.add_argument("--window_ratio",type=float,default=1.0)
    p.add_argument("--guidance_mod", default=None, choices=list(GUIDANCE_BANDS),
                   help="which imagery guides the upsampling (s2 = 13-band L2A, s1 = 2-band "
                        "VV/VH). Default: match the feature arm, so s1 features are guided by "
                        "S1 imagery instead of cross-modally by S2.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AnyUp uses 2e-4, but our loss surface is flatter (LR/HR feats start "
                        "already aligned), so a higher LR converges faster")
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="floor the cosine decay reaches at the end of training")
    p.add_argument("--warmup_frac", type=float, default=0.05,
                   help="fraction of total steps for linear LR warmup before cosine decay")
    p.add_argument("--baseline_batches", type=int, default=20,
                   help="batches to average the bilinear baseline over")
    p.add_argument("--qk_dim", type=int, default=128)
    p.add_argument("--down_reg", type=float, default=0.1,     # AnyUp downsampling_regularization default
                   help="weight of anyup_down loss (0 to disable)")
    p.add_argument("--proj_head", action=argparse.BooleanOptionalAction, default=False,
                   help="learned 1x1 conv projecting upsampled ps4-space features into the HR "
                        "target's space before anyup_hr; --no-proj_head to A/B against stock AnyUp")
    p.add_argument("--linear_baseline", action=argparse.BooleanOptionalAction, default=True,
                   help="co-train a bilinear+linear-head baseline to attribute mAnyUp's gain to "
                        "guidance-driven upsampling vs. a plain linear ps4->ps1 map")
    p.add_argument("--stage_to_tmpdir", action="store_true",
                   help="copy feature/S2 dirs to $SLURM_TMPDIR for faster reads")
    p.add_argument("--time_pool", default="mean", choices=list(TIME_POOLS),
                   help="how the S2 series is collapsed into the guidance composite. This is "
                        "baked into the trained weights -- a checkpoint must be EVALUATED with "
                        "the same setting (exp/pastis/lp_cached_features.py --time_pool), so it is "
                        "recorded in the checkpoint filename and enforced at load time.")
    p.add_argument("--out_dir", default=None,
                   help="default: checkpoints/manyup/<lr_cfg>__to__<hr_cfg>, so parallel runs "
                        "of different pairs never share a dir")
    p.add_argument("--ckpt_every", type=int, default=5, help="save every N epochs")
    p.add_argument("--sanity", action="store_true", help="one batch then exit")
    args = p.parse_args()

    AnyUp, Cosine_MSE = _import_anyup(args.anyup_repo, args.arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.out_dir is None:
        args.out_dir = f"checkpoints/manyup/{args.lr_cfg}__to__{args.hr_cfg}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    froot = Path(args.features_root)
    lr_dir, hr_dir = froot / args.lr_cfg, froot / args.hr_cfg
    data_root = Path(args.data_root)

    # Optionally stage the big feature dirs + S2 images to node-local disk.
    if args.stage_to_tmpdir:
        lr_dir, hr_dir = stage_to_tmpdir([lr_dir, hr_dir])
        # S2 images live under data_root/pastis_r_<split>/s2_images; stage the split dir.
        s2_split = data_root / f"pastis_r_{args.split}"
        (s2_split_staged,) = stage_to_tmpdir([s2_split])
        data_root = s2_split_staged.parent   # so PairedFeatureDataset finds pastis_r_<split>/s2_images

    if args.guidance_mod is None:
        args.guidance_mod = guidance_mod_for(args.lr_cfg)
    guide_bands = GUIDANCE_BANDS[args.guidance_mod]
    print(f"guidance: {args.guidance_mod} ({guide_bands}-band) from {data_root}")

    ds = PairedFeatureDataset(lr_dir, hr_dir, data_root, args.split, half=args.id_half,
                              time_pool=args.time_pool, guidance_mod=args.guidance_mod)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Held-out TEST sample for the per-epoch viz. Feature dirs (possibly staged) contain the test
    # split too; S2 test images come from the ORIGINAL data_root (only train S2 was staged).
    viz_sample = None
    try:
        # guidance_mod MUST match the train loader: the model's image encoder is built for that
        # band count, so an s2-guided viz sample crashes an s1-guided model (13 vs 2 channels).
        test_ds = PairedFeatureDataset(lr_dir, hr_dir, Path(args.data_root), "test",
                                       time_pool=args.time_pool,
                                       guidance_mod=args.guidance_mod)
        viz_sample = test_ds[0]   # first common test sample: (lr, hr, s2)
    except RuntimeError as e:
        print(f"viz disabled: no usable test sample ({e})")

    # peek one sample for shapes
    lr0, hr0, _ = ds[0]
    D, gh, gw = lr0.shape
    _, GH, GW = hr0.shape
    print(f"LR feats {tuple(lr0.shape)}  ->  HR target {tuple(hr0.shape)}  (upsample {gh}x{gw} -> {GH}x{GW})")
    assert lr0.shape[0] == hr0.shape[0], "LR and HR feature dims (D) must match to share the upsampler"

    # Identifier for this run's artefacts. Carries the modality arm and BOTH grids, e.g.
    #   s2_ps16_4x4_to_ps4_16x16            (mean guidance pooling, the default)
    #   s2s1_ps16_4x4_to_ps4_16x16_median
    # The cfg names are oe_<size>_<mods>_ps<N>_tile<M>[_img<K>], so modality and patch size
    # are parsed straight out of them rather than passed in again.
    lr_mods, lr_ps = _cfg_bits_shared(args.lr_cfg)
    hr_mods, hr_ps = _cfg_bits_shared(args.hr_cfg)
    # Arms should match; if someone crosses them, name both so the mismatch is visible.
    mods = lr_mods if lr_mods == hr_mods else f"{lr_mods}to{hr_mods}"
    tp_tag = "" if args.time_pool == "mean" else f"_{args.time_pool}"
    # Tag the guidance only when it is NOT the arm's default, so existing names are unchanged.
    g_tag = "" if args.guidance_mod == guidance_mod_for(args.lr_cfg) else f"_g{args.guidance_mod}"
    a_tag = "" if args.arch == "anyup" else f"_{args.arch}"
    # Attention window and consistency-loss weight change the TRAINED MODEL, not just the run,
    # so two variants of one LR->HR pair must not share a filename. Tagged against the values
    # every pre-flag checkpoint was trained with (AnyUp's own window 0.1 / down_reg 0.1) rather
    # than against argparse's defaults, so existing checkpoint paths keep resolving.
    w_tag = "" if args.window_ratio == 0.1 else f"_w{args.window_ratio:g}"
    dr_tag = "" if args.down_reg == 0.1 else f"_dr{args.down_reg:g}"
    var_tag = f"{tp_tag}{g_tag}{a_tag}{w_tag}{dr_tag}"
    run_tag = f"{mods}_ps{lr_ps}_{gh}x{gw}_to_ps{hr_ps}_{GH}x{GW}{var_tag}"
    print(f"run tag: {run_tag}")

    # 13-channel guidance is the ONLY architectural change vs stock AnyUp. Train from scratch.
    mk = dict(input_dim=guide_bands, qk_dim=args.qk_dim,
              # Both archs take window_ratio (it gates the cross-attention mask, not the
              # architecture), so setting it only for "manyup" silently left --arch anyup on
              # AnyUp's own 0.1 default while the checkpoint recorded the requested value.
              window_ratio=args.window_ratio)
    if args.arch == "manyup":
        mk["feat_dim"] = args.feat_dim        # the post-attention transform needs the feature dim
        mk["transform_depth"] = args.transform_depth
    model = AnyUp(**mk).to(device).train()

    # Optional pixel-wise linear head (1x1 conv, D->D). AnyUp pools VALUES from the LR feats, so
    # it assumes LR and HR share a feature space -- true when they're one backbone at two
    # resolutions. Our LR (ps4) and HR (ps1) targets are DIFFERENT patch sizes, so OlmoEarth may
    # produce systematically different features; forcing anyup_hr to match them directly is then
    # ill-posed. The head lets the upsampled ps4-space map be linearly PROJECTED into ps1-space:
    # we only require it can PREDICT the HR target, not equal it. anyup_down stays in ps4-space
    # (on the raw, pre-projection upsampled map), so the fidelity anchor is unaffected.
    proj_head = nn.Conv2d(D, D, kernel_size=1).to(device) if args.proj_head else None
    params = list(model.parameters()) + (list(proj_head.parameters()) if proj_head else [])
    print(f"mAnyUp params: {sum(p.numel() for p in model.parameters())}"
          + (f" + proj_head {sum(p.numel() for p in proj_head.parameters())}" if proj_head else ""))

    criterion = Cosine_MSE()
    opt = torch.optim.AdamW(params, lr=args.lr)

    # Warmup + cosine LR schedule, stepped PER BATCH. total_steps drives the cosine period; the
    # first warmup_frac of steps ramp linearly 0 -> lr, then cosine-decay lr -> lr_min. Both the
    # mAnyUp optimizer and the baseline's share the SAME schedule so the attribution comparison
    # (mAnyUp vs bilinear+linear head) stays fair -- otherwise one would train under a decaying LR
    # and the other a constant one.
    total_steps = args.epochs * len(loader)
    def make_sched(o):
        return _warmup_cosine(o, total_steps, args.warmup_frac, args.lr, args.lr_min)
    sched = make_sched(opt)

    # ---- baseline 1: NAIVE bilinear upsample of LR feats (no guidance, no learning). One-shot.
    baseline = _bilinear_baseline(loader, criterion, GH, GW, device, n_batches=args.baseline_batches)
    print(f"[baseline] bilinear (no learning) HR loss = {baseline:.4f}")

    # ---- baseline 2 (optional): bilinear upsample + a LEARNED 1x1 conv (D->D), co-trained on the
    # same data. This isolates how much of mAnyUp's gain comes from the guidance-driven upsampling
    # vs. just a linear ps4->ps1 projection. If mAnyUp (with proj_head) barely beats THIS, the win
    # is mostly the linear map, not the guidance. Trained in lockstep below with its own optimizer.
    lin_head = nn.Conv2d(D, D, kernel_size=1).to(device) if args.linear_baseline else None
    lin_opt = torch.optim.AdamW(lin_head.parameters(), lr=args.lr) if lin_head else None
    lin_sched = make_sched(lin_opt) if lin_opt else None
    if lin_head:
        print(f"[baseline] co-training bilinear+linear head ({sum(p.numel() for p in lin_head.parameters())} params)")

    for epoch in range(args.epochs):
        model.train()
        running = {"hr": 0.0, "down": 0.0, "lin": 0.0}
        for bi, (lr, hr, s2) in enumerate(loader):
            lr, hr, s2 = lr.to(device), hr.to(device), s2.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(s2, lr, (GH, GW))            # guidance, LR feats, target grid (ps4-space)
                # HR loss sees the projection (ps4-space -> ps1-space); identity if no head.
                pred_hr = proj_head(pred) if proj_head is not None else pred
                loss_hr = criterion(pred_hr, hr)["total"]
                loss = loss_hr
                loss_down = torch.tensor(0.0, device=device)
                if args.down_reg > 0:
                    # down-reg stays in ps4-space: downsample the RAW (pre-projection) upsampled
                    # map and match the LR input feats -- the anchor is native to the upsampler.
                    down = F.interpolate(pred.float(), size=(gh, gw), mode="area")
                    loss_down = criterion(down, lr)["total"] * args.down_reg
                    loss = loss + loss_down

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()                                       # warmup+cosine, per batch

            # Co-train the bilinear+linear baseline on the same batch (independent optimizer, no
            # guidance, no learned upsampling -- just a linear map on bilinearly-upsampled LR).
            loss_lin = torch.tensor(0.0, device=device)
            if lin_head is not None:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    up = F.interpolate(lr.float(), size=(GH, GW), mode="bilinear", align_corners=False)
                    loss_lin = criterion(lin_head(up), hr)["total"]
                lin_opt.zero_grad()
                loss_lin.backward()
                lin_opt.step()
                lin_sched.step()                               # same schedule -> fair comparison

            running["hr"] += loss_hr.item()
            running["down"] += float(loss_down)
            running["lin"] += float(loss_lin)
            if bi % 50 == 0:
                print(f"epoch {epoch} batch {bi}/{len(loader)}  learning rate={sched.get_last_lr()[0]:.2e}  "
                      f"upsampling loss={loss_hr.item():.4f} consistency loss={float(loss_down):.4f}"
                      + (f" lin={float(loss_lin):.4f}" if lin_head is not None else ""))
            if args.sanity:
                print("sanity: one batch done, exiting.")
                return

        n = len(loader)
        mean_hr, mean_lin = running["hr"] / n, running["lin"] / n
        vs = baseline - mean_hr
        lin_str = ""
        if lin_head is not None:
            # The honest attribution: how much mAnyUp beats a learned linear map (not just raw
            # bilinear). If this gap is ~0, the guidance-driven upsampling isn't adding much.
            lin_str = (f"  | lin_head={mean_lin:.4f}  (mAnyUp vs lin_head: {mean_lin - mean_hr:+.4f})")
        print(f"== epoch {epoch} done  hr={mean_hr:.4f}  down={running['down']/n:.4f}  "
              f"| bilinear={baseline:.4f}  (mAnyUp {'beats' if vs > 0 else 'WORSE than'} "
              f"bilinear by {vs:+.4f}){lin_str}")

        if viz_sample is not None:
            # One folder per run instead of a single shared viz/ where every config's
            # test0_epNNN.png collided. The folder name carries modality and both grids, so
            # runs are distinguishable on disk without opening them.
            viz_dir = out_dir / "viz" / run_tag
            viz_dir.mkdir(parents=True, exist_ok=True)
            save_epoch_viz(model, proj_head, viz_sample, GH, GW, device,
                           viz_dir / f"{run_tag}_ep{epoch:03d}.png", epoch, run_tag)

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            # Tag non-default guidance pooling in the name (empty for mean, so the existing
            # mean-trained checkpoint paths keep working). vars(args) below records it either way.
            # var_tag (built once above) carries every knob that changes the trained model:
            # time_pool, guidance_mod, arch, window_ratio, down_reg. Keeping one source of
            # truth stops the printed run tag and the saved filename from disagreeing.
            ckpt = out_dir / (f"manyup_{args.transform_depth}transform_{args.lr_cfg}"
                              f"_to_{args.hr_cfg}{var_tag}_ep{epoch}.pth")
            torch.save({"model": model.state_dict(),
                        "proj_head": proj_head.state_dict() if proj_head else None,
                        "args": vars(args), "epoch": epoch,
                        "input_dim": guide_bands, "qk_dim": args.qk_dim,
                        "arch": args.arch, "feat_dim": args.feat_dim,
                        "transform_depth": args.transform_depth,
                        "window_ratio": args.window_ratio},
                       ckpt)
            print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
