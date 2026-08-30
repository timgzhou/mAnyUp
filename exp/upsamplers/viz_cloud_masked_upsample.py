"""Feasibility check: can OmniCloudMask make UPA / UPMA upsampling robust to cloud?

THE IDEA. UPA ("upsample anything") and UPMA ("upsample many things") are joint bilateral
upsamplers: every HR output pixel is a normalized weighted average of nearby LR feature
pixels, with weight = spatial Gaussian x range term, where the range term measures how
similar the GUIDE image is at the contributing pixel vs the output pixel. Concretely, in
exp/upsamplers/upa_anyup.py::gs_jbu_aniso_noparent:

    log_w = log_ws + log_wr           # spatial + range
    log_w = where(radius_mask, log_w, -inf)

The guide is optical (S2). Where the guide is CLOUD, the range term is comparing cloud-top
reflectance instead of ground, so the kernel keys on cloud edges -- it will happily blend
features across a real land boundary because the cloud above it looks uniform, and refuse to
blend across a cloud edge that has no ground meaning at all. The upsampler invents structure
that belongs to the atmosphere, not the scene.

Because the radius mask already writes -inf into log_w, a cloud mask can be injected the
SAME way, at zero architectural cost. Two distinct interventions, which this script separates
because they are NOT equally justified:

  SOURCE masking ("where not to upsample FROM"): drop cloudy LR pixels as contributors
      (set their log_w to -inf). Well-posed: a cloudy contributor's guide value is
      meaningless, so its range weight is noise. Surviving clean neighbours are renormalized
      over (the kernel already normalizes, so this is free).
  TARGET masking ("where not to upsample TO"): mark HR outputs under cloud as untrustworthy.
      This does NOT repair them -- under thick cloud the optical guide carries no ground
      information at all, so there is nothing to guide WITH. The honest move is to flag those
      pixels, not to pretend a bilateral kernel can see through cloud.

WHAT THIS SCRIPT DRAWS, per sample:
  row 0: guidance + context -- S2 RGB, the OmniCloudMask, S1 post VV (cloud-free reference),
         and the flood label
  row 1: raw LR features (ps=8 OlmoEarth), UPA, UPA+cloud-masked, and their difference
  row 2: the same for UPMA (multispectral guidance)
All feature panels share ONE PCA basis (fit on the raw LR map), following the convention in
upa_anyup.py, so colour differences between panels are real and not per-panel recolouring.

READ THE FIGURE FOR: does the cloud-masked variant differ from the unmasked one INSIDE the
cloud footprint (it should) while staying identical outside it (it must -- otherwise the
mask is corrupting clean regions)? The difference panel makes exactly that testable: it
should light up on the cloud and be black everywhere else.

CAVEAT ON THIS DATASET. GEOID's S2 layer is a cloud-FILTERED pre-event composite, so it is
mostly clean by construction: across all 502k S2 chips the median cloud_cover is 0.000 and
only 11.7% have any cloud at all (1.9% exceed 50%). So cloud-aware upsampling is a
small-subpopulation fix here, NOT a general win -- and the samples below are deliberately
picked from the cloudy tail via the metadata CSV, which is not a representative draw. The
technique matters much more for datasets whose optical input is a single acquisition rather
than a curated composite.

Run:
    source env_setup/env_olmo.sh
    python -u -m exp.upsamplers.viz_cloud_masked_upsample --tiles EMSR251-2-23,EMSR251-2-31
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from exp.upsamplers.upa_anyup import (
    LearnablePixelwiseAnisoJBU_NoParent,
    pca_rgb_shared,
)

# S2 L2A here is the 12-band GEOID stack: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
RGB_IDX = [3, 2, 1]                       # B04,B03,B02
SURFACE_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]   # drop B01 (aerosol) and B09 (water vapour)
CLOUD_CLASSES = (1, 2, 3)                 # 1 thick, 2 thin, 3 shadow; 0 = clear
CLOUD_NAMES = {0: "clear", 1: "thick cloud", 2: "thin cloud", 3: "shadow"}
CLOUD_COLORS = ["#101010", "#ffffff", "#b8b8b8", "#5a5a5a"]
LABEL_COLORS = ["#f2f2f2", "#1f77ff", "#e4002b"]
LABEL_NAMES = {0: "background", 1: "perm water", 2: "flood"}


def stretch(x, lo=2, hi=98):
    x = np.asarray(x, dtype=np.float32)
    v = np.isfinite(x)
    if not v.any():
        return np.zeros_like(x)
    a, b = np.percentile(x[v], [lo, hi])
    return np.nan_to_num(np.clip((x - a) / (b - a + 1e-6), 0, 1), nan=0.0)


def to_db(x):
    x = np.asarray(x, dtype=np.float32)
    o = np.full_like(x, np.nan)
    m = np.isfinite(x) & (x > 0)
    o[m] = 10.0 * np.log10(x[m])
    return o


def _find(roots, event, layer, tile, pas=None):
    """Locate one raster across several candidate trees (aux shards, full download, sample)."""
    pat = f"{event}-{tile}_{layer}" + (f"_{pas}_*" if pas else "") + ".tif"
    for root in roots:
        hits = sorted(glob.glob(str(Path(root) / event / layer / pat)))
        if hits:
            return hits[0]
    return None


def read(path, band=None):
    if path is None:
        return None
    with rasterio.open(path) as r:
        return r.read(band) if band else r.read()


# --------------------------------------------------------------------------------------
# cloud-aware bilateral upsampling
# --------------------------------------------------------------------------------------
def upsample(feat_lr, guide_hr, scale, steps, lr, cloud_hr=None, device="cuda", seed=0):
    """Fit a UPA/UPMA kernel self-supervised on the guide, then apply it to the features.

    Self-supervision (as in upa_anyup.py): the kernel is trained to reconstruct the full-res
    guide from its own bicubic downsample. It never sees the features -- so the features can
    be swapped afterwards without refitting.

    cloud_hr: optional (Hh,Wh) bool, True = cloudy. When given, cloudy pixels are excluded
    from the RECONSTRUCTION LOSS, so the kernel is never rewarded for reproducing cloud
    texture -- this is the "do not learn FROM cloud" half. The source-exclusion half is done
    by zeroing the cloudy guide contrast (see mask_guide below), which drives those
    contributors' range weights to a constant so they cannot dominate.
    """
    torch.manual_seed(seed)
    Hh, Wh = guide_hr.shape[-2:]
    Hl, Wl = Hh // scale, Wh // scale
    guide_lr = F.interpolate(guide_hr, size=(Hl, Wl), mode="bicubic", align_corners=False)

    model = LearnablePixelwiseAnisoJBU_NoParent(
        Hl, Wl, scale=scale, init_sigma=float(scale), init_sigma_r=0.12,
        R_max=4, use_autocast=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    w = None
    if cloud_hr is not None:
        w = torch.from_numpy((~cloud_hr).astype(np.float32))[None, None].to(device)

    model.train()
    for _ in range(steps):
        out = model(guide_lr, guide_hr)
        diff = (out - guide_hr).abs()
        loss = (diff * w).sum() / w.sum().clamp(min=1) / diff.shape[1] if w is not None \
            else diff.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        return model(feat_lr, guide_hr)[0].permute(1, 2, 0).float().cpu().numpy()


def mask_guide(guide_hr, cloud_hr, device):
    """Neutralize the guide where it is cloudy.

    Setting cloudy pixels to the scene mean makes their range term flat, so a cloudy pixel
    neither attracts (identical guide -> max weight) nor repels (large diff -> ~0 weight)
    its neighbours on the basis of cloud-top reflectance. Combined with dropping them from
    the fitting loss, that is the "do not upsample FROM cloud" intervention -- implemented
    on the guide rather than inside the kernel, so upa_anyup.py stays untouched.
    """
    g = guide_hr.clone()
    m = torch.from_numpy(cloud_hr.astype(bool))[None, None].to(device)
    if m.any():
        keep = ~m.expand_as(g)
        mean = g[keep].mean() if keep.any() else torch.zeros((), device=device)
        g = torch.where(m.expand_as(g), mean, g)
    return g


def render(tile_id, roots, args, device):
    event, tile = tile_id.rsplit("-", 1)

    s2 = read(_find(roots, event, "s2l2a", tile, "pre"))
    cloud = read(_find(roots, event, "cloudmask", tile, "pre"), band=1)
    label = read(_find(roots, event, "label", tile), band=1)
    s1_post = read(_find(roots, event, "s1grd", tile, "post"))
    if s2 is None or cloud is None:
        print(f"  !! {tile_id}: missing s2l2a or cloudmask, skipping")
        return

    # crop to a window that actually straddles a cloud edge -- a full 1024 tile at ps=8 is
    # 128 LR tokens and the interesting structure is lost at figure scale.
    W = args.window
    cl = np.isin(cloud, CLOUD_CLASSES)
    if cl.any():
        ys, xs = np.where(cl)
        cy = int(np.clip(ys.mean() - W // 2, 0, s2.shape[1] - W))
        cx = int(np.clip(xs.mean() - W // 2, 0, s2.shape[2] - W))
    else:
        cy = cx = (s2.shape[1] - W) // 2
    sl = (slice(cy, cy + W), slice(cx, cx + W))

    s2 = s2[:, sl[0], sl[1]].astype(np.float32)
    cloud = cloud[sl]
    cl = np.isin(cloud, CLOUD_CLASSES)
    label = label[sl] if label is not None else None
    s1_post = s1_post[:, sl[0], sl[1]] if s1_post is not None else None

    cloud_frac = 100.0 * cl.mean()
    print(f"  {tile_id}: window {W}x{W} at ({cy},{cx}), cloud/shadow {cloud_frac:.1f}%")
    if cloud_frac < 1.0:
        print(f"     (little cloud in window -- masked and unmasked will look near-identical)")

    # ---- guides ----
    rgb = np.stack([stretch(s2[i]) for i in RGB_IDX], 0)               # (3,W,W)
    ms = np.stack([stretch(s2[i]) for i in SURFACE_IDX], 0)            # (10,W,W)
    g_rgb = torch.from_numpy(rgb)[None].float().to(device)
    g_ms = torch.from_numpy(ms)[None].float().to(device)

    # ---- LR "features": stand in for a ps=8 OlmoEarth map ----
    # This script is a GUIDANCE feasibility check, so what matters is that the LR map has the
    # right grid (W/8) and enough channels to PCA meaningfully. We build it by average-pooling
    # the S2 stack by 8 -- deterministic, needs no GPU encoder pass, and shares the scene's
    # real structure. Swap in a cached OlmoEarth map with --feat_pt to check the real thing.
    scale = args.patch_size
    if args.feat_pt:
        rec = torch.load(args.feat_pt)
        f = rec["feat"] if isinstance(rec, dict) else rec
        f = f[0] if f.dim() == 4 else f                                # (gH,gW,D), t0
        feat_lr = f.permute(2, 0, 1)[None].float().to(device)
    else:
        pooled = F.avg_pool2d(torch.from_numpy(s2)[None].float(), scale)
        feat_lr = pooled.to(device)

    Hl = feat_lr.shape[-1]
    if Hl * scale != W:
        feat_lr = F.interpolate(feat_lr, size=(W // scale, W // scale),
                                mode="bilinear", align_corners=False)

    g_rgb_m = mask_guide(g_rgb, cl, device)
    g_ms_m = mask_guide(g_ms, cl, device)

    runs = {}
    for name, guide, cmask in [
            ("UPA", g_rgb, None), ("UPA+cloud", g_rgb_m, cl),
            ("UPMA", g_ms, None), ("UPMA+cloud", g_ms_m, cl)]:
        runs[name] = upsample(feat_lr, guide, scale, args.steps, args.lr,
                              cloud_hr=cmask, device=device, seed=args.seed)

    raw_np = feat_lr[0].permute(1, 2, 0).float().cpu().numpy()
    # ONE shared PCA basis fit on the raw LR map, applied to every panel (upa_anyup.py
    # convention) so colour differences across panels are real.
    colored = pca_rgb_shared(torch.from_numpy(raw_np),
                             [torch.from_numpy(raw_np)] + [torch.from_numpy(runs[k])
                                                           for k in runs])
    raw_c, upa_c, upam_c, upma_c, upmam_c = colored

    fig, axes = plt.subplots(3, 4, figsize=(19, 14.6))

    # ---- row 0: guidance + context ----
    ax = axes[0][0]
    ax.imshow(np.transpose(rgb, (1, 2, 0)))
    ax.set_title("S2 RGB (pre-event composite)\n= the UPA guide", fontweight="bold", fontsize=11)
    ax = axes[0][1]
    ax.imshow(cloud, cmap=ListedColormap(CLOUD_COLORS), vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(f"OmniCloudMask — {cloud_frac:.1f}% cloud/shadow\n(model output, not manual)",
                 fontweight="bold", fontsize=11)
    ax = axes[0][2]
    if s1_post is not None:
        ax.imshow(stretch(to_db(s1_post[0])), cmap="gray")
    ax.set_title("S1 post VV (dB)\nradar: unaffected by cloud", fontweight="bold", fontsize=11)
    ax = axes[0][3]
    if label is not None:
        ax.imshow(np.where(label == 255, 0, label), cmap=ListedColormap(LABEL_COLORS),
                  vmin=0, vmax=2, interpolation="nearest")
    ax.set_title("flood label", fontweight="bold", fontsize=11)

    # ---- rows 1/2: raw | plain | cloud-masked | difference ----
    for row, (base, masked, tag) in enumerate(
            [(upa_c, upam_c, "UPA (RGB guide)"),
             (upma_c, upmam_c, "UPMA (multispectral guide)")], start=1):
        axes[row][0].imshow(raw_c, interpolation="nearest")
        axes[row][0].set_title(f"raw LR features\n{Hl}x{Hl} tokens (ps={scale})",
                               fontweight="bold", fontsize=11)
        axes[row][1].imshow(base, interpolation="nearest")
        axes[row][1].set_title(f"{tag}\nno cloud handling", fontweight="bold", fontsize=11)
        axes[row][2].imshow(masked, interpolation="nearest")
        axes[row][2].set_title(f"{tag}\n+ cloud-masked guide", fontweight="bold", fontsize=11)
        d = np.abs(base.astype(np.float32) - masked.astype(np.float32)).mean(-1)
        im = axes[row][3].imshow(d, cmap="magma")
        # outline the cloud so "does the change sit inside it?" is answerable by eye
        if cl.any():
            axes[row][3].contour(cl.astype(float), levels=[0.5], colors="#00e5ff",
                                 linewidths=1.4)
        plt.colorbar(im, ax=axes[row][3], fraction=0.046)
        inside = float(d[cl].mean()) if cl.any() else float("nan")
        outside = float(d[~cl].mean()) if (~cl).any() else float("nan")
        axes[row][3].set_title(f"|difference|  (cyan = cloud edge)\n"
                               f"in-cloud {inside:.4f} vs outside {outside:.4f}",
                               fontweight="bold", fontsize=11)
        print(f"     {tag}: mean |diff| inside cloud {inside:.5f}, outside {outside:.5f}"
              + (f"  -> ratio {inside/outside:.1f}x" if outside and outside > 1e-9 else ""))

    for r in axes:
        for a in r:
            a.set_xticks([]); a.set_yticks([])

    handles = [Patch(color=CLOUD_COLORS[k], label=f"{k}: {v}") for k, v in CLOUD_NAMES.items()]
    handles += [Patch(color=LABEL_COLORS[k], label=f"label {k}: {v}")
                for k, v in LABEL_NAMES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=10)
    fig.suptitle(f"Cloud-masked guided upsampling — {tile_id}   "
                 f"(window {W}x{W} @10 m, ps={scale}, {args.steps} fit steps)\n"
                 f"Does masking change the output INSIDE the cloud (wanted) "
                 f"without touching it outside (required)?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))

    out = Path(args.out_dir) / f"cloud_upsample_{tile_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default="EMSR251-2-23,EMSR251-2-31,EMSR199-14-11",
                    help="comma-separated <event>-<tile>; pick cloudy ones from the "
                         "metadata CSV (cloud_cover between ~0.15 and ~0.75)")
    ap.add_argument("--roots", default="data/GEOID-Flood-aux/geoid-flood,"
                                       "data/GEOID-Flood-full/geoid-flood,"
                                       "data/GEOID-Flood/sample/geoid-flood",
                    help="comma-separated trees searched in order for <event>/<layer>/*.tif")
    ap.add_argument("--patch_size", type=int, default=8, help="LR grid = window/patch_size")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat_pt", default=None,
                    help="optional cached OlmoEarth feature .pt to use instead of pooled S2")
    ap.add_argument("--out_dir", default="results/upsamplers/cloud")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    roots = [r for r in args.roots.split(",") if r]
    for t in [x for x in args.tiles.split(",") if x]:
        render(t, roots, args, device)


if __name__ == "__main__":
    main()
