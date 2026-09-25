"""Compare feature upsamplers on one PASTIS sample: raw OlmoEarth LR features -> UPA -> AnyUp.

Extracted from UPA-vs-ANYUP.ipynb. Three ways to get a 64x64xD feature map out of a coarse
(gH,gW,D) OlmoEarth feature map, all guided by the same S2 RGB image:

  1. raw       -- the low-res features, no upsampling (shown at native gHxgW)
  2. UPA       -- per-image test-time optimized joint bilateral upsampler. Fits an anisotropic
                  pixelwise JBU to reconstruct the HR RGB from its own bicubic downsample
                  (self-supervised, never sees the features), then applies the learned kernels
                  to the feature map.
  3. UPMA      -- "upsample many things": identical machinery to UPA, but guided by the full
                  multispectral S2 stack instead of RGB. NIR/SWIR carry crop-boundary structure
                  RGB misses. Bands are percentile-stretched per channel so none dominates the
                  L1 target, and the bilateral range term averages (not sums) over channels so
                  sigma_r keeps the same meaning at any band count.
  4. AnyUp     -- pretrained feature upsampler pulled from torch.hub (wimmerth/anyup), frozen.

Optionally a panel with the native per-pixel (patch size 1) features as a reference. The last
panel is the PASTIS ground-truth semantic map, so you can judge whether the structure each
upsampler invents actually lines up with the real parcel boundaries.

All feature panels are colored with ONE shared PCA basis, fit on the raw low-res map, so color
differences between panels are real and not per-panel recoloring.

Note the methods are fed differently normalized guidance: UPA gets the percentile-stretched
uint8 display RGB, UPMA the percentile-stretched multispectral stack, AnyUp min-max +
ImageNet-normalized float RGB (what its pretrained weights expect). That is a confound if you
read the figure as a pure architecture comparison. UPA vs UPMA *is* a clean ablation, though --
same kernel and optimizer, only the guidance band count differs.

    source env_setup/env_olmo.sh        # AnyUp pulls torch.hub; UPA/UPMA need cuda
    python exp/upsamplers/upa_anyup.py
    python exp/upsamplers/upa_anyup.py --img_idx 3 --guide_bands all
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")        # headless cluster: never try an interactive backend (it hangs)

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from exp.common.paths import FEATURES
from manyup.upsamplers.jbu import LearnablePixelwiseAnisoJBU_NoParent, fit_jbu  # noqa: F401

RGB_BANDS = [3, 2, 1]        # B04/B03/B02 = R,G,B in the 13-band stack
# Surface-structure bands for UPMA guidance: B02-B08A + B11/B12. Drops B01 (aerosol, 60m),
# B09 (water vapour) and B10 (cirrus) -- atmospheric bands that carry no crop-parcel structure,
# and which a per-channel stretch would otherwise amplify to noise and give equal weight in
# the range term.
SURFACE_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
DISPLAY_PX = 128             # all feature panels rendered at this size (nearest, keeps blocks)

# PASTIS class names + colormap (same 20-class scheme as visualize_olmoearth_pastis.py).
CLASSES = [
    'background', 'meadow', 'soft_winter_wheat', 'corn', 'winter_barley',
    'winter_rapeseed', 'spring_barley', 'sunflower', 'grapevine', 'beet',
    'winter_triticale', 'winter_durum_wheat', 'fruits_vegetables_flowers',
    'potatoes', 'leguminous_fodder', 'soybeans', 'orchard', 'mixed_cereal',
    'sorghum', 'void_label',
]
CMAP = plt.get_cmap('tab20', 20)
IGNORE_INDEX = -1            # targets.pt stores ignore as -1; remapped to 19 (void) for display

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TIME_POOLS = ("mean", "median")   # how a (T,...) stack is collapsed to a single frame


def time_pool(x: torch.Tensor, mode: str = "mean", dim: int = 0) -> torch.Tensor:
    """Collapse the time axis of a (T,...) stack to a single frame.

    Guidance for UPA/UPMA/AnyUp/mAnyUp is a single image, so the PASTIS time series has to be
    reduced first. "mean" is the historical default. "median" is the robust alternative: PASTIS
    S2 series carry undetected cloud, haze and shadow frames, and a mean drags those bright/dark
    outliers into the composite, softening exactly the parcel boundaries the guided upsamplers
    key on. A per-pixel median rejects them as long as they are the minority over T.

    Note this only changes the GUIDANCE composite -- the feature-side reduction stays whatever
    the head does, so a change here is attributable to guidance quality alone.
    """
    if mode == "mean":
        return x.mean(dim)
    if mode == "median":
        return x.median(dim).values
    raise ValueError(f"unknown time_pool {mode!r}, expected one of {TIME_POOLS}")


# ---------------------------------------------------------------------------
# UPA / UPMA: per-image optimized anisotropic joint bilateral upsampling
# ---------------------------------------------------------------------------
def percentile_stretch(x: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0) -> np.ndarray:
    """(C,H,W) -> per-channel percentile stretch into [0,1].

    Robust alternative to per-channel min-max: satellite bands routinely have a handful of
    specular / edge-artifact pixels that would otherwise set the whole range and crush the
    real dynamic range into a sliver. Matches what raw_rgb() already does for display, so the
    3-band and 13-band guidance paths differ only in band count.
    """
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, lo_p, axis=(1, 2), keepdims=True)
    hi = np.percentile(x, hi_p, axis=(1, 2), keepdims=True)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def UPA(HR_img, lr_modaliry, fit_steps: int = 50):
    """Upsample Anything: RGB guidance.

    HR_img: (H,W,3) uint8 array. lr_modaliry: [1,C,Hl,Wl] cuda. -> [1,C,H,W].
    """
    hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    return fit_jbu(hr, lr_modaliry, fit_steps)


def UPMA(guide, lr_modality, fit_steps: int = 50):
    """Upsample Many things: full multispectral guidance.

    Identical machinery to UPA -- same kernel, same parametrization, same optimizer -- the
    only difference is that the guide carries every S2 band instead of just RGB, so both the
    reconstruction target and the bilateral range term see the full spectrum. The extra bands
    (NIR/SWIR especially) carry crop-boundary structure that RGB misses, which is exactly the
    signal the anisotropic kernels are trying to find.

    guide: (Cg,H,W) float array, ALREADY percentile-stretched to [0,1] (no /255 here -- that
    would re-divide an already-normalized guide and collapse the range term).
    lr_modality: [1,C,Hl,Wl] cuda. -> [1,C,H,W].
    """
    hr = torch.from_numpy(np.asarray(guide, dtype=np.float32)).unsqueeze(0).cuda()
    return fit_jbu(hr, lr_modality, fit_steps)


# ---------------------------------------------------------------------------
# display helpers (PCA / raw RGB), matching visualize_features.py conventions
# ---------------------------------------------------------------------------
def raw_rgb(s2_t: torch.Tensor) -> np.ndarray:
    """(13,H,W) one timestep -> (H,W,3) float in [0,1], percentile-stretched."""
    rgb = s2_t[RGB_BANDS].float().numpy().transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, 2, (0, 1)), np.percentile(rgb, 98, (0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def pca_rgb_shared(fit: torch.Tensor, apply_to):
    """Fit top-3 PCA + min-max on `fit` (H,W,D); apply that SAME basis to every map in
    `apply_to`. Shared basis => identical feature vectors get identical colors, so color
    differences across panels are real (not per-panel recoloring)."""
    Hf, Wf, D = fit.shape
    xf = fit.reshape(Hf * Wf, D).float().numpy()
    mean = xf.mean(0, keepdims=True)
    cov = ((xf - mean).T @ (xf - mean)) / max(Hf * Wf - 1, 1)
    _, evecs = np.linalg.eigh(cov)
    dirs = evecs[:, -3:][:, ::-1]
    proj_fit = (xf - mean) @ dirs
    lo, hi = proj_fit.min(0), proj_fit.max(0)
    out = []
    for m in apply_to:
        H, W, _ = m.shape
        x = m.reshape(H * W, D).float().numpy()
        proj = ((x - mean) @ dirs).reshape(H, W, 3)
        out.append(np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1))
    return out


def nearest_resize(img: np.ndarray, size: int = DISPLAY_PX) -> np.ndarray:
    H, W = img.shape[:2]
    yi = (np.arange(size) * H // size).clip(0, H - 1)
    xi = (np.arange(size) * W // size).clip(0, W - 1)
    return img[yi][:, xi]


def _norm_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """(3,H,W) -> per-channel min-max then ImageNet standardization (what AnyUp expects)."""
    flat = rgb.reshape(*rgb.shape[:-3], 3, -1)
    lo = flat.amin(-1).reshape(*rgb.shape[:-3], 3, 1, 1)
    hi = flat.amax(-1).reshape(*rgb.shape[:-3], 3, 1, 1)
    return ((rgb - lo) / (hi - lo + 1e-6) - _IMAGENET_MEAN) / _IMAGENET_STD


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # exp/upsamplers/upa_anyup.py -> repo root is parents[2]; .parent pointed at
    # exp/upsamplers/ and made data_splits resolve to a nonexistent path.
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2],
                   help="repo root (holds exp/ and data/)")
    p.add_argument("--features_root", type=Path,
                   default=FEATURES)
    p.add_argument("--lr_dir", default="oe_base_s2_ps4_tile64",
                   help="low-res feature dir to upsample (16x16x768 -> scale 4 up to 64x64)")
    p.add_argument("--ref_dir", default="oe_base_s2_ps1_tile32",
                   help="native per-pixel features for the optional reference panel")
    p.add_argument("--split", default="test")
    p.add_argument("--img_idx", type=int, default=1)
    p.add_argument("--fit_steps", type=int, default=50,
                   help="UPA/UPMA test-time optimization steps")
    p.add_argument("--guide_bands", default="surface",
                   help="UPMA guidance bands: 'surface' (B02-B08A,B11,B12), 'all' (13 bands), "
                        "or a comma-separated index list e.g. '3,2,1,7'")
    p.add_argument("--out", type=Path, default=Path("results/pastis/feature_viz/upa_anyup.png"))
    args = p.parse_args()

    repo = args.repo
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()   # before any olmoearth_pretrain.evals import

    data_splits = repo / "data" / "pastis_olmoearth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load the low-res OlmoEarth features + guidance for this one sample ----
    # .mean(0) collapses the stored TIME axis: (T,gH,gW,D) -> (gH,gW,D). The coarse gHxgW grid
    # comes from OlmoEarth's patch size, not from spatial pooling.
    lr = torch.load(args.features_root / args.lr_dir / f"pastis_r_{args.split}" /
                    f"{args.img_idx}.pt").float().mean(0)                       # (gH,gW,D)
    gH, gW, D = lr.shape
    s2 = torch.load(data_splits / f"pastis_r_{args.split}" / "s2_images" /
                    f"{args.img_idx}.pt")                                       # (T,13,64,64)
    rgb_disp = raw_rgb(s2.float().mean(0))                                      # (64,64,3) in [0,1]
    out_hw = (s2.shape[-2], s2.shape[-1])                                       # (64,64)
    print(f"[load] lr {tuple(lr.shape)}  s2 {tuple(s2.shape)}  scale={out_hw[0] // gH}")

    # ---- ground truth: one (N,64,64) int64 tensor for the whole split, indexed by img_idx ----
    targets = torch.load(data_splits / f"pastis_r_{args.split}" / "targets.pt")
    gt = targets[args.img_idx].numpy()                                          # (64,64), -1..18
    gt_disp = np.where(gt == IGNORE_INDEX, 19, gt)                              # ignore -> void_label
    print(f"[gt] classes present: "
          f"{[CLASSES[c] for c in np.unique(gt_disp).astype(int) if 0 <= c <= 19]}")

    # ============ method 1: raw low-res features (no upsampling) ============
    lr_hw = lr                                                                  # (gH,gW,D)

    # ============ method 2: UPA (per-image optimized JBU) ============
    # guidance here is the percentile-stretched DISPLAY rgb, round-tripped through uint8
    # (UPA divides by 255 internally) -- deliberately different from AnyUp's normalization.
    hr_img_255 = (rgb_disp * 255).astype(np.uint8)                              # HxWx3 uint8
    lr_feat_in = lr.permute(2, 0, 1).unsqueeze(0).to(device)                    # (1,D,gH,gW)
    upa_hr = UPA(hr_img_255, lr_feat_in, fit_steps=args.fit_steps)              # (1,D,64,64)
    upa_hw = upa_hr.squeeze(0).permute(1, 2, 0).float().cpu()                   # (64,64,D)
    print(f"[UPA] fit {args.fit_steps} steps -> {tuple(upa_hw.shape)}")

    # ============ method 3: UPMA (same JBU, full multispectral guidance) ============
    if args.guide_bands == "surface":
        guide_bands = SURFACE_BANDS
    elif args.guide_bands == "all":
        guide_bands = list(range(s2.shape[1]))
    else:
        guide_bands = [int(b) for b in args.guide_bands.split(",")]
    # time-mean first (same as the RGB path), then per-band percentile stretch to [0,1] so no
    # single band dominates the L1 reconstruction target or the range term.
    guide_ms = percentile_stretch(s2.float().mean(0)[guide_bands].numpy())      # (Cg,64,64)
    upma_hr = UPMA(guide_ms, lr_feat_in, fit_steps=args.fit_steps)              # (1,D,64,64)
    upma_hw = upma_hr.squeeze(0).permute(1, 2, 0).float().cpu()                 # (64,64,D)
    print(f"[UPMA] bands={guide_bands} ({len(guide_bands)}ch) fit {args.fit_steps} steps "
          f"-> {tuple(upma_hw.shape)}")

    # ============ method 3: pretrained AnyUp (torch.hub, frozen) ============
    s2_full = torch.load(data_splits / f"pastis_r_{args.split}" / "s2_images" /
                         f"{args.img_idx}.pt").float()
    rgb_guide = _norm_rgb(s2_full.mean(0)[RGB_BANDS])                           # (3,64,64)
    anyup = torch.hub.load("wimmerth/anyup", "anyup_multi_backbone",
                           use_natten=False, pretrained=True).to(device).eval()
    with torch.no_grad():
        anyup_hr = anyup(rgb_guide.unsqueeze(0).to(device),
                         lr.permute(2, 0, 1).unsqueeze(0).to(device),
                         output_size=out_hw)                                    # (1,D,64,64)
    anyup_hw = anyup_hr.squeeze(0).permute(1, 2, 0).float().cpu()               # (64,64,D)
    print(f"[AnyUp] -> {tuple(anyup_hw.shape)}")

    # ============ optional native reference (ps1 tile64) ============
    ref_hw = None
    ref_path = args.features_root / args.ref_dir / f"pastis_r_{args.split}" / f"{args.img_idx}.pt"
    if ref_path.exists():
        cand = torch.load(ref_path).float().mean(0)                             # (64,64,D)
        if cand.shape[-1] == D:
            ref_hw = cand
        else:
            print(f"ref D={cand.shape[-1]} != {D}; skipping reference panel")
    else:
        print(f"no reference at {ref_path}; skipping reference panel")

    # ---- color every feature panel with ONE shared PCA basis (fit on the low-res map) ----
    maps = [lr_hw, upa_hw, upma_hw, anyup_hw] + ([ref_hw] if ref_hw is not None else [])
    colored = pca_rgb_shared(lr_hw, maps)

    # panels carry a kind flag: "rgb" panels are plain images, "label" needs the categorical
    # colormap with fixed vmin/vmax so colors mean the same class across every figure.
    panels = [(rgb_disp, "raw RGB (mean T)", "rgb"),
              (nearest_resize(colored[0], DISPLAY_PX), f"OlmoEarth raw\n({gH}x{gW}x{D})", "rgb"),
              (nearest_resize(colored[1], DISPLAY_PX), f"UPA (RGB guide)\n-> 64x64x{D}", "rgb"),
              (nearest_resize(colored[2], DISPLAY_PX),
               f"UPMA ({len(guide_bands)}-band guide)\n-> 64x64x{D}", "rgb"),
              (nearest_resize(colored[3], DISPLAY_PX), f"AnyUp -> 64x64x{D}", "rgb")]
    if ref_hw is not None:
        panels.append((nearest_resize(colored[4], DISPLAY_PX),
                       f"{args.ref_dir}\n(native 64x64x{D})", "rgb"))
    # GT last, at native 64x64 -- nearest_resize would be lossless here but imshow's
    # interpolation="nearest" already keeps the label blocks crisp.
    panels.append((gt_disp, "ground truth\n(PASTIS semantic)", "label"))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.8))
    for ax, (img, ttl, kind) in zip(axes, panels):
        if kind == "label":
            ax.imshow(img, cmap=CMAP, vmin=0, vmax=19, interpolation="nearest")
        else:
            ax.imshow(img)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    present = [i for i in np.unique(gt_disp).astype(int) if 0 <= i <= 19]
    handles = [mpatches.Patch(color=CMAP(i), label=f"{i}: {CLASSES[i]}") for i in present]
    fig.legend(handles=handles, bbox_to_anchor=(1.005, 0.98), loc="upper left", fontsize=7)

    fig.suptitle(f"{args.lr_dir} -- {args.split} #{args.img_idx} | shared PCA basis "
                 f"(fit on raw features)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

# python -u exp/upsamplers/upa_anyup.py