"""Visualize GEOID-Flood samples: all modalities, both timesteps, + labels.

GEOID-Flood (https://huggingface.co/datasets/links-ads/geoid-flood, arXiv:2608.02315) is
a flood-segmentation benchmark built from 219 Copernicus EMS Rapid Mapping activations
across 65 countries (2016-2026). It is NOT the same dataset as ImpactMesh-Flood
(ibm-esa-geospatial) that exp/viz/visualize_impactmesh.py plots -- different publisher,
different tiling, different label scheme. The two are easy to confuse because both are
CEMS-derived multimodal flood sets; keep them separate.

Layout per sample:
  row 0: S1 GRD VV / VH, pre and post (the T=2 change-detection signal)
  row 1: S1 RTC VV / VH, pre and post (terrain-corrected variant of the same acquisitions)
  row 2: S2 L2A pre-event RGB + SWIR false-colour, cloud mask, DEM
  row 3: the label, its component floodmask / permwater / validity layers, and the label
         drawn over the post-event S1 VV so the AOI edge is visible against real imagery

TEMPORAL STRUCTURE. Unlike ImpactMesh's four phases, GEOID gives S1 at exactly two dates
(pre, post) and S2 only PRE-event -- a cloud-filtered composite. So there is no post-event
optical view: the change signal is S1-only. That is what makes this dataset a clean fit
for the S1 pre/post framing already used for UrbanSARFloods.

LABEL SEMANTICS (the part that silently breaks training if ignored):
    0   = background
    1   = permanent water
    2   = flooded water
    255 = IGNORE -- outside the CEMS-mapped area, NOT background
The 255 class is the analyst's delineation boundary clipped to the tile, exactly analogous
to ImpactMesh's -1. Folding it into class 0 scores predictions against labels that were
never drawn, so it is rendered explicitly (grey + hatch) here and must be passed as
ignore_index downstream.

S1 GRD/RTC ship as LINEAR sigma0, not dB. We convert with 10*log10 for display (and the
loader does the same), because the OlmoEarth S1 encoder was pretrained on dB.

Run (needs rasterio; any env with rasterio+matplotlib works):
    python -u -m exp.viz.visualize_geoid_flood
    python -u -m exp.viz.visualize_geoid_flood --root data/GEOID-Flood-full/geoid-flood
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

SAMPLE_ROOT = "data/GEOID-Flood/sample/geoid-flood"
OUT_DIR = "results/dataset_samples"

# Fixed, non-random selection spanning the label regimes you actually hit (percentages are
# of the 1024x1024 tile, measured from the sample split):
#   EMSR712-3-42   21.1% flood, 12.8% ignore  -- the headline case: lots of flood AND a
#                                                visible AOI edge, both classes readable.
#   EMSR712-3-49   11.6% flood, 50.4% ignore  -- HALF the tile is unmapped. Shows why 255
#                                                cannot be treated as background.
#   EMSR712-10-7    7.0% flood,  0.0% ignore  -- fully mapped tile, 5.6% permanent water:
#                                                the perm-water/flood distinction is the
#                                                whole difficulty of class 1 vs 2.
#   EMSR712-3-32   21.0% flood,  3.1% ignore  -- near-fully-mapped, flood-dominated.
SAMPLES = ["EMSR712-3-42", "EMSR712-3-49", "EMSR712-10-7", "EMSR712-3-32"]

CLASS_NAMES = {0: "background", 1: "permanent water", 2: "flooded water", 255: "ignore (unmapped)"}
CLASS_COLORS = ["#f2f2f2", "#1f77ff", "#e4002b"]     # 0, 1, 2
IGNORE_COLOR = "#9e9e9e"


def to_db(x: np.ndarray) -> np.ndarray:
    """Linear sigma0 -> dB. Zeros/negatives are nodata in GRD, so mask them to NaN first."""
    x = np.asarray(x, dtype=np.float32)
    out = np.full_like(x, np.nan)
    valid = np.isfinite(x) & (x > 0)
    out[valid] = 10.0 * np.log10(x[valid])
    return out


def stretch(x, lo=2, hi=98):
    """Percentile stretch to [0,1], NaN-safe (NaN -> 0)."""
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros_like(x)
    p_lo, p_hi = np.percentile(x[valid], [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    out = np.clip((x - p_lo) / (p_hi - p_lo), 0, 1)
    return np.nan_to_num(out, nan=0.0)


def find(root: Path, event: str, layer: str, tile: str, pas: str | None = None):
    """Locate one raster. Filenames carry an acquisition timestamp we don't know up front,
    so glob on the stable prefix and take the first match."""
    pat = f"{event}-{tile}_{layer}" + (f"_{pas}_*" if pas else "") + ".tif"
    hits = sorted(glob.glob(str(root / event / layer / pat)))
    if not hits and pas:
        hits = sorted(glob.glob(str(root / event / layer / f"{event}-{tile}_{layer}_{pas}_*.tif")))
    return Path(hits[0]) if hits else None


def read(path, band=None):
    if path is None:
        return None, {}
    with rasterio.open(path) as r:
        arr = r.read(band) if band else r.read()
    m = re.search(r"_(\d{8})T(\d{6})", Path(path).name)
    date = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else ""
    return arr, {"date": date}


def _blank(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10, color="#888")
    ax.set_xticks([]); ax.set_yticks([])


def render(root: Path, sample: str, out_dir: Path):
    event, tile = sample.rsplit("-", 1)

    grd_pre, m_gp = read(find(root, event, "s1grd", tile, "pre"))
    grd_post, m_gq = read(find(root, event, "s1grd", tile, "post"))
    rtc_pre, m_rp = read(find(root, event, "s1rtc", tile, "pre"))
    rtc_post, m_rq = read(find(root, event, "s1rtc", tile, "post"))
    s2, m_s2 = read(find(root, event, "s2l2a", tile, "pre"))
    cloud, _ = read(find(root, event, "cloudmask", tile, "pre"), band=1)
    dem, _ = read(find(root, event, "dem", tile), band=1)
    label, _ = read(find(root, event, "label", tile), band=1)
    floodmask, _ = read(find(root, event, "floodmask", tile), band=1)
    permwater, _ = read(find(root, event, "permwater", tile), band=1)
    validity, _ = read(find(root, event, "validity", tile), band=1)

    if label is None:
        print(f"  !! no label for {sample}, skipping")
        return

    tot = label.size
    frac = {c: 100.0 * int((label == c).sum()) / tot for c in (0, 1, 2, 255)}

    fig = plt.figure(figsize=(21, 21.5))
    gs = fig.add_gridspec(4, 4, hspace=0.22, wspace=0.06)

    # ---- row 0: S1 GRD, the pre/post change signal (dB) ----
    grd_panels = [
        (grd_pre, 0, f"GRD VV pre\n{m_gp.get('date','')}"),
        (grd_pre, 1, f"GRD VH pre\n{m_gp.get('date','')}"),
        (grd_post, 0, f"GRD VV post\n{m_gq.get('date','')}"),
        (grd_post, 1, f"GRD VH post\n{m_gq.get('date','')}"),
    ]
    for col, (arr, b, title) in enumerate(grd_panels):
        ax = fig.add_subplot(gs[0, col])
        if arr is None:
            _blank(ax, "s1grd missing")
        else:
            ax.imshow(stretch(to_db(arr[b])), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel("S1 GRD (dB)", fontsize=12, fontweight="bold")

    # ---- row 1: S1 RTC, same acquisitions, terrain-corrected ----
    rtc_panels = [
        (rtc_pre, 0, f"RTC VV pre\n{m_rp.get('date','')}"),
        (rtc_pre, 1, f"RTC VH pre\n{m_rp.get('date','')}"),
        (rtc_post, 0, f"RTC VV post\n{m_rq.get('date','')}"),
        (rtc_post, 1, f"RTC VH post\n{m_rq.get('date','')}"),
    ]
    for col, (arr, b, title) in enumerate(rtc_panels):
        ax = fig.add_subplot(gs[1, col])
        if arr is None:
            _blank(ax, "s1rtc missing")
        else:
            ax.imshow(stretch(to_db(arr[b])), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel("S1 RTC (dB)", fontsize=12, fontweight="bold")

    # ---- row 2: S2 (PRE-EVENT ONLY), cloud mask, DEM ----
    ax = fig.add_subplot(gs[2, 0])
    if s2 is not None:
        # S2 L2A band order is B01..B12 (12 bands): RGB = B04,B03,B02 -> idx 3,2,1
        ax.imshow(np.dstack([stretch(s2[3]), stretch(s2[2]), stretch(s2[1])]))
        ax.set_xticks([]); ax.set_yticks([])
    else:
        _blank(ax, "s2l2a missing")
    ax.set_title(f"S2 RGB (B4,B3,B2) — PRE only\n{m_s2.get('date','')}", fontsize=10, fontweight="bold")
    ax.set_ylabel("S2 L2A / static", fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[2, 1])
    if s2 is not None:
        # SWIR false colour B11,B8,B04 -> idx 10,7,3; open water goes dark
        ax.imshow(np.dstack([stretch(s2[10]), stretch(s2[7]), stretch(s2[3])]))
        ax.set_xticks([]); ax.set_yticks([])
    else:
        _blank(ax, "s2l2a missing")
    ax.set_title("S2 SWIR false-colour (B11,B8,B4)\nwater dark", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[2, 2])
    if cloud is not None:
        cc = ListedColormap(["#101010", "#ffffff", "#b0b0b0", "#5a5a5a"])
        ax.imshow(np.where(cloud == 255, 0, cloud), cmap=cc, vmin=0, vmax=3, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        pct = 100.0 * float((np.isin(cloud, (1, 2, 3))).sum()) / cloud.size
        ax.set_title(f"OmniCloudMask (pre)\n{pct:.1f}% cloud/shadow", fontsize=10, fontweight="bold")
    else:
        _blank(ax, "cloudmask missing")
        ax.set_title("OmniCloudMask (pre)", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[2, 3])
    if dem is not None:
        im = ax.imshow(dem, cmap="terrain")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        _blank(ax, "dem missing")
    ax.set_title("Copernicus DEM (m)", fontsize=10, fontweight="bold")

    # ---- row 3: label + its components + overlay ----
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    lab_vis = np.where(label == 255, 0, label)
    ign = label == 255

    ax = fig.add_subplot(gs[3, 0])
    ax.imshow(lab_vis, cmap=cmap, norm=norm, interpolation="nearest")
    # draw ignore explicitly on top: flat tint + hatch, so it survives greyscale printing
    ov = np.zeros((*label.shape, 4)); ov[ign] = (0.62, 0.62, 0.62, 0.95)
    ax.imshow(ov)
    if ign.any():
        ax.contourf(ign.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"])
        for c in ax.collections:
            c.set_edgecolor("#4d4d4d"); c.set_linewidth(0.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("LABEL (3-class + ignore)", fontsize=10, fontweight="bold")
    ax.set_ylabel("labels", fontsize=12, fontweight="bold")

    for col, (arr, name) in enumerate(
            [(floodmask, "floodmask (CEMS flood)"),
             (permwater, "permwater (permanent water)"),
             (validity, "validity (1 = imaged & mapped)")], start=1):
        ax = fig.add_subplot(gs[3, col])
        if arr is None:
            _blank(ax, f"{name} missing")
        else:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            name += f"\n{100.0*float((arr==1).sum())/arr.size:.1f}% set"
        ax.set_title(name, fontsize=10, fontweight="bold")

    handles = [Patch(color=CLASS_COLORS[c], label=f"{c}: {CLASS_NAMES[c]}  ({frac[c]:.1f}%)")
               for c in (0, 1, 2)]
    handles.append(Patch(facecolor=IGNORE_COLOR, hatch="//", edgecolor="#4d4d4d",
                         label=f"255: {CLASS_NAMES[255]}  ({frac[255]:.1f}%)\nNOT background"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.055))

    fig.suptitle(
        f"GEOID-Flood — {sample}   (event {event})\n"
        f"1024x1024 @10 m · S1 GRD+RTC pre/post (VV,VH) · S2 L2A PRE-only (12 bands) · DEM · 3-class label\n"
        f"background {frac[0]:.1f}%   permanent water {frac[1]:.1f}%   flood {frac[2]:.1f}%   "
        f"ignore/unmapped {frac[255]:.1f}%",
        fontsize=15, fontweight="bold", y=0.995)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"geoid_flood_{sample}.png"
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}   flood={frac[2]:.1f}% perm={frac[1]:.1f}% ignore={frac[255]:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=SAMPLE_ROOT,
                    help="tree root holding <event>/<layer>/*.tif (default: the HF sample/)")
    ap.add_argument("--samples", default=",".join(SAMPLES),
                    help="comma-separated <event>-<tile> ids")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    root = Path(args.root)
    for s in [x for x in args.samples.split(",") if x]:
        render(root, s, Path(args.out_dir))


if __name__ == "__main__":
    main()
