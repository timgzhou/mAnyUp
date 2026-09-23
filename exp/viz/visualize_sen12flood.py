"""Visualize one SEN12FLOOD scene: S2 RGB (B04,B03,B02) + S1 (VV,VH), before/after flood.

Scene 0140 has a clean FLOODING False->True transition:
  before: 2019-02-25 (S2)  ~ 2019-03-02 (S1)   FLOODING=False
  after : 2019-03-12 (S2)  ~ 2019-03-13 (S1)   FLOODING=True
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

BASE = "/scratch/timz/mAnyUp/data/sen12flood"
OUT = "/scratch/timz/mAnyUp/sen12flood_sample.png"

SCENE = "0140"
# (label, s2_date, s1_date)
STEPS = [
    ("before", "2019_02_25", "2019_03_02"),
    ("after", "2019_03_12", "2019_03_13"),
]


def s2_dir(date):
    return f"{BASE}/sen12floods_s2_source/sen12floods_s2_source/sen12floods_s2_source_{SCENE}_{date}"


def s1_dir(date):
    return f"{BASE}/sen12floods_s1_source/sen12floods_s1_source/sen12floods_s1_source_{SCENE}_{date}"


def s2_label(date):
    gj = (
        f"{BASE}/sen12floods_s2_labels/sen12floods_s2_labels/"
        f"sen12floods_s2_labels_{SCENE}_{date}/labels.geojson"
    )
    return json.load(open(gj))["properties"]["FLOODING"]


def read_band(path, out_hw=None):
    """Read a single-band tif, optionally resampling to out_hw=(H,W)."""
    with rasterio.open(path) as r:
        if out_hw is None:
            return r.read(1).astype(np.float32)
        return r.read(
            1, out_shape=out_hw, resampling=Resampling.bilinear
        ).astype(np.float32)


def stretch(x, lo=2, hi=98):
    """Per-channel percentile stretch to [0,1] for display."""
    p_lo, p_hi = np.percentile(x, [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    return np.clip((x - p_lo) / (p_hi - p_lo), 0, 1)


def load_s2_rgb(date):
    d = s2_dir(date)
    # 10m bands are 512x512; use that as the target grid
    with rasterio.open(f"{d}/B04.tif") as r:
        H, W = r.height, r.width
    r_ = read_band(f"{d}/B04.tif", (H, W))
    g_ = read_band(f"{d}/B03.tif", (H, W))
    b_ = read_band(f"{d}/B02.tif", (H, W))
    rgb = np.dstack([stretch(r_), stretch(g_), stretch(b_)])
    return rgb


def load_s1(date):
    d = s1_dir(date)
    vv = read_band(f"{d}/VV.tif")  # dB float32
    vh = read_band(f"{d}/VH.tif")
    return stretch(vv), stretch(vh)


fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for row, (tag, s2d, s1d) in enumerate(STEPS):
    flooding = s2_label(s2d)
    rgb = load_s2_rgb(s2d)
    vv, vh = load_s1(s1d)

    # false-color S1 composite: R=VV, G=VH, B=VV/VH-ish
    s1_rgb = np.dstack([vv, vh, stretch(vv - vh)])

    panels = [
        (rgb, f"S2 RGB (B4,B3,B2)\n{s2d.replace('_','-')}"),
        (s1_rgb, f"S1 (VV,VH,VV-VH)\n{s1d.replace('_','-')}"),
        (vv, f"S1 VV\n{s1d.replace('_','-')}"),
        (vh, f"S1 VH\n{s1d.replace('_','-')}"),
    ]
    for col, (img, title) in enumerate(panels):
        ax = axes[row, col]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    color = "red" if str(flooding).lower() in ("true", "1") else "green"
    axes[row, 0].set_ylabel(
        f"{tag.upper()}\nFLOODING={flooding}",
        fontsize=13,
        color=color,
        fontweight="bold",
        rotation=0,
        labelpad=55,
        va="center",
    )

fig.suptitle(
    f"SEN12FLOOD scene {SCENE} — image-level flood label (polygon = tile footprint, not a mask)",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches="tight")
print("saved:", OUT)
