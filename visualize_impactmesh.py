"""Visualize ImpactMesh-Flood samples: all modalities, all 4 timesteps, + label mask.

Layout per sample: rows = modality view (S2 RGB, S2 false-colour SWIR, S1 VV, S1 VH),
        cols = the 4 timesteps (pre-month, pre-event, event, post-event).
Bottom row: static layers (DEM, flood mask), the class legend, and the mask over event RGB.

SAMPLES is a fixed, non-random list -- four test-split tiles chosen to span the failure modes
you actually hit when reading these labels:

  EMSR150_1_30UXE_x614855_y5973395  Northern England, Storm Eva. The original sample. Its
      "pre-event" S1 (2015-12-08) postdates Storm Desmond (4-6 Dec 2015), so the catchment is
      ALREADY flooded at t1 -- "pre-event" is relative to the EMS activation, not to
      hydrological conditions. 1.4% flood, 16.6% nodata.
  EMSR773_1_30SYJ_x715915_y4339925  Spain. 98.9% flood, 0% nodata -- a near-saturated tile,
      the opposite extreme, and a check that the -1 overlay draws nothing when there is no -1.
  EMSR561_2_36KYG_x718035_y8175475  Mozambique. 76.1% flood WITH 12.4% nodata -- both classes
      well represented, so the AOI edge is visible cutting through real flood.
  EMSR753_1_33PTP_x303465_y1309615  Nigeria. 7.3% flood, 77.1% nodata -- nodata-dominated. Read
      the hatched region as UNLABELLED, not as "no flood"; this is the tile that shows why.

The -1 class is Copernicus EMS AOI nodata: the analyst's delineation polygon clipped into a
square tile. It is genuinely unlabelled -- NOT background -- so it is drawn explicitly (grey +
diagonal hatch) in both the mask panel and the RGB overlay. Treating it as class 0 during
training or evaluation silently scores predictions against labels that were never made.
"""

import numpy as np
import rasterio
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

BASE = "/scratch/timz/rs-change-detection/data/ImpactMesh-Flood/data"
OUT_TMPL = "/scratch/timz/rs-change-detection/impactmesh_sample{suffix}.png"

# Fixed selection (see module docstring). Order is meaningful: original first, then
# saturated / mixed / nodata-dominated.
SAMPLES = [
    ("EMSR150_1_30UXE_x614855_y5973395", ""),
    ("EMSR773_1_30SYJ_x715915_y4339925", "_2"),
    ("EMSR561_2_36KYG_x718035_y8175475", "_3"),
    ("EMSR753_1_33PTP_x303465_y1309615", "_4"),
]

PHASES = ["pre-month", "pre-event", "event", "post-event"]

# -1 = AOI nodata (unlabelled), 0 = no flood, 1 = flood
NODATA_RGBA = (0.62, 0.62, 0.62, 0.75)


def stretch(x, lo=2, hi=98):
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros_like(x)
    p_lo, p_hi = np.percentile(x[valid], [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    return np.clip((x - p_lo) / (p_hi - p_lo), 0, 1)


def hatch_overlay(ax, sel, color, hatch, lw=0.0):
    """Draw `sel` (bool mask) as a hatched region via contourf.

    imshow of an RGBA layer can only give a flat tint; the AOI nodata region needs to be
    distinguishable from ordinary dark pixels even when printed greyscale, so it gets a
    hatch pattern on top of the tint.
    """
    if not sel.any():
        return
    ax.contourf(sel.astype(float), levels=[0.5, 1.5], colors="none",
                hatches=[hatch], extend="neither")
    # contourf hatch colour follows the edge colour of its collections
    for coll in ax.collections:
        coll.set_edgecolor(color)
        coll.set_linewidth(lw)


def render(sample, suffix):
    s1 = xr.open_zarr(f"{BASE}/S1RTC/{sample}_S1RTC.zarr.zip")
    s2 = xr.open_zarr(f"{BASE}/S2L2A/{sample}_S2L2A.zarr.zip")

    s1_dates = s1.attrs["datetime"].split(";")
    s2_dates = s2.attrs["datetime"].split(";")
    # orbit geometry changes between timesteps and makes naive t-differences encode
    # look-direction change as well as water -- surface it in the title.
    s1_orbits = s1.attrs.get("sat_orbit_state", "").split(";")
    s1_relorb = s1.attrs.get("sat_relative_orbit", "").split(";")
    s2_cloud = s2.attrs.get("eo_cloud_cover", "").split(";")

    s1_bands = list(s1.band.values)
    s2_bands = list(s2.band.values)
    s1_arr = s1.bands.values.astype(np.float32)  # (4, 2, 256, 256)
    s2_arr = s2.bands.values.astype(np.float32)  # (4, 12, 256, 256)

    with rasterio.open(f"{BASE}/MASK/{sample}_annotation_flood.tif") as r:
        mask = r.read(1)
    with rasterio.open(f"{BASE}/DEM/{sample}_DEM.tif") as r:
        dem = r.read(1).astype(np.float32)

    i_vv, i_vh = s1_bands.index("vv"), s1_bands.index("vh")
    i_r, i_g, i_b = s2_bands.index("B04"), s2_bands.index("B03"), s2_bands.index("B02")
    i_swir, i_nir = s2_bands.index("B11"), s2_bands.index("B08")

    fig = plt.figure(figsize=(19, 21))
    gs = fig.add_gridspec(5, 4, hspace=0.28, wspace=0.06)

    row_defs = [
        ("S2 RGB\n(B4,B3,B2)", "s2_rgb"),
        ("S2 SWIR false-colour\n(B11,B8,B4) — water dark", "s2_swir"),
        ("S1 VV (dB)", "s1_vv"),
        ("S1 VH (dB)", "s1_vh"),
    ]

    for row, (label, kind) in enumerate(row_defs):
        for t in range(4):
            ax = fig.add_subplot(gs[row, t])
            if kind == "s2_rgb":
                img = np.dstack([stretch(s2_arr[t, i_r]), stretch(s2_arr[t, i_g]),
                                 stretch(s2_arr[t, i_b])])
                ax.imshow(img)
                date = s2_dates[t][:10]
                if t < len(s2_cloud) and s2_cloud[t]:
                    date += f"  ({float(s2_cloud[t]):.0f}% cloud)"
            elif kind == "s2_swir":
                img = np.dstack([stretch(s2_arr[t, i_swir]), stretch(s2_arr[t, i_nir]),
                                 stretch(s2_arr[t, i_r])])
                ax.imshow(img)
                date = s2_dates[t][:10]
            elif kind == "s1_vv":
                ax.imshow(stretch(s1_arr[t, i_vv]), cmap="gray")
                date = s1_dates[t][:10]
                if t < len(s1_orbits) and s1_orbits[t]:
                    date += f"  ({s1_orbits[t][:3]}, rel {s1_relorb[t]})"
            else:
                ax.imshow(stretch(s1_arr[t, i_vh]), cmap="gray")
                date = s1_dates[t][:10]

            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t{t} — {PHASES[t]}\n{date}", fontsize=11, fontweight="bold")
            else:
                ax.set_title(date, fontsize=9)
            if t == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")

    # ---- bottom row: static layers + mask ----
    ax = fig.add_subplot(gs[4, 0])
    im = ax.imshow(dem, cmap="terrain")
    ax.set_title("DEM (Copernicus, m)", fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

    # flood mask: -1 = AOI nodata/unlabelled, 0 = no flood, 1 = flood
    cmap = ListedColormap(["#9e9e9e", "#1a3a5c", "#ff2d55"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax = fig.add_subplot(gs[4, 1])
    ax.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
    hatch_overlay(ax, mask == -1, "#4d4d4d", "//")
    ax.set_title("FLOOD MASK (per-pixel)\nhatched = AOI nodata", fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    n_inv = int((mask == -1).sum()); n_no = int((mask == 0).sum()); n_fl = int((mask == 1).sum())
    tot = mask.size
    handles = [
        plt.Line2D([0], [0], marker="s", ls="", ms=12, color="#ff2d55",
                   label=f"flood  ({n_fl:,} px, {100*n_fl/tot:.1f}%)"),
        plt.Line2D([0], [0], marker="s", ls="", ms=12, color="#1a3a5c",
                   label=f"no flood ({n_no:,} px, {100*n_no/tot:.1f}%)"),
        plt.Line2D([0], [0], marker="s", ls="", ms=12, color="#9e9e9e",
                   label=f"AOI nodata / -1 ({n_inv:,} px, {100*n_inv/tot:.1f}%)\n"
                         f"UNLABELLED — not background"),
    ]
    ax = fig.add_subplot(gs[4, 2])
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=11)
    ax.axis("off")

    # mask overlaid on the event-time S2 RGB -- both flood AND nodata, so the AOI edge is
    # not mistaken for label geometry.
    ax = fig.add_subplot(gs[4, 3])
    base_img = np.dstack([stretch(s2_arr[2, i_r]), stretch(s2_arr[2, i_g]),
                          stretch(s2_arr[2, i_b])])
    ax.imshow(base_img)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask == 1] = [1.0, 0.18, 0.33, 0.55]
    overlay[mask == -1] = NODATA_RGBA
    ax.imshow(overlay)
    hatch_overlay(ax, mask == -1, "#3a3a3a", "//")
    ax.set_title("mask over S2 RGB @ event\nred = flood, hatched grey = nodata",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"ImpactMesh-Flood — sample {sample}\n"
        f"S1 (2 bands) + S2 (12 bands) x 4 timesteps, 256x256 @10m, + DEM + per-pixel flood mask\n"
        f"flood {100*n_fl/tot:.1f}%   no-flood {100*n_no/tot:.1f}%   AOI nodata {100*n_inv/tot:.1f}%",
        fontsize=15, fontweight="bold", y=0.995,
    )
    out = OUT_TMPL.format(suffix=suffix)
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    print(f"  mask classes: flood={n_fl}, no-flood={n_no}, nodata={n_inv}")


if __name__ == "__main__":
    for sample, suffix in SAMPLES:
        render(sample, suffix)
