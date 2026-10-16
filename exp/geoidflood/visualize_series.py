"""Figures of GEOID S1-series samples (manyup/data/datasets/geoidflood_series.py).

One PNG per chip (val split: its fixed window), from different events, chips with a flood
fraction >= --min_flood:
  top      VV (dB) at the dates around GEOID's post scene, same grayscale for all panels;
           the post date is framed, orbit in the title
  bottom   the label; VV(post) - VV(previous date on the same orbit); median VV of each
           class over the whole 6-month window (flood pixels should drop at the flood)

    python -m exp.geoidflood.visualize_series --n 6
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from manyup.data import All  # noqa: E402
from manyup.data.datasets.geoidflood_series import GeoidSeriesDataset  # noqa: E402
from manyup.data.time import from_days  # noqa: E402

OUT = Path("results/geoidseries/samples")
# class colours (fixed per class): background neutral, permanent water blue, flood orange
BG, WATER, FLOOD, IGNORED = "#b9b8b3", "#2a78d6", "#eb6834", "#ffffff"
INK, MUTED = "#0b0b0b", "#52514e"
CLASSES = (("background", BG), ("permanent water", WATER), ("flood", FLOOD))
VMIN, VMAX = -25.0, 0.0                   # VV dB display range


def figure(s, path: Path, n_dates: int = 16):
    m = s.modalities["S1"]
    vv = m.data[:, 0].numpy()                                         # (T, H, W) dB
    ok = m.pixel_valid.numpy()
    days = m.time.start.numpy()
    orbit = m.aux["orbit"][:, 0, 0].numpy().astype(int)
    label = s.target.numpy()
    post = int(np.argmin(np.abs(days - s.meta["post_day"])))
    lo = max(0, min(post - n_dates // 2, len(days) - n_dates))
    shown = list(range(lo, min(lo + n_dates, len(days))))
    prev = [t for t in range(post) if orbit[t] == orbit[post]]
    same_orbit = bool(prev)
    prev = prev[-1] if prev else (post - 1 if post > 0 else None)   # else: previous date, any orbit

    fig = plt.figure(figsize=(16, 8.4), facecolor="white")
    gs = fig.add_gridspec(3, 9, height_ratios=[1, 1, 1.35], hspace=0.42, wspace=0.08,
                          width_ratios=[1] * 8 + [0.25])
    for i, t in enumerate(shown):
        ax = fig.add_subplot(gs[i // 8, i % 8])
        ax.imshow(np.where(ok[t], vv[t], np.nan), cmap="gray", vmin=VMIN, vmax=VMAX)
        d = from_days(int(days[t]))
        ax.set_title(f"{d:%Y-%m-%d} · orb {orbit[t]}", fontsize=8,
                     color=INK if t == post else MUTED, fontweight="bold" if t == post else None)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(FLOOD if t == post else "#dddcd7")
            sp.set_linewidth(3 if t == post else 0.8)

    ax = fig.add_subplot(gs[2, 0:2])
    lab = np.where(label == 255, 3, label)
    ax.imshow(lab, cmap=ListedColormap([BG, WATER, FLOOD, IGNORED]), vmin=0, vmax=3,
              interpolation="nearest")
    ax.set_title("label", fontsize=9, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CLASSES],
              labels=[n for n, _ in CLASSES], fontsize=7, loc="lower left",
              bbox_to_anchor=(0, -0.32), ncol=3, frameon=False, handlelength=1)

    ax = fig.add_subplot(gs[2, 2:4])
    if prev is not None:
        diff = np.where(ok[post] & ok[prev], vv[post] - vv[prev], np.nan)
        im = ax.imshow(diff, cmap="RdBu", vmin=-10, vmax=10)      # diverging, gray-ish midpoint
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
        cb.set_label("dB (blue: brighter, red: darker)", fontsize=7, color=MUTED)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        which = "same-orbit" if same_orbit else "(other-orbit)"
        ax.set_title(f"VV post − previous {which} date ({from_days(int(days[prev])):%m-%d})",
                     fontsize=9, color=INK)
    ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(gs[2, 5:9])
    dates = [from_days(int(d)) for d in days]
    for c, (name, col) in enumerate(CLASSES):
        sel = label == c
        if sel.sum() < 20:
            continue
        med = [np.median(vv[t][sel & ok[t]]) if (sel & ok[t]).sum() else np.nan
               for t in range(len(days))]
        ax.plot(dates, med, color=col, lw=2, marker="o", ms=3.5, label=name)
    ax.axvline(dates[post], color=MUTED, lw=1, ls="--")
    ax.text(dates[post], ax.get_ylim()[1], " GEOID post", fontsize=7, color=MUTED, va="top")
    ax.set_ylabel("median VV (dB)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color="#eeede9", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("median VV per class, whole window", fontsize=9, color=INK)

    frac = (label == 2).sum() / max((label != 255).sum(), 1)
    fig.suptitle(f"{s.id}  ·  {len(days)} dates in window  ·  flood {frac:.0%} of mapped pixels",
                 fontsize=11, color=INK, x=0.125, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--min_flood", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = GeoidSeriesDataset(a.split, {"S1": All()})
    rng = np.random.default_rng(a.seed)
    seen, made = set(), 0
    for i in rng.permutation(len(ds)):
        event = ds.sample_id(int(i)).split("-")[0]
        if event in seen:
            continue
        s = ds[int(i)]
        lab = s.target.numpy()
        mapped = (lab != 255).sum()
        if not mapped or (lab == 2).sum() / mapped < a.min_flood:
            continue
        seen.add(event)
        path = OUT / f"{a.split}_{s.id.replace('@', '_')}.png"
        figure(s, path)
        print(path, flush=True)
        made += 1
        if made >= a.n:
            break


if __name__ == "__main__":
    main()
