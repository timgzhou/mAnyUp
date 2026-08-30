"""Plot the GEOID-Flood patch-size study: ps8tile128 vs ps4tile64 vs ps1tile16.

Reads results/geoidflood/lp.csv (written by exp/geoidflood/lp.py) and draws the one
comparison the study is for: at a FIXED 16x16 token grid, how does flood IoU move as each
token's ground footprint goes 80 m -> 40 m -> 10 m?

Three panels:
  1. headline mIoU(BG, FL) vs metres-per-token, one line per head (concat / diff)
  2. per-class IoU (BG / PW / FL) as grouped bars per config -- shows whether a config's
     headline is carried by background or by the flood class that actually matters
  3. flood precision vs recall per config -- a coarse-token config can hold IoU while
     trading sharp boundaries for blanket over-prediction, and only P/R exposes that

Run:
    python -u -m exp.viz.plot_geoid_patchsize
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/geoidflood/lp.csv"
OUT = "results/geoidflood/patchsize_study.png"

HEAD_STYLE = {"concat": ("#1f77ff", "o", "-"), "diff": ("#e4002b", "s", "--")}
CLASS_COLORS = {"BG": "#9e9e9e", "PW": "#1f77ff", "FL": "#e4002b"}


def load(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("metres_per_token", "patch_size", "tile_size", "grid"):
            r[k] = int(r[k]) if r.get(k) not in (None, "") else None
        for k in list(r):
            if k.startswith(("miou", "iou_", "BG_", "PW_", "FL_", "overall", "macro")):
                r[k] = float(r[k]) if r[k] not in (None, "") else float("nan")
    # keep the LAST row per (features, head) -- a re-run appends rather than replaces
    dedup = {}
    for r in rows:
        dedup[(r["features"], r["head"])] = r
    return list(dedup.values())


def cfg_label(r) -> str:
    return f"ps{r['patch_size']}tile{r['tile_size']}\n{r['metres_per_token']} m/token"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = load(Path(args.csv))
    if not rows:
        print(f"no rows in {args.csv}")
        return
    heads = sorted({r["head"] for r in rows})
    # x axis is metres-per-token; order coarse -> fine so the "more context" end is left
    configs = sorted({r["metres_per_token"] for r in rows}, reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # -- panel 1: headline vs m/token --
    ax = axes[0]
    for head in heads:
        color, marker, ls = HEAD_STYLE.get(head, ("#333", "^", ":"))
        xs, ys = [], []
        for m in configs:
            hit = [r for r in rows if r["head"] == head and r["metres_per_token"] == m]
            if hit:
                xs.append(m); ys.append(hit[0]["miou_BG_FL"])
        ax.plot(range(len(xs)), ys, marker=marker, ls=ls, color=color, lw=2, ms=9,
                label=f"head={head}")
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"{y:.3f}", (i, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, color=color)
    ref = [r for r in rows if r["metres_per_token"] == configs[0]]
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([cfg_label(next(r for r in rows if r["metres_per_token"] == m))
                        for m in configs], fontsize=9)
    ax.set_ylabel("val mIoU (background, flood)", fontsize=11)
    ax.set_title("Headline: mIoU(BG, FL)\nfixed 16x16 token grid in every config",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=10)

    # -- panel 2: per-class IoU bars --
    ax = axes[1]
    head0 = "concat" if "concat" in heads else heads[0]
    sub = [next(r for r in rows if r["head"] == head0 and r["metres_per_token"] == m)
           for m in configs
           if any(r["head"] == head0 and r["metres_per_token"] == m for r in rows)]
    width, cls = 0.26, ["BG", "PW", "FL"]
    x = np.arange(len(sub))
    for i, c in enumerate(cls):
        vals = [r[f"{c}_iou"] for r in sub]
        ax.bar(x + (i - 1) * width, vals, width, label=c, color=CLASS_COLORS[c])
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg_label(r) for r in sub], fontsize=9)
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_title(f"Per-class IoU (head={head0})\nPW is not a change signal — reported apart",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=10)

    # -- panel 3: flood precision vs recall --
    ax = axes[2]
    for head in heads:
        color, marker, _ = HEAD_STYLE.get(head, ("#333", "^", ":"))
        for m in configs:
            hit = [r for r in rows if r["head"] == head and r["metres_per_token"] == m]
            if not hit:
                continue
            r = hit[0]
            ax.scatter(r["FL_recall"], r["FL_precision"], s=150, color=color,
                       marker=marker, edgecolor="k", zorder=3,
                       label=f"head={head}" if m == configs[0] else None)
            ax.annotate(f"ps{r['patch_size']}t{r['tile_size']}",
                        (r["FL_recall"], r["FL_precision"]),
                        textcoords="offset points", xytext=(7, -4), fontsize=9)
    ax.set_xlabel("flood recall", fontsize=11)
    ax.set_ylabel("flood precision", fontsize=11)
    ax.set_title("Flood class: precision vs recall\ncoarse tokens can buy IoU with "
                 "over-prediction", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=10)

    fig.suptitle("GEOID-Flood — OlmoEarth frozen-feature patch-size study "
                 "(ps8tile128 vs ps4tile64 vs ps1tile16)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"saved: {args.out}")

    # also print the table, so a run leaves a readable summary in the log
    print(f"\n{'config':16s} {'head':7s} {'m/tok':>6s} {'mIoU(BG,FL)':>12s} "
          f"{'FL IoU':>8s} {'FL P':>7s} {'FL R':>7s} {'PW IoU':>8s}")
    for r in sorted(rows, key=lambda r: (-r["metres_per_token"], r["head"])):
        print(f"ps{r['patch_size']}tile{r['tile_size']:<10d} {r['head']:7s} "
              f"{r['metres_per_token']:6d} {r['miou_BG_FL']:12.4f} {r['FL_iou']:8.4f} "
              f"{r['FL_precision']:7.4f} {r['FL_recall']:7.4f} {r['PW_iou']:8.4f}")


if __name__ == "__main__":
    main()
