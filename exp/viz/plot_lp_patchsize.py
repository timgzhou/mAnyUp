"""Plot lp_pa2px linear-probe accuracy vs. encoder patch size (PASTIS, OlmoEarth base).

Reads results/pastis/lp_olmoearth_pastis.csv and plots the patch-size sweep at fixed
tile_size=64 -- the configs visualized in exp/viz/visualize_features.py -- so the metric
curve lines up with the feature maps in that figure.

DEFAULT_CONFIGS uses ps1_tile32 for the ps=1 point. tile_size only controls how the image is
split for encoding (smaller tiles are an approximation: independent sub-tiles do not
cross-attend), and the ps1 rows span 0.49-0.52 mIoU across tile1/tile8/tile32/tile64. tile32
sits with tile8/tile64 at the top of that range, whereas tile1 (0.49) is the outlier -- at
tile1 every 1x1 tile is encoded alone, so there is no spatial context at all. The point is
still drawn hollow and called out, because it IS a different extraction setting than the
tile64 configs used for every other patch size.

    source env_setup/env_olmo.sh    # (or any env with matplotlib)
    python -u -m exp.viz.plot_lp_patchsize
    python -u -m exp.viz.plot_lp_patchsize --metric test_overall_acc
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless cluster: never try an interactive backend
import matplotlib.pyplot as plt

RESULTS_CSV = "results/pastis/lp_olmoearth_pastis.csv"

# (patch_size, feature dir). Mirrors the --dirs list used for the feature-map figure.
DEFAULT_CONFIGS = [
    (1, "oe_base_s2_ps1_tile32"),       # tile32 stand-in; see module docstring
    (2, "oe_base_s2_ps2_tile64"),
    (4, "oe_base_s2_ps4_tile64"),
    (8, "oe_base_s2_ps8_tile64"),
    (16, "oe_base_s2_ps16_tile64"),
]
EXACT_TILE = 64                # configs at this tile_size are the clean comparison


def load_rows(csv_path: Path, head_mode: str) -> list[dict]:
    with open(csv_path) as f:
        return [r for r in csv.DictReader(f) if r["head_mode"] == head_mode]


def pick(rows: list[dict], features: str, metric: str) -> float | None:
    """Latest value of `metric` for `features` (CSV is append-only, so last row wins)."""
    hits = [r for r in rows if r["features"] == features and r.get(metric)]
    if not hits:
        return None
    return float(sorted(hits, key=lambda r: r["timestamp"])[-1][metric])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default=RESULTS_CSV)
    ap.add_argument("--head_mode", default="lp_pa2px")
    ap.add_argument("--metric", default="test_miou",
                    choices=["test_miou", "test_overall_acc"])
    ap.add_argument("--out", default="feature_viz/lp_pa2px_vs_patchsize.png")
    args = ap.parse_args()

    rows = load_rows(Path(args.results_csv), args.head_mode)
    if not rows:
        raise SystemExit(f"ERROR: no {args.head_mode} rows in {args.results_csv}")

    xs, ys, names, is_sub = [], [], [], []
    for ps, feat in DEFAULT_CONFIGS:
        val = pick(rows, feat, args.metric)
        if val is None:
            print(f"WARNING: no {args.head_mode} row for {feat}; omitting ps={ps}")
            continue
        xs.append(ps); ys.append(val); names.append(feat)
        is_sub.append(not feat.endswith(f"_tile{EXACT_TILE}"))
        print(f"  ps={ps:<3} {feat:<26} {args.metric}={val:.3f}")

    if not xs:
        raise SystemExit("ERROR: no matching rows to plot")

    metric_label = {"test_miou": "test mIoU",
                    "test_overall_acc": "test overall accuracy"}[args.metric]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, ys, "-", color="#2b6cb0", lw=2, zorder=2)
    # Solid = tile64 (like-for-like); hollow = the ps1_tile1 stand-in.
    for x, y, sub in zip(xs, ys, is_sub):
        ax.plot(x, y, "o", ms=9, zorder=3, color="#2b6cb0",
                markerfacecolor="white" if sub else "#2b6cb0", mew=2)

    # Token grid is 64/ps per side -- the quantity the patch size actually controls.
    for x, y in zip(xs, ys):
        g = 64 // x
        ax.annotate(f"{y:.2f}\n{g}x{g}", (x, y), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9,
                    color="#1a365d", linespacing=1.25)

    ax.set_xscale("log", base=2)
    # Descending patch size left to right, i.e. coarse -> fine features, matching the column
    # order of the feature-map figures. invert_xaxis() rather than reordering the data: the
    # axis is numeric/log2, so the spacing stays proportional to patch size either way.
    ax.invert_xaxis()
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ps{x}" for x in xs])
    ax.set_xlabel("encoder patch size  (token grid = 64/ps per side)      coarse → fine")
    ax.set_ylabel(metric_label)
    ax.set_title(f"PASTIS linear probe ({args.head_mode}) vs. patch size\n"
                 f"OlmoEarth base, Sentinel-2, frozen encoder", fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(y=0.22)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
