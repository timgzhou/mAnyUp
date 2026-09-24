"""Plot lp_pa2px vs. anyup across the patch-size sweep (PASTIS, OlmoEarth base).

Same style as exp/viz/plot_lp_ens_vs_patchsize.py, so the patch-size figures in results/pastis/feature_viz/
read as one series:
  lp_pa2px  -- linear probe straight on the (time-averaged) low-res features
  anyup     -- AnyUp guided upsampling to 64x64 before the same per-pixel probe

Restricted to PLOT_PATCH_SIZES (ps8/ps4/ps2) -- the middle of the sweep, where both heads are
clean tile64 extractions. Widen that set to bring ps16/ps1 back.

x runs coarse -> fine, matching the feature-map column order in visualize_features.py.

    source env_setup/env_olmo.sh    # (or any env with matplotlib)
    python -u -m exp.viz.plot_lp_anyup_vs_patchsize
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless cluster: never try an interactive backend
import matplotlib.pyplot as plt

from exp.viz.plot_lp_patchsize import DEFAULT_CONFIGS, RESULTS_CSV, load_rows, pick

BASE, ANYUP = "lp_pa2px", "anyup"
C_BASE, C_ANYUP = "#2b6cb0", "#c05621"
C_FILL = "#2f855a"

PLOT_PATCH_SIZES = (8, 4, 2)   # subset of DEFAULT_CONFIGS to draw; see module docstring


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default=RESULTS_CSV)
    ap.add_argument("--metric", default="test_miou",
                    choices=["test_miou", "test_overall_acc"])
    ap.add_argument("--out", default="results/pastis/feature_viz/lp_anyup_vs_patchsize.png")
    args = ap.parse_args()

    csv_path = Path(args.results_csv)
    rows_b = load_rows(csv_path, BASE)
    rows_a = load_rows(csv_path, ANYUP)

    # reversed(): plot coarse -> fine, so resolution increases to the right.
    xs, ys_b, ys_a = [], [], []
    for ps, feat in reversed(DEFAULT_CONFIGS):
        if ps not in PLOT_PATCH_SIZES:
            continue
        v_b, v_a = pick(rows_b, feat, args.metric), pick(rows_a, feat, args.metric)
        if v_b is None or v_a is None:
            missing = BASE if v_b is None else ANYUP
            print(f"WARNING: no {missing} row for {feat}; omitting ps={ps}")
            continue
        xs.append(ps); ys_b.append(v_b); ys_a.append(v_a)
        print(f"  ps={ps:<3} {feat:<26} {BASE}={v_b:.3f}  {ANYUP}={v_a:.3f}"
              f"  delta={v_a - v_b:+.3f}")

    if not xs:
        raise SystemExit(f"ERROR: no configs with BOTH {BASE} and {ANYUP}; nothing to plot")

    metric_label = {"test_miou": "test mIoU",
                    "test_overall_acc": "test overall accuracy"}[args.metric]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # CATEGORICAL x positions (0..n-1): the configs are evenly spaced this way, and the
    # ps/grid pair is spelled out in the tick labels.
    px = list(range(len(xs)))

    ax.fill_between(px, ys_b, ys_a, color=C_FILL, alpha=0.13, zorder=1)
    ax.plot(px, ys_b, "-", color=C_BASE, lw=2, zorder=2, label="lp_pa2px (low-res probe)")
    ax.plot(px, ys_a, "-", color=C_ANYUP, lw=2, zorder=2, label="anyup (guided upsample -> 64x64)")
    for x, y_b, y_a in zip(px, ys_b, ys_a):
        for y, c in ((y_b, C_BASE), (y_a, C_ANYUP)):
            ax.plot(x, y, "o", ms=8, zorder=3, color=c, mew=2)
        # Label above/below whichever curve is on top, so the two never collide.
        hi, lo = max(y_b, y_a), min(y_b, y_a)
        c_hi = C_ANYUP if y_a >= y_b else C_BASE
        c_lo = C_BASE if y_a >= y_b else C_ANYUP
        ax.annotate(f"{hi:.2f}", (x, hi), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=c_hi)
        ax.annotate(f"{lo:.2f}", (x, lo), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.5, color=c_lo)

    ax.set_ylabel(metric_label)
    ax.set_title("PASTIS linear probe: feature upsampling vs. patch size\n"
                 "OlmoEarth base, Sentinel-2, frozen encoder", fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(y=0.20)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

    ax.set_xlabel("encoder patch size  (token grid = 64/ps per side)      coarse → fine")
    ax.set_xticks(px)
    ax.set_xticklabels([f"ps{x}\n{64 // x}x{64 // x}" for x in xs])
    ax.margins(x=0.08)
    # The CSV stores mIoU rounded to 2dp (lp_cached_features.py writes f"{miou:.2f}"), so every
    # plotted value carries about +-0.01; small differences between configs are not resolved.
    fig.text(0.5, -0.035, "mIoU is recorded to 2 decimals, so each value carries about +-0.01; "
             "small differences between points are not resolved.",
             ha="center", fontsize=7.5, color="#4a5568", style="italic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
