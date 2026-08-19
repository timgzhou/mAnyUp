"""Plot lp_pa2px vs. lp_pa2px_ens across the patch-size sweep (PASTIS, OlmoEarth base).

Same configs as exp/viz/plot_lp_patchsize.py, but two curves instead of one:
  lp_pa2px      -- a single probe on the time-averaged features
  lp_pa2px_ens  -- temporal ensemble: one probe per timestep, predictions averaged

Both curves on one panel, with the gap between them shaded. x runs coarse -> fine (largest
patch size / smallest token grid on the left), matching the column order of the feature-map
figures in visualize_features.py.

See plot_lp_patchsize.py's docstring for why ps=1 uses a tile32 extraction (there is no
ps1_tile64 run); that point is drawn hollow.

    source env_setup/env_olmo.sh    # (or any env with matplotlib)
    python -u -m exp.viz.plot_lp_ens_vs_patchsize
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless cluster: never try an interactive backend
import matplotlib.pyplot as plt

from exp.viz.plot_lp_patchsize import (DEFAULT_CONFIGS, EXACT_TILE, RESULTS_CSV,
                                       load_rows, pick)

SINGLE, ENS = "lp_pa2px", "lp_pa2px_ens"
C_SINGLE, C_ENS, C_GAIN = "#2b6cb0", "#c05621", "#2f855a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default=RESULTS_CSV)
    ap.add_argument("--metric", default="test_miou",
                    choices=["test_miou", "test_overall_acc"])
    ap.add_argument("--out", default="feature_viz/lp_pa2px_ens_vs_patchsize.png")
    args = ap.parse_args()

    csv_path = Path(args.results_csv)
    rows_s = load_rows(csv_path, SINGLE)
    rows_e = load_rows(csv_path, ENS)

    xs, ys_s, ys_e, is_sub = [], [], [], []
    # reversed(): plot coarse -> fine (ps16 leftmost), so resolution increases to the right.
    for ps, feat in reversed(DEFAULT_CONFIGS):
        v_s, v_e = pick(rows_s, feat, args.metric), pick(rows_e, feat, args.metric)
        if v_s is None or v_e is None:
            missing = SINGLE if v_s is None else ENS
            print(f"WARNING: no {missing} row for {feat}; omitting ps={ps}")
            continue
        xs.append(ps); ys_s.append(v_s); ys_e.append(v_e)
        is_sub.append(not feat.endswith(f"_tile{EXACT_TILE}"))
        print(f"  ps={ps:<3} {feat:<26} {SINGLE}={v_s:.3f}  {ENS}={v_e:.3f}  "
              f"delta={v_e - v_s:+.3f}")

    if not xs:
        raise SystemExit("ERROR: no configs with BOTH heads present; nothing to plot")

    metric_label = {"test_miou": "test mIoU",
                    "test_overall_acc": "test overall accuracy"}[args.metric]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # CATEGORICAL x positions (0..n-1) rather than the patch size itself: the configs are
    # evenly spaced this way, and the ps/grid pair is spelled out in the tick labels.
    px = list(range(len(xs)))

    # --- the two curves, with the gap shaded ---
    ax.fill_between(px, ys_s, ys_e, color=C_GAIN, alpha=0.13, zorder=1)
    ax.plot(px, ys_s, "-", color=C_SINGLE, lw=2, zorder=2, label="lp_pa2px (single probe)")
    ax.plot(px, ys_e, "-", color=C_ENS, lw=2, zorder=2, label="lp_pa2px_ens (temporal ensemble)")
    for x, y_s, y_e, sub in zip(px, ys_s, ys_e, is_sub):
        for y, c in ((y_s, C_SINGLE), (y_e, C_ENS)):
            ax.plot(x, y, "o", ms=8, zorder=3, color=c,
                    markerfacecolor="white" if sub else c, mew=2)
        ax.annotate(f"{y_e:.2f}", (x, y_e), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=C_ENS)
        ax.annotate(f"{y_s:.2f}", (x, y_s), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.5, color=C_SINGLE)

    ax.set_ylabel(metric_label)
    ax.set_title("PASTIS linear probe: temporal ensembling vs. patch size\n"
                 "OlmoEarth base, Sentinel-2, frozen encoder", fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(y=0.20)
    handles, labels = ax.get_legend_handles_labels()
    if any(is_sub):
        hollow = plt.Line2D([], [], marker="o", ls="none", ms=8, color="gray",
                            markerfacecolor="white", mew=2)
        handles, labels = handles + [hollow], labels + [f"no tile{EXACT_TILE} run (see docstring)"]
    ax.legend(handles, labels, fontsize=8.5, loc="lower right", framealpha=0.95)

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
