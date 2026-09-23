"""PASTIS mIoU vs END-TO-END TRAINING COST (wall-clock GPU time).

Companion to plot_miou_vs_speed.py, which plots mIoU against *inference* arithmetic cost
(GMACs/sample). That axis answers "what does one forward pass cost"; this one answers
"what did it cost to produce this point at all" -- the number you pay once to get a
trained model.

The two regimes pay very different bills, so the total is assembled differently:
  LP  (frozen backbone): extract features ONCE (the dominant term), then train a linear
      probe on the cached tensors. total = extraction wall time + probe train time.
  FFT (full finetune):   no cached features -- the backbone runs every epoch with
      gradients. total = the job's wall time, end to end.

Reading extraction as part of LP's cost is the honest comparison: without it there is
nothing to probe. Source numbers and their provenance live in results/bench/train_time.csv.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/bench/train_time.csv"
OUT = "results/bench/miou_vs_train_time.png"

# Shape = patch size, matching plot_miou_vs_speed.py so the two figures read together.
PS_MARKER = {1: "o", 2: "s", 4: "^", 8: "D", 16: "v"}
LP_COLOR = "#00a05a"      # tile-64 green, the only tile in this figure
FFT_COLOR = "#7d3cc6"     # same purple as the GMACs plot's finetune stars


def fmt_time(sec: float) -> str:
    if sec < 3600:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.1f} h"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=CSV)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    fig, ax = plt.subplots(figsize=(11, 7))
    xs_all, ys_all = [], []
    placed: list[tuple[float, float]] = []
    xs_all_pre = [float(r["total_sec"]) / 3600.0 for r in rows]

    for r in rows:
        x = float(r["total_sec"]) / 3600.0        # hours
        y = float(r["test_miou"])
        ps = int(r["patch_size"])
        fft = r["regime"] == "fft"
        xs_all.append(x)
        ys_all.append(y)
        ax.scatter(x, y, s=340 if fft else 150,
                   color=FFT_COLOR if fft else LP_COLOR,
                   marker="*" if fft else PS_MARKER.get(ps, "o"),
                   edgecolor="black", linewidth=0.7, zorder=4 if fft else 3)

        # Stack labels that would overprint (log x, so compare in log space).
        n_hit = sum(1 for px, py in placed
                    if abs(np.log10(x) - np.log10(px)) < 0.10 and abs(y - py) < 0.015)
        # The rightmost point would run its label off-axis, so anchor that one to the left.
        right_edge = x > 0.6 * max(xs_all_pre)
        dx, ha = ((-11, "right") if right_edge or n_hit else (9, "left"))
        # Break out the two LP terms; FFT has a single wall-clock number.
        if fft:
            detail = fmt_time(float(r["total_sec"]))
        else:
            detail = (f"{fmt_time(float(r['total_sec']))}\n"
                      f"(extract {fmt_time(float(r['extract_sec']))}"
                      f" + probe {fmt_time(float(r['train_sec']))})")
        dy = 8 - 26 * n_hit
        ax.annotate(f"{r['label']}\n{detail}", xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, linespacing=1.3, ha=ha,
                    color=FFT_COLOR if fft else "black",
                    fontweight="bold" if fft else "normal")
        placed.append((x, y))

    ax.set_xscale("log")
    ax.set_xlabel("end-to-end training cost (GPU-hours, log)")
    ax.set_ylabel("test mIoU (lp_pa2px / finetune)")
    ax.set_title("Accuracy vs training cost\nupper-LEFT is better (x is cost)")
    ax.grid(alpha=0.3, which="both")

    handles = [plt.Line2D([], [], ls="", marker=PS_MARKER[ps], ms=9, color=LP_COLOR,
                          mec="black", label=f"LP patch {ps} (tile 64, S2)")
               for ps in (16, 8, 4)]
    handles.append(plt.Line2D([], [], ls="", marker="*", ms=15, color=FFT_COLOR,
                              mec="black", label="full finetune (s2s1, img64, 64 ep)"))
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle("PASTIS accuracy vs end-to-end training cost "
                 "(OlmoEarth-base, tile 64, image_size 64)\n"
                 "LP = feature extraction + probe training;  FFT = full finetune wall time",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
