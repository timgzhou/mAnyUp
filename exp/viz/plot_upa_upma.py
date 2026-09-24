"""Grouped bars: lr_bilinear vs. UPA vs. UPMA vs. AnyUp, per patch size (PASTIS, OlmoEarth base).

Reads results/upsamplers/upsampler_pa2pa.csv -- written by exp/upsamplers/eval_pa2pa.py, which
is a DIFFERENT protocol from the lp_* sweeps in results/pastis/lp_olmoearth_pastis.csv: one
pa2pa head is trained on the low-res features, then each upsampler is swapped in at eval time.
So the honest within-figure baseline is that run's own lr_bilinear control, not lp_pa2px.

Bars, not a line sweep: only ps16 and ps4 have been evaluated, and two points per method would
draw a trend the data cannot support. Groups run coarse -> fine left to right, matching the
other patch-size figures.

lp_pa2px is drawn as a dashed reference line per group ONLY when --lp_ref is passed: it is a
separately trained head from the other CSV (2-decimal mIoU), so it is a rough sanity marker,
not a comparable bar.

--eval_mode selects which protocol to draw, and the two are never mixed in one figure:
  frozen_shared (default) -- one head trained on LR tokens, applied to every upsampler
  retrained               -- a per-method probe trained on that method's own upsampled features
                             (eval_pa2pa.py --retrain_head), which removes the asymmetry where
                             UPA/UPMA are judged by a head that never saw upsampled features
They write to different default filenames, so both can coexist in results/pastis/feature_viz/.

    source env_setup/env_olmo.sh    # (or any env with matplotlib)
    python -u -m exp.viz.plot_upa_upma
    python -u -m exp.viz.plot_upa_upma --n_test 64            # the quick-eval subset instead
    python -u -m exp.viz.plot_upa_upma --eval_mode retrained  # the per-method-probe protocol
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless cluster: never try an interactive backend
import matplotlib.pyplot as plt

from exp.viz.plot_lp_patchsize import RESULTS_CSV as LP_CSV, load_rows, pick

UPS_CSV = "results/upsamplers/upsampler_pa2pa.csv"

# Ordered coarse -> fine, matching the other patch-size figures. Only these two configs have
# been run through eval_pa2pa.py.
CONFIGS = [(16, "oe_base_s2_ps16_tile64"), (4, "oe_base_s2_ps4_tile64")]
METHODS = ["lr_bilinear", "upa", "upma", "anyup"]
COLORS = {"lr_bilinear": "#a0aec0", "upa": "#2b6cb0", "upma": "#6b46c1", "anyup": "#c05621"}
LABELS = {"lr_bilinear": "lr_bilinear (control)", "upa": "UPA",
          "upma": "UPMA (multispectral guide)", "anyup": "AnyUp"}


def load_ups(csv_path: Path, n_test: str, time_pool: str, eval_mode: str) -> list[dict]:
    """Rows for one eval setting and one protocol, with the known-corrupt rows dropped.

    eval_mode keeps the two protocols apart -- frozen_shared (one LR-trained head for every
    method) and retrained (a per-method probe on that method's own upsampled features). Mixing
    them in one figure would compare heads, not upsamplers. Rows predating the column are
    frozen_shared.

    The 2026-08-11T01:32:32 upma/anyup pair is field-shifted (acc=57.8, n_test=0.3136 -- the
    mIoU column holds what is really the accuracy, etc.). Those exact values were re-run clean
    on 08-12, so the guard is a sanity check on the columns rather than a timestamp blocklist:
    anything whose numeric fields are out of range is not a usable row."""
    out = []
    for r in csv.DictReader(open(csv_path)):
        try:
            miou, acc, n = float(r["test_miou"]), float(r["test_overall_acc"]), float(r["n_test"])
        except ValueError:
            continue
        if not (0 <= miou <= 1 and 0 <= acc <= 1 and n >= 1 and n == int(n)):
            print(f"WARNING: dropping malformed row {r['timestamp']} {r['method']} "
                  f"(miou={r['test_miou']} acc={r['test_overall_acc']} n_test={r['n_test']})")
            continue
        if r["n_test"] != n_test:
            continue
        if (r.get("eval_mode") or "frozen_shared") != eval_mode:
            continue
        # lr_bilinear never touches the guidance image, so it carries no time_pool value.
        if r["method"] != "lr_bilinear" and r["time_pool"] != time_pool:
            continue
        out.append(r)
    return out


def pick_ups(rows: list[dict], features: str, method: str, metric: str) -> float | None:
    """Latest value of `metric` for (features, method) -- the CSV is append-only."""
    hits = [r for r in rows if r["features"] == features and r["method"] == method
            and r.get(metric)]
    if not hits:
        return None
    return float(sorted(hits, key=lambda r: r["timestamp"])[-1][metric])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default=UPS_CSV)
    ap.add_argument("--metric", default="test_miou",
                    choices=["test_miou", "test_overall_acc"])
    ap.add_argument("--n_test", default="1984",
                    help="which eval size to plot (1984 = full test split, 64 = quick subset)")
    ap.add_argument("--time_pool", default="mean", choices=["mean", "median"])
    ap.add_argument("--eval_mode", default="frozen_shared",
                    choices=["frozen_shared", "retrained"],
                    help="which protocol to plot; see load_ups. 'retrained' needs rows from "
                         "eval_pa2pa.py --retrain_head")
    ap.add_argument("--lp_ref", action="store_true",
                    help="overlay lp_pa2px from the lp CSV as a dashed per-group reference "
                         "(different protocol -- see module docstring)")
    ap.add_argument("--out", default=None,
                    help="default: results/pastis/feature_viz/upa_upma_vs_patchsize[_retrained].png, so the "
                         "two protocols never overwrite each other")
    args = ap.parse_args()
    if args.out is None:
        suffix = "_retrained" if args.eval_mode == "retrained" else ""
        args.out = f"results/pastis/feature_viz/upa_upma_vs_patchsize{suffix}.png"

    rows = load_ups(Path(args.results_csv), args.n_test, args.time_pool, args.eval_mode)
    if not rows:
        raise SystemExit(f"ERROR: no rows in {args.results_csv} with n_test={args.n_test}, "
                         f"time_pool={args.time_pool}, eval_mode={args.eval_mode}")

    vals = {}          # (ps, method) -> value
    for ps, feat in CONFIGS:
        for m in METHODS:
            v = pick_ups(rows, feat, m, args.metric)
            if v is None:
                print(f"WARNING: no {m} row for {feat} at n_test={args.n_test}; bar omitted")
            vals[(ps, m)] = v
        got = {m: vals[(ps, m)] for m in METHODS if vals[(ps, m)] is not None}
        print(f"  ps={ps:<3} {feat:<24} " + "  ".join(f"{m}={v:.4f}" for m, v in got.items()))

    metric_label = {"test_miou": "test mIoU",
                    "test_overall_acc": "test overall accuracy"}[args.metric]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    width = 0.19
    for j, m in enumerate(METHODS):
        # Center the group of len(METHODS) bars on each integer x.
        offs = (j - (len(METHODS) - 1) / 2) * width
        xs_j, ys_j = [], []
        for i, (ps, _) in enumerate(CONFIGS):
            if vals[(ps, m)] is not None:
                xs_j.append(i + offs); ys_j.append(vals[(ps, m)])
        ax.bar(xs_j, ys_j, width=width, color=COLORS[m], label=LABELS[m], zorder=2)
        for x, y in zip(xs_j, ys_j):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color="#2d3748")

    if args.lp_ref:
        lp_rows = load_rows(Path(LP_CSV), "lp_pa2px")
        for i, (ps, feat) in enumerate(CONFIGS):
            v = pick(lp_rows, feat, args.metric)
            if v is None:
                continue
            half = len(METHODS) * width / 2
            ax.plot([i - half, i + half], [v, v], "--", color="#1a202c", lw=1.3, zorder=4)
            ax.annotate(f"lp_pa2px {v:.2f}", (i + half, v), textcoords="offset points",
                        xytext=(3, 0), va="center", fontsize=7.5, color="#1a202c")

    ax.set_ylabel(metric_label)
    mode_title = {"frozen_shared": "one shared head trained on LR tokens",
                  "retrained": "a separate head retrained per method"}[args.eval_mode]
    ax.set_title(f"PASTIS pa2pa: feature upsamplers, {mode_title}\n"
                 f"OlmoEarth base, Sentinel-2, frozen encoder "
                 f"(n_test={args.n_test}, time_pool={args.time_pool})", fontsize=11)
    ax.set_xlabel("encoder patch size  (token grid = 64/ps per side)      coarse → fine")
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([f"ps{ps}\n{64 // ps}x{64 // ps}" for ps, _ in CONFIGS])
    ax.grid(alpha=0.3, axis="y", zorder=0)
    ax.margins(y=0.18)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

    note = {
        "frozen_shared": (
            "Same trained pa2pa head per config, upsampler swapped in at eval -- so bars are "
            "comparable to each other and to lr_bilinear,\nbut not to the separately trained "
            "lp_pa2px runs in results/pastis/."),
        "retrained": (
            "Each bar has its OWN probe, trained on that method's upsampled features "
            "(lr_bilinear included) -- so a method is no longer\npenalized for shifting the "
            "feature distribution away from what a shared head was fit to."),
    }[args.eval_mode]
    fig.text(0.5, -0.055, note, ha="center", fontsize=7.5, color="#4a5568", style="italic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
