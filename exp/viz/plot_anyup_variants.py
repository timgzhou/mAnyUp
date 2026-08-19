"""Plot test mIoU for four PASTIS heads, grouped by feature config.

Two axes of comparison in one figure -- guidance (none vs AnyUp) x time handling (single probe
vs per-timestep ensemble):

  lp_pa2px      no guidance:  1x1 conv D->C*p^2, sub-pixel unfold        -> one probe
  lp_pa2px_ens  as pa2px, but ONE PROBE PER TIMESTEP, logits averaged
  anyup         AnyUp-upsampled mean-T features, mean-T RGB guidance     -> one probe
  anyup_t1_ens  per-timestep AnyUp AND guidance, ONE PROBE PER TIMESTEP, logits averaged

The two _ens variants are the point: averaging per-timestep LOGITS from independent probes is
not algebraically the same as probing time-averaged features, so the pa2px->pa2px_ens and
anyup->anyup_t1_ens gaps each isolate what temporal ensembling buys, with and without guidance.
(The dropped middle variants -- anyup_t2, anyup_t1, anyup_t2_ens -- all sat within +-0.01 of
plain anyup, i.e. tied at the CSV's 2-decimal resolution.)

Only configs with at least --min_variants of the four are plotted, so a config with a single
lonely bar does not read as a comparison. --configs restricts which feature sets are shown.

    source env_setup/env_olmo.sh    # (or any env with matplotlib)
    python -u -m exp.viz.plot_anyup_variants
    python -u -m exp.viz.plot_anyup_variants --configs ps4,ps8   # substring match
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless cluster: never try an interactive backend
import matplotlib.pyplot as plt

RESULTS_CSV = "results/pastis/lp_olmoearth_pastis.csv"

# Order reads as an ablation: baseline, +spatial guidance, +temporal ensembling, +both.
#   lp_pa2px      no guidance, one probe          -> LR
#   anyup         AnyUp spatial guidance          -> +spatial
#   lp_pa2px_ens  per-timestep probes, no guidance-> +temporal
#   anyup_t1_ens  guidance AND per-timestep probes-> +both
VARIANTS = ["lp_pa2px", "anyup", "lp_pa2px_ens", "anyup_t1_ens"]
PRETTY = ["LR", "+spatial", "+temporal", "+both"]
COLORS = ["#a0aec0", "#3182ce", "#ed8936", "#c05621"]


def load(csv_path: Path) -> dict[tuple[str, str], float]:
    """(features, head_mode) -> latest test_miou. CSV is append-only, so last row wins."""
    best: dict[tuple[str, str], tuple[str, float]] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["head_mode"] not in VARIANTS or not r["test_miou"]:
                continue
            key = (r["features"], r["head_mode"])
            ts = r["timestamp"]
            if key not in best or ts > best[key][0]:
                best[key] = (ts, float(r["test_miou"]))
    return {k: v for k, (_, v) in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default=RESULTS_CSV)
    ap.add_argument("--min_variants", type=int, default=3,
                    help="skip configs with fewer than this many variants present")
    ap.add_argument("--configs", default=None,
                    help="comma-separated substrings; keep only feature configs matching one "
                         "(e.g. 'ps4,ps8'). Default: every config in the CSV.")
    ap.add_argument("--clean", action="store_true",
                    help="presentation mode: ablation labels in the legend, no title, no x axis, "
                         "no footnote -- just bars, the y axis, and value labels")
    ap.add_argument("--out", default="feature_viz/anyup_variants_miou.png")
    args = ap.parse_args()

    vals = load(Path(args.results_csv))
    if not vals:
        raise SystemExit(f"ERROR: no anyup* rows in {args.results_csv}")

    configs = sorted({f for f, _ in vals})
    if args.configs:
        pats = [p for p in args.configs.split(",") if p]
        configs = [c for c in configs if any(p in c for p in pats)]
        if not configs:
            raise SystemExit(f"ERROR: no config matches --configs {args.configs}")
    configs = [c for c in configs
               if sum((c, v) in vals for v in VARIANTS) >= args.min_variants]
    if not configs:
        raise SystemExit(f"ERROR: no config has >= {args.min_variants} variants")
    # Within the kept set, order by patch size then tile size for a readable x axis.
    def ps_tile(name: str):
        ps = tile = 10**6
        for tok in name.split("_"):
            if tok.startswith("ps") and tok[2:].isdigit():
                ps = int(tok[2:])
            elif tok.startswith("tile") and tok[4:].isdigit():
                tile = int(tok[4:])
        return ps, tile
    configs.sort(key=ps_tile)

    labels = PRETTY if args.clean else VARIANTS
    # One config: widen the bars so a single group fills the axes instead of a thin cluster.
    single = len(configs) == 1
    width = 0.62 if (args.clean and single) else 0.19
    fig, ax = plt.subplots(figsize=(6.4, 5.0) if (args.clean and single)
                           else (1.9 * len(configs) + 3.2, 5.0))
    shown = [v for v in VARIANTS if any((c, v) in vals for c in configs)]
    for i, (variant, color, label) in enumerate(zip(VARIANTS, COLORS, labels)):
        xs, ys = [], []
        for j, cfg in enumerate(configs):
            v = vals.get((cfg, variant))
            if v is None:
                continue
            xs.append(j + (i - (len(shown) - 1) / 2) * width)
            ys.append(v)
        bars = ax.bar(xs, ys, width, label=label, color=color, zorder=2)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:.2f}", (b.get_x() + b.get_width() / 2, y),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=13 if args.clean else 6.5,
                        rotation=0 if args.clean else 90, color="#1a202c")

    # Bars start at a nonzero floor: every value sits in 0.45-0.60 and a 0-based axis would
    # squash the differences the plot exists to show.
    lo = min(vals[(c, v)] for c in configs for v in VARIANTS if (c, v) in vals)
    hi = max(vals[(c, v)] for c in configs for v in VARIANTS if (c, v) in vals)
    ax.set_ylim(max(0.0, lo - 0.06), hi + 0.05)
    ax.grid(alpha=0.3, axis="y", zorder=0)

    if args.clean:
        ax.set_ylabel("mIoU", fontsize=15)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xticks([])                       # config is stated on the slide, not the axis
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=13, ncol=len(shown), loc="upper center",
                  frameon=False, columnspacing=1.4, handlelength=1.3)
    else:
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace("oe_base_", "") for c in configs], fontsize=8.5)
        ax.set_ylabel("test mIoU")
        ax.set_title("PASTIS linear probe: guidance x temporal ensembling\n"
                     "OlmoEarth base, frozen encoder + frozen AnyUp upsampler", fontsize=11)
        ax.legend(fontsize=8.5, ncol=4, loc="upper center", framealpha=0.95)
        fig.text(0.5, -0.02,
                 "y axis is truncated (does not start at 0). mIoU is recorded to 2 decimals, so "
                 "each bar carries about +-0.01; gaps under that are not resolved.",
                 ha="center", fontsize=7.5, color="#4a5568", style="italic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # Text summary: the t1 -> t1_ens gap is the headline the figure is built around.
    print(f"\n{'config':<24}" + "".join(f"{v:<15}" for v in VARIANTS) + "anyup_t1_ens - anyup")
    for cfg in configs:
        row = "".join(f"{vals.get((cfg, v), float('nan')):<15.2f}" for v in VARIANTS)
        t1, ens = vals.get((cfg, "anyup")), vals.get((cfg, "anyup_t1_ens"))
        gap = f"{ens - t1:+.2f}" if (t1 is not None and ens is not None) else "--"
        print(f"{cfg.replace('oe_base_', ''):<24}{row}{gap}")


if __name__ == "__main__":
    main()
