"""Performance vs. inference GMACs, fixed-resolution OlmoEarth grid + mixed-resolution overlay.
Plain matplotlib/seaborn, test-time only (no training hours/samples-per-sec in this figure).
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

RESULTS = Path("/scratch/timz/mAnyUp/results")
HERE = Path(__file__).parent

# task -> (test metric column, higher_is_better, display label)
METRIC = {
    "pastis": ("test_miou", True, "mIoU"),
    "biomassters": ("test_rmse", False, "RMSE (t/ha)"),
    "yieldsat_ml24": ("test_rmse", False, "pixel RMSE (t/ha)"),
    "geoidflood": ("test_miou_headline", True, "mIoU (bg+flood)"),
}
TASK_LABEL = {"pastis": "PASTIS", "biomassters": "BioMassters",
              "yieldsat_ml24": "YieldSAT", "geoidflood": "GEOID-Flood"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def best_row(rows, ps, ts, mode, metric_col, higher, model_size="tiny", train_mode="fft"):
    cand = [r for r in rows if r["model_size"] == model_size and r["mode"] == train_mode
            and r["temporal_mode"] == mode and int(r["patch_size"]) == ps
            and int(r["tile_size"]) == ts and r.get(metric_col) not in (None, "")]
    if not cand:
        return None
    key = (lambda r: float(r["val_primary"]))
    return max(cand, key=key) if higher else min(cand, key=key)


def flop_row(rows, ps, ts, mode):
    suffix = "ts" if mode == "series" else "pd"
    run = f"ps{ps}ts{ts}_{suffix}"
    cand = [r for r in rows if r["run"] == run]
    return cand[-1] if cand else None


def fixed_res_points(task):
    metric_col, higher, label = METRIC[task]
    oe = read_csv(RESULTS / task / "olmoearth.csv")
    fl = read_csv(RESULTS / task / "flops.csv")
    points = []
    for ps in (2, 4, 8, 16):
        for ts in sorted({ps, 32, 64}):
            if ts % ps or 64 % ts:
                continue
            for mode in ("series", "per_timestep"):
                row = best_row(oe, ps, ts, mode, metric_col, higher)
                frow = flop_row(fl, ps, ts, mode)
                if row is None or frow is None:
                    continue
                points.append({"ps": ps, "ts": ts, "mode": mode,
                               "test_metric": float(row[metric_col]),
                               "gmacs": float(frow["gmacs"])})
    return points, label, higher


def main():
    mixed = json.load(open(HERE / "mixed_res_data.json"))

    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    tasks = ["pastis", "biomassters", "yieldsat_ml24", "geoidflood"]

    mode_color = {"series": "#2a78d6", "per_timestep": "#eb6834"}
    ps_marker = {2: "o", 4: "^", 8: "s", 16: "D"}
    ts_fill = {}   # filled per-task below, since which ts values exist varies

    for ax, task in zip(axes.flat, tasks):
        points, label, higher = fixed_res_points(task)
        ax.set_xscale("log")
        # a log axis spanning less than ~2 decades (GEOID-Flood: 0.17-4) packs major AND
        # minor ticks close enough to overlap under the default locator; pin major ticks to
        # powers of ten and switch minor ticks off instead of fighting the auto formatter.
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(TASK_LABEL[task], fontsize=15, fontweight="bold")
        ax.set_xlabel("GMACs / sample (log)")
        ax.set_ylabel(label)

        # tile size -> fill style (solid = ts==ps, open = ts==64, half = ts==32)
        def fill_kwargs(ps, ts):
            if ts == ps:
                return dict(markerfacecolor="C0")   # placeholder, overwritten by color below
            if ts == 32:
                return dict(fillstyle="left")
            return dict(fillstyle="none")            # ts == 64

        for p in points:
            kw = {}
            if p["ts"] == p["ps"]:
                kw["markerfacecolor"] = mode_color[p["mode"]]
                kw["fillstyle"] = "full"
            elif p["ts"] == 32:
                kw["fillstyle"] = "left"
                kw["markerfacecolor"] = mode_color[p["mode"]]
                kw["markerfacecoloralt"] = "white"
            else:   # ts == 64
                kw["fillstyle"] = "none"
            ax.plot(p["gmacs"], p["test_metric"], marker=ps_marker[p["ps"]],
                    markersize=9, color=mode_color[p["mode"]],
                    markeredgecolor=mode_color[p["mode"]], linestyle="none", **kw)

        # mixed-resolution overlay: star, accent color, fill = selector bucket
        for mp in mixed.get(task, []):
            is_vp = mp["bucket"] == "value_predictor"
            ax.plot(mp["gmacs"], mp["test_metric"], marker="*", markersize=20,
                    color="#c98500", markeredgecolor="#8a5c00",
                    fillstyle=("none" if is_vp else "full"), linestyle="none", zorder=5)

        if task == "geoidflood":
            ax.text(0.03, 0.03, "mixed-res flood results use a different task\n"
                    "(flood-only IoU, full series) -- not shown here",
                    transform=ax.transAxes, fontsize=9, color="gray", va="bottom")

        if not higher:
            ax.invert_yaxis()
        sns.despine(ax=ax)

    # -- legend (shared, figure-level) --
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=mode_color["series"],
               markeredgecolor=mode_color["series"], markersize=10, label="series"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=mode_color["per_timestep"],
               markeredgecolor=mode_color["per_timestep"], markersize=10, label="per_timestep"),
    ]
    ps_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                         markeredgecolor="gray", markersize=10, label=f"ps{ps}")
                 for ps, m in ps_marker.items()]
    fill_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="gray", markersize=10, fillstyle="full", label="ts=ps"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markerfacecoloralt="white",
               markeredgecolor="gray", markersize=10, fillstyle="left", label="ts=32"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=10, fillstyle="none", label="ts=64"),
    ]
    mixed_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#c98500",
               markeredgecolor="#8a5c00", markersize=16, fillstyle="full", label="mixed-res: random pick"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="#8a5c00", markersize=16, fillstyle="none", label="mixed-res: value predictor"),
    ]
    fig.legend(handles=handles + ps_handles + fill_handles + mixed_handles,
              loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)

    fig.suptitle("Fixed- & mixed-resolution compute scatter (OlmoEarth tiny, FFT)",
                 fontsize=17, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    out_png = HERE / "flops_scatter.png"
    out_pdf = HERE / "flops_scatter.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
