"""Fixed-resolution OlmoEarth tiny on the regular grid, no crops (rounds 43, 47, 48): the same
runs drawn twice, in the style of exp.viz.shift_grid_tradeoff.

    python -m exp.viz.space_time_ablation
      -> figures/patch_size_ablation.pdf            monthly composites only, a point per patch size
      -> figures/temporal_resolution_ablation.pdf   16 px patches only, a point per input cadence
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

TASKS = {"pastis": ("PASTIS", "mIoU", True), "biomassters": ("BioMassters", "RMSE", False),
         "yieldsat_raw": ("YieldSAT", "RMSE", False)}
# inference GMACs per sample, traced with --flops: {task: {cadence: {patch size: GMACs}}}
GMACS = {
    "pastis": {"year": {16: 0.38, 8: 1.73, 4: 10.55}, "season": {16: 1.73, 8: 10.55, 4: 100.19},
               "all": {16: 6.26, 8: 53.66, 4: 673.2}, "raw": {16: 62.70, 8: 783.57}},
    "biomassters": {"year": {16: 0.36, 8: 1.68, 4: 10.32},
                    "season": {16: 1.68, 8: 10.32, 4: 99.28},
                    "all": {16: 5.14, 8: 42.09, 4: 512.9}},
    "yieldsat_raw": {"year": {16: 0.27, 8: 1.20, 4: 6.84}, "season": {16: 0.80, 8: 4.00, 4: 34.33},
                     "all": {16: 2.10, 8: 13.60, 4: 137.8}, "raw": {16: 39.57, 8: 477.10}},
}
STEPS = {"year": "one composite", "season": "seasonal composites", "all": "monthly composites",
         "raw": "every acquisition"}
SET2 = ("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3")        # ColorBrewer Set2: lines, markers
DARK2 = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")       # the same hues darker: line labels
MARKS = (("D", 8.5), ("o", 8), ("s", 7.5), ("^", 8.5))
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def rows(task):
    """-> DataFrame (steps, P, x GMACs, y test metric), seeds averaged."""
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False)
    r = f.run.astype(str)
    g = f[r.str.contains(r"_none_native_set_(?:lr(?:all|year|raw)_)?(?:lrps\d_)?lr0.001_ep\d+"
                         r"(?:_seed1)?$")].drop_duplicates("run", keep="last").copy()
    g["P"] = g.lr_patch.astype(int)
    g["steps"] = g.run.str.extract(r"_lr(all|year|raw)_")[0].fillna("season")
    g = g.groupby(["steps", "P"]).test_primary.mean().reset_index(name="y")
    g["x"] = [GMACS[task].get(s, {}).get(p) for s, p in zip(g.steps, g.P)]
    return g.dropna(subset=["x"])


def draw(by, out):
    """by "steps": a line per cadence, markers = patch size; by "P": the reverse."""
    other = "P" if by == "steps" else "steps"
    lines = list(STEPS) if by == "steps" else [16, 8, 4]
    marks = [16, 8, 4] if by == "steps" else list(STEPS)
    name = (lambda v: STEPS[v]) if by == "steps" else (lambda v: f"{v} px patches")
    mname = (lambda v: f"{v} px patches") if by == "steps" else (lambda v: STEPS[v])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, (task, (title, metric, up)) in zip(axes, TASKS.items()):
        ax.set_facecolor(SURFACE)
        g = rows(task)
        for i, v in enumerate(lines):
            c = g[g[by] == v].sort_values("x")
            if not len(c):
                continue
            ax.plot(c.x, c.y, "-", color=SET2[i], lw=2.6, zorder=2)
            for r in c.itertuples():
                m, ms = MARKS[marks.index(getattr(r, other))]
                ax.plot(r.x, r.y, m, ms=ms, color=SET2[i], mec=SURFACE, mew=1.2, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 3)))   # 3+ decades wide
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.margins(x=0.09, y=0.12)
        if not up:
            ax.invert_yaxis()
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Inference GMACs per sample")
        ax.set_ylabel(f"{metric} (higher is better)" if up else
                      f"{metric} (lower is better; axis flipped)")
    h = [plt.Line2D([], [], color=SET2[i], lw=2.5) for i in range(len(lines))]
    lab = [name(v) for v in lines]
    h += [plt.Line2D([], [], marker=m, ms=ms, ls="none", color=MUTED, mec=SURFACE, mew=1.2)
          for (m, ms), _ in zip(MARKS, marks)]
    lab += [mname(v) for v in marks]
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=10,
               columnspacing=1.6, handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=2.2)
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(Path(out).with_suffix(".png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def draw_single(out, by, fixed):
    """One line: `by` ("steps" or "P") varies, coloured and labelled point by point; the other
    is `fixed` (a patch size, or a cadence)."""
    other = "P" if by == "steps" else "steps"
    order = list(STEPS) if by == "steps" else [16, 8, 4]
    name = (lambda v: STEPS[v]) if by == "steps" else (lambda v: f"{v} px patches")
    note = (f"all models: {fixed} px patches on the regular grid" if by == "steps"
            else f"all models: {STEPS[fixed]}, regular grid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, (task, (title, metric, up)) in zip(axes, TASKS.items()):
        ax.set_facecolor(SURFACE)
        g = rows(task)
        c = g[g[other] == fixed].sort_values("x")
        ax.plot(c.x, c.y, "-", color="#c9c8c2", lw=2.6, zorder=2)
        for r in c.itertuples():
            i = order.index(getattr(r, by))
            ax.plot(r.x, r.y, "o", ms=10, color=SET2[i], mec=SURFACE, mew=1.2, zorder=3)
            ax.annotate(name(getattr(r, by)), (r.x, r.y), xytext=(10, -4), textcoords="offset points",
                        ha="left", va="top", fontsize=9.5, color=DARK2[i], fontweight="bold",
                        zorder=4)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 3)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.margins(x=0.09, y=0.14)
        ax.set_xlim(right=ax.get_xlim()[1] * 14)         # room for the last label
        if not up:
            ax.invert_yaxis()
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Inference GMACs per sample")
        ax.set_ylabel(f"{metric} (higher is better)" if up else
                      f"{metric} (lower is better; axis flipped)")
    h = [plt.Line2D([], [], marker="o", ms=9, ls="none", color=SET2[i], mec=SURFACE, mew=1.2)
         for i in range(len(order))]
    h.append(plt.Line2D([], [], ls="none"))
    fig.legend(h, [name(v) for v in order] + [note],
               loc="lower center", ncol=len(order) + 1, frameon=False, fontsize=10, columnspacing=1.6,
               handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=2.2)
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(Path(out).with_suffix(".png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    draw_single("figures/patch_size_ablation.pdf", "P", "all")
    draw_single("figures/temporal_resolution_ablation.pdf", "steps", 16)


if __name__ == "__main__":
    main()
