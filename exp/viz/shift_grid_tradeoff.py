"""Spatial resolution only (no seasonal composites): the monthly on-grid model at one patch size
against the same model plus same-size tokens on a shifted grid (--shift_stride, round 47).

    python -m exp.viz.shift_grid_tradeoff  -> figures/spatial_resolution_ablation.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from exp.upsamplers.finetune_inject import TILE, shifted_count

TASKS = {"pastis": ("PASTIS", "mIoU", True), "biomassters": ("BioMassters", "RMSE", False),
         "yieldsat_raw": ("YieldSAT", "RMSE", False)}
BASE_GMACS = {"pastis": {16: 6.26, 8: 53.66, 4: 673.2},        # traced monthly series (--flops)
              "biomassters": {16: 5.14, 8: 42.09, 4: 512.9},
              "yieldsat_raw": {16: 2.10, 8: 13.60, 4: 137.8}}
# ps4 is left out: only its sparse stride-16 / stride-8 additions were run (6% / 25% more
# tokens), not the matching stride-4 grid, so it is not a fair test at that patch size
COL = {16: "#66c2a5", 8: "#fc8d62"}        # ColorBrewer Set2: lines and markers
TEXT = {16: "#1b9e77", 8: "#d95f02"}       # Dark2, the same hues darker: the token labels
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def rows(task):
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False)
    r = f.run.astype(str)
    g = f[r.str.contains("lrall") & r.str.contains("native_set")
          & r.str.contains("_none_|_shift")].drop_duplicates("run", keep="last").copy()
    g["P"] = g.lr_patch.astype(int)
    g["S"] = g.run.str.extract(r"_shift(\d+)")[0].fillna(0).astype(int)
    g = g[(g.S == 0) | (g.S == g.P)]     # the doubling only: the half-shifted grid of the SAME
    #                                       patch size (16 -> 41 tokens, 64 -> 145)
    d = pd.read_csv(f"results/{task}/inject_ncrops.csv")
    d = d[d.run.str.contains("_shift")].drop_duplicates("run", keep="last").set_index("run")
    g["x"] = g.run.map(d.gmacs)
    g.loc[g.S == 0, "x"] = g.P.map(BASE_GMACS[task])
    return g


MARK = {0: ("D", 9, "on-grid tokens only"),                 # stride -> marker, size, legend
        16: ("o", 8.5, "+ the half-shifted grid (off-grid tokens)")}


def layout(ax, P=16):
    """The 64 px tile with the P px patch centres: on the patch grid (16 for P = 16) and on the
    half-shifted grid (25), whose windows overlap four on-grid patches each and hang out of
    the tile at its edges, where the missing pixels are zero padding."""
    from exp.upsamplers.finetune_inject import shifted_positions
    pad, h = P // 2, P // 2
    on = [P * k + h for k in range(TILE // P)]                     # centres 8, 24, 40, 56
    off = [p + h for p in shifted_positions(P, P)]                 # centres 0, 16, ..., 64
    ax.set_facecolor(SURFACE)
    ax.add_patch(plt.Rectangle((-pad, -pad), TILE + 2 * pad, TILE + 2 * pad, fc="#efeee9",
                               ec="#cfcec8", hatch="////", lw=0, zorder=0))
    ax.add_patch(plt.Rectangle((0, 0), TILE, TILE, fc=SURFACE, ec=INK, lw=1.4, zorder=1))
    for v in on:                                                   # the on-grid patches
        for u in on:
            ax.add_patch(plt.Rectangle((u - h, v - h), P, P, fc="none", ec="#b9b8b1", lw=1.1,
                                       zorder=2))
    ex = (off[1], off[2])                                          # one off-grid window, shaded
    ax.add_patch(plt.Rectangle((ex[0] - h, ex[1] - h), P, P, fc=COL[16], alpha=0.3, ec=TEXT[16],
                               lw=1.6, ls="--", zorder=3))
    edge = (off[0], off[3])                                        # one at the edge: padded
    ax.add_patch(plt.Rectangle((edge[0] - h, edge[1] - h), P, P, fc=COL[16], alpha=0.3,
                               ec=TEXT[16], lw=1.6, ls="--", zorder=3))
    uu, vv = zip(*[(u, v) for v in on for u in on])
    ax.plot(uu, vv, "D", ms=7, color=COL[16], mec=SURFACE, mew=1, zorder=4)
    uu, vv = zip(*[(u, v) for v in off for u in off])
    ax.plot(uu, vv, "o", ms=7.5, color=SURFACE, mec=TEXT[16], mew=1.8, zorder=4)
    ax.annotate("overlaps 4\non-grid patches", (ex[0] + h, ex[1] - h), xytext=(4, -2),
                textcoords="offset points", fontsize=8.5, color=TEXT[16], fontweight="bold",
                bbox=dict(fc=SURFACE, ec="none", pad=1.2, alpha=0.9),
                va="top", zorder=5)
    ax.annotate("half outside:\nzero padding", (edge[0] + h, edge[1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color=TEXT[16], fontweight="bold",
                bbox=dict(fc=SURFACE, ec="none", pad=1.2, alpha=0.9),
                va="center", zorder=5)
    ax.set_xlim(-pad - 1, TILE + pad + 1)
    ax.set_ylim(TILE + pad + 1, -pad - 1)                          # image rows run down
    ax.set_aspect("equal")
    ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_yticks([0, 16, 32, 48, 64])
    ax.tick_params(length=0, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Where the {P} px tokens sit", loc="left", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(f"{len(on) ** 2} on-grid + {len(off) ** 2} half-shifted = "
                  f"{len(on) ** 2 + len(off) ** 2} tokens a month (px)")
    h_on = plt.Line2D([], [], marker="D", ms=7, ls="none", color=COL[16], mec=SURFACE)
    h_off = plt.Line2D([], [], marker="o", ms=7, ls="none", color=SURFACE, mec=TEXT[16], mew=1.8)
    h_pad = plt.Rectangle((0, 0), 1, 1, fc="#efeee9", hatch="////", ec="#cfcec8", lw=0.5)
    ax.legend([h_on, h_off, h_pad], ["on-grid centre", "half-shifted centre", "zero padding"],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
              fontsize=9, handletextpad=0.3, columnspacing=1.0)


def main():
    plt.rcParams.update({"hatch.color": "#cfcec8", "hatch.linewidth": 0.8, "pdf.fonttype": 42, "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": (1, 1, 1, 0.82)})
    layout(axes[3])
    for ax, (task, (title, metric, up)) in zip(axes[:3], TASKS.items()):
        ax.set_facecolor(SURFACE)
        g = rows(task)
        for P, col in COL.items():
            c = g[g.P == P].sort_values("x")
            ax.plot(c.x, c.test_primary, "-", color=col, lw=2.6, zorder=2)
            for r in c.itertuples():
                m, ms, _ = MARK[16 if r.S else 0]
                ax.plot(r.x, r.test_primary, m, ms=ms, color=col, mec=SURFACE, mew=1.2, zorder=3)
                n = (TILE // P) ** 2 + (shifted_count(P, r.S) if r.S else 0)
                # an off-grid point clearly worse than its base: label it below, clear of the base's
                base = c[c.S == 0].test_primary.iloc[0]
                worse = (base - r.test_primary) if up else (r.test_primary - base)
                span = g.test_primary.max() - g.test_primary.min()
                above = r.S != 0 and worse < 0.04 * span     # (a tie still goes above)
                ax.annotate(f"{n} tokens", (r.x, r.test_primary),
                            xytext=(9, -5) if r.S == 0 else (0, 9) if above else (0, -9),
                            textcoords="offset points",
                            ha="left" if r.S == 0 else "center",
                            va="top" if r.S == 0 or not above else "bottom", fontsize=9, color=TEXT[P],
                            fontweight="bold", zorder=4)
        ax.set_xscale("log")                    # each panel its own range, plain-number ticks
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.margins(x=0.09, y=0.14)
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
    h = [plt.Line2D([], [], color=c, lw=2.5) for c in COL.values()]
    lab = [f"{P} px patches" for P in COL]
    h += [plt.Line2D([], [], marker=m, ms=ms, ls="none", color=MUTED, mec=SURFACE, mew=1.2)
          for m, ms, _ in MARK.values()]
    lab += [t for _, _, t in MARK.values()]
    h.append(plt.Line2D([], [], ls="none"))
    lab.append("token counts are per month")
    fig.legend(h, lab, loc="lower left", ncol=6, frameon=False, fontsize=10,
               columnspacing=1.6, handletextpad=0.5, bbox_to_anchor=(0.02, 0.0))
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=2.2)
    out = Path("figures/spatial_resolution_ablation.pdf")           # vector, fonts embedded as TrueType
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
