"""Prithvi-EO-2.0 tiny-TL on PASTIS (S2 only; scripts/slurm/mixres_prithvi.sh): the mixed models
against the fixed patch sizes, in the style of spacetime_tradeoff's spacetime_baselines.png but
on their own axis (a different backbone and input: not comparable with the OlmoEarth curves).
Seed means, bars the seed range. -> figures/prithvi_baselines.png

    python -m exp.viz.prithvi_tradeoff
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import res_pairs_tradeoff as rp  # noqa: E402
from exp.viz.spacetime_tradeoff import PAIR, pair_name  # noqa: E402

MODEL = "prithvi_tiny"
COLS = ["run", "model.model_size", "model.lr_patch", "model.crop_patch", "select.crop_px",
        "select.quadrant.q_s", "train.seed", "test_miou", "gmacs", "model.crop_time"]


def runs(task):
    d = pd.read_csv(f"results/{task}/mixres.csv", low_memory=False,
                    usecols=lambda c: c in COLS).reindex(columns=COLS)
    d = d[d["model.model_size"] == MODEL].drop_duplicates("run", keep="last")
    d["q_s"] = d["select.quadrant.q_s"].fillna(0)
    return d


def mean(d, keys):
    g = d.groupby(keys)
    return pd.DataFrame({"x": g.gmacs.mean(), "y": g.test_miou.mean(), "lo": g.test_miou.min(),
                         "hi": g.test_miou.max(), "seeds": g.size()}).reset_index()


def draw():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": rp.MUTED,
                         "axes.labelcolor": rp.MUTED, "xtick.color": rp.MUTED,
                         "ytick.color": rp.MUTED, "text.color": rp.INK,
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(10, 6.6), facecolor=rp.SURFACE)
    ax.set_facecolor(rp.SURFACE)
    sea, mon = runs("pastis_s2"), runs("pastis_s2_mon")
    style = {True: ("-", "D", rp.INK), False: ("--", "s", rp.SURFACE)}   # monthly, seasonal
    for monthly, d in ((True, mon), (False, sea[sea.q_s == 0])):
        ls, mk, mfc = style[monthly]
        p = mean(d, ["model.lr_patch"]).sort_values("x")
        ax.plot(p.x, p.y, ls + mk, color=rp.INK, lw=1.2, ms=7, mfc=mfc, mec=rp.INK, mew=1.4,
                alpha=0.75, zorder=3)
        for r in p.itertuples():
            ps = int(r._1)
            # S16 monthly and S10 seasonal sit together near 1.3 GMACs: one right, one left
            side = {(True, 16): 1, (False, 10): -1}.get((monthly, ps), 0)
            ax.annotate(rf"$S_{{{ps}}}T_{{{'mon' if monthly else 'sea'}}}$", (r.x, r.y),
                        xytext=(9 * side, 0) if side else (0, 8 if monthly else -8),
                        textcoords="offset points",
                        ha={1: "left", -1: "right", 0: "center"}[side],
                        va="center" if side else "bottom" if monthly else "top", fontsize=9,
                        zorder=5)
    ours = sea[sea.q_s > 0].copy()
    ours["geo"] = [f"c{c}" + (f"p{int(p)}" if p > 0 else "")
                   for c, p in zip(ours["select.crop_px"], ours["model.crop_patch"].fillna(0))]
    ours["frac"] = ours["model.crop_time"] == "frac"
    drawn = []
    # crops at their month's time (crop_time frac) in full, at the season's slot faded behind
    for (ps, geo, frac), d in ours.groupby(["model.lr_patch", "geo", "frac"]):
        p = mean(d, ["q_s"]).sort_values("x")
        col = PAIR[(int(ps), geo)]
        if not frac:
            ax.plot(p.x, p.y, "--*", color=col, lw=1.4, ms=10, mec=col, alpha=0.4, zorder=5)
            continue
        ax.plot(p.x, p.y, "-*", color=col, lw=2.4, ms=14, mec=col, mew=1.4, zorder=6)
        multi = p[p.seeds > 1]
        ax.errorbar(multi.x, multi.y, yerr=[multi.y - multi.lo, multi.hi - multi.y], fmt="none",
                    ecolor=col, elinewidth=1.4, capsize=4, zorder=6)
        drawn.append((int(ps), geo))
    ax.set_xscale("log")
    ax.grid(True, color=rp.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("GMACs per sample (log scale)")
    ax.set_ylabel("test mIoU")
    drawn = [k for k in PAIR if k in drawn]                         # legend in PAIR order
    h = [plt.Line2D([], [], color=PAIR[k], marker="*", ms=13, lw=2.4) for k in drawn]
    h += [plt.Line2D([], [], color=rp.MUTED, marker="*", ms=10, lw=1.4, ls="--", alpha=0.5),
          plt.Line2D([], [], color=rp.INK, marker="D", ms=7, lw=1.2, alpha=0.75),
          plt.Line2D([], [], color=rp.INK, marker="s", ms=7, mfc=rp.SURFACE, ls="--", lw=1.2,
                     alpha=0.75)]
    lab = [f"ours {pair_name(*k)}" for k in drawn]
    lab += ["ours, crops without month position", "fixed resolution, monthly", "fixed resolution, seasonal"]
    ax.legend(h, lab, loc="lower right", frameon=False, fontsize=9.5,
              title="ours: entropy + attention, cross attention", title_fontsize=9.5,
              alignment="left")
    fig.suptitle("PASTIS, Prithvi-EO-2.0 tiny: mixed resolution against fixed patch sizes",
                 x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("figures/prithvi_baselines.png")
    fig.savefig(out, dpi=150, facecolor=rp.SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(draw())
