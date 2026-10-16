"""Accuracy against compute: mixed-resolution curves (grid-placed crops, one model tested at
many crop counts) and the fixed-resolution OlmoEarth tiny models.

    python -m exp.viz.res_pairs_tradeoff [--grid_mode shared|c2f]  -> figures/res_pairs_tradeoff_<mode>_<variant>.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TASKS = {   # task -> (title, metric, higher is better, task of the monthly baselines)
    "pastis": ("PASTIS", "mIoU", True, "pastis"),
    "biomassters": ("BioMassters", "RMSE", False, "biomassters"),
    "yieldsat_raw": ("YieldSAT", "RMSE", False, "yieldsat_ml24"),
}
ARMS = {    # (season patch, crop patch) -> (label, colour)
    ("16", "16"): ("ps16 seasons + ps16 month crops", "#4a3aa7"),   # uniform variant only
    ("16", "8"): ("ps16 seasons + ps8 month crops", "#2a78d6"),
    ("16", "4"): ("ps16 seasons + ps4 month crops", "#eb6834"),
    ("8", "4"): ("ps8 seasons + ps4 month crops", "#1baf7a"),
}
TILE_PX = 64
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COV_COUNTS = (64, 128, 256, 512, 1024, 2048)        # crop counts whose coverage is printed
MONTHLY = {ps: f"$S_{{{ps}}}T_{{month}}$" for ps in (16, 8, 4)}
SEASONAL = {ps: f"$S_{{{ps}}}T_{{season}}$" for ps in (16, 8, 4)}
MODE_LABEL = {"shared": "ps4grid", "c2f": "c2f"}   # display name only; CLI value / filenames unchanged


def arm(run):
    c = re.search(r"_cpx(\d+)", run)
    return ("8" if "_lrps8" in run else "16", c.group(1) if c else "4")


def kind(run):
    """How the training crop count was drawn: uniform 0..2048, a set, a set with 2x epochs, or
    one dedicated count (ded: crops injected at block 0, ded3: at block 3)."""
    m = re.search(r"_cnt([\d-]+)", run)
    if not m:
        return "uniform"
    if "-" not in m.group(1):
        return "ded3" if "_L3_" in run else "ded"
    return "set_long" if re.search(r"_ep(100|20)(_|$)", run) else "set"


def curves(task, grid_mode, variant):
    """-> one row per (pair, crop count): a one-model variant tested at every count, or the
    dedicated models each at its own count."""
    d = pd.read_csv(f"results/{task}/inject_ncrops.csv")
    d = d[(d.grid_mode == grid_mode) & (d.pick == "grid") & d.run.str.contains("_grid_")]
    d = d.drop_duplicates(["run", "n_test", "pick"], keep="last")
    d["arm"], d["kind"] = d.run.map(arm), d.run.map(kind)
    d = d[d.kind == variant]
    if grid_mode == "stagger":      # own-grid lattice (round 40): caps 196 / 512 / 1024, and no
        # logged coverage -> computed: windows on the crop's own grid never overlap
        months = {"pastis": 11.74, "biomassters": 10.77, "yieldsat_raw": 6.57}[task]
        px = d.arm.map(lambda k: int(k[1]))
        d = d.assign(coverage=(d.n_test * px ** 2 / (months * TILE_PX ** 2)).clip(upper=1))
    elif variant.startswith("ded"):
        d = d[d.n_test == d.n_max]
    else:
        d = d[d.n_max == 2048]
    return d.groupby(["arm", "n_test"]).agg(y=("test_primary", "mean"), x=("gmacs", "mean"),
                                            cov=("coverage", "mean")).reset_index()


def fixed(task, base):
    """-> [(label, GMACs, metric, monthly?)]"""
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False)
    f_all = f
    f = f[f.run.astype(str).str.contains("_none_native_set_(?:lrps\\d_)?lr0.001_ep\\d+(?:_seed1)?$")]
    nc = pd.read_csv(f"results/{task}/inject_ncrops.csv")
    zero = nc[(nc.n_test == 0) & (nc.n_max == 2048)]
    out = []
    for ps in (16, 8, 4):
        y = f[f.lr_patch == ps].drop_duplicates("run", keep="last").test_primary.mean()
        if ps == 4:                                         # ps4 seasons: traced once (--flops)
            x = {"pastis": 100.19, "biomassters": 99.28, "yieldsat_raw": 34.33}[task]
        else:                                               # = a mixed model with 0 crops
            x = zero[zero.run.str.contains("_lrps8") == (ps == 8)].gmacs.mean()
        out.append((SEASONAL[ps], x, y, False))
    if task == "yieldsat_raw":      # monthly models on the SAME data as the mixed ones (the
        # ml24 baselines see one chosen date per month); GMACs traced with --flops
        gm = {16: 2.10, 8: 13.60, 4: 137.80}
        for ps in (16, 8, 4):
            tag = "_none_native_lrall_lr" if ps == 16 else f"_none_lrall_lrps{ps}_lr"
            r = f_all[f_all.run.astype(str).str.contains(tag)]
            if len(r):
                out.append((MONTHLY[ps], gm[ps], r.test_primary.iloc[-1], True))
        return out
    o = pd.read_csv(f"results/{base}/olmoearth.csv")
    o = o[(o["mode"] == "fft") & (o.model_size == "tiny") & (o.temporal_mode == "series") & (o.tile_size == 64)
          & (o.lr == 1e-3) & (o.epochs == o.epochs.min())]
    fl = pd.read_csv(f"results/{base}/flops.csv", usecols=range(6))
    fl = fl[fl.model_size == "tiny"].drop_duplicates("run", keep="last").set_index("run").gmacs
    for ps in (16, 8, 4):
        y = o[o.patch_size == ps].test_primary.iloc[0]
        out.append((MONTHLY[ps], fl[f"ps{ps}ts64_ts"], y, True))
    return out


def place_coverage(fig, pending, fixed_pts, fs=6.5):
    """Coverage labels beside the mixed points, in the curve's colour: each takes the first of
    eight spots around its point that is clear of every marker, fixed-model label and coverage
    label already placed (boxes in points; the layout must be final)."""
    k = 72 / fig.dpi
    taken = {}                                              # ax -> [(x0, y0, x1, y1)]
    for ax, x, y, _, _ in pending:                          # the mixed markers themselves
        px, py = ax.transData.transform((x, y)) * k
        taken.setdefault(ax, []).append((px - 3.5, py - 3.5, px + 3.5, py + 3.5))
    for ax, x, y, label, above in fixed_pts:                # fixed markers and their labels
        px, py = ax.transData.transform((x, y)) * k
        w = 4.6 * (len(label.replace("$", "").replace("{", "").replace("}", "").replace("_", "")))
        y0 = py + 9 if above else py - 9 - 10
        taken.setdefault(ax, []).extend([(px - 6, py - 6, px + 6, py + 6),
                                         (px - w / 2, y0, px + w / 2, y0 + 10)])
    spots = [(0, 8), (0, -8), (-9, 5), (9, -5), (9, 5), (-9, -5), (0, 15), (0, -15), (-16, 0),
             (16, 0), (-12, 12), (12, -12), (12, 12), (-12, -12), (0, 22), (0, -22)]
    for ax, x, y, text, col in pending:
        px, py = ax.transData.transform((x, y)) * k
        w, h = 0.62 * fs * len(text), fs + 1
        for dx, dy in spots:
            box = (px + dx - w / 2, py + dy - h / 2, px + dx + w / 2, py + dy + h / 2)
            if all(box[2] < t[0] or t[2] < box[0] or box[3] < t[1] or t[3] < box[1]
                   for t in taken[ax]):
                break
        taken[ax].append(box)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center",
                    va="center", fontsize=fs, color=col, zorder=6)


VARIANTS = {   # variant -> (file tag, subtitle)
    "uniform": ("uniform", "ONE model per curve; training crop count drawn uniformly from "
                           "0..CAP; tested at every count (SEEDS)"),
    "set": ("set", "ONE model per curve; training crop count drawn from {0, 64, 128, 256, 512, "
                   "1,024, 2,048}; tested at every count (1 seed)"),
    "set_long": ("set_long", "ONE model per curve; training crop count drawn from {0, 64, ..., "
                             "2,048}, twice the epochs; tested at every count (1 seed)"),
    "ded": ("dedicated", "one DEDICATED model per point, trained and tested at that crop count; "
                         "crops join at block 0 (1 seed)"),
    "ded+ded3": ("dedicated_L0_vs_L3",
                 "one DEDICATED model per point (1 seed).  Solid: crops join at block 0.  "
                 "Dashed, open markers: crops join at block 3 (cheaper: 3 blocks without "
                 "joint attention)"),
    "ded3": ("dedicated_L3", "one DEDICATED model per point, trained and tested at that crop "
                             "count; crops join at block 3 (1 seed)"),
}


def draw(grid_mode, variant, overlay=None, suffix="", note="", mixed=True, subtitle=None,
         footnote=None):
    """overlay(ax, task, higher_is_better, pending): extra series drawn on each panel;
    suffix: appended to the file name; note: an extra footnote line; mixed=False: no grid-crop
    curves (nor their coverage legend); subtitle / footnote: replace the default texts."""
    tag, sub = VARIANTS[variant]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.7), facecolor=SURFACE)
    n = 0
    pending, fixed_pts = [], []         # coverage labels to place; fixed points to keep clear of
    for ax, (task, (title, metric, up, base)) in zip(axes, TASKS.items()):
        ax.set_facecolor(SURFACE)
        both = variant == "ded+ded3"
        g = curves(task, grid_mode, "ded" if both else variant)
        g3 = curves(task, grid_mode, "ded3") if both else g.iloc[:0]
        n += len(g3) if both else len(g)
        for key, (label, col) in (ARMS.items() if mixed else ()):
            c = g[g.arm == key].sort_values("n_test")
            c3 = g3[g3.arm == key].sort_values("n_test")
            ax.plot(c3.x, c3.y, "--o", color=col, lw=1.6, ms=6, mfc=SURFACE, mec=col, mew=1.5,
                    zorder=3)
            if not len(c):
                continue
            ax.plot(c.x, c.y, "-o", color=col, lw=2, ms=6 if variant.startswith("ded") else 4.5,
                    mec=SURFACE, mew=0.8, label=label, zorder=3)
            for r in c[c.n_test.isin(COV_COUNTS + ((196,) if grid_mode == "stagger" else ()))
                       ].itertuples():
                pending.append((ax, r.x, r.y, f"{100 * r.cov:.0f}%", col))
        for label, x, y, monthly in fixed(task, base):
            ax.plot(x, y, "D" if monthly else "s", ms=8, color=INK if monthly else SURFACE,
                    mec=INK, mew=1.5, zorder=4, ls="none")
            above = up == monthly           # monthly on the better side, seasonal on the other
            ax.annotate(label, (x, y), xytext=(0, 9 if above else -9),
                        textcoords="offset points", ha="center",
                        va="bottom" if above else "top", fontsize=9, zorder=5)
            fixed_pts.append((ax, x, y, label, above))
        if overlay is not None:
            overlay(ax, task, up, pending)
        ax.set_xscale("log")
        if not up:
            ax.invert_yaxis()
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("GMACs per sample (log scale)")
        ax.set_ylabel(f"test {metric}" + ("" if up else "  (axis flipped: up is better)"))
    if not n:
        plt.close(fig)
        return None
    h, drawn = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="D", ls="none", ms=8, color=INK),
          plt.Line2D([], [], marker="s", ls="none", ms=8, color=SURFACE, mec=INK, mew=1.5)]
    if mixed:
        h.append(plt.Line2D([], [], marker=r"$68\%$", ls="none", ms=17, color=MUTED, mew=0))
    if variant == "ded+ded3":
        h.append(plt.Line2D([], [], ls="--", marker="o", lw=1.6, ms=6, color=MUTED, mfc=SURFACE,
                            mew=1.5))
    lab = drawn + ["fixed resolution, monthly input", "fixed resolution, seasonal input"]
    if mixed:
        lab.append("coverage: share of (month, pixel) positions under a crop, in the curve's "
                   "colour")
    if variant == "ded+ded3":
        lab.append("same pair, crops join at block 3 (coverage as the block-0 point "
                   "with the same count)")
    fig.legend(h, lab, loc="lower left", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.005, 0.09))
    fig.suptitle("Mixed-resolution OlmoEarth tiny against fixed patch sizes: accuracy for compute",
                 x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold")
    sub = sub.replace("CAP", "the cap (196 ps16 / 512 ps8 / 1,024 ps4 crops)"
                      if grid_mode == "stagger" else "2,048")
    sub = sub.replace("SEEDS", "1 seed" if grid_mode == "stagger" else "2 seeds")
    fig.text(0.012, 0.915, subtitle if subtitle is not None else "Mixed models: " + sub,
             fontsize=10.5, color=MUTED)
    fig.text(0.012, 0.012, footnote if footnote is not None else
             "Fixed models: S = patch size in px (16 / 8 / 4), T = input steps (seasonal or "
             "monthly composites).\nExtra month crops sit on a fixed staggered grid "
             f"(--grid_mode {grid_mode}"
             + "): not random, not learned."
             + ("  Each crop size sits on its OWN patch grid (ps16 crops on the season grid); "
                "coverage is computed, not measured." if grid_mode == "stagger" else "")
             + "\nSeasonal points: mean of 2 seeds.  Monthly points: 1 seed." + note,
             fontsize=8.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.175, 1, 0.91))
    place_coverage(fig, pending, fixed_pts)
    out = Path(f"figures/res_pairs_tradeoff_{grid_mode}_{tag}{suffix}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_mode", default="shared", choices=("shared", "c2f", "stagger"))
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "font.family": "DejaVu Sans"})
    for variant in VARIANTS:
        out = draw(a.grid_mode, variant)
        print(out or f"({variant}: no results yet)")


if __name__ == "__main__":
    main()
