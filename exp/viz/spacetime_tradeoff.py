"""Space then time (selectors.SpaceTime, scripts/slurm/mixres_spacetime.sh) on the accuracy-for-
compute plot of exp.viz.res_pairs_tradeoff, PASTIS, one panel per setup, against the fixed-
resolution OlmoEarth tiny models:

    space + time   n locations, each with one / two of its seasons:
                   entropy + attention  highest-entropy windows; the seasons the readout attends
                                      to most in the crop-free pass (no trained scorer)
                   entropy + learned  highest-entropy windows; the learned season score alone
                                      (season_residual xattn_conv, oracle ListNet, no prior)
                   grid + grid        n-window lattice; every 4th / 2nd present month
                   random + random    n random windows; random season(s) per location
    space only     n locations x every present month (the spatial table): entropy (legacy
                   sent_srall_warm runs; GMACs of the location oracle's trace, the same crops +
                   the uniform pass), grid
    time only      every location, one -> two seasons: learned (as above) / random

Also draws the fixed-resolution models against grid + grid, random + random and entropy (train-
set threshold) + attention (and with cross attention), one season, both setups on one axis
(figures/spacetime_baselines.png), and writes the LaTeX table of the space + time arms
(figures/spacetime_table.tex).

    python -m exp.viz.spacetime_tradeoff
"""
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp.viz import res_pairs_tradeoff as rp

SETUPS = {   # setup -> (lr patch, ns, legacy entropy run tag, crop geometry tag, title)
    "s16m8": (16, (8, 16, 32), "_cpx8_st8_poscenter_sent_srall_warm_lr0.001_ep50", "c8",
              "ps16 seasons + ps8 month crops"),
    "s8m4": (8, (32, 64, 128), "_poscenter_sent_srall_warm_lrps8_lr0.001_ep50", "c4",
             "ps8 seasons + ps4 month crops"),
}
ST = [   # run tag after ps<lr>_, keep, label, colour, line style, marker
    ("entropy-top+attention-every_location-one", "one",
     "entropy + attention, one season", "#b71c1c", "-", "^"),
    ("entropy-top+attention-every_location-half", "half",
     "entropy + attention, two seasons", "#b71c1c", "--", "^"),
    ("entropy-top+season_residual-every_location-one-resoraclexattn_conv-noneprior", "one",
     "entropy + learned, one season", "#e8710a", "-", "v"),
    ("entropy-top+season_residual-every_location-half-resoraclexattn_conv-noneprior", "half",
     "entropy + learned, two seasons", "#e8710a", "--", "v"),
    ("grid-top+none-month_grid4", "one", "grid + grid, one season", "#2a78d6", "-", "o"),
    ("grid-top+none-month_grid2", "half", "grid + grid, two seasons", "#2a78d6", "--", "o"),
    ("random-top+season_random-every_location-one", "one", "random + random, one season",
     "#8c8b86", "-", "o"),
    ("random-top+season_random-every_location-half", "half", "random + random, two seasons",
     "#8c8b86", "--", "o"),
]
QUAD = [   # quadrant replication of entropy + attention with the spatial threshold of the
    # training distribution (scripts/slurm/mixres_quadrant_rep.sh, arm global): keep, tag, label
    ("one", "-hhone-crop-score", "entropy (global threshold) + attention, one season", "-"),
    ("half", "-hhhalf-hlhalf-crop-score", "entropy (global threshold) + attention, two seasons",
     "--"),
]
QUAD_COL = "#1b7a3e"
BASE_COL = "#8c8b86"    # content-blind baselines (grid, random) on the baselines figure
# the same, one season, with cross attention in pass 2 (model.attn cross_hr: crop tokens read the
# season tokens and the sample's other crops, season tokens never read crops, so inference
# reuses the crop-free pass; ATTN=cross_hr in mixres_quadrant_rep.sh): tag, label, colour
XHR = ("-hhone-crop-score-xhr", "entropy (global threshold) + attention, one season, cross attn",
       "#00838f")
# ours as ONE run (train.stage1_*, SCHED=50 in mixres_quadrant_rep.sh): 50 epochs with the
# location oracle (joint attention), then 50 with entropy + attention and cross attention (B)
SCHED = "-hhone-crop-score-xhr-s1o50"
S4M2 = (4, "ps4 seasons + ps2 month crops")     # 4 px windows, each four ps2 crop tokens (c4p2)
TIME = [   # run tag, label, colour
    ("season_residual-every_location-{k}-resoraclexattn_conv-noneprior_n0",
     "time only (every location): learned, one -> two seasons", "#f4b37a"),
    ("season_random-every_location-{k}_n0", "time only (every location): random, one -> two "
     "seasons", "#c9c7c0"),
]


def mixres():
    rows = [r for r in csv.DictReader(open("results/pastis/mixres.csv"))
            if "_seed0_" in r["run"] and "_frozen_" not in r["run"] and r["test_miou"]
            and "_warmlr_" not in r["run"]      # (warm from the fixed-resolution model)
            and not r["train.init_ckpt"].startswith("mixres_")]   # (follow-ups of a mixres run)
    return pd.DataFrame([(r["run"], int(r["select.n"]), float(r["test_miou"]),
                          float(r["gmacs"] or "nan"), r["train.init_ckpt"]) for r in rows],
                        columns=["run", "n", "y", "x", "init"]).drop_duplicates("run", keep="last")


def sched_points(ps, geo):
    """-> q_s, x, y (mean over seeds), lo / hi (seed range), seeds of the one-run schedule B at
    season patch ps and crop geometry geo (c8: 8 px crops; c10p5: 10 px windows as ps5 crops)."""
    m = pd.read_csv("results/pastis/mixres.csv", low_memory=False, usecols=["run", "test_miou", "gmacs"])
    m = m[m.run.str.match(rf"mixres_pastis_ps{ps}_quadg-skip-qs[\d.]+-qt0\.5{SCHED}_n0_{geo}_seed\d+_")]
    m = m.drop_duplicates("run", keep="last")
    m["q_s"] = m.run.str.extract(r"-qs([\d.]+)-")[0].astype(float)
    return m.groupby("q_s").agg(x=("gmacs", "mean"), y=("test_miou", "mean"), lo=("test_miou", "min"),
                                hi=("test_miou", "max"), seeds=("run", "size")).reset_index()


def st_points(d, ps, tag):
    p = d[d.run.str.startswith(f"mixres_pastis_ps{ps}_{tag}_n")]
    return p.groupby("n").agg(x=("x", "mean"), y=("y", "mean")).reset_index()


SHARED_K = 0.25     # ps8 rerun: every q_s from the location oracle at this k (LOC_K=0.25)
ORACLE_K = {16: (0.03125, 0.0625, 0.125), 8: (0.125, 0.25, 0.5)}   # location oracles on disk


def own_k(ps, q_s):
    """The location oracle a q_s run warms from by default (mixres_quadrant_rep.sh loc: the
    nearest budget on a log scale; k counts ps4 cells, q_s windows: k = q_s / 4 at ps16)."""
    q = q_s / (4 if ps == 16 else 1)
    return min(ORACLE_K[ps], key=lambda k: abs(math.log(k / q)))


def quad_points(d, ps, tag, shared=None):
    """-> q_s, x, y of the global-threshold quadrant runs (every q_s run so far), each warm
    from the location oracle at its own budget, or (shared = k) all from the one at k."""
    p = d[d.run.str.match(rf"mixres_pastis_ps{ps}_quadg-skip-qs[\d.]+-qt0\.5{tag}_n0_")].copy()
    p["q_s"] = p.run.str.extract(r"-qs([\d.]+)-")[0].astype(float)
    k = p.init.str.extract(r"_k([\d.]+)_")[0].astype(float)
    want = shared if shared else p.q_s.map(lambda q: own_k(ps, q))
    p = p[(k - want).abs() < 0.01 * want]          # (names round k: k0.0312)
    return p.groupby("q_s").agg(x=("x", "mean"), y=("y", "mean")).reset_index()


def space_entropy(setup):
    ps, ns, tag, _, _ = SETUPS[setup]
    f = pd.read_csv("results/pastis/inject_fft.csv", low_memory=False,
                    usecols=["run", "test_primary"])
    fl = pd.read_csv("results/pastis/oracle_flops.csv")
    fl = fl[(fl.setup == setup) & (fl.selector == "oracle") & (fl.stride == 4)]
    fl = fl.drop_duplicates("n", keep="last").set_index("n").gmacs
    rows = []
    for n in ns:
        run = f"inject_fft_pastis_tiny_oracle_bothalf_L0_k{n / 256:.3g}_loc_native_set{tag}"
        r = f[f.run.astype(str) == run]
        if len(r):
            rows.append((n, fl[n], r.test_primary.iloc[-1]))
    return pd.DataFrame(rows, columns=["n", "x", "y"])


def space_grid(d, setup):
    ps, ns, _, geo, _ = SETUPS[setup]
    p = d[d.run.str.match(rf"mixres_pastis_ps{ps}_grid-top_n\d+_{geo}_warm_")]
    return p[p.n.isin(ns)].groupby("n").agg(x=("x", "mean"), y=("y", "mean")).reset_index()


def draw(d):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": rp.MUTED,
                         "axes.labelcolor": rp.MUTED, "xtick.color": rp.MUTED,
                         "ytick.color": rp.MUTED, "text.color": rp.INK,
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.9), facecolor=rp.SURFACE)
    fixed = rp.fixed("pastis", "pastis")
    for ax, (setup, (ps, ns, _, _, title)) in zip(axes, SETUPS.items()):
        ax.set_facecolor(rp.SURFACE)
        for label, x, y, monthly in fixed:
            ax.plot(x, y, "D" if monthly else "s", ms=8, color=rp.INK if monthly else rp.SURFACE,
                    mec=rp.INK, mew=1.5, zorder=4, ls="none")
            ax.annotate(label, (x, y), xytext=(0, 9 if monthly else -9),
                        textcoords="offset points", ha="center",
                        va="bottom" if monthly else "top", fontsize=9, zorder=5)
        e, g = space_entropy(setup), space_grid(d, setup)
        ax.plot(e.x, e.y, "-s", color="#7b1fa2", lw=2, ms=6, mec=rp.SURFACE, zorder=4,
                label="space only (all months): entropy")
        ax.plot(g.x, g.y, "-s", color="#9fc3ec", lw=2, ms=6, mec=rp.SURFACE, zorder=3,
                label="space only (all months): grid")
        for tag, _, label, col, ls, mk in ST:
            p = st_points(d, ps, tag)
            ax.plot(p.x, p.y, ls + mk, color=col, lw=2, ms=7 if mk in "^v" else 5.5,
                    mfc=col if ls == "-" else rp.SURFACE, mec=col, mew=1.4, zorder=5, label=label)
            if mk == "^" and ls == "-":
                for r in p.itertuples():
                    ax.annotate(f"n={r.n}", (r.x, r.y), xytext=(0, 9), textcoords="offset points",
                                ha="center", fontsize=7.5, color=col, zorder=6)
        for keep, tag, label, ls in QUAD:
            p = quad_points(d, ps, tag)
            ax.plot(p.x, p.y, ls + "^", color=QUAD_COL, lw=2, ms=7,
                    mfc=QUAD_COL if ls == "-" else rp.SURFACE, mec=QUAD_COL, mew=1.4, zorder=6,
                    label=label)
        p = quad_points(d, ps, XHR[0])
        ax.plot(p.x, p.y, "-P", color=XHR[2], lw=2, ms=8, mec=rp.SURFACE, zorder=6, label=XHR[1])
        for tag, label, col in TIME:
            p = pd.concat([d[d.run.str.startswith(f"mixres_pastis_ps{ps}_{tag.format(k=k)}_")]
                           for k in ("one", "half")])
            ax.plot(p.x, p.y, ":D", color=col, lw=1.8, ms=5.5, mec=rp.SURFACE, zorder=4,
                    label=label)
        ax.set_xscale("log")
        ax.grid(True, color=rp.GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"PASTIS, {title}", loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("GMACs per sample (log scale)")
        ax.set_ylabel("test mIoU")
    h, lab = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="D", ls="none", ms=8, color=rp.INK),
          plt.Line2D([], [], marker="s", ls="none", ms=8, color=rp.SURFACE, mec=rp.INK, mew=1.5)]
    lab += ["fixed resolution, monthly input", "fixed resolution, seasonal input"]
    fig.legend(h, lab, loc="lower left", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.005, 0.095))
    fig.suptitle("Choosing where AND when: space then time, against fixed patch sizes",
                 x=0.012, y=0.99, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.012, 0.935, "Mixed models pick n locations (n = 8 / 16 / 32 ps8 windows; "
             "32 / 64 / 128 ps4 cells), then one or two seasons of months at each; full FT, "
             "native_set, block 0, 1 seed", fontsize=10.5, color=rp.MUTED)
    fig.text(0.012, 0.012,
             "Entropy: the model's own crop-free pass (its GMACs are included); global threshold: "
             "a location is picked when its entropy passes the val split's (1 - q_s) quantile "
             "(adaptive crops per tile).\nAttention: the "
             "readout's attention over seasons in that same pass, no trained scorer. Cross "
             "attention: crop tokens attend to the season tokens and each other, season tokens "
             "never to crops (the crop-free pass is reused).\nLearned: a "
             "season score trained on the true per-season loss ranking (labels at training "
             "only).\nGrid time: every 4th / 2nd present month. Fixed models: S = patch size in "
             "px, T = seasonal or monthly composites (seasonal: mean of 2 seeds).",
             fontsize=8.5, color=rp.MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.3, 1, 0.92))
    out = Path("figures/spacetime_tradeoff.png")
    fig.savefig(out, dpi=150, facecolor=rp.SURFACE)
    plt.close(fig)
    return out


PAIR = {   # (season patch, crop geometry) -> colour: one hue per season patch, dark for the
    # pair whose crops are ~half the season patch, light for the other
    (16, "c8"): "#1565c0", (16, "c12"): "#90caf9", (14, "c14p7"): "#ad1457",
    (12, "c12p6"): "#6d4c41", (10, "c10p5"): "#2e7d32", (10, "c8"): "#a5d6a7",
    (8, "c4"): "#00838f", (8, "c6"): "#80deea", (6, "c6p3"): "#e65100", (6, "c4"): "#ffb74d",
    (4, "c4p2"): "#6a1b9a"}


def pair_name(ps, geo):
    c = re.match(r"c(\d+)(?:p(\d+))?", geo)
    return rf"$S_{{{ps}}}T_{{sea}}+S_{{{c.group(2) or c.group(1)}}}T_{{mon}}$"   # where the label sits beside its curve


def draw_ours(ax, ps, geo):
    """Ours at one resolution pair (seed means, bars: seed range), in the pair's colour."""
    p = sched_points(ps, geo).sort_values("x")
    if not len(p):
        return False
    col = PAIR[(ps, geo)]
    ax.plot(p.x, p.y, "-*", color=col, lw=2.4, ms=14, mec=col, mew=1.4, zorder=6)
    multi = p[p.seeds > 1]
    ax.errorbar(multi.x, multi.y, yerr=[multi.y - multi.lo, multi.hi - multi.y], fmt="none",
                ecolor=col, elinewidth=1.4, capsize=4, zorder=6)
    return True


def draw_baselines(d):
    """The fixed-resolution models against the mixed models with one season per location
    (entropy (global threshold) + attention, cross attention, one run), grid + grid and
    random + random, every resolution pair on one axis (figures/spacetime_baselines.png)."""
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": rp.MUTED,
                         "axes.labelcolor": rp.MUTED, "xtick.color": rp.MUTED,
                         "ytick.color": rp.MUTED, "text.color": rp.INK,
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(10, 6.6), facecolor=rp.SURFACE)
    ax.set_facecolor(rp.SURFACE)
    fixed = rp.fixed("pastis", "pastis")
    style = {True: ("-", "D", rp.INK), False: ("--", "s", rp.SURFACE)}   # monthly, seasonal
    for monthly, (ls, mk, mfc) in style.items():
        pts = sorted((x, y) for _, x, y, m in fixed if m == monthly)
        ax.plot(*zip(*pts), ls + mk, color=rp.INK, lw=1.2, ms=7, mfc=mfc, mec=rp.INK, mew=1.4,
                alpha=0.75, zorder=3)
    for label, x, y, monthly in fixed:
        ps = re.search(r"S_\{(\d+)\}", label).group(1)
        # monthly labels above their point, seasonal below, but for the two that would sit
        # on our curve / the seasonal labels
        up = monthly and ps not in ("10", "16", "8", "6")
        right = monthly and ps == "10"                       # (between S6 season and ours)
        ax.annotate(rf"$S_{{{ps}}}T_{{{'mon' if monthly else 'sea'}}}$", (x, y),
                    xytext=(8, -2) if right else (0, 8 if up else -8), textcoords="offset points",
                    ha="left" if right else "center",
                    va="center" if right else "bottom" if up else "top", fontsize=9, zorder=5)
    base = {"grid": ("--", "o"), "random": (":", "d")}
    for tag, keep, *_ in ST:
        kind = tag.split("-")[0]
        if kind not in base or keep != "one":
            continue
        for ps, *_ in SETUPS.values():
            p = st_points(d, ps, tag)
            ax.plot(p.x, p.y, base[kind][0] + base[kind][1], color=BASE_COL, lw=1.6, ms=6,
                    alpha=0.55, zorder=4)
    drawn = [k for k in PAIR if draw_ours(ax, *k)]
    ax.set_xscale("log")
    ax.grid(True, color=rp.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("GMACs per sample (log scale)")
    ax.set_ylabel("test mIoU")
    h = [plt.Line2D([], [], color=PAIR[k], marker="*", ms=13, lw=2.4) for k in drawn]
    h += [plt.Line2D([], [], color=rp.INK, marker="D", ms=7, lw=1.2, alpha=0.75),
         plt.Line2D([], [], color=rp.INK, marker="s", ms=7, mfc=rp.SURFACE, ls="--", lw=1.2,
                    alpha=0.75),
         plt.Line2D([], [], color=BASE_COL, marker="o", ms=6, ls="--", lw=1.6, alpha=0.55),
         plt.Line2D([], [], color=BASE_COL, marker="d", ms=6, ls=":", lw=1.6, alpha=0.55)]
    lab = [f"ours {pair_name(*k)}" for k in drawn]
    lab += ["fixed resolution, monthly", "fixed resolution, seasonal", "grid + grid",
            "random + random"]
    ax.legend(h, lab, loc="lower right", ncol=2, frameon=False, fontsize=9.5,
              title="ours: entropy + attention, cross attention", title_fontsize=9.5,
              alignment="left")
    fig.suptitle("PASTIS: mixed resolution against fixed patch sizes",
                 x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("figures/spacetime_baselines.png")
    fig.savefig(out, dpi=150, facecolor=rp.SURFACE)
    plt.close(fig)
    return out


def table(d):
    none = {16: .431, 8: .523}       # no crops (the spatial / time tables)
    cells = {}
    for setup, (ps, ns, *_) in SETUPS.items():
        e = space_entropy(setup).set_index("n")
        for keep in ("one", "half"):
            for n in ns:
                if n in e.index:
                    cells[(keep, "all", setup, n)] = (e.y[n], e.x[n])
    for tag, keep, label, *_ in ST:
        for setup, (ps, ns, *_) in SETUPS.items():
            p = st_points(d, ps, tag).set_index("n")
            for n in ns:
                if n in p.index:
                    cells[(keep, label.split(",")[0], setup, n)] = (p.y[n], p.x[n])
    for keep, tag, *_ in QUAD:          # q_s = n / candidate windows (64 | 256)
        for key, suffix in (("global", ""), ("global_xhr", "-xhr")):   # full / cross attention
            for setup, (ps, ns, *_) in SETUPS.items():
                p = quad_points(d, ps, tag + suffix).set_index("q_s")
                for n in ns:
                    q = n / (64 if ps == 16 else 256)
                    if q in p.index:
                        cells[(keep, key, setup, n)] = (p.y[q], p.x[q])
    names = [("grid + grid", "Grid"), ("random + random", "Random"),
             ("entropy + learned", "Entropy + learned"),
             ("entropy + attention", "Entropy + attention"),
             ("global", "\\quad global threshold"),
             ("global_xhr", "\\quad\\quad cross attention")]
    ref = ("all", "Entropy, all months")     # space only, every present month (not bolded)
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lcccccc}",
         r"\toprule",
         r" & \multicolumn{3}{c}{ps16 sea + ps8 mon} & \multicolumn{3}{c}{ps8 sea + ps4 mon} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
         r"$n$ & 8 & 16 & 32 & 32 & 64 & 128 \\", r"\midrule",
         r"No crops & \multicolumn{3}{c}{" + f"{none[16]:.3f}" + r"} & \multicolumn{3}{c}{"
         + f"{none[8]:.3f}" + r"} \\"]
    for keep, head in (("one", "one season"), ("half", "two seasons")):
        L += [r"\midrule", r"\multicolumn{7}{l}{\emph{" + head + r"}} \\"]
        cols = [(s, n) for s, (_, ns, *_) in SETUPS.items() for n in ns]
        best = {c: max(cells[(keep, k, *c)][0] for k, _ in names if (keep, k, *c) in cells)
                for c in cols}
        for k, nm in names + [ref] * (keep == "half"):
            if k == "all":
                L.append(r"\cmidrule(lr){1-7}")
            row = []
            for c in cols:
                if (keep, k, *c) not in cells:
                    row.append("")
                    continue
                y, x = cells[(keep, k, *c)]
                v = f"{y:.3f}"
                v = r"\textbf{" + v + "}" if y == best[c] and k != "all" else v
                row.append(v + r" {\scriptsize(" + f"{x:.1f}" + ")}")
            L.append(f"{nm} & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{PASTIS test mIoU (GMACs per sample in parentheses) when choosing both "
          r"\emph{where} and \emph{when}: each method picks $n$ spatial locations and, at each, "
          r"one or two seasons of monthly crops ($\approx n \times \frac{12}{4} \times M$ or "
          r"$n \times \frac{12}{2} \times M$ additional tokens, $M$ modalities). Grid: a regular "
          r"lattice of locations, every 4th / 2nd month. Random: random locations and seasons. "
          r"Entropy + learned: the highest-entropy locations of the model's own crop-free pass, "
          r"each with the seasons of lowest learned score (trained on the true per-season loss "
          r"ranking; no labels at test). Entropy + attention: the same locations, each with the "
          r"seasons the readout attends to most in the crop-free pass (no trained scorer). "
          r"Entropy, all months: the same locations with every month (space only, for "
          r"reference). Global threshold: the same, but a location is picked when its "
          r"entropy exceeds the validation split's $(1 - n / N)$ quantile ($N$ candidate "
          r"windows), so each tile gets as many crops as it needs; the GMACs are the test-set "
          r"average. Every row uses full self-attention over season and crop tokens except "
          r"cross attention: crop tokens attend to the season tokens and to the tile's other "
          r"crops, season tokens never to crops, so the crop-free pass is reused and only the "
          r"crop tokens are computed again. Entropy GMACs include the crop-free pass. Fixed-resolution references: "
          r"S16 seasonal .429 (1.7), S16 monthly .462 (6.3), S8 seasonal .525 (10.6), "
          r"S8 monthly .572 (53.7), S4 seasonal .570 (100.2), S4 monthly .614 (673.2).}",
          r"\label{tab:spacetime}", r"\end{table}"]
    out = Path("figures/spacetime_table.tex")
    out.write_text("\n".join(L) + "\n")
    return out


def main():
    d = mixres()
    print(draw(d))
    print(draw_baselines(d))
    print(table(d))


if __name__ == "__main__":
    main()
