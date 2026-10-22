"""Where to put the high-res crops: every label oracle against its content-blind controls, the
per-(month, cell) crop curves and the fixed-resolution models, on the accuracy-for-compute plot
of exp.viz.res_pairs_tradeoff. One figure per setup (and per candidate stride for ps16 + ps8):

  s16m8   ps16 seasons + ps8 crops, n = 8 / 16 / 32 windows' worth of crops (rounds 49 - 56)
  s8m4    ps8 seasons + ps4 crops,  n = 32 / 64 / 128 ps4 cells' worth   (rounds 55 - 56)

Every location-constrained arm spends n x present months crops (n windows at every month, or
the same number of crops as (season, window) pairs: every present month of a picked season).
Stars: label oracles (read the TRUE label at test time -- ceilings); circles: content-blind.
    location oracle   the n windows with the highest loss of the LR-only model (all seasons)
    season oracle     the highest-loss (season, window) pairs, per-season loss maps
    worse half        highest-loss windows, each its higher-loss half of seasons
    better half       the same windows, each its LOWER-loss half of seasons
    location grid / random      n windows on a lattice / at random, every month
    season grid / random        (season, window) pairs: a lattice per season shifted one coset
                                per season / at random; same budget as the season oracle
    staggered grid (per month)  crops per (month, cell) on a lattice staggered across months
                                (--grid_mode shared), one model dedicated to each count
    fully random (per month)    crops at uniformly random (month, pixel), dedicated models
x = GMACs per sample (location arms: exp.bench.oracle_flops, incl. the oracles' LR passes;
per-month curves: their own sweep). Missing points: runs still training.

    python -m exp.viz.oracle_season_tradeoff
"""
import matplotlib.pyplot as plt
import pandas as pd

from exp.viz import res_pairs_tradeoff as rp

SETUPS = {   # setup -> (run tag after "native_set", stride tag fn, ns, per-month pair, title)
    "s16m8": (lambda st: "_cpx8" + ("" if st == 4 else f"_st{st}") + "_poscenter_lr",
              (8, 16, 32), ("16", "8"), "ps16 seasons + ps8 crops"),
    "s8m4": (lambda st: ("" if st == 4 else f"_st{st}") + "_poscenter_lrps8_lr",
             (32, 64, 128), ("8", "4"),
             "ps8 seasons + ps4 crops"),
}
ORACLES = [  # selector, label, colour
    ("oracle", "location oracle", "#d81b8c"),
    ("oracle_season", "season oracle", "#7b1fa2"),
    ("oracle_tophalf", "worse-half seasons oracle", "#e8710a"),
    ("oracle_bothalf", "better-half seasons oracle", "#b71c1c"),
]
CONTROLS = [  # selector, label, colour, line style
    ("grid", "location grid", "#2a78d6", "-"),
    ("random", "location random", "#8c8b86", "-"),
    ("season_grid", "season grid", "#2a78d6", "--"),
    ("season_random", "season random", "#8c8b86", "--"),
]
PER_MONTH = [  # selector in the run name, label, colour
    ("grid", "staggered grid, per (month, cell)", "#1a7f5a"),
    ("random", "fully random, per (month, pixel)", "#9cbf8f"),
]


def location_points(task, setup, sel, stride):
    """-> DataFrame (n, x, y): test metric from inject_fft.csv, GMACs from oracle_flops.csv."""
    tag, ns = SETUPS[setup][0](stride), SETUPS[setup][1]
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False, usecols=["run", "test_primary"])
    try:
        fl = pd.read_csv(f"results/{task}/oracle_flops.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["n", "x", "y"])
    fl = fl[(fl.setup == setup) & (fl.selector == sel) & (fl.stride == stride)]
    fl = fl.drop_duplicates("n", keep="last").set_index("n").gmacs
    rows = []
    for n in ns:
        run = f"_tiny_{sel}_L0_k{n / 256:.3g}_loc_native_set{tag}0.001_ep"
        r = f[f.run.astype(str).str.contains(run, regex=False)].drop_duplicates("run", keep="last")
        if len(r) and n in fl.index:
            rows.append((n, fl[n], r.test_primary.iloc[-1]))
    return pd.DataFrame(rows, columns=["n", "x", "y"])


def per_month_points(task, pair, sel):
    """Dedicated per-(month, cell) models (round 44 grid / round 56 random), each at its own
    crop count, tested with its own pick."""
    d = pd.read_csv(f"results/{task}/inject_ncrops.csv")
    d = d[d.run.str.contains(f"_{sel}_L0_") & d.run.str.contains("_free_")]
    d = d[d.run.map(rp.arm) == pair]
    d = d[d.run.map(rp.kind) == "ded"]
    if sel == "grid":
        d = d[d.grid_mode == "shared"]
    d = d[(d.pick == sel) & (d.n_test == d.n_max)].drop_duplicates(["run", "n_test"], keep="last")
    return d.groupby("n_test").agg(x=("gmacs", "mean"), y=("test_primary", "mean")).reset_index()


def make_overlay(setup, stride):
    pair = SETUPS[setup][2]

    def overlay(ax, task, up, pending):
        for sel, label, col in PER_MONTH:
            p = per_month_points(task, pair, sel)
            ax.plot(p.x, p.y, ":o", color=col, lw=1.8, ms=4.5, mec=rp.SURFACE, mew=0.6, zorder=3,
                    label=label)
        for sel, label, col, ls in CONTROLS:
            p = location_points(task, setup, sel, 4)
            ax.plot(p.x, p.y, ls + "o", color=col, lw=1.8, ms=5.5,
                    mfc=col if ls == "-" else rp.SURFACE, mec=col, mew=1.3, zorder=4, label=label)
        for sel, label, col in ORACLES:
            st = stride if sel != "random" else 4
            p = location_points(task, setup, sel, st)
            ax.plot(p.x, p.y, "-*", color=col, lw=2.1, ms=12, mec=rp.SURFACE, mew=0.7, zorder=5,
                    label=label)
            if sel == "oracle_bothalf":
                for r in p.itertuples():
                    pending.append((ax, r.x, r.y, f"n={r.n}", col))
    return overlay


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": rp.MUTED,
                         "axes.labelcolor": rp.MUTED, "xtick.color": rp.MUTED,
                         "ytick.color": rp.MUTED, "text.color": rp.INK,
                         "font.family": "DejaVu Sans"})
    for setup, (_, ns, _, title) in SETUPS.items():
        for stride in ((8, 6, 4) if setup == "s16m8" else (4, 2)):
            unit = "ps8 windows" if setup == "s16m8" else "ps4 cells"
            sub = (f"{title}: crops chosen by a label oracle (stars) or content-blind (circles); "
                   f"budget n = {' / '.join(map(str, ns))} {unit} x present months; full FT, "
                   "native_set, block 0 (1 seed)"
                   + f"; oracle candidate stride {stride} px")
            foot = (
                "Fixed models: S = patch size in px, T = seasonal or monthly composites (seasonal: "
                "mean of 2 seeds).  Oracles read the TRUE label at test time: ceilings, not "
                "methods; their GMACs include the LR-only pass that scores the candidates.\n"
                "Location arms crop n windows at EVERY present month; season arms crop (season, "
                "window) pairs = every present month of that season, same number of crops.  "
                "Better / worse half: the highest-loss windows, each its lower / higher-loss half "
                "of seasons.\nSeason grid: a lattice per season shifted one coset per season.  "
                "Per-(month, cell) curves: one model dedicated to each crop count, crops on a "
                "staggered lattice or at random (month, pixel).  Missing points: still training.")
            suffix = f"_oracles_{setup}" + ("" if setup == "s8m4" and stride == 4
                                            else f"_st{stride}")
            print(rp.draw("shared", "ded", make_overlay(setup, stride), suffix, mixed=False,
                          subtitle=sub, footnote=foot, ncol=4))


if __name__ == "__main__":
    main()
