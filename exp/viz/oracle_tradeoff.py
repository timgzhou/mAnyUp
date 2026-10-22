"""Location-constrained mixed resolution on the accuracy-for-compute plot of
exp.viz.res_pairs_tradeoff, against the fixed-resolution models only (its grid-crop curves are
left out: they place crops per (month, cell), not per location). Every line: ps16 seasons + ps8
crops at n locations, each at EVERY present month, n = 8 / 16 / 32 (rounds 49 / 52 / 53):
    oracle            the n highest-loss windows of the LR model vs the TRUE label, candidate
                      windows centred on every ps4 cell (neighbours overlap by half)
    oracle, stride 6  the same on 10 x 10 evenly spread windows, <= 2 px overlap (--loc_stride 6)
    oracle, stride 8  the same on the 8 x 8 non-overlapping windows (--loc_stride 8)
    grid              a content-blind lattice of the 8 x 8 windows (--selector grid)
    random            n random locations (windows centred on random ps4 cells)

x = inference GMACs per sample from `finetune_inject --flops` (results/<task>/inject_flops.csv),
INCLUDING the oracle's crop-free LR pass that picks the locations. Missing runs are skipped.

    python -m exp.viz.oracle_tradeoff  -> figures/res_pairs_tradeoff_shared_dedicated_oracle.png
"""
import re

import matplotlib.pyplot as plt
import pandas as pd

from exp.viz import res_pairs_tradeoff as rp

K = {"0.0312": 8, "0.0625": 16, "0.125": 32}                  # k_frac tag -> locations
SERIES = [  # (selector, run tag after cpx8, label, colour, marker, size, n labels)
    ("oracle", "_st8", "oracle locations, stride 8 (no overlap; label-informed)", "#7b1fa2",
     "*", 14, True),
    ("oracle", "_st6", "oracle locations, stride 6 (<= 2 px overlap; label-informed)", "#e8710a",
     "*", 12, False),
    ("oracle", "", "oracle locations, stride 4 (label-informed)", "#d81b8c", "*", 11, False),
    ("grid", "", "grid locations (regular lattice)", "#2a78d6", "o", 7, False),
    ("random", "", "random locations", "#8c8b86", "o", 6, False),
]


def points(task, sel, tag):
    """-> DataFrame (n, x GMACs, y test primary) for one location-constrained native_set arm."""
    pat = (rf"_{sel}_L0_k(0\.0312|0\.0625|0\.125)_loc_native_set_cpx8{tag}"
           r"_poscenter_lr0\.001_ep\d+$")
    hit = lambda col: col.astype(str).str.match(rf".*{pat}")          # noqa: E731
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False)
    f = f[hit(f.run)].drop_duplicates("run", keep="last")
    y = {K[re.search(pat, r).group(1)]: v for r, v in zip(f.run, f.test_primary)}
    fl = pd.read_csv(f"results/{task}/inject_flops.csv", usecols=range(6))
    fl = fl[hit(fl.run)].drop_duplicates("run", keep="last")
    x = {K[re.search(pat, r).group(1)]: v for r, v in zip(fl.run, fl.gmacs)}
    return pd.DataFrame([(n, x[n], y[n]) for n in sorted(y) if n in x], columns=["n", "x", "y"])


def overlay(ax, task, up, pending):
    for sel, tag, label, col, mk, ms, labelled in SERIES:
        p = points(task, sel, tag)
        ax.plot(p.x, p.y, "-" + mk, color=col, lw=2.2 if mk == "*" else 1.8, ms=ms,
                mec=rp.SURFACE, mew=0.8, zorder=5 if mk == "*" else 4, label=label)
        if labelled:
            for r in p.itertuples():
                pending.append((ax, r.x, r.y, f"{r.n} loc", col))


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": rp.MUTED,
                         "axes.labelcolor": rp.MUTED, "xtick.color": rp.MUTED,
                         "ytick.color": rp.MUTED, "text.color": rp.INK,
                         "font.family": "DejaVu Sans"})
    sub = ("Mixed models: ps16 seasons + ps8 crops at n LOCATIONS, each at EVERY present month; "
           "n = 8 / 16 / 32 left to right on each line; full FT, native_set readout, block 0 "
           "(1 seed)")
    foot = ("Fixed models: S = patch size in px (16 / 8 / 4), T = input steps (seasonal or "
            "monthly composites).  Seasonal points: mean of 2 seeds.  Monthly points: 1 seed.\n"
            "Oracle: the n windows where the LR-only model's loss vs the TRUE label is highest "
            "(reads labels at test time: a ceiling, not a method); its GMACs (measured, --flops) "
            "include that LR pass.\nStride 4: a candidate window centred on every ps4 cell "
            "(neighbours overlap by half); stride 6: 10 x 10 windows, <= 2 px overlap; stride 8: the "
            "8 x 8 non-overlapping windows. Missing "
            "points: runs still training.")
    print(rp.draw("shared", "ded", overlay, "_oracle", mixed=False, subtitle=sub, footnote=foot))


if __name__ == "__main__":
    main()
