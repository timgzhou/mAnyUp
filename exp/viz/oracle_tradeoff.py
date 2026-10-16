"""The label-error oracle (rounds 49 / 53) on the accuracy-for-compute plot of
exp.viz.res_pairs_tradeoff, against the fixed-resolution models only (the grid-crop curves are
left out: they place crops per (month, cell), not per location): ps16 seasons + ps8 crops at the n
locations with the highest LR loss vs the TRUE label, every present month, n = 8 / 16 / 32.

x = inference GMACs per sample from `finetune_inject --flops` (results/<task>/inject_flops.csv),
INCLUDING the oracle's crop-free LR pass that picks the locations.

    python -m exp.viz.oracle_tradeoff  -> figures/res_pairs_tradeoff_shared_dedicated_oracle.png
"""
import re

import matplotlib.pyplot as plt
import pandas as pd

from exp.viz import res_pairs_tradeoff as rp

ORACLE = "#c0127a"
K = {"0.0312": 8, "0.0625": 16, "0.125": 32}                  # k_frac tag -> locations


def oracle_points(task, stride=""):
    """-> DataFrame (n, x GMACs, y test primary) for the native_set oracle arms."""
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False)
    pat = (rf"_oracle_L0_k(0\.0312|0\.0625|0\.125)_loc_native_set_cpx8{stride}"
           r"_poscenter_lr0\.001_ep\d+$")
    hit = lambda col: col.astype(str).str.match(rf".*{pat}")          # noqa: E731
    f = f[hit(f.run)].drop_duplicates("run", keep="last")
    y = {K[re.search(pat, r).group(1)]: v for r, v in zip(f.run, f.test_primary)}
    fl = pd.read_csv(f"results/{task}/inject_flops.csv", usecols=range(6))
    fl = fl[hit(fl.run)].drop_duplicates("run", keep="last")
    x = {K[re.search(pat, r).group(1)]: v for r, v in zip(fl.run, fl.gmacs)}
    rows = [(n, x[n], y[n]) for n in sorted(y) if n in x]
    return pd.DataFrame(rows, columns=["n", "x", "y"])


def overlay(ax, task, up, pending):
    p = oracle_points(task)
    if not len(p):
        return
    ax.plot(p.x, p.y, "-*", color=ORACLE, lw=2.2, ms=13, mec=rp.SURFACE, mew=0.8, zorder=5,
            label="ps16 seasons + ps8 month crops at ORACLE locations (label-informed)")
    for r in p.itertuples():
        pending.append((ax, r.x, r.y, f"{r.n} loc", ORACLE))


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": rp.MUTED,
                         "axes.labelcolor": rp.MUTED, "xtick.color": rp.MUTED,
                         "ytick.color": rp.MUTED, "text.color": rp.INK,
                         "font.family": "DejaVu Sans"})
    sub = ("Oracle: ps16 seasons + ps8 crops at the n locations (n loc) where the LR-only model's "
           "loss vs the TRUE label is highest, each at EVERY present month; full FT, native_set "
           "readout, crops join at block 0 (1 seed)")
    foot = ("Fixed models: S = patch size in px (16 / 8 / 4), T = input steps (seasonal or "
            "monthly composites).  Seasonal points: mean of 2 seeds.  Monthly points: 1 seed.\n"
            "Oracle GMACs (measured, --flops) include its crop-free LR pass that picks the "
            "locations; it reads labels at test time, so it is a ceiling, not a method.")
    print(rp.draw("shared", "ded", overlay, "_oracle", mixed=False, subtitle=sub, footnote=foot))


if __name__ == "__main__":
    main()
