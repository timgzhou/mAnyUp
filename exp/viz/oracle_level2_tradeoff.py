"""The two-layer (hierarchical) label oracles (rounds 59 / 60, finetune_inject --level2) on the
accuracy-for-compute plot of exp.viz.res_pairs_tradeoff, in the style of
exp.viz.oracle_season_tradeoff. One figure per setup:

  s16m8  layer 1 = ps16 seasons + ps8 month crops (better-half oracle, stride 8, n1 = 8 / 16 / 32)
         layer 2 = m4 (months at ps4) / d8 (raw dates at ps8) / d4 (raw dates at ps4)
  s8m4   layer 1 = ps8 seasons + ps4 month crops (better-half oracle, n1 = 32 / 64 / 128)
         layer 2 = m2 (months at ps2) / d2 (raw dates at ps2)

Layer 2 = the n2 = n1 / 2, n1, 2 n1 finest cells under a layer-1 crop where the layer-1 model's
loss is highest, each its lower-loss half of the cropped months. Each layer-1 budget n1 fans out
into its own short line: layer 1 + the extra fine-tune alone (l2none, n2 = 0), then n2 rising.
x = GMACs per sample (exp.bench.oracle_flops --level2, incl. both oracle passes). Missing points:
runs still training.

    python -m exp.viz.oracle_level2_tradeoff
"""
import re
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from exp.viz import oracle_season_tradeoff as ost
from exp.viz import res_pairs_tradeoff as rp

SETUPS = {   # setup -> (n1 budgets, run tag before "_l2"/"_warm", tag after "_warm", arms, title)
    "s16m8": ((8, 16, 32), "_cpx8_st8_poscenter", "", ("m4", "d8", "d4"),
              "ps16 seasons + ps8 month crops -> finer layer 2"),
    "s8m4": ((32, 64, 128), "_poscenter", "_lrps8", ("m2", "d2"),
             "ps8 seasons + ps4 month crops -> finer layer 2"),
}
ARM_STYLE = {   # arm -> (label, colour, marker)
    "m4": ("+ months at ps4 (M4)", "#d81b8c", "D"),
    "d8": ("+ raw dates at ps8 (D8)", "#e8710a", "s"),
    "d4": ("+ raw dates at ps4 (D4)", "#7b1fa2", "^"),
    "m2": ("+ months at ps2 (M2)", "#d81b8c", "D"),
    "d2": ("+ raw dates at ps2 (D2)", "#7b1fa2", "^"),
}
RANDOM = "#8c8b86"
LAYER1 = "#b71c1c"
CTRL = {"pastis": "d", "yieldsat_raw": "d", "biomassters": "m"}   # the random control's arm


def results(task):
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False,
                    usecols=["run", "test_primary"])
    f = f[f.run.astype(str).str.match(r".*_lr0\.001_ep\d+$")]
    return f.drop_duplicates("run", keep="last").set_index("run").test_primary


def flops(task):
    try:
        fl = pd.read_csv(f"results/{task}/level2_flops.csv")
    except FileNotFoundError:
        return {}
    fl = fl.drop_duplicates(["setup", "selector", "n", "n2"], keep="last")
    return {(r.setup, r.selector, r.n, r.n2): r.gmacs for r in fl.itertuples()}


def value(res, task, setup, n1, tag):
    """test primary of the layer-1 better-half run at n1 + `tag` (e.g. _l2m4n16_warm)."""
    _, pre, post, _, _ = SETUPS[setup]
    k = format(n1 / 256, ".3g")
    pat = re.escape(f"inject_fft_{task}_tiny_oracle_bothalf_L0_k{k}_loc_native_set{pre}{tag}"
                    f"{post}_lr0.001_ep") + r"\d+$"
    hit = res[res.index.str.match(pat)]
    if not len(hit) or not done(hit.index[-1]):
        return None
    return float(hit.iloc[-1])


@cache
def done(run):
    """a chain's links each log a row; only the finished run's row counts"""
    st = Path(f"checkpoints/{run}_state.pt")
    return st.exists() and bool(torch.load(st, weights_only=False, map_location="cpu").get("done"))


def fan(res, fl, task, setup, n1, sel, arm, pick):
    """(x, y, n2) of one layer-1 budget: the l2none point, then n2 = n1 / 2, n1, 2 n1."""
    pts = []
    base = value(res, task, setup, n1, "_warm")
    g = fl.get((setup, "l2none", n1, 0))
    if base is not None and g is not None:
        pts.append((g, base, 0))
    for n2 in (n1 // 2, n1, 2 * n1):
        v = value(res, task, setup, n1, f"_l2{arm}n{n2}{pick}_warm")
        g = fl.get((setup, sel, n1, n2))
        if v is not None and g is not None:
            pts.append((g, v, n2))
    return pts


def make_overlay(setup):
    ns, _, _, arms, _ = SETUPS[setup]
    pair = ("16", "8") if setup == "s16m8" else ("8", "4")
    l1_stride = 8 if setup == "s16m8" else 4

    def overlay(ax, task, up, pending):
        res, fl = results(task), flops(task)
        p = ost.per_month_points(task, pair, "grid")
        ax.plot(p.x, p.y, ":o", color="#1a7f5a", lw=1.8, ms=4.5, mec=rp.SURFACE, mew=0.6,
                zorder=3, label="staggered grid, per (month, cell)")
        p = ost.location_points(task, setup, "season_grid", 4)
        ax.plot(p.x, p.y, "--o", color="#2a78d6", lw=1.8, ms=5.5, mfc=rp.SURFACE, mec="#2a78d6",
                mew=1.3, zorder=4, label="season grid (layer 1 only, content-blind)")
        p = ost.location_points(task, setup, "oracle_bothalf", l1_stride)
        ax.plot(p.x, p.y, "-*", color=LAYER1, lw=2.1, ms=12, mec=rp.SURFACE, mew=0.7, zorder=5,
                label="better-half oracle, layer 1 only")
        # layer 1 + the same extra fine-tune, no level 2 (the fans' starting points)
        base = [fan(res, fl, task, setup, n1, "", "", "")[:1] for n1 in ns]
        bx = [b[0][0] for b in base if b]
        by = [b[0][1] for b in base if b]
        ax.plot(bx, by, "--*", color=LAYER1, lw=1.4, ms=10, mfc=rp.SURFACE, mec=LAYER1,
                mew=1.2, zorder=5, label="layer 1 + extra fine-tune (no layer 2)")
        for n1, b in zip(ns, base):
            if b:
                pending.append((ax, b[0][0], b[0][1], f"n1={n1}", LAYER1))
        series = [(a, "") for a in arms] + [(CTRL[task] + arms[0][1], "random")]
        for arm, pick in series:
            if arm not in arms:
                continue
            sel = f"l2{arm}" + ("r" if pick else "")
            label, col, mk = ARM_STYLE[arm]
            if pick:
                label, col = (f"{label[2:]} at random cells + months (BioMassters: months)", RANDOM)
            first = True
            for n1 in ns:
                pts = fan(res, fl, task, setup, n1, sel, arm, pick)
                if len(pts) < 2:
                    continue
                xs, ys, _ = zip(*pts)
                ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=6)
                ax.plot(xs[1:], ys[1:], mk, color=col, ms=7, mec=rp.SURFACE, mew=0.7, zorder=7,
                        mfc=rp.SURFACE if pick else col, label=label if first else None)
                if pick:
                    ax.lines[-1].set_markeredgecolor(col)
                first = False
    return overlay


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": rp.MUTED,
                         "axes.labelcolor": rp.MUTED, "xtick.color": rp.MUTED,
                         "ytick.color": rp.MUTED, "text.color": rp.INK,
                         "font.family": "DejaVu Sans"})
    for setup, (ns, _, _, arms, title) in SETUPS.items():
        unit = "ps8 windows" if setup == "s16m8" else "ps4 cells"
        fine = "ps4" if setup == "s16m8" else "ps2"
        sub = (f"{title}: two-layer label oracles; layer 1 n1 = {' / '.join(map(str, ns))} "
               f"{unit}, layer 2 n2 = n1/2, n1, 2 n1 {fine} cells (d8: n2/4 ps8 windows); "
               "full FT warm from layer 1, native_set, block 0 (1 seed)")
        foot = (
            "Fixed models: S = patch size in px, T = seasonal or monthly composites.  Oracles read "
            "the TRUE label at test time: ceilings, not methods.\nLayer 2: the highest-loss fine "
            "cells under a layer-1 crop (loss of the layer-1 model's output), each its lower-loss "
            "half of the cropped months (each month crop's own head map); months = composites, "
            "dates = every raw acquisition of them.\nEach n1 fans out from 'layer 1 + extra "
            "fine-tune' (same warm start and epochs, no layer 2) through n2 rising.  GMACs include "
            "both oracle passes (crop-free LR pass and layer-1 pass).  Random: same budget, random "
            "cells and months.  Missing points: still training.")
        print(rp.draw("shared", "ded", make_overlay(setup), f"_oracles_level2_{setup}",
                      mixed=False, subtitle=sub, footnote=foot, ncol=4))


if __name__ == "__main__":
    main()
