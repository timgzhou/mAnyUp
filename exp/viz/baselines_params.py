"""MiSTR against the published baselines: accuracy vs compute, marker area = parameters added to
the fixed model, the size each method has to store and train on top of it
-> figures/baselines_params{,_biomassters}.{png,pdf}. Per configuration the learning rate with the
best val metric. A training-free method adds none: Upsample Anything's per-image kernels are
discarded after use and their cost is in its GMACs. Whatever falls past a break in the y axis
goes in a compressed strip, so the other methods keep their scale; RMSE runs downward, better is
always up. Left out: bilinear (the control), vid-TLDR's r .5,.25,.125 (PASTIS mIoU .10-.15) and
AnyUp's head-only arm (the joint one is better at the same cost).

    python -m exp.viz.baselines_params [pastis|biomassters]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import datasets_tradeoff as dt  # noqa: E402
from exp.viz import res_pairs_tradeoff as rp  # noqa: E402

OURS = "#eb6834"
MISTR_ADDED = 193                       # the crop embedding (192) + the global entropy threshold
APT_ADDED = 184_897                     # (uses the crop embedding too)
ANYUP_NET = 878_080                     # the upsampler; AnyUp and Upsample Anything's features
#                                         add a pixel head, 192 -> one output per class
# base: the fixed model, encoder + head + readout (from the checkpoints, less the unused hr_type)
TASKS = {
    "pastis": dict(title="PASTIS", metric="test mIoU", outputs=19, base=7_141_889, lower=False,
                   ylim=(.40, .61), strip=(.26, .40), ticks={.3: "0.300", .35: "0.350"}),
    "biomassters": dict(title="BioMassters", metric="test RMSE", outputs=1, base=6_252_545,
                        lower=True, ylim=(41, 32.2), strip=(51, 41),
                        ticks={44: "44", 48: "48"}),
}


def count(n):
    return f"{round(n / 1000)}k" if n >= 1000 else f"{n}"


def added(name, n, digits, base):
    """A legend label: the parameters added and their share of the fixed model."""
    head, _, tail = name.partition(" (")
    return f"{head} ({tail + ', ' if tail else ''}+{count(n)}, {n / base:.{digits}%})"


AREA_FLOOR, AREA_MAX = 22, 330          # pt^2: a zero-parameter marker, AnyUp's
STAR = 95                                # pt^2, not to scale: area(193) is too small to read


def area(n, top):
    return AREA_FLOOR + n / top * (AREA_MAX - AREA_FLOOR)


def rows(d, fixed, lower):
    """The configuration's points: best-val lr per patch size and schedule."""
    for col, v in fixed.items():
        d = d[d[col].astype(str) == str(v)]
    d = d[d["baseline.r"].astype(str) != "0.5,0.25,0.125"]
    d = d.assign(sel=d.val_primary.fillna(d.test_primary) * (-1 if lower else 1))
    keys = ["model.lr_patch", "baseline.r", "baseline.keep"]
    return d.sort_values("sel").groupby(keys, dropna=False).tail(1)


def main(task="pastis"):
    t = TASKS[task]
    head = (192 + 1) * t["outputs"]
    anyup_added, base_n = ANYUP_NET + head, t["base"]
    MISTR = added("MiSTR (ours", MISTR_ADDED, 3, base_n)
    APT, TLDR, ANYUP = (added("APT", APT_ADDED, 1, base_n), added("vid-TLDR", 0, 0, base_n),
                        added("AnyUp", anyup_added, 1, base_n))
    UPA = f"Upsample Anything (+0 to {count(head)}, <0.1%)"
    # key, its rows' fixed values, parameters added, colour, legend label (None: same entry as the
    # one above). All circles: area compares across methods only within one shape.
    methods = (
        ({"baseline.method": "apt", "baseline.tokens": "seasons"}, APT_ADDED, "#2a78d6", APT),
        ({"baseline.method": "apt", "baseline.tokens": "months"}, APT_ADDED, "#2a78d6", None),
        ({"baseline.method": "tldr"}, 0, "#1baf7a", TLDR),
        ({"baseline.method": "upsample", "baseline.upsampler": "anyup",
          "train.head_only": False}, anyup_added, "#eda100", ANYUP),
        ({"baseline.method": "upsample", "baseline.upsampler": "upa",
          "baseline.space": "features"}, head, "#e87ba4", UPA),
        ({"baseline.method": "upsample", "baseline.upsampler": "upa",
          "baseline.space": "predictions"}, 0, "#e87ba4", None),
    )
    dt.XT = "-xhr"
    d = dt.rows(task)
    fx = dt.fixed(task, d)
    base = {p: x for f, p, x, _ in fx if f == "sea"}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 11})
    fig, (ax, low) = plt.subplots(2, 1, figsize=(4.6, 4.9), sharex=True, facecolor=rp.SURFACE,
                                  gridspec_kw=dict(height_ratios=[5, 1], hspace=.05))
    ax.set_facecolor(rp.SURFACE)
    low.set_facecolor(rp.SURFACE)
    for fam, ls, mk, mfc, label in (("sea", "--", "s", rp.SURFACE, "fixed resolution, seasonal"),
                                    ("mon", "-", "D", rp.INK, "fixed resolution, monthly")):
        p = sorted((x, y) for f, _, x, y in fx if f == fam and x <= 170)
        ax.plot(*zip(*p), ls, color=rp.INK, lw=1, marker=mk, ms=3.5, mfc=mfc, alpha=.75, zorder=2,
                label=label)
    b = d[d["baseline.method"].notna()]
    for fixed, n, col, label in methods:
        r = rows(b, fixed, t["lower"])
        in_main = r.test_primary.between(*sorted(t["ylim"]))
        for a, q in ((ax, r[in_main]), (low, r[~in_main])):
            a.scatter(q.gmacs, q.test_primary, s=area(n, anyup_added), color=col, alpha=.45,
                      edgecolors=col, linewidths=1, zorder=3, clip_on=False,
                      label=label if a is ax else None)
    pts = dt.ours(d, task, 16, "one", base.get(16))
    pts += dt.ours(d, task, 8, "one", base.get(8))
    for ps, geo in dt.PAIRS:
        if (ps, geo) not in ((16, "c8"), (8, "c4")):
            pts += dt.sched_pair(d, task, ps, geo, base.get(ps))
    print(pd.DataFrame(sorted(pts), columns=["gmacs", "metric"]).round(3).to_string(index=False))
    ax.scatter(*zip(*pts), s=STAR, marker="*", color=OURS, edgecolors=rp.SURFACE, linewidths=.4,
               zorder=5, label=MISTR)
    ax.set_xscale("log")
    ax.set_xlim(1.4, 200)
    low.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax.set(ylim=t["ylim"], title=t["title"])
    low.set(ylim=t["strip"], xlabel="GMACs per sample")
    low.set_yticks(list(t["ticks"]), list(t["ticks"].values()))
    fig.supylabel(t["metric"], x=.0, fontsize=11)
    for a in (ax, low):
        a.minorticks_off()
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=.2, lw=.5)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    for a, y in ((ax, 0), (low, 1)):        # the y axis break
        a.plot([-.012, .012], [y - .012 * (1 + 4 * y), y + .012 * (1 + 4 * y)], color=rp.INK, lw=.8,
               transform=a.transAxes, clip_on=False)
    h, lab = ax.get_legend_handles_labels()
    h, lab = h + [plt.Line2D([], [], alpha=0)], lab + [""]
    # (filled by column: MiSTR alone on its row, the fixed models share the last)
    order = [lab.index(x) for x in (MISTR, APT, UPA, "fixed resolution, seasonal", "", ANYUP, TLDR,
                                    "fixed resolution, monthly")]
    leg = fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center",
                     bbox_to_anchor=(.45, -.78), bbox_transform=low.transAxes, frameon=True, ncol=2,
                     fontsize=11, fancybox=False, handlelength=1.6, handletextpad=.3,
                     columnspacing=.7,
                     labelspacing=.45, borderaxespad=0)
    leg.get_frame().set(edgecolor=rp.MUTED, facecolor="none", linewidth=.6)
    for t, x in zip(leg.legend_handles, [lab[i] for i in order]):
        if hasattr(t, "set_sizes"):
            t.set_sizes([75 if x == MISTR else 32])
    key = [plt.scatter([], [], s=area(n, anyup_added), color="#8a8984", alpha=.45,
                       edgecolors="#8a8984") for n in (0, APT_ADDED, anyup_added)]
    key.insert(1, plt.scatter([], [], s=STAR, marker="*", color=OURS, edgecolors=rp.SURFACE,
                              linewidths=.4))
    ax.legend(key, ["0", f"{MISTR_ADDED}", count(APT_ADDED), count(anyup_added)],
              title="parameters added",
              loc="lower right", ncol=2, frameon=False, fontsize=10,
              title_fontproperties={"size": 11, "weight": "bold"}, handletextpad=.5,
              columnspacing=1.1, labelspacing=.9, borderpad=.5)
    fig.tight_layout(pad=.2)
    out = "figures/baselines_params" + ("" if task == "pastis" else f"_{task}")
    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(f"{out}.{ext}", dpi=dpi, facecolor=rp.SURFACE, bbox_inches="tight",
                    pad_inches=.02)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
