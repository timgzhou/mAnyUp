"""Accuracy vs compute: MiSTR against the published baselines (docs/baselines.md), PASTIS and
BioMassters -> figures/baselines_tradeoff.{png,pdf}. Per configuration the learning rate with the
best val metric, then the mean over seeds. RMSE axes inverted, so up is better.

    python -m exp.viz.baselines_tradeoff
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import datasets_tradeoff as dt  # noqa: E402

TASKS = (("pastis", "PASTIS (crop types)", "test mIoU", False),
         ("biomassters", "BioMassters (biomass)", "test RMSE", True))
ARM = re.compile(r"^mixres_\w+?_ps(?P<ps>\d+)_(?P<tag>(?:upsample|tldr|apt)-[^_]*(?:_lr)?)_n\d+_")
LABEL = (("upsample-seasons-bilinear", "bilinear (seasons)", "0.6", "o"),
         ("upsample-months-bilinear", "bilinear (months)", "0.6", "s"),
         ("upsample-seasons-anyup", "AnyUp (seasons)", "C0", "o"),
         ("upsample-months-anyup", "AnyUp (months)", "C0", "s"),
         ("upsample-seasons-upa", "Upsample Anything", "C2", "o"),
         ("tldr-months", "vid-TLDR", "C3", "^"),
         ("apt-seasons", "APT (seasons)", "C4", "o"),
         ("apt-months", "APT (months)", "C4", "s"))


def baselines(d, lower):
    """(family, GMACs, metric) of every baseline configuration."""
    m = d.run.astype(str).str.extract(ARM)
    d = d[m.tag.notna()].assign(ps=m.ps, tag=m.tag)
    d = d[d.gmacs.notna()]
    out = []
    for (ps, tag), r in d.groupby(["ps", "tag"]):
        fit = r.groupby("train.lr", dropna=False).agg(v=("val_primary", "mean"), g=("gmacs", "mean"),
                                                     t=("test_primary", "mean"))
        fit = fit.sort_values("v", ascending=lower, na_position="last").iloc[0]
        fam = next((k for k, *_ in LABEL if tag.startswith(k)), None)
        out.append((fam, fit.g, fit.t))
    return pd.DataFrame(out, columns=["fam", "g", "t"])


def main():
    dt.XT = "-xhr"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (task, title, ylabel, lower) in zip(axes, TASKS):
        d = dt.rows(task)
        fx = dt.fixed(task, d)
        for fam, style, mk in (("sea", "--", "s"), ("mon", "-", "D")):
            pts = sorted((x, y) for f, _, x, y in fx if f == fam and x <= 150)
            if pts:
                ax.plot(*zip(*pts), style + mk, color="k", ms=3, lw=.8, label=f"fixed ({fam})")
        base = {p: x for f, p, x, _ in fx if f == "sea"}
        for i, (ps, geo) in enumerate(k for k in dt.PAIRS):
            pts = dt.ours(d, task, ps, "one", base.get(ps))
            if pts:
                ax.plot(*zip(*pts), "*-", color="C1", ms=8, lw=.8,
                        label=None if i else "MiSTR (ours)")
        b = baselines(d, lower)
        for fam, label, col, mk in LABEL:
            r = b[b.fam == fam].sort_values("g")
            if len(r):
                ax.plot(r.g, r.t, mk, color=col, ms=4, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("GMACs per sample")
        ax.set_ylabel(ylabel + (" (flipped direction)" if lower else ""))
        ax.set_title(title)
        if lower:
            ax.invert_yaxis()
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/baselines_tradeoff.{ext}", dpi=200)
    print("wrote figures/baselines_tradeoff.png")


if __name__ == "__main__":
    main()
