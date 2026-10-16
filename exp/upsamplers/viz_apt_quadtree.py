"""Visualize the ACTUAL APT algorithm (Choudhury et al. 2025, arXiv:2510.18091, Section 3.1):
a strict quadtree over patch size (16 -> 8 -> 4 -> 2 px on a 64px tile), each level kept
coarse if its entropy is below that level's threshold tau_i, else split into 4 children and
recursed on -- no token budget (the paper has none), no time pooling (the paper has no time
axis at all: every month here is scored and recursed on INDEPENDENTLY, confirmed against the
paper text 2026-10-08, see [[apt-entropy-selector]]). This is distinct from this codebase's
--apt_res (2-level, ps16-vs-ps4, FIXED TOKEN BUDGET k, not the paper's threshold mechanism).

The paper's own tau values (5.75, 4.0) are meaningless here: our entropy sums over EVERY band
of EVERY modality (this codebase's extension beyond the paper's single-channel score), so raw
magnitudes are ~10x theirs and differ a lot by dataset. Instead, per (dataset, level), tau is
swept at a few PERCENTILES of that level's own score distribution over present cells -- directly
comparable across datasets/levels despite the scale difference, and guaranteed non-degenerate
(never "splits everything" or "splits nothing" by construction of using real data's own
quantiles).

Model-free (APT scores raw pixels; no checkpoint/model build, see viz_apt_picks.py's apt_stub,
reused here). One PNG per sample: results/paper/apt_quadtree/<task>_sample<i>.png -- rows are
tau-percentile presets (loose/medium/tight, i.e. high/mid/low percentile -- HIGHER percentile
tau means MORE cells pass the "entropy < tau" keep-coarse test, so fewer splits; the labels
describe the resulting behaviour, not the percentile number), columns are present months. Each
kept patch is drawn as an outlined box sized to its own patch size (16/8/4/2 px), colour-coded
by size, so the quadtree structure is visible directly on the imagery. Run:

    python -u -m exp.upsamplers.viz_apt_quadtree --tasks pastis,biomassters,geoidflood,yieldsat_raw
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers.finetune_inject import InjectFT, loader  # noqa: E402
from exp.upsamplers.viz_apt_picks import apt_stub, luma, luma_stretch, slot_label  # noqa: E402

OUT_DIR = Path("results/paper/apt_quadtree")
TILE = 64
GRID_SIZES = (4, 8, 16, 32)             # coarsest -> finest; patch sizes 16, 8, 4, 2 px
PATCH_SIZES = {4: 16, 8: 8, 16: 4, 32: 2}
LEVEL_COLOR = {16: "#2ca02c", 8: "#1f77b4", 4: "#ffcc00", 2: "#ff3333"}  # ps -> colour


def tau_presets(obj, ms, present, B, percentiles=(75, 50, 25)):
    """taus[level_i] for every preset: the given percentile of that level's OWN NORMALIZED
    score distribution over present cells (real data, this sample's batch) -- must use
    normalize=True here to match _apt_quadtree_mask's own default, since raw (un-normalized)
    entropy isn't comparable across grid sizes in the first place (see _apt_scores'
    docstring: a finer cell has fewer pixels, so it cannot reach the same entropy ceiling no
    matter how diverse its content is -- confirmed empirically, e.g. PASTIS ps16 vs ps2 raw
    medians ~4.5 vs ~2.0 mean-per-band bits, vs. ~0.90 and ~0.98 once normalized). A HIGHER
    percentile tau means MORE cells satisfy "entropy < tau" (kept coarse), so fewer splits --
    labelled "loose" (few splits) at the high percentile end, "tight" (many splits) at low.

    Percentiles are taken over NONZERO-entropy cells only. Some datasets (e.g. yieldsat_raw)
    have a large masked/off-field background with exactly 0 entropy at every level (measured:
    55-65% of present cells on yieldsat_raw sample 0) -- including that mass in the percentile
    pushes the 50th/25th percentile itself to 0, which makes tau=0 everywhere and the "entropy
    < tau" keep-coarse test never true, degenerately splitting EVERY cell including the
    background (caught visually: yieldsat_raw medium/tight rows both showed tau=-0.0 at every
    level and split the whole tile, 2026-10-08). Excluding the zero-entropy mass keeps the
    percentiles meaningful (anchored to the CONTENT cells' own spread) regardless of how much
    background a dataset has."""
    old_apt_time = obj.apt_time
    obj.apt_time = "none"
    try:
        scores = {g: obj._apt_scores(ms, present, B, g, normalize=True)
                 for g in GRID_SIZES[:-1]}                           # finest has no tau
    finally:
        obj.apt_time = old_apt_time
    labels = {75: "loose (few splits)", 50: "medium", 25: "tight (many splits)"}
    presets = []
    for p in percentiles:
        taus = []
        for g in GRID_SIZES[:-1]:
            s = scores[g]
            vals = s[present[:, :, None, None].expand_as(s)]
            vals = vals[vals > 1e-6]
            if vals.numel() == 0:
                taus.append(0.0)
                continue
            taus.append(float(vals.quantile(p / 100)))
        presets.append((labels.get(p, f"p{p}"), tuple(taus)))
    return presets


def draw_quadtree(ax, out: dict, t: int):
    """out: {grid_size: mask (B,T,g,g) bool} from _apt_quadtree_mask, sample 0, month t. Draws
    every kept cell at its own patch-size footprint, colour-coded by size (see LEVEL_COLOR)."""
    for g in GRID_SIZES:
        ps = PATCH_SIZES[g]
        mask_t = out[g][0, t].cpu().numpy()
        yy, xx = np.nonzero(mask_t)
        for y, x in zip(yy, xx):
            rect = mpatches.Rectangle((x * ps, y * ps), ps, ps, linewidth=1.0,
                                      edgecolor=LEVEL_COLOR[ps], facecolor="none")
            ax.add_patch(rect)


def label_image(task, target) -> np.ndarray:
    """(64, 64, 3) in [0,1] -- a generic rendering of this task's target, for the new raw-row's
    label column (user: "add a first row where we show just raw inputs, and the corresponding
    label... see if apt quadtree is selecting regions we find hard or interesting"). Generic
    over task.kind rather than hardcoding per-task class names/colours, so it works for all 5
    registered tasks without new per-task plumbing:
      classification (PASTIS, geoidflood, geoidseries/_flood): each class index -> a distinct
        colour (tab20, cycled), ignore_index (-1) -> mid-grey.
      regression (BioMassters, yieldsat_ml24/_raw): percentile-stretched viridis; yieldsat's
        (2,64,64) target is [z-scored yield, field id] (NaN = unlabelled, see _YieldSat.target)
        -- only channel 0 is shown, NaN -> white (same "no label here" convention `label_image`
        ends up giving the plain classification ignore case via grey, kept visually distinct
        since NaN truly means "no pixel-level label" vs ignore meaning "excluded from scoring").
    """
    t = target[0]
    if task.kind == "classification":
        t = t.cpu().numpy()
        cmap = plt.get_cmap("tab20")
        n_classes = max(int(t[t >= 0].max()) + 1 if (t >= 0).any() else 1, 1)
        img = np.full(t.shape + (3,), 0.5)                      # ignore_index: mid-grey
        for c in range(n_classes):
            img[t == c] = cmap(c % 20)[:3]
        return img
    # regression: yieldsat's target is (2, 64, 64) [z-scored value, field id] -- others (1,64,64)
    val = t[0] if t.ndim == 3 else t
    val = val.float().cpu().numpy()
    finite = np.isfinite(val)
    if not finite.any():
        return np.ones(val.shape + (3,))                        # nothing labelled: blank white
    lo, hi = np.percentile(val[finite], 2), np.percentile(val[finite], 98)
    norm = np.clip((val - lo) / (hi - lo + 1e-6), 0, 1)
    img = plt.get_cmap("viridis")(norm)[..., :3]
    img[~finite] = 1.0                                           # unlabelled: white
    return img


def render_sample(task, ms, present, obj, sample_i: int, out_path: Path, percentiles,
                  split: str = "test", target=None):
    present0 = present[0]
    months = present0.nonzero(as_tuple=True)[0].tolist()
    if not months:
        return
    presets = tau_presets(obj, ms, present, 1, percentiles)
    stretch = luma_stretch(ms, task, months, sample_i, split)  # shared across every panel

    masks_by_preset = []
    for label, taus in presets:
        out = obj._apt_quadtree_mask(ms, present, 1, GRID_SIZES, taus)
        n_by_ps = {PATCH_SIZES[g]: int(out[g][0, months].sum().item()) for g in GRID_SIZES}
        masks_by_preset.append((label, taus, out, n_by_ps))

    # row 0: raw imagery, no quadtree overlay, plus one extra LABEL column at the end -- lets
    # the splits below be judged against what's actually in the scene and what the task cares
    # about, rather than eyeballed in isolation (user: "see if apt quadtree is selecting
    # regions we find hard or interesting").
    nrows, ncols = len(presets) + 1, len(months) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.6 * nrows), squeeze=False)
    for col, t in enumerate(months):
        ax = axes[0, col]
        ax.imshow(luma(ms, task, t, stretch))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slot {t}\n{slot_label(ms.timestamps[0], t)}", fontsize=8)
        if col == 0:
            ax.set_ylabel("raw input\n(no overlay)", fontsize=7.5, labelpad=8)
    label_ax = axes[0, ncols - 1]
    if target is not None:
        label_ax.imshow(label_image(task, target))
    label_ax.set_xticks([]); label_ax.set_yticks([])
    label_ax.set_title("label", fontsize=8)
    for row, (label, taus, out, n_by_ps) in enumerate(masks_by_preset, start=1):
        axes[row, ncols - 1].axis("off")                        # no label repeat on these rows
        for col, t in enumerate(months):
            ax = axes[row, col]
            ax.imshow(luma(ms, task, t, stretch))
            draw_quadtree(ax, out, t)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                counts = "  ".join(f"ps{ps}:{n}" for ps, n in sorted(n_by_ps.items(), reverse=True))
                tau_str = " / ".join(f"{tv:.1f}" for tv in taus)
                ax.set_ylabel(f"{label}\n\ntau={tau_str}\n\n{counts}", fontsize=7.5, labelpad=8)
    legend = [mpatches.Patch(edgecolor=LEVEL_COLOR[ps], facecolor="none", label=f"ps{ps}")
             for ps in sorted(LEVEL_COLOR, reverse=True)]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{task.name} test sample {sample_i} -- real APT quadtree "
                f"(tau = percentile of each level's own score dist, no time pooling, no "
                f"budget)", fontsize=10)
    fig.tight_layout(rect=(0.09, 0.05, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


MAX_COLS = 16   # figures wider than this (e.g. geoidseries' up-to-~60-date chips) are
                # unreadable -- cap how many present months a sample selection will show;
                # render_sample itself has no limit, this is purely the sample PICKER's bound


def pick_dense_sample(task, dl_factory, n: int, scan: int = 60) -> list[int]:
    """Indices of the n test-split samples (in scan-order) with the MOST present months,
    capped at MAX_COLS -- for tasks whose per-sample present-month count varies a lot (e.g.
    geoidseries/geoidseries_flood: real per-event Sentinel-1 revisit coverage ranges from ~1
    valid date to ~60 in a 6-month window, see [[apt-entropy-selector]] -- a sparse-event chip
    landing in the loader's first few test-split indices would otherwise silently produce an
    uninformative 1-2-column figure). Reads `scan` samples ahead (present-month count only,
    cheap -- no scoring) and returns the n richest, in their ORIGINAL dataset order (not sorted
    by richness) so a multi-sample batch still reads left-to-right as the dataset's own
    ordering, just skipping the sparse ones."""
    dl = dl_factory()
    counts = []
    for i, (ms, ss, season, present, pairs, extra, target) in enumerate(dl):
        if i >= scan:
            break
        counts.append((i, min(int(present[0].sum().item()), MAX_COLS)))
    counts.sort(key=lambda ic: -ic[1])
    return sorted(i for i, _ in counts[:n])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", default="pastis,biomassters,geoidflood,yieldsat_raw")
    ap.add_argument("--n", type=int, default=4, help="test samples per dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--percentiles", default="75,50,25",
                    help="comma-separated percentiles of each level's own score distribution "
                         "to use as tau presets (rows), high->low")
    ap.add_argument("--pick_dense", action="store_true",
                    help="pick the n test samples with the most present months (scanning "
                         "ahead, see pick_dense_sample) instead of just the first n -- useful "
                         "for geoidseries/geoidseries_flood, whose per-chip S1 coverage varies "
                         "widely and whose first test-split chips may be a sparse-coverage "
                         "event; off by default (the first n is the right choice for every "
                         "other task, where present-month count barely varies sample to "
                         "sample and reading dataset order top-to-bottom is more natural)")
    a = ap.parse_args()
    percentiles = tuple(int(p) for p in a.percentiles.split(","))

    for name in a.tasks.split(","):
        task = TASKS[name]
        obj = apt_stub(task)
        dl_factory = lambda: loader(task, a.split, batch_size=1, num_workers=2,   # noqa: E731
                                    shuffle=False)
        wanted = set(pick_dense_sample(task, dl_factory, a.n)) if a.pick_dense else None
        dl = dl_factory()
        it = iter(dl)
        i = 0
        shown = 0
        while shown < a.n:
            ms, ss, season, present, pairs, extra, target = next(it)
            if wanted is not None and i not in wanted:
                i += 1
                continue
            months = present[0].nonzero(as_tuple=True)[0].tolist()[:MAX_COLS]
            present_capped = torch.zeros_like(present)
            present_capped[0, months] = True
            render_sample(task, ms, present_capped, obj, i, OUT_DIR / f"{name}_sample{i}.png",
                         percentiles, a.split, target)
            i += 1
            shown += 1


if __name__ == "__main__":
    main()
