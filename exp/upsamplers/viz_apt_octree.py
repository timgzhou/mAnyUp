"""Visualize the user's 2026-10-09 space-time octree idea: "start with ps16 and t=all,
calculate entropy, and if above threshold then split in either time dimension or space
dimension, or both, depending on the after-split entropy -- this is almost info gain."

A true 3D (y, x, t) octree (InjectFT._apt_octree_mask, finetune_inject.py), generalizing the
"real APT" quadtree (viz_apt_quadtree.py, space-only, every month scored independently) to let
the SAME recursion also choose to stay coarse in TIME -- a crop spanning several months,
collapsed into one pixel+slot-averaged token (apt_pool's existing averaging convention), when
pooling over time doesn't lose much entropy. No token budget (user: "I wanna see the
tokenization that would come out of it... without training at all") -- pure entropy threshold,
swept at a few PERCENTILES of real per-node entropy (same reasoning as viz_apt_quadtree.py:
this codebase's per-band-summed entropy has no universal "good" absolute value, and differs by
dataset/band count).

Model-free (reuses viz_apt_picks.py's apt_stub/luma/luma_stretch/slot_label, and
viz_apt_quadtree.py's label_image/pick_dense_sample). One PNG per sample:
results/paper/apt_octree/<task>_sample<i>.png -- rows are tau presets, columns are present
months (+ a label column), each present month's panel also carries a raw-input row (same
layout as viz_apt_quadtree.py). A leaf spanning MULTIPLE months is drawn on EVERY month panel
it covers, hatched (vs. a plain outline for a single-month leaf) so "this box is the same
pixel-averaged crop across several months" is visible without comparing adjacent panels by
eye. Box colour by patch size (same LEVEL_COLOR convention as viz_apt_quadtree.py). Run:

    python -u -m exp.upsamplers.viz_apt_octree --tasks pastis,biomassters,geoidflood,yieldsat_raw
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
from exp.upsamplers.finetune_inject import loader  # noqa: E402
from exp.upsamplers.viz_apt_picks import apt_stub, luma, luma_stretch, slot_label  # noqa: E402
from exp.upsamplers.viz_apt_quadtree import (MAX_COLS, label_image,  # noqa: E402
                                             pick_dense_sample)

OUT_DIR = Path("results/paper/apt_octree")
LEVEL_COLOR = {16: "#2ca02c", 8: "#1f77b4", 4: "#ffcc00", 2: "#ff3333"}  # ps -> colour


def tau_presets(obj, ms, present, percentiles=(75, 50, 25), scan_tau: float = 1e9):
    """Percentiles of real per-node entropy, sampled by running _apt_octree_mask once at a
    very permissive tau (scan_tau, effectively "always split" down to ps2/single-month) and
    recording every node's h_parent ALONG THE WAY -- not just the leaves' own entropy (leaves
    at a permissive tau are already low-entropy by construction, which would bias the
    percentiles toward only the "stops early" end of the real distribution). Reuses
    InjectFT._apt_octree_recurse's own pooled-entropy computation by calling a thin
    instrumented copy of the recursion here rather than modifying the production method to
    log internals it has no other reason to expose."""
    pixels = obj._apt_pixels(ms)
    Hpx = pixels.shape[1]
    all_h = []

    def recurse(pixels_b, y0, x0, ps, months, r0, r1):
        rank_months = months[r0:r1]
        if not rank_months:
            return
        block = pixels_b[y0:y0 + ps, x0:x0 + ps][:, :, rank_months, :]
        h = obj._pool_entropy(block)
        all_h.append(h)
        can_space, can_time = ps > 2, (r1 - r0) > 1
        if h < scan_tau or not (can_space or can_time):
            return
        hp, rm = ps // 2, (r0 + r1) // 2
        if can_space:
            for dy in (0, 1):
                for dx in (0, 1):
                    recurse(pixels_b, y0 + dy * hp, x0 + dx * hp, hp, months, r0, r1)
        if can_time:
            recurse(pixels_b, y0, x0, ps, months, r0, rm)
            recurse(pixels_b, y0, x0, ps, months, rm, r1)

    months = present[0].nonzero(as_tuple=True)[0].tolist()
    for gy in range(Hpx // 16):
        for gx in range(Hpx // 16):
            recurse(pixels[0], gy * 16, gx * 16, 16, months, 0, len(months))

    labels = {75: "loose (few splits)", 50: "medium", 25: "tight (many splits)"}
    arr = np.array(all_h) if all_h else np.array([0.0])
    return [(labels.get(p, f"p{p}"), float(np.percentile(arr, p))) for p in percentiles]


def draw_octree(ax, leaves, t: int):
    """leaves: _apt_octree_mask's per-sample leaf list [(y0, x0, ps, months), ...]. Draws
    every leaf whose `months` includes t, at its own (y0, x0, ps) footprint, colour-coded by
    ps (LEVEL_COLOR) -- a leaf spanning >1 month is hatched (vs. a plain outline for a single-
    month leaf) so a crop that stayed coarse IN TIME is visually distinct from one that's
    simply this month's own ps-sized pick."""
    for y0, x0, ps, months in leaves:
        if t not in months:
            continue
        hatch = "///" if len(months) > 1 else None
        rect = mpatches.Rectangle((x0, y0), ps, ps, linewidth=1.0,
                                  edgecolor=LEVEL_COLOR[ps], facecolor="none", hatch=hatch)
        ax.add_patch(rect)
        if len(months) > 1:          # span length, lower-right corner of the box -- user:
            ax.text(x0 + ps - 0.5, y0 + ps - 0.5, str(len(months)), fontsize=5, color="black",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.75))


def render_sample(task, ms, present, obj, sample_i: int, out_path: Path, percentiles,
                  split: str = "test", target=None):
    present0 = present[0]
    months = present0.nonzero(as_tuple=True)[0].tolist()
    if not months:
        return
    presets = tau_presets(obj, ms, present, percentiles)
    stretch = luma_stretch(ms, task, months, sample_i, split)

    leaves_by_preset = []
    for label, tau in presets:
        leaves = obj._apt_octree_mask(ms, present, 1, tau)[0]
        n_by_ps = {}
        n_merged = 0
        for _, _, ps, ms_ in leaves:
            n_by_ps[ps] = n_by_ps.get(ps, 0) + 1
            if len(ms_) > 1:
                n_merged += 1
        leaves_by_preset.append((label, tau, leaves, n_by_ps, n_merged))

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

    for row, (label, tau, leaves, n_by_ps, n_merged) in enumerate(leaves_by_preset, start=1):
        axes[row, ncols - 1].axis("off")
        for col, t in enumerate(months):
            ax = axes[row, col]
            ax.imshow(luma(ms, task, t, stretch))
            draw_octree(ax, leaves, t)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                counts = "  ".join(f"ps{ps}:{n}" for ps, n in sorted(n_by_ps.items(), reverse=True))
                ax.set_ylabel(f"{label}\n\ntau={tau:.2f}\n\n{counts}\n{n_merged} time-merged",
                             fontsize=7.5, labelpad=8)
    legend = [mpatches.Patch(edgecolor=LEVEL_COLOR[ps], facecolor="none", label=f"ps{ps}")
             for ps in sorted(LEVEL_COLOR, reverse=True)]
    legend.append(mpatches.Patch(edgecolor="black", facecolor="none", hatch="///",
                                 label="spans >1 month"))
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8,
              bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{task.name} test sample {sample_i} -- space-time octree "
                f"(tau = percentile of real per-node entropy, no budget, greedy "
                f"gain-per-token axis choice)", fontsize=10)
    fig.tight_layout(rect=(0.09, 0.05, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", default="pastis,biomassters,geoidflood,yieldsat_raw")
    ap.add_argument("--n", type=int, default=4, help="test samples per dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--percentiles", default="75,50,25",
                    help="comma-separated percentiles of real per-node entropy to use as tau "
                         "presets (rows), high->low")
    ap.add_argument("--pick_dense", action="store_true",
                    help="pick the n test samples with the most present months instead of "
                         "just the first n -- see viz_apt_quadtree.pick_dense_sample")
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
