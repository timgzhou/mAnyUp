"""Side-by-side comparison of --apt_octree's actual TRAINING tokenization against --selector
grid's, at the SAME token budget (k_frac 1/6 and 1/3) -- built to investigate why apt_octree
(entropy-driven, sparse, variable-resolution placement) is NOT beating grid (a content-blind
regular lattice) in the round-46 training results (2026-10-13): PASTIS miou 0.528-0.544
(octree) vs 0.576 (grid), YieldSAT_raw rmse 1.336-1.369 (octree) vs 1.327 (grid). The question
this answers visually: is apt_octree actually concentrating its budget on genuinely
informative regions, or is its entropy criterion picking something that LOOKS plausible per-
cell but leaves large swaths of the tile under-covered relative to grid's guaranteed spread?

Unlike viz_apt_octree.py (which sweeps entropy PERCENTILES with no budget, for exploring tree
shape), this script uses the REAL CALIBRATED tau values training actually ran with
(results/apt_octree_tau.json) so the token count is roughly budget-matched to grid's own k.

Model-free for both arms: apt_octree via InjectFT._apt_octree_mask (same apt_stub pattern as
viz_apt_picks.py), grid via the same grid_scores() + topk the training forward pass uses, no
checkpoint needed since grid is content-blind and apt_octree only reads raw pixels.

One PNG per sample: results/paper/apt_octree_vs_grid/<task>_sample<i>.png -- 2 rows (grid,
apt_octree) x 2 budget columns-of-panels (k=1/6, k=1/3), each a strip of every present month's
thumbnail with that arm's picks overlaid, token count annotated. Run:

    python -u -m exp.upsamplers.viz_apt_octree_vs_grid --tasks pastis,yieldsat_raw
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers.finetune_inject import grid_scores, loader, OUT_GRID  # noqa: E402
from exp.upsamplers.viz_apt_picks import apt_stub, luma, luma_stretch, slot_label  # noqa: E402
from exp.upsamplers.viz_apt_quadtree import label_image, pick_dense_sample  # noqa: E402

OUT_DIR = Path("results/paper/apt_octree_vs_grid")
TAU_FILE = Path("results/apt_octree_tau.json")
LEVEL_COLOR = {16: "#2ca02c", 8: "#1f77b4", 4: "#ffcc00", 2: "#ff3333"}
MAX_COLS = 8          # present months to show per sample (matches viz_apt_quadtree's cap)


def load_taus(task_name: str) -> dict[str, float]:
    """{'0.16667': tau, '0.33333': tau} for this task from the calibration sweep -- the EXACT
    values round 46's training used (see scripts/slurm/inject.sh / yieldsat/inject.sh), so the
    octree row here shows what training actually tokenized, not an exploratory percentile."""
    data = json.loads(TAU_FILE.read_text())[task_name]
    return {k: v["tau"] for k, v in data.items()}


def grid_mask(present: torch.Tensor, H: int, k: int) -> torch.Tensor:
    """(T, H, H) bool: --selector grid's own picks at budget k -- global top-k over the
    (T, H, H) score volume grid_scores returns (same selection training's forward() does for
    the grid arm, see finetune_inject.py's `scores = grid_scores(...)` branch), phase 0 (no
    --grid_phase random, matching evaluation, not training-time jitter)."""
    scores = grid_scores(present, H, k, "stagger", random_phase=False)[0]       # (T, H, H)
    flat = scores.flatten()
    k_eff = min(k, int(present[0].sum()) * H * H)
    idx = flat.topk(k_eff).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.view_as(scores)                                                # (T, H, H)


def draw_grid_cells(ax, mask_t: np.ndarray, cell_px: int):
    """Outline every ps4 cell mask_t (H, H) marks True, at its 4px (cell_px) footprint --
    grid's picks are always ps4, one fixed size, unlike apt_octree's mixed resolutions."""
    yy, xx = np.nonzero(mask_t)
    for y, x in zip(yy, xx):
        rect = mpatches.Rectangle((x * cell_px, y * cell_px), cell_px, cell_px,
                                  linewidth=1.0, edgecolor=LEVEL_COLOR[4], facecolor="none")
        ax.add_patch(rect)


def draw_octree(ax, leaves, t: int):
    """Same convention as viz_apt_octree.py's draw_octree: outline every leaf covering month
    t at its own (y0, x0, ps) pixel footprint, hatched if it spans >1 month."""
    for y0, x0, ps, months in leaves:
        if t not in months:
            continue
        hatch = "///" if len(months) > 1 else None
        rect = mpatches.Rectangle((x0, y0), ps, ps, linewidth=1.0,
                                  edgecolor=LEVEL_COLOR[ps], facecolor="none", hatch=hatch)
        ax.add_patch(rect)
        if len(months) > 1:
            ax.text(x0 + ps - 0.5, y0 + ps - 0.5, str(len(months)), fontsize=5, color="black",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.75))


def render_sample(task, ms, present, obj, taus, sample_i: int, out_path: Path,
                  split: str = "test", target=None):
    present0 = present[0]
    months = present0.nonzero(as_tuple=True)[0].tolist()[:MAX_COLS]
    if not months:
        return
    present_capped = torch.zeros_like(present)
    present_capped[0, months] = True
    stretch = luma_stretch(ms, task, months, sample_i, split)
    H, cell_px = OUT_GRID, 64 // OUT_GRID
    n_present = len(months)

    budgets = [("k=1/6", 1 / 6, "0.16667"), ("k=1/3", 1 / 3, "0.33333")]
    rows = []                              # (row_label, budget_label, kind, data, n_tokens)
    for blabel, kfrac, taukey in budgets:
        k = round(kfrac * n_present * H * H)
        gmask = grid_mask(present_capped, H, k)                                 # (T, H, H)
        rows.append(("grid", blabel, "grid", gmask.cpu().numpy(), int(gmask.sum())))
        tau = taus[taukey]
        leaves = obj._apt_octree_mask(ms, present_capped, 1, tau, ps_min=4)[0]
        rows.append(("apt_octree", blabel, "octree", leaves, len(leaves)))

    nrows, ncols = len(rows), n_present + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.5 * nrows), squeeze=False)
    for row, (arm, blabel, kind, data, n_tok) in enumerate(rows):
        for col, t in enumerate(months):
            ax = axes[row, col]
            ax.imshow(luma(ms, task, t, stretch))
            if kind == "grid":
                draw_grid_cells(ax, data[t], cell_px)
            else:
                draw_octree(ax, data, t)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"slot {t}\n{slot_label(ms.timestamps[0], t)}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{arm}\n{blabel}\n{n_tok} tokens", fontsize=7.5, labelpad=8)
        label_ax = axes[row, ncols - 1]
        if row == 0 and target is not None:
            label_ax.imshow(label_image(task, target))
            label_ax.set_title("label", fontsize=8)
        label_ax.set_xticks([]); label_ax.set_yticks([])
        if row != 0:
            label_ax.axis("off")

    legend = [mpatches.Patch(edgecolor=LEVEL_COLOR[ps], facecolor="none", label=f"ps{ps}")
             for ps in sorted(LEVEL_COLOR, reverse=True)]
    legend.append(mpatches.Patch(edgecolor="black", facecolor="none", hatch="///",
                                 label="octree: spans >1 month"))
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8,
              bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{task.name} test sample {sample_i} -- grid vs apt_octree at matched budget "
                f"(real calibrated tau, round 46)", fontsize=10)
    fig.tight_layout(rect=(0.09, 0.05, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", default="pastis,yieldsat_raw")
    ap.add_argument("--n", type=int, default=4, help="test samples per dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pick_dense", action="store_true",
                    help="pick the n test samples with the most present months")
    a = ap.parse_args()

    for name in a.tasks.split(","):
        task = TASKS[name]
        obj = apt_stub(task)
        taus = load_taus(name)
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
            render_sample(task, ms, present, obj, taus, i, OUT_DIR / f"{name}_sample{i}.png",
                         a.split, target)
            i += 1
            shown += 1


if __name__ == "__main__":
    main()
