"""Visualize what --selector apt (and --apt_pool / --apt_res) would tokenize: for 4 test
samples of each dataset, every present month's RGB (or SAR false-colour) thumbnail with the
ps4 cells APT would inject highlighted, plus the H_space + H_time score map that drove the
picks.

Model-free: APT scores raw pixels (InjectFT._apt_scores / _apt_pool_mask / _apt_res_mask), so
this never builds or loads a trained checkpoint -- a bare stand-in object with the attributes
those methods read (task, apt_merge_mse, apt_time) is enough. CPU or GPU both work; GPU only
matters for speed.

One PNG per sample: results/paper/apt_picks/<task>_sample<i>.png, one row per arm (apt top-k,
apt_pool at two thresholds, apt_res) x (score map | one panel per present month). apt_pool
picks are filled + merged runs coloured one colour per run; apt_res picks outline ps4 cells
it split into (same style as plain apt) and additionally draw a blue 16px box around every
ps16 cell it chose to KEEP coarse (not split). Run:

    python -u -m exp.upsamplers.viz_apt_picks --tasks pastis,biomassters,geoidflood,yieldsat_raw
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import NamedTuple  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers.finetune_inject import InjectFT, loader, OUT_GRID  # noqa: E402

OUT_DIR = Path("results/paper/apt_picks")
H = OUT_GRID          # 16: the ps4 cell grid side
CELL_PX = 64 // H      # 4 px per cell


def apt_stub(task, apt_merge_mse: float = 0.01):
    """A bare object carrying only what _apt_scores / _apt_pixels / _apt_pool_mask read --
    no model, no checkpoint: APT is scored from raw pixels alone."""
    obj = InjectFT.__new__(InjectFT)
    obj.task = task
    obj.apt_merge_mse = apt_merge_mse
    obj.apt_time = "sum"
    return obj


def draw_cells16(ax, mask16_t: np.ndarray):
    """Outline every ps16 cell mask16_t (4, 4) marks True, at the ps16 (16px) footprint size --
    4x a ps4 cell's side -- so apt_res's kept-coarse crops are visually distinct from its
    split-to-ps4 crops (drawn by draw_cells at the normal ps4 size)."""
    yy, xx = np.nonzero(mask16_t)
    for y, x in zip(yy, xx):
        rect = mpatches.Rectangle((x * 4 * CELL_PX, y * 4 * CELL_PX), 4 * CELL_PX, 4 * CELL_PX,
                                  linewidth=1.4, edgecolor="#00bfff", facecolor="none")
        ax.add_patch(rect)


def slot_label(timestamps, t: int) -> str:
    """"YYYY-MM" for OlmoEarth slot t, from ms.timestamps[0] (T,3) (day, month, year) -- month
    is OlmoEarth's own 0-indexed convention (0=Jan..11=Dec, confirmed directly against a real
    sample's 24 slots, 2026-10-09), +1 for the usual 1-indexed calendar label. Used in panel
    titles instead of the bare slot index "month {t}" (confusing on yieldsat_raw, whose 24
    slots run Sep-Aug across TWO calendar years ending at harvest -- slot 8 is NOT "the 9th
    month of the season" in any obvious sense without this, see [[apt-entropy-selector]])."""
    _, month0, year = (int(v) for v in timestamps[t].tolist())
    return f"{year}-{month0 + 1:02d}"


def _rgb_bands(task):
    """(oe_name, idx): the OlmoEarth attribute name and channel indices for task.rgb's proxy
    bands (S2_RGB / the SAR VV,VH,VV stand-in) -- shared by luma() and luma_stretch()."""
    from olmoearth_pretrain.data.constants import Modality
    mod_ds, bands = next(iter(task.rgb.items()))
    oe_name = task.olmoearth.modalities[mod_ds]
    order = [b.upper() for b in Modality.get(oe_name).band_order]
    idx = [order.index(b.upper()) for b in bands]
    return oe_name, idx


class YieldSatRawFrames:
    """True-reflectance S2 RGB per OlmoEarth calendar-month SLOT, for yieldsat_raw ONLY.

    WHY this exists: ms.sentinel2_l2a (what luma() reads for every other dataset) is
    OlmoEarth's own PER-BAND normalized representation (manyup/data/normalize.py: each band
    independently min-max or mean/std rescaled to its OWN distribution's range), not true
    reflectance. That normalization does not preserve cross-band ratios, so no amount of
    percentile stretching or gamma correction on ms can recover natural-looking colour -- the
    information needed (a shared radiometric reference across R/G/B) was already discarded
    before the tensor reaches this script. Confirmed by comparing against
    figures/paper/datasets/example_yieldsat.png (exp/viz/paper/datasets.py's yieldsat_example,
    which reads the SAME raw packed store this class uses) -- that figure's natural
    brown/green/tan progression is only possible because it reads true reflectance directly,
    never through ms. See [[apt-entropy-selector]].

    Reads the raw packed .npy store directly (manyup.data.datasets.yieldsat), the SAME files
    YieldSatDataset itself loads from, so values are true Sentinel-2 reflectance*1e4 (nodata=0
    outside the field polygon -- an actual zero, unlike ms's near-constant-but-nonzero fill
    after per-band normalization).

    A YieldSatDataset's test split is index-order deterministic with shuffle=False (confirmed:
    Paired.__getitem__(i) -> self.ds[i] directly, DataLoader with sampler=None and
    num_workers>0 still yields in index order, workers only prefetch ahead in parallel) -- so
    sample index i here lines up with the i-th item loader() yields in the SAME viz script's
    main loop, with no need to thread sample ids through loader()'s own return tuple (which
    does not expose them at all: `extra` only carries frac/target).

    Per OlmoEarth slot (PoolYears(2, "mean") pools possibly-MULTIPLE raw acquisitions into one
    24-slot calendar grid), picks ONE representative raw acquisition for display (user-
    confirmed 2026-10-09: simpler and matches the reference figure's own convention of one
    real unblended acquisition per shown date, not an attempt to replicate PoolYears' own
    reducer in reflectance units) -- the raw acquisition whose (year, month) matches the
    slot's, with the most clear-and-inside-polygon pixels among that month's candidates."""

    def __init__(self, split: str = "test"):
        from manyup.data import All
        from manyup.data.datasets.yieldsat import (CLEAR_SCL, YieldSatDataset, _array,
                                                    parse_id, read_window)
        self._clear_scl, self._array, self._parse_id, self._read_window = (
            CLEAR_SCL, _array, parse_id, read_window)
        self.ds = YieldSatDataset(split, {"S2": All()})

    def __call__(self, sample_i: int, months: list[int], timestamps) -> dict[int, np.ndarray]:
        """timestamps: ms.timestamps[0] (T, 3) (day, month, year) per slot -- OlmoEarth's own
        per-slot calendar date, the only place the (month, year) a slot stands for survives
        into the loaded batch (PoolYears' own per-acquisition `sources` list does not survive
        collation -- checked directly, 2026-10-09). Returns {month_t: (64,64,3) in [0,1]},
        stretched + gamma'd the same way as luma() (same GAMMA, its own percentile range over
        just the returned months -- a yieldsat_raw sample's raw reflectance doesn't carry the
        same near-constant-fill problem ms's normalized version does, since nodata is a true
        0 excluded by read_window's own fill, but the SAME exclusion-before-percentile
        discipline is applied here for consistency, not because it was observed necessary)."""
        sample_id = self.ds._native_ids("test")[sample_i]
        field, r0, c0 = self._parse_id(sample_id)
        f = self.ds.fields[field]
        s2 = self._read_window(self._array(self.ds._path("s2", field)), r0, c0, 64)   # (A,12,64,64)
        scl = self._read_window(self._array(self.ds._path("scl", field)), r0, c0, 64)  # (A,64,64)
        clear = np.isin(scl, self._clear_scl)
        raw_days = np.asarray(f["raw_days"])
        from manyup.data.time import from_days
        raw_dates = [from_days(int(d)) for d in raw_days]
        inside = (s2[:, 1:4] != 0).any(1)                        # (A, 64, 64): field polygon

        chosen = {}
        for t in months:
            # ms.timestamps' month is OlmoEarth's own 0-indexed convention (0=Jan..11=Dec,
            # confirmed directly against every one of a real sample's 24 slots, 2026-10-09 --
            # NOT the 1-indexed convention manyup.data.time.month_bounds/datetime.date use
            # everywhere else in this codebase). +1 to compare against d.month (Python's
            # datetime, 1-indexed) -- first cut compared them directly and silently matched
            # every slot to the WRONG calendar month's acquisitions (e.g. slot 9, true October,
            # was rendering September's real imagery; slot 8, true September, found zero
            # matches and silently fell back to ms's bad-colour path) -- caught by printing
            # every slot's (month, year) against real acquisition dates directly, not visible
            # from the rendered image alone since the wrong month still looked plausible.
            _, month0, year = (int(v) for v in timestamps[t].tolist())
            month = month0 + 1
            cand = [a for a, d in enumerate(raw_dates) if d.year == year and d.month == month]
            if not cand:
                continue
            quality = [(clear[a] & inside[a]).sum() for a in cand]
            chosen[t] = cand[int(np.argmax(quality))]

        rgb_idx = [3, 2, 1]                                      # B04, B03, B02 (true colour)
        frames_raw = {t: s2[a][rgb_idx].transpose(1, 2, 0).astype(np.float32)
                      for t, a in chosen.items()}
        stacked = np.stack(list(frames_raw.values())) if frames_raw else np.zeros((1, 64, 64, 3))
        inside_stacked = np.stack([inside[chosen[t]] for t in chosen]) if frames_raw \
            else np.zeros((1, 64, 64), dtype=bool)
        content = stacked[inside_stacked] if inside_stacked.any() else stacked.reshape(-1, 3)
        lo, hi = np.percentile(content, 2), np.percentile(content, 98)
        out = {}
        for t, img in frames_raw.items():
            stretched = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) ** GAMMA
            stretched[~inside[chosen[t]]] = 1.0                   # outside polygon: white
            out[t] = stretched
        return out


GAMMA = 0.8   # matches exp/viz/paper/datasets.py's yieldsat_example (dataset_visualization/
              # poster_timeseries.py's _stretch): gamma < 1 brightens shadows/midtones, same
              # natural-looking tone curve as the reference dataset-overview figure, instead
              # of the flatter look a plain linear percentile stretch gives.


class LumaStretch(NamedTuple):
    lo: float
    hi: float
    mode_val: np.ndarray | None   # (3,) the detected PER-CHANNEL fill value, or None
    is_fill: bool                  # whether mode_val should be painted white (a real fill region)
    raw_frames: dict[int, np.ndarray] | None = None   # yieldsat_raw only -- see YieldSatRawFrames


_YIELDSAT_RAW_FRAMES: dict[str, "YieldSatRawFrames"] = {}   # split -> cached reader (opens the
                                                             # packed index once, not per sample)


def luma_stretch(ms, task, months: list[int], sample_i: int | None = None,
                 split: str = "test") -> LumaStretch:
    """A SHARED percentile-stretch range for luma(), computed once over every present month of
    this sample so a field's true colour trend across the season is visible (not exaggerated/
    hidden by re-stretching each month's panel independently -- the original per-month stretch
    made consecutive months of the same field swing between unrelated-looking colours on
    yieldsat_raw, see [[apt-entropy-selector]]).

    Also excludes the FILL/background mass before computing percentiles: yieldsat_raw's field
    polygon covers only part of the 64x64 tile, and the rest is filled with a near-constant
    value PER CHANNEL (confirmed on a real sample: R~0.34, G~0.3, B~0.283 -- three DIFFERENT
    constants, not one shared RGB triple; NOT flagged invalid by OlmoEarth's own
    sentinel2_l2a_mask, which is all-False/present there -- it's genuinely "present" data from
    the dataset's own pixel_valid convention, just outside the polygon). Including that mass in
    the percentile pins the 2nd-percentile black point at the fill value and visibly washes out
    the real field content. Detected via a PER-CHANNEL MODE heuristic (each channel's own most
    repeated exact value, by construction the fill constant for that band) rather than a
    hardcoded threshold, so this works for any dataset/fill value without per-dataset tuning --
    on a dataset with no large uniform fill region (PASTIS, BioMassters), each channel's mode is
    just whatever value happens to repeat most, a tiny fraction of pixels, and is_fill is False
    (luma() leaves those pixels alone). A pixel counts as fill only when ALL 3 channels
    simultaneously match their own channel's mode (first cut used a single scalar mode pooled
    across channels, which never matched any real pixel since R/G/B fill at different values --
    caught by checking the fraction of pixels flagged was 0% despite a large navy background
    clearly present in the rendered image, 2026-10-09)."""
    oe_name, idx = _rgb_bands(task)
    x = getattr(ms, oe_name)[0, ..., months, :][..., idx].float().cpu().numpy()  # (64,64,n_months,3)
    flat = x.reshape(-1, 3)
    mode_val = np.empty(3)
    for c in range(3):
        vals, counts = np.unique(np.round(flat[:, c], 4), return_counts=True)
        mode_val[c] = vals[counts.argmax()]
    is_fill_px = (np.abs(flat - mode_val) < 1e-3).all(-1)   # per-pixel: ALL channels match
    is_fill = is_fill_px.mean() > 0.15                       # only treat it as fill if LARGE
    content = flat[~is_fill_px] if is_fill else flat
    if content.size == 0:
        content = flat
    lo, hi = np.percentile(content, 2), np.percentile(content, 98)

    raw_frames = None
    if task.name == "yieldsat_raw" and sample_i is not None:
        # ms's per-band-normalized units can't be stretched back to natural colour -- see
        # YieldSatRawFrames' own docstring for the full reasoning. Read true reflectance
        # directly from the packed store instead, for display only (scoring/masks elsewhere
        # in both viz scripts stay on ms, unaffected).
        if split not in _YIELDSAT_RAW_FRAMES:
            _YIELDSAT_RAW_FRAMES[split] = YieldSatRawFrames(split)
        raw_frames = _YIELDSAT_RAW_FRAMES[split](sample_i, months, ms.timestamps[0])
    return LumaStretch(float(lo), float(hi), mode_val if is_fill else None, is_fill, raw_frames)


def luma(ms, task, month_t: int, stretch: LumaStretch | None = None) -> np.ndarray:
    """(64, 64, 3) natural-colour (S2) or false-colour (S1) RGB of sample 0's month month_t,
    percentile + gamma stretched for display (same tone curve as the reference dataset-overview
    figure, figures/paper/datasets/example_yieldsat.png -- see [[apt-entropy-selector]]). ms is
    the full BATCH (B=1 here) as loader() gives it -- (B, 64, 64, T, C) per modality. Picks the
    first guidance modality's rgb proxy bands (task.rgb), same convention as every other viz
    script in this repo (S2_RGB / the SAR VV,VH,VV stand-in). stretch: an optional LumaStretch
    from luma_stretch() shared across a sample's months -- falls back to this month's OWN
    percentile range, no gamma, no fill-painting (the original behaviour) when not given, for
    callers that render a single month in isolation."""
    if stretch is not None and stretch.raw_frames is not None and month_t in stretch.raw_frames:
        return stretch.raw_frames[month_t]    # yieldsat_raw: true-reflectance frame, see above
    oe_name, idx = _rgb_bands(task)
    x = getattr(ms, oe_name)[0, ..., month_t, idx].float().cpu().numpy()   # (64, 64, 3)
    if stretch is None:
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    img = np.clip((x - stretch.lo) / (stretch.hi - stretch.lo + 1e-6), 0, 1) ** GAMMA
    if stretch.is_fill:
        is_fill_px = (np.abs(x - stretch.mode_val) < 1e-3).all(-1)   # ALL channels match
        img[is_fill_px] = 1.0                                        # fill pixels: solid white
    return img


def draw_cells(ax, mask_t: np.ndarray, color_by_run: dict):
    """Outline every ps4 cell mask_t marks True; if it's a merged apt_pool run (color_by_run
    has it), fill with that run's colour (same colour across the run's representative months
    -- only the FIRST month of a run is ever True in the mask, so this just fills that one
    cell; the merge is conveyed by the legend / caption, not by painting every member month,
    since non-representative months never get a crop cell to paint)."""
    yy, xx = np.nonzero(mask_t)
    for y, x in zip(yy, xx):
        fc = color_by_run.get((y, x), "none")
        rect = mpatches.Rectangle((x * CELL_PX, y * CELL_PX), CELL_PX, CELL_PX,
                                  linewidth=1.1, edgecolor="#ffcc00",
                                  facecolor=fc, alpha=0.55 if fc != "none" else 1.0)
        ax.add_patch(rect)


def render_sample(task, ms, present, rows: list[tuple], k_frac: float, sample_i: int,
                  out_path: Path, split: str = "test"):
    """ms, present: the full BATCH as loader() gives them (B=1 -- one sample loaded at a
    time), so _apt_scores / _apt_pool_mask see exactly the shapes InjectFT.forward passes
    them (no model build needed: APT is scored from raw pixels alone).
    rows: [(label, obj_or_None, kind), ...] -- obj_or_None shares the task's score map
    (computed once, from the first row) when None; kind in {"plain", "pool", "res"} picks
    which of apt / apt_pool / apt_res drove that row. split: must match the loader()'s own
    split -- threaded to luma_stretch for yieldsat_raw's raw-pixel reader (YieldSatRawFrames),
    which opens its OWN YieldSatDataset instance and must agree on which split's ids it is
    indexing into."""
    present0 = present[0]                                             # (T,) this sample's
    months = present0.nonzero(as_tuple=True)[0].tolist()
    if not months:
        return
    k = round(k_frac * len(months) * H * H)   # same budget convention as InjectFT.forward
    stretch = luma_stretch(ms, task, months, sample_i, split)  # shared across every panel

    base_obj = next(o for _, o, _ in rows if o is not None)
    scores = base_obj._apt_scores(ms, present, 1, H)[0]                 # (T, H, H)
    cell_score = scores.masked_fill(~present0[:, None, None], 0).sum(0).cpu().numpy()  # (H,H)

    from manyup.seasonal import select
    mask_plain = select(scores[None].float(), present, k)[0].cpu().numpy()            # (T,H,H)

    cmap = plt.get_cmap("tab10")
    row_data = []   # (label, mask (T,H,H) np, groups or None, run_colors, [n_merged], [mask16])
    for label, obj, kind in rows:
        if kind == "plain":
            row_data.append((label, mask_plain, None, {}))
        elif kind == "pool":
            mask_t, groups = obj._apt_pool_mask(ms, present, 1, H, k)
            mask = mask_t[0].cpu().numpy()
            merged = {(t, y, x): v for (b, t, y, x), v in groups.items() if len(v) > 1}
            run_colors = {(y, x): cmap(i % 10) for i, (t, y, x) in enumerate(merged)}
            row_data.append((label, mask, groups, run_colors, len(merged)))
        elif kind == "res":
            mask4_t, res_masks = obj._apt_res_mask(ms, present, 1, k)
            mask4 = mask4_t[0].cpu().numpy()
            mask16 = res_masks[16][0].cpu().numpy()
            # a split ps16 cell's 4x4 ps4 children are all True together, so count split
            # cells via the ps16 grid directly (16px blocks), not the ps4 mask's shape.
            n_split = int(mask4_t[0].view(mask4_t.shape[1], 4, 4, 4, 4).any(-1).any(-2).sum().item())
            row_data.append((label, mask4, None, {}, n_split, mask16))
        else:
            raise ValueError(kind)

    n_months = len(months)
    nrows = len(rows)
    ncols = n_months + 1    # score map + one panel per present month
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.3 * nrows))
    for col, t in enumerate([None] + months):
        for row, entry in enumerate(row_data):
            label = entry[0]
            ax = axes[row, col]
            if t is None:
                ax.imshow(cell_score, cmap="magma")
                if row == 0:
                    ax.set_title(f"score (sum over\n{n_months} months)", fontsize=8)
                if row == nrows - 1:
                    ax.set_xlabel("H_space + H_time", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            mask = entry[1]
            run_colors = entry[3]
            ax.imshow(luma(ms, task, t, stretch))
            draw_cells(ax, mask[t], run_colors)
            if len(entry) > 5:                                  # "res" row: also has mask16
                draw_cells16(ax, entry[5][t])
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"slot {t}\n{slot_label(ms.timestamps[0], t)}", fontsize=8)
            if col == 1:
                if len(entry) > 5:
                    tag = f"\n({entry[4]} split)"
                elif len(entry) > 4:
                    tag = f"\n({entry[4]} merged)"
                else:
                    tag = ""
                ax.set_ylabel(label + tag, fontsize=9)
    fig.suptitle(f"{task.name} test sample {sample_i} -- k_frac={k_frac:.3g}, "
                f"budget {k} crops", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", default="pastis,biomassters,geoidflood,yieldsat_raw")
    ap.add_argument("--n", type=int, default=4, help="test samples per dataset")
    ap.add_argument("--k_frac", type=float, default=1 / 6)
    ap.add_argument("--split", default="test")
    ap.add_argument("--demo_merge_mse", type=float, default=1.5,
                    help="a second --apt_pool row at this (much looser) threshold, so the "
                         "merge mechanism is visibly demonstrated even on datasets where the "
                         "default 0.01 threshold finds no mergeable runs among the cells apt "
                         "actually picks (observed on PASTIS / YieldSAT test samples -- their "
                         "highest-scoring cells genuinely change a lot month to month; the "
                         "cells that WOULD merge at 0.01 are low-score background apt never "
                         "picks anyway, see [[apt-entropy-selector]])")
    a = ap.parse_args()

    for name in a.tasks.split(","):
        task = TASKS[name]
        default_obj = apt_stub(task, apt_merge_mse=0.01)
        demo_obj = apt_stub(task, apt_merge_mse=a.demo_merge_mse)
        rows = [("apt (top-k)", default_obj, "plain"),
                ("apt_pool\n(mse<0.01)", default_obj, "pool"),
                (f"apt_pool\n(mse<{a.demo_merge_mse:g}, demo)", demo_obj, "pool"),
                ("apt_res\n(ps16 vs ps4)", default_obj, "res")]
        dl = loader(task, a.split, batch_size=1, num_workers=2, shuffle=False)
        it = iter(dl)
        for i in range(a.n):
            ms, ss, season, present, pairs, extra, target = next(it)
            render_sample(task, ms, present, rows, a.k_frac, i,
                         OUT_DIR / f"{name}_sample{i}.png", a.split)


if __name__ == "__main__":
    main()
