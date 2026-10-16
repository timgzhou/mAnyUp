"""Paper material, model: the mixed-resolution pipeline on one PASTIS test tile, from the
output of exp.viz.paper.collect.

    python -m exp.viz.paper.model [--tile 40091@r0c0]

Writes to figures/paper/model/<tile>/:
    pipeline.{png,pdf}     every stage on one shared time axis (12 month columns, 4 seasons):
                           acquisitions -> monthly composites -> seasonal composites -> season
                           features -> month query (score + picks) -> date query (score + picks)
                           -> the mixed-resolution sparse feature map -> prediction
    volume.{png,pdf}       the output tokens alone, as three stacks of time slices (season ps16
                           tokens, sparse month ps4 tokens, sparser date ps4 tokens)
    materials/             every panel of both as its own PNG (nearest-neighbour upscaled),
                           plus materials.json describing them, for assembling a figure by hand

Colours: S2 true colour (B04, B03, B02) and S1 as R = VV, G = VH, B = VV - VH, each with ONE
stretch for acquisitions, months and seasons (see Render). Features: the first
three principal components of all final tokens together (season + month + date crops + the
crop-free season tokens), one basis for every feature panel. Query scores: viridis, one scale
per level; picked cells are outlined.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

from manyup.data.datasets.pastis import CLASS_NAMES
from manyup.data.time import from_days

OUT = Path("figures/paper/model")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
INK, MUTED, RULE, ACCENT = "#111111", "#555555", "#c9c9c9", "#e4572e"
_tab = plt.get_cmap("tab20").colors
PASTIS_COLORS = np.array([(0, 0, 0)] + [_tab[i] for i in range(1, 19)] + [(1, 1, 1)])
MONTHS = ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep")
SEASONS = ("Oct–Dec", "Jan–Mar", "Apr–Jun", "Jul–Sep")
MOD_NAME = {"S2": "Sentinel-2", "S1A": "Sentinel-1"}


# ---------------------------------------------------------------- rendering
def stretch(x, lo, hi, gamma=1.0):
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1) ** gamma


class Render:
    """Raw frames -> RGB, one stretch per level and modality.
    S2 acquisitions: per band, the median over dates of each frame's 1st / 99th percentile (a
    few cloudy dates do not set it). S2 monthly / seasonal MEANS mix clear and cloudy dates
    (PASTIS has no cloud mask), so they are far brighter: one range for the three bands (1st to
    99.5th percentile of the level) and gamma 0.5, so cloudy composites stay readable and
    white stays white. S1: the monthly composites' 2-98% per channel, at every level."""

    def __init__(self, images):
        r2 = images["raw"]["S2"]
        x = r2["data"][r2["valid"]][:, [2, 1, 0]].numpy()
        self.s2 = {"raw": (np.median(np.percentile(x, 1, axis=(2, 3)), 0),
                           np.median(np.percentile(x, 99, axis=(2, 3)), 0), 0.85)}
        for level in ("month", "season"):
            m2 = images[level]["S2"]
            x = m2["data"][m2["valid"]][:, [2, 1, 0]].numpy()
            self.s2[level] = (*np.percentile(x, [1, 99.5]), 0.5)
        m1 = images["month"]["S1A"]
        x = self._s1(m1["data"][m1["valid"]].numpy())
        self.s1 = (np.percentile(x, 2, axis=(0, 2, 3)), np.percentile(x, 98, axis=(0, 2, 3)))

    @staticmethod
    def _s1(d):
        return np.stack([d[:, 0], d[:, 1], d[:, 0] - d[:, 1]], 1)

    def __call__(self, mod, frame, level="raw"):
        """frame (C, H, W) -> (H, W, 3) in [0, 1]"""
        f = frame.numpy()[None]
        if mod == "S2":
            lo, hi, g = self.s2[level]
            x = f[:, [2, 1, 0]]
        else:
            x, (lo, hi), g = self._s1(f), self.s1, 1.0
        return stretch(x[0].transpose(1, 2, 0), lo, hi, g)


class PCA:
    def __init__(self, feats: list[torch.Tensor]):
        X = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats]).double()
        self.mean = X.mean(0)
        _, _, V = torch.pca_lowrank(X - self.mean, q=3, center=False, niter=6)
        self.V = V
        P = (X - self.mean) @ V
        self.lo, self.hi = np.percentile(P.numpy(), 2, axis=0), np.percentile(P.numpy(), 98, axis=0)

    def __call__(self, f: torch.Tensor) -> np.ndarray:
        P = ((f.double() - self.mean) @ self.V).numpy()
        return stretch(P, self.lo, self.hi)


def sparse_rgba(rgb, y, x, n=16):
    """Token colours (k, 3) at cells (y, x) of an n x n grid -> (n, n, 4), empty = transparent."""
    img = np.zeros((n, n, 4), np.float32)
    img[y, x, :3] = rgb
    img[y, x, 3] = 1.0
    return img


def save_png(img, path, px=512):
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    k = max(1, round(px / a.shape[0]))
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    mode = "RGBA" if a.shape[-1] == 4 else "RGB"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, mode).resize((a.shape[1] * k, a.shape[0] * k), Image.NEAREST).save(path)


# ---------------------------------------------------------------- figure helpers
class Canvas:
    """Axes placed in inches from the top-left corner."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W, H), facecolor="white")

    def ax(self, x, y, w, h, frame=RULE, lw=0.4):
        ax = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(lw), sp.set_color(frame)
        return ax

    def text(self, x, y, s, **kw):
        kw.setdefault("color", INK)
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def line(self, x0, y0, x1, y1, **kw):
        kw.setdefault("color", RULE), kw.setdefault("lw", 0.6)
        self.fig.add_artist(plt.Line2D([x0 / self.W, x1 / self.W],
                                       [1 - y0 / self.H, 1 - y1 / self.H], **kw))

    def image(self, x, y, s, img, grid=0, picks=None, frame=RULE, lw=0.4, **kw):
        ax = self.ax(x, y, s, s, frame, lw)
        n = img.shape[0]
        ax.imshow(img, interpolation="nearest", extent=(0, n, n, 0), **kw)
        if grid:                                       # token grid lines
            for v in range(1, grid):
                ax.axhline(v * n / grid, color="white", lw=0.3, alpha=0.7)
                ax.axvline(v * n / grid, color="white", lw=0.3, alpha=0.7)
        if picks is not None:                          # picked cells: outlined
            for yy, xx in zip(*np.nonzero(picks)):
                ax.add_patch(Rectangle((xx, yy), 1, 1, fill=False, ec=ACCENT, lw=0.45))
        return ax

    def empty(self, x, y, s):
        self.fig.patches.append(Rectangle((x / self.W, 1 - (y + s) / self.H), s / self.W,
                                          s / self.H, fill=False, ec=RULE, lw=0.4,
                                          ls=(0, (2, 2)), transform=self.fig.transFigure))

    def save(self, path):
        for ext in ("png", "pdf"):
            self.fig.savefig(path.with_suffix("." + ext), dpi=300, bbox_inches="tight",
                             pad_inches=0.1)
        plt.close(self.fig)
        print("wrote", path.with_suffix(".png"))


def by_month(month_of, valid):
    """date index lists per month (dates in time order), for dates inside the month grid."""
    out = [[] for _ in range(12)]
    for d, (m, ok) in enumerate(zip(month_of.tolist(), valid.tolist())):
        if ok and m >= 0:
            out[m].append(d)
    return out


# ---------------------------------------------------------------- the pipeline figure
def pipeline(o, dest: Path):
    im = o["images"]
    render = Render(im)
    mods = list(im["raw"])
    lvl, mod, step = o["tok_level"].numpy(), o["tok_mod"].numpy(), o["tok_step"].numpy()
    ty, tx, tmonth = o["tok_y"].numpy(), o["tok_x"].numpy(), o["tok_month"].numpy()
    pca = PCA([o["F_free"], o["F_season"], o["tok"]])
    tok_rgb = pca(o["tok"])
    present = o["present"].numpy()
    season_of = o["season_of_month"].numpy()
    dates = {m: by_month(o["date_month"][m], o["date_valid"][m]) for m in mods}
    mat, meta = dest / "materials", {}

    def keep(name, img, note):
        save_png(img, mat / f"{name}.png")
        meta[name] = note

    cw, left, gap = 1.0, 2.05, 0.06            # month column width, label gutter, panel gap
    big = cw - gap                              # one image per month
    ncol = 3                                    # date thumbnails per month: 3 per line
    small = (cw - gap) / ncol - 0.02
    lines = {m: max(-(-len(v) // ncol) for v in dates[m]) for m in mods}
    row_dates = {m: lines[m] * (small + 0.02) for m in mods}
    W = left + 12 * cw + 0.35 + 2.3
    heights = [0.5] + [row_dates[m] for m in mods] + [big] * 2 + [big] + [big] + \
              [row_dates[m] for m in mods] + [big, big, row_dates["S2"]]
    H = sum(heights) + 0.22 * len(heights) + 7 * 0.34 + 0.4
    c = Canvas(W, H)
    X = lambda mth: left + mth * cw + gap / 2  # noqa: E731

    y = 0.2
    for mth in range(12):
        c.text(X(mth) + big / 2, y + 0.12, MONTHS[mth], ha="center", va="center", fontsize=9,
               weight="bold")
    for s in range(4):
        c.line(X(3 * s), y + 0.3, X(3 * s + 2) + big, y + 0.3, color=INK, lw=0.8)
        c.text(X(3 * s + 1) + big / 2, y + 0.4, f"season {s + 1} · {SEASONS[s]}", ha="center",
               va="center", fontsize=7.5, color=MUTED)
    y += 0.62

    def stage(n, title, sub, y):
        c.line(0.15, y - 0.05, W - 0.15, y - 0.05)
        c.text(0.18, y + 0.14, f"{n}  {title}", fontsize=9.5, weight="bold", va="center")
        c.text(0.18, y + 0.34, sub, fontsize=7.5, color=MUTED, va="center")
        return y + 0.56

    def label(y, h, text):
        c.text(left - 0.12, y + h / 2, text, ha="right", va="center", fontsize=7.5)

    def date_row(y, m, draw):
        """draw(date index, x, y, size) for every date of modality m, 3 per line per month."""
        for mth in range(12):
            for j, d in enumerate(dates[m][mth]):
                draw(d, X(mth) + (j % ncol) * (small + 0.02), y + (j // ncol) * (small + 0.02),
                     small)
        return y + row_dates[m] + 0.14

    # 1 acquisitions
    y = stage("1", "Acquisitions", "every date of each modality, on its own dates "
              "(only the dates inside the 12-month grid are shown)", y)
    for m in mods:
        r = im["raw"][m]
        n_in = sum(len(v) for v in dates[m])
        label(y, row_dates[m], f"{MOD_NAME[m]}\n{n_in} dates")

        def draw(d, x, yy, s, m=m, r=r):
            img = render(m, r["data"][d])
            c.image(x, yy, s, img)
            keep(f"1_acquisition_{m}_{d:02d}_{from_days(int(r['start'][d])):%Y%m%d}", img,
                 f"{MOD_NAME[m]} acquisition {from_days(int(r['start'][d]))}")
        y = date_row(y, m, draw)

    # 2 monthly composites
    y = stage("2", "Monthly composites", "mean of each month's acquisitions, clouds included (PASTIS has no cloud "
              "mask) · the month-level crops are cut from these", y)
    for m in mods:
        r = im["month"][m]
        label(y, big, f"{MOD_NAME[m]}\n12 months")
        for mth in range(12):
            if not bool(r["valid"][mth]):
                c.empty(X(mth), y, big)
                continue
            img = render(m, r["data"][mth], "month")
            c.image(X(mth), y, big, img)
            keep(f"2_month_{m}_{mth:02d}_{MONTHS[mth]}", img, f"{MOD_NAME[m]} monthly mean")
        y += big + 0.14

    # 3 seasonal composites + their ps16 features
    y = stage("3", "Seasonal composites → OlmoEarth at patch size 16",
              "mean of 3 months · 4 × 4 tokens per season · features: first 3 principal "
              "components, crop-free pass", y)
    label(y, big, "S2 · S1 composite →\nps16 season tokens")
    for s in range(4):
        for j, m in enumerate(mods):
            r = im["season"][m]
            img = render(m, r["data"][s], "season")
            c.image(X(3 * s + j), y, big, img, grid=4)
            keep(f"3_season_{m}_{s}", img, f"{MOD_NAME[m]} seasonal mean, season {s + 1}")
        f = pca(o["F_free"][s])
        c.image(X(3 * s + 2), y, big, f, frame=INK, lw=0.8)
        c.text(X(3 * s + 2) - gap / 2, y + big / 2, "→", ha="center", va="center", fontsize=9)
        keep(f"3_season_features_cropfree_{s}", f,
             "season tokens (4x4, ps16), crop-free pass, PCA colours")
    y += big + 0.14

    # 4 month query
    k = int(o["month_mask"].sum())
    y = stage("4", "Month-level query", f"a score for every (month, ps4 cell): 12 × 16 × 16 "
              f"· the top {k} ({k / present.sum() / 256:.0%}) are cropped from the monthly "
              f"composites at patch size 4 (outlined)", y)
    sc = o["month_score"].numpy()
    lo, hi = np.percentile(sc[present], [1, 99])
    label(y, big, f"query score\n{k} picks")
    for mth in range(12):
        if not present[mth]:
            c.empty(X(mth), y, big)
            continue
        pk = o["month_mask"][mth].numpy()
        c.image(X(mth), y, big, sc[mth], picks=pk, cmap="viridis", vmin=lo, vmax=hi)
        c.text(X(mth) + big / 2, y + big + 0.07, f"{int(pk.sum())}", ha="center", va="center",
               fontsize=6.5, color=ACCENT)
        keep(f"4_month_query_score_{mth:02d}_{MONTHS[mth]}",
             plt.get_cmap("viridis")(stretch(sc[mth], lo, hi))[..., :3],
             "month-level query score, 16x16 ps4 cells")
        keep(f"4_month_query_mask_{mth:02d}_{MONTHS[mth]}", pk.astype(np.float32),
             "month-level picks (white = cropped)")
    y += big + 0.26

    # 5 date query
    n_date = int((lvl == 1).sum())
    y = stage("5", "Date-level query", f"a score for every (modality, date, ps4 cell) · the "
              f"top {n_date} over both modalities are cropped from single acquisitions at "
              f"patch size 4 (outlined)", y)
    allz = np.concatenate([o["date_score"][m].numpy()[o["date_cand"][m].numpy()] for m in mods])
    lo, hi = np.percentile(allz, [1, 99])
    picked = {}
    for j, m in enumerate(mods):
        z = o["date_score"][m].numpy()
        pk = np.zeros(z.shape, bool)
        sel = (lvl == 1) & (mod == j)
        pk[step[sel], ty[sel], tx[sel]] = True
        picked[m] = pk
        label(y, row_dates[m], f"{MOD_NAME[m]}\n{int(pk.sum())} picks")

        def draw(d, x, yy, s, m=m, z=z, pk=pk):
            c.image(x, yy, s, z[d], picks=pk[d] if pk[d].any() else None, cmap="viridis",
                    vmin=lo, vmax=hi)
            day = from_days(int(im["raw"][m]["start"][d]))
            keep(f"5_date_query_score_{m}_{d:02d}_{day:%Y%m%d}",
                 plt.get_cmap("viridis")(stretch(z[d], lo, hi))[..., :3],
                 "date-level query score, 16x16 ps4 cells")
            keep(f"5_date_query_mask_{m}_{d:02d}_{day:%Y%m%d}", pk[d].astype(np.float32),
                 "date-level picks")
        y = date_row(y, m, draw)

    # 6 the mixed-resolution feature map
    y = stage("6", "Output: mixed-resolution, sparse, temporal feature map",
              "all tokens attend to each other in every block · dense ps16 season tokens, "
              "sparse ps4 month tokens, sparser ps4 date tokens (white = no token)", y)
    y6 = y
    label(y, big, "season tokens\nps16 · 4 × 4 × 4")
    for s in range(4):
        f = pca(o["F_season"][s])
        c.line(X(3 * s), y + big / 2, X(3 * s + 2) + big, y + big / 2, color=RULE, lw=0.8)
        ax = c.image(X(3 * s + 1), y, big, f, grid=4, frame=INK, lw=0.8)
        ax.set_zorder(3)
        keep(f"6_tokens_season_{s}", f, "final season tokens (4x4, ps16), PCA colours")
    y += big + 0.14
    label(y, big, f"month tokens\nps4 · {k} of {int(present.sum()) * 256}")
    for mth in range(12):
        sel = (lvl == 0) & (step == mth)
        img = sparse_rgba(tok_rgb[sel], ty[sel], tx[sel])
        c.image(X(mth), y, big, img)
        keep(f"6_tokens_month_{mth:02d}_{MONTHS[mth]}", img,
             "final month-crop tokens (ps4), PCA colours, transparent = no token")
    y += big + 0.14
    n_cand = int(sum(o["date_cand"][m][:, 0, 0].sum() for m in mods)) * 256
    label(y, row_dates["S2"], f"date tokens\nps4 · {n_date} of {n_cand:,}")
    for j, m in enumerate(mods):
        if not picked[m].any():
            continue

        def draw(d, x, yy, s, j=j, m=m):
            sel = (lvl == 1) & (mod == j) & (step == d)
            img = sparse_rgba(tok_rgb[sel], ty[sel], tx[sel])
            c.image(x, yy, s, img)
            if sel.any():
                day = from_days(int(im["raw"][m]["start"][d]))
                keep(f"6_tokens_date_{m}_{d:02d}_{day:%Y%m%d}", img,
                     "final date-crop tokens (ps4), PCA colours")
        y = date_row(y, m, draw)
    # prediction and label, right of stage 6
    target = o["target"].numpy()
    pred = o["logits"].argmax(0).numpy()
    lab = PASTIS_COLORS[np.where(target < 0, 19, target)]
    px = left + 12 * cw + 0.35
    side = 1.05
    c.image(px, y6, side, PASTIS_COLORS[pred])
    c.text(px + side / 2, y6 - 0.07, "prediction", ha="center", fontsize=7.5)
    c.image(px + side + 0.1, y6, side, lab)
    c.text(px + 1.5 * side + 0.1, y6 - 0.07, "label", ha="center", fontsize=7.5)
    ok = target >= 0
    c.text(px, y6 + side + 0.14, f"pixel accuracy {(pred[ok] == target[ok]).mean():.0%} on "
           f"this tile", fontsize=6.5, color=MUTED)
    classes = [k_ for k_ in np.unique(np.concatenate([pred.ravel(), target[ok].ravel()]))]
    c.fig.legend(handles=[Patch(fc=PASTIS_COLORS[k_], ec="0.5", lw=0.4, label=CLASS_NAMES[k_])
                          for k_ in classes], loc="upper left", frameon=False, fontsize=6.2,
                 handlelength=1.0, labelspacing=0.3,
                 bbox_to_anchor=(px / W, 1 - (y6 + side + 0.25) / H))
    keep("7_prediction", PASTIS_COLORS[pred], "argmax prediction, 64x64 px")
    keep("7_label", lab, "label, 64x64 px (white = void)")
    c.save(dest / "pipeline")
    (mat / "materials.json").write_text(json.dumps(meta, indent=1))
    return pca, tok_rgb, dates


# ---------------------------------------------------------------- the token volume
def volume(o, dest: Path, pca, tok_rgb, dates):
    """The output tokens as three stacks of time slices seen at an angle: time runs to the
    right; each slice is the 64 x 64 px tile at one time step."""
    lvl, mod, step = o["tok_level"].numpy(), o["tok_mod"].numpy(), o["tok_step"].numpy()
    ty, tx = o["tok_y"].numpy(), o["tok_x"].numpy()
    raw = o["images"]["raw"]
    mods = list(raw)
    day0 = int(o["images"]["month"]["S2"]["start"][0])
    day1 = int(o["images"]["month"]["S2"]["end"][-1])
    fig = plt.figure(figsize=(13, 6.4), facecolor="white")
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.9])
    ax.set_xlim(-2.2, 13.6), ax.set_ylim(-0.25, 5.2), ax.set_aspect("equal"), ax.axis("off")
    shear = Affine2D.from_values(0.42, 0.26, 0, 1.0, 0, 0)       # a slice: narrow, leaning

    def slab(img, x, y0, edge=RULE, lw=0.5, z=1):
        tr = shear + Affine2D().translate(x, y0) + ax.transData
        ax.imshow(img, extent=(0, 1, 0, 1), transform=tr, interpolation="nearest", zorder=z,
                  clip_on=False)
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=edge, lw=lw, transform=tr, zorder=z,
                               clip_on=False))

    def t2x(day):                       # time -> x, months 1 unit apart
        return 12 * (day - day0) / (day1 - day0 + 1)

    rows = {"season": 3.7, "month": 1.95, "date": 0.2}
    for s in range(4):
        slab(pca(o["F_season"][s]), 3 * s + 1.0, rows["season"], INK, 0.6)
    ms = o["images"]["month"]["S2"]
    for m in range(12):
        sel = (lvl == 0) & (step == m)
        img = sparse_rgba(tok_rgb[sel], ty[sel], tx[sel])
        slab(img, t2x((int(ms["start"][m]) + int(ms["end"][m])) / 2) - 0.2, rows["month"])
    for j, m in enumerate(mods):
        for d in [d for v in dates[m] for d in v]:
            sel = (lvl == 1) & (mod == j) & (step == d)
            if not sel.any():
                continue
            slab(sparse_rgba(tok_rgb[sel], ty[sel], tx[sel]), t2x(int(raw[m]["start"][d])) - 0.2,
                 rows["date"])
    n_m, n_d = int((lvl == 0).sum()), int((lvl == 1).sum())
    for name, yy, txt in (("season", rows["season"], "season tokens\npatch size 16\n4 seasons × 4 × 4, dense"),
                          ("month", rows["month"], f"month tokens\npatch size 4\n{n_m} of 12 × 16 × 16"),
                          ("date", rows["date"], f"date tokens\npatch size 4\n{n_d}, on {len({(a, b) for a, b in zip(mod[lvl == 1], step[lvl == 1])})} dates")):
        ax.text(-2.15, yy + 0.6, txt, fontsize=9, va="center", ha="left", color=INK)
    for m in range(12):
        ax.text(m + 0.5, -0.12, MONTHS[m], ha="center", va="top", fontsize=8.5, color=MUTED)
        ax.plot([m, m], [-0.06, 0.0], color=RULE, lw=0.6)
    ax.plot([0, 12], [-0.03, -0.03], color=RULE, lw=0.6)
    ax.annotate("", xy=(12.6, -0.03), xytext=(12, -0.03),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.text(12.65, -0.03, "time", va="center", fontsize=8.5, color=MUTED)
    fig.text(0.02, 0.95, "Mixed-resolution, sparse, temporal feature map", fontsize=12,
             weight="bold", color=INK)
    fig.text(0.02, 0.915, "each slice is the 64 × 64 px tile at one time step · colour = first "
             "3 principal components of the final tokens · empty cells hold no token",
             fontsize=8.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"volume.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote", dest / "volume.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", default="40091@r0c0")
    ap.add_argument("--suffix", default="", help="collect file suffix, e.g. _randmonths")
    a = ap.parse_args()
    name = a.tile.replace("@", "_") + a.suffix
    o = torch.load(OUT / f"collect_{name}.pt", weights_only=False)
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    pca, tok_rgb, dates = pipeline(o, dest)
    volume(o, dest, pca, tok_rgb, dates)


if __name__ == "__main__":
    main()
