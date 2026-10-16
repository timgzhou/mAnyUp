"""Paper material, readout over a whole low-res cell: one ps16 block (16 x 16 px = 4 x 4 ps4
cells) with every token that covers any part of it, and where each ps4 cell's prediction comes
from -- cells with high-res crops read the crops, cells without read the season tokens.

    python -m exp.viz.paper.readout_block [--tiles 40091@r0c0 ...] [--n 6] [--block by,bx]

Writes to figures/paper/model/readout_block/:
    cand<rank>_<tile>_b<by><bx>.{png,pdf}   one figure per candidate block, best first
    candidates.json                          the ranking and each block's numbers
    summary.json                             over all given tiles: cells with / without crops,
                                             their accuracy and their weight on season tokens

Each figure:
    left     the tile: label, season tokens only, all tokens (the block outlined)
    centre   time axis. Season tokens: each one's 16 x 16 px class map. Month and date tokens:
             one panel per month / date that has a crop in the block -- the block's 4 x 4 grid
             of ps4 cells, each cropped cell filled with the 4 x 4 px class map of its token,
             uncropped cells left empty
    right    the temporal pool over the block (season tokens only -> all tokens -> label), and
             per ps4 cell: how many crops cover it, the pool's total weight on the season
             tokens, and on the ps4 tokens

Candidate blocks (ranked): >= 2 label classes, some cells with crops and some without, and a
large gain from the crops. See exp.viz.paper.readout for the readout itself and for what
"season tokens only" means.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.patches import Patch, Rectangle

from exp.viz.paper.model import INK, MONTHS, MUTED, OUT, PASTIS_COLORS, RULE, save_png
from exp.viz.paper.readout import NS, cell_tokens, season_only, spread
from manyup.data.datasets.pastis import CLASS_NAMES

EMPTY = np.array([0.94, 0.94, 0.94])
LANE = {"season": 6.35, "month": 3.3, "date": 0.25}


def cell_stats(o) -> dict:
    """Per ps4 cell (16, 16): crops covering it, weight on season tokens, px correct with the
    season tokens only and with all tokens, labelled px."""
    n = np.zeros((16, 16))
    np.add.at(n, (o["tok_y"].numpy(), o["tok_x"].numpy()), 1)
    w = o["wgt"][:, :NS].sum(1).view(16, 16).numpy()
    target = o["target"]
    ok = rearrange((target >= 0).float(), "(y i) (x j) -> y x (i j)", i=4, j=4)
    out = {"crops": n, "w_season": w, "labelled": ok.sum(-1).numpy()}
    for name, pred in (("season", season_only(o).argmax(0)), ("all", o["logits"].argmax(0))):
        hit = rearrange(((pred == target) & (target >= 0)).float(), "(y i) (x j) -> y x (i j)",
                        i=4, j=4)
        out[f"hit_{name}"] = hit.sum(-1).numpy()
    return out


def rank_blocks(o) -> list[tuple[float, tuple[int, int], dict]]:
    st = cell_stats(o)
    target = o["target"]
    out = []
    for by in range(4):
        for bx in range(4):
            c = (slice(4 * by, 4 * by + 4), slice(4 * bx, 4 * bx + 4))
            blk = target[16 * by:16 * by + 16, 16 * bx:16 * bx + 16]
            lab = st["labelled"][c].sum()
            classes = len(blk[blk >= 0].unique())
            free = int((st["crops"][c] == 0).sum())
            if lab < 128 or classes < 2 or not 2 <= free <= 13:
                continue
            a_s, a_a = st["hit_season"][c].sum() / lab, st["hit_all"][c].sum() / lab
            fm = st["crops"][c] == 0
            free_acc = st["hit_all"][c][fm].sum() / max(st["labelled"][c][fm].sum(), 1)
            score = 2 * (a_a - a_s) + a_a + 0.5 * free_acc + 0.3 * min(classes, 4) \
                - 0.05 * abs(free - 6)
            out.append((float(score), (by, bx), dict(
                classes=classes, cells_without_crops=free, crops=int(st["crops"][c].sum()),
                acc_season=float(a_s), acc_all=float(a_a), acc_cells_without_crops=float(free_acc))))
    return sorted(out, key=lambda v: -v[0])


def figure(o, by: int, bx: int, path: Path, facts: dict | None = None) -> dict:
    target = o["target"].numpy()
    lr = season_only(o).argmax(0).numpy()
    hr = o["logits"].argmax(0).numpy()
    st = cell_stats(o)
    mat = path.parent / (path.name + "_materials")
    bl = (slice(16 * by, 16 * by + 16), slice(16 * bx, 16 * bx + 16))
    cs = (slice(4 * by, 4 * by + 4), slice(4 * bx, 4 * bx + 4))
    cells = [(by * 4 + i, bx * 4 + j) for i in range(4) for j in range(4)]
    toks = {(y, x): cell_tokens(o, y * 16 + x) for y, x in cells}
    FW, FH = 16.0, 8.0
    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    fig.text(0.012, 0.962, "Readout over one low-res cell: where high-res tokens exist they "
             "take over, elsewhere the season tokens answer", fontsize=13, weight="bold",
             color=INK)
    fig.text(0.012, 0.928, "one ps16 block = 16 × 16 px = 4 × 4 ps4 cells · every token predicts "
             "per pixel at its own patch size · each ps4 cell pools the tokens covering it",
             fontsize=9, color=MUTED)

    def frame(ax, color=RULE, lw=0.5):
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color), sp.set_linewidth(lw)

    def square(x0, y0, w):
        return fig.add_axes([x0, y0, w, w * FW / FH])

    # ---- left: the tile
    ok = target >= 0
    lab_rgb = PASTIS_COLORS[np.where(ok, target, 19)]
    tiles = [("label · 64 × 64 px tile", lab_rgb),
             (f"season tokens only (ps16) · {(lr[ok] == target[ok]).mean():.0%} px correct",
              PASTIS_COLORS[lr]),
             (f"all tokens (ps16 + ps4) · {(hr[ok] == target[ok]).mean():.0%} px correct",
              PASTIS_COLORS[hr])]
    for j, (name, img) in enumerate(tiles):
        ax = square(0.012, 0.635 - j * 0.295, 0.118)
        ax.imshow(img, extent=(0, 64, 64, 0), interpolation="nearest")
        for v in range(16, 64, 16):
            ax.axhline(v, color="white", lw=0.5, alpha=0.8)
            ax.axvline(v, color="white", lw=0.5, alpha=0.8)
        ax.add_patch(Rectangle((16 * bx, 16 * by), 16, 16, fill=False, ec=INK, lw=1.8))
        frame(ax)
        ax.set_title(name, fontsize=7.5, color=INK, loc="left", pad=3)

    # ---- centre: tokens on a time axis, all drawn in the block's frame
    ax = fig.add_axes([0.155, 0.05, 0.585, 0.85])
    ax.set_xlim(-0.2, 12.2), ax.set_ylim(-1.1, 9.0), ax.set_aspect("equal"), ax.axis("off")
    for m in range(12):
        ax.text(m + 0.5, -0.32, MONTHS[m], ha="center", va="center", fontsize=8.5, color=MUTED)
    for m in range(13):
        ax.plot([m, m], [-0.12, 0.0], color=RULE, lw=0.7)
        ax.plot([m, m], [0, 8.75], color=RULE, lw=0.4, ls=(0, (1, 3)), zorder=0)
    ax.plot([0, 12], [0, 0], color=INK, lw=0.8)

    def thumb(cx, y0, size, img, edge=INK, lw=0.9, cell_grid=True):
        ax.imshow(img, extent=(cx - size / 2, cx + size / 2, y0, y0 + size),
                  interpolation="nearest", zorder=3)
        if cell_grid:                             # the 4 x 4 ps4 cells of the block
            for v in range(1, 4):
                u = size * v / 4
                ax.plot([cx - size / 2 + u] * 2, [y0, y0 + size], color="white", lw=0.5, zorder=4)
                ax.plot([cx - size / 2, cx + size / 2], [y0 + u] * 2, color="white", lw=0.5,
                        zorder=4)
        ax.add_patch(Rectangle((cx - size / 2, y0), size, size, fill=False, ec=edge, lw=lw,
                               zorder=5))

    # season tokens: the whole 16 x 16 px map
    q_of = lambda y, x: y * 16 + x  # noqa: E731
    seasons = [t for t in toks[cells[0]] if t["level"] == "season"]
    for t in seasons:
        lg = torch.stack([o["cand"][q_of(y, x), t["k"]] for y, x in cells])
        full = rearrange(lg, "(i j) c (u v) -> c (i u) (j v)", i=4, u=4)
        img = PASTIS_COLORS[full.argmax(0).numpy()]
        cx = (t["t0"] + t["t1"]) / 2
        yb = LANE["season"]
        ax.add_patch(Rectangle((t["t0"] + 0.06, yb + 0.25), t["t1"] - t["t0"] - 0.12, 0.1,
                               fc="#8a8a8a", ec="none"))
        thumb(cx, yb + 0.55, 1.7, img, cell_grid=False)
        save_png(img, mat / f"season_{t['k']}_classmap_16px.png")

    # month / date tokens: one block-frame panel per month / date with a crop in the block
    def panels(level, key):
        groups = {}
        for (y, x), ts in toks.items():
            for t in ts:
                if t["level"] == level:
                    groups.setdefault(key(t), []).append((y, x, t))
        return sorted(groups.items(), key=lambda kv: kv[1][0][2]["t0"])

    counts = {}
    for level, key in (("month", lambda t: t["month"]), ("date", lambda t: (t["mod"], t["date"]))):
        gs = panels(level, key)
        counts[level] = (sum(len(v) for _, v in gs), len(gs))
        if not gs:
            continue
        size = min(0.95, 11.6 / len(gs) - 0.1)
        cxs = spread([(v[0][2]["t0"] + v[0][2]["t1"]) / 2 for _, v in gs], size + 0.1, 0, 12)
        yb = LANE[level]
        for (k, members), cx in zip(gs, cxs):
            img = np.tile(EMPTY, (16, 16, 1))
            for y, x, t in members:
                i, j = y - 4 * by, x - 4 * bx
                img[4 * i:4 * i + 4, 4 * j:4 * j + 4] = PASTIS_COLORS[
                    t["logits"].argmax(0).view(4, 4).numpy()]
            t0 = members[0][2]
            if level == "month":
                ax.add_patch(Rectangle((t0["t0"] + 0.06, yb), 0.88, 0.1, fc="#8a8a8a", ec="none"))
                ax.plot([cx, t0["t0"] + 0.5], [yb + 0.75, yb + 0.1], color=MUTED, lw=0.6)
            else:
                ax.plot([t0["t0"]] * 2, [0, yb], color=INK, lw=1.2)
                ax.plot([cx, t0["t0"]], [yb + 0.75, yb], color=MUTED, lw=0.6)
            thumb(cx, yb + 0.8, size, img)
            name = t0["name"] + (" S1" if level == "date" and t0["mod"] != "S2" else "")
            ax.text(cx, yb + 0.8 + size + 0.06, name, ha="center", va="bottom",
                    fontsize=7 if len(gs) <= 12 else 6, color=MUTED)
            save_png(img, mat / f"{level}_{name.replace(' ', '')}_block.png")
    n_s = len(seasons)
    heads = {"season": f"Season tokens · patch size 16 · {n_s} tokens, each predicts the whole "
                       f"16 × 16 px block for 3 months",
             "month": f"Month tokens · patch size 4 · {counts['month'][0]} crops in "
                      f"{counts['month'][1]} months · each fills one ps4 cell (4 × 4 px) of the "
                      f"block; grey = not cropped",
             "date": f"Date tokens · patch size 4 · {counts['date'][0]} crops on "
                     f"{counts['date'][1]} dates · each fills one ps4 cell; grey = not cropped"}
    tops = {"season": LANE["season"] + 2.35, "month": LANE["month"] + 2.05, "date": LANE["date"] + 2.05}
    for lv in LANE:
        ax.text(0, tops[lv], heads[lv], fontsize=8.3, color=INK, weight="bold", va="bottom")
    ax.text(0, -0.85, "every panel is the same 16 × 16 px block; colours: the class each token's "
            "own logits give per pixel", fontsize=7.5, color=MUTED, va="center")

    # ---- right: the pool over the block, and where each cell's answer comes from
    x0, wd, gap = 0.762, 0.07, 0.008
    fig.text(x0, 0.875, "Temporal pool, per ps4 cell", fontsize=8.3, weight="bold", color=INK)
    blk_t = target[bl]
    okb = blk_t >= 0
    cols = [("season tokens\nonly", lr[bl]), ("all tokens", hr[bl]),
            ("label", np.where(okb, blk_t, 19))]
    for j, (name, m) in enumerate(cols):
        a2 = square(x0 + j * (wd + gap), 0.60, wd)
        a2.imshow(PASTIS_COLORS[m], extent=(0, 16, 16, 0), interpolation="nearest")
        for v in range(4, 16, 4):
            a2.axhline(v, color="white", lw=0.5), a2.axvline(v, color="white", lw=0.5)
        frame(a2, INK, 1.0)
        a2.set_title(name, fontsize=8, color=INK, pad=3)
        if j < 2:
            a2.set_xlabel(f"{(m == blk_t)[okb].mean():.0%} px correct", fontsize=7.5, color=INK,
                          labelpad=3)
        save_png(PASTIS_COLORS[m], mat / f"block_{['season_only', 'all_tokens', 'label'][j]}.png")
    crops, ws = st["crops"][cs], st["w_season"][cs]
    maps = [("crops covering\nthe cell", crops, "Oranges", 0, max(crops.max(), 1), "{:.0f}"),
            ("weight on\nseason tokens", ws, "Blues", 0, 1, "{:.2f}"),
            ("weight on\nps4 tokens", 1 - ws, "Oranges", 0, 1, "{:.2f}")]
    for j, (name, m, cmap, lo, hi, fmt) in enumerate(maps):
        a3 = square(x0 + j * (wd + gap), 0.27, wd)
        a3.imshow(m, cmap=cmap, vmin=lo, vmax=hi * 1.25, extent=(0, 4, 4, 0),
                  interpolation="nearest")
        for v in range(1, 4):
            a3.axhline(v, color="white", lw=0.6), a3.axvline(v, color="white", lw=0.6)
        for i in range(4):
            for jj in range(4):
                a3.text(jj + 0.5, i + 0.5, fmt.format(m[i, jj]), ha="center", va="center",
                        fontsize=6.3, color=INK if m[i, jj] < 0.6 * hi * 1.25 else "white")
        frame(a3, INK, 1.0)
        a3.set_title(name, fontsize=8, color=INK, pad=3)
    fig.text(x0, 0.815, "the block's prediction, 16 × 16 px", fontsize=7.5, color=MUTED)
    fig.text(x0, 0.515, "per ps4 cell (4 × 4 cells)", fontsize=7.5, color=MUTED)
    seen = set(np.unique(blk_t[okb]).tolist()) | set(np.unique(lr[bl]).tolist()) \
        | set(np.unique(hr[bl]).tolist())
    for ts in toks.values():
        seen |= {int(t["logits"].mean(1).argmax()) for t in ts}
    classes = sorted(seen)
    fig.legend(handles=[Patch(fc=PASTIS_COLORS[k], ec="0.5", lw=0.4, label=CLASS_NAMES[k])
                        for k in classes], loc="upper left", frameon=False, fontsize=7.5,
               ncol=2 if len(classes) > 5 else 1, handlelength=1.0, labelspacing=0.35,
               columnspacing=0.8, bbox_to_anchor=(x0 - 0.004, 0.21))
    for ext in ("png", "pdf"):
        fig.savefig(path.parent / f"{path.name}.{ext}", dpi=220 if ext == "png" else 300)
    plt.close(fig)
    return {"tile": o["tile"], "block": [by, bx], "file": path.name + ".png", **(facts or {})}


def summary(data: dict) -> dict:
    """Over all tiles: cells without crops vs with, their accuracy and season weight."""
    keys = ("crops", "w_season", "labelled", "hit_season", "hit_all")
    s = {k: np.concatenate([cell_stats(o)[k].ravel() for o in data.values()]) for k in keys}
    out = {"tiles": list(data), "cells": int(len(s["crops"]))}
    for name, m in (("without_crops", s["crops"] == 0), ("with_crops", s["crops"] > 0)):
        lab = max(s["labelled"][m].sum(), 1)
        out[name] = {"cells": int(m.sum()),
                     "mean_weight_on_season_tokens": float(s["w_season"][m].mean()) if m.any() else None,
                     "px_acc_season_only": float(s["hit_season"][m].sum() / lab),
                     "px_acc_all_tokens": float(s["hit_all"][m].sum() / lab)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiles", nargs="+", default=["40091@r0c0"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--block", default=None, help="by,bx on the 4 x 4 ps16 grid (first tile)")
    a = ap.parse_args()
    data = {t: torch.load(OUT / f"collect_{t.replace('@', '_')}{a.suffix}.pt", weights_only=False)
            for t in a.tiles}
    dest = OUT / "readout_block"
    dest.mkdir(parents=True, exist_ok=True)
    sm = summary(data)
    (dest / "summary.json").write_text(json.dumps(sm, indent=1))
    print(json.dumps(sm, indent=1))
    if a.block:
        chosen = [(0.0, a.tiles[0], tuple(map(int, a.block.split(","))), {})]
    else:
        chosen = sorted(((s, t, b, f) for t, o in data.items() for s, b, f in rank_blocks(o)),
                        key=lambda v: -v[0])[:a.n]
    index = []
    for rank, (s, t, (by, bx), f) in enumerate(chosen, 1):
        tn = t.replace("@", "_")
        name = f"block_{tn}_b{by}{bx}" if a.block else f"cand{rank:02d}_{tn}_b{by}{bx}"
        index.append(figure(data[t], by, bx, dest / name, {"score": round(s, 3), **f}))
        print(name, f)
    if not a.block:
        (dest / "candidates.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
