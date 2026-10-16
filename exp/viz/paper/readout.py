"""Paper material, readout: how one label cell is predicted from the tokens that cover it
(NativeSetReadout, --readout native_set), on tiles of exp.viz.paper.collect.

    python -m exp.viz.paper.readout [--tiles 40091@r0c0 ...] [--n 8] [--cell y,x]

Writes to figures/paper/model/readout/:
    cand<rank>_<tile>_y<y>x<x>.{png,pdf}   one figure per candidate cell, best first
    candidates.json                         the ranking and each cell's tokens
    <same name>_materials/                  the panels of that figure as PNGs

Each figure, for one ps4 cell (4 x 4 label px, outlined in orange everywhere):
    left     the tile: label, the prediction of the ps16 season tokens alone, and of all tokens
    centre   every token covering the cell on a time axis, each with the per-pixel class map
             its own logits give at its own patch size -- a ps16 season token predicts its
             whole 16 x 16 px block (the cell outlined inside it), a ps4 month / date token
             predicts exactly the cell's 4 x 4 px -- and its attention weight
    right    the temporal pool: the weighted sum of those logits, over the ps16 block and
             zoomed on the cell: season tokens only -> all tokens -> label

The readout, per ps4 cell p (no spatial neighbours: only tokens whose footprint covers p):
    logits_i   one linear head D -> 19 x 16 x 16 on every token. A ps16 season token's map is
               its 16 x 16 px block; p reads its own 4 x 4 px of it. A ps4 crop token's map is
               area-averaged 4x to its 4 x 4 px footprint, which is p.
    a_i        score(f_i) + log(#months of the season) for a season token,
               score(f_i) + log sigmoid(query score) for a crop
    out(p)     sum_i softmax_i(a_i) logits_i
"Season tokens only" is the same pass with the softmax restricted to the season tokens (what the
readout gives at a cell without crops); those tokens have still attended to the crops.

Candidate cells (ranked): on a class border (>= 2 label classes in the cell's ps16 block, more
for >= 2 in the cell itself), 1-4 month and 1-6 date crops, where the season tokens alone get
the cell wrong and all tokens together get it right.
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

from exp.viz.paper.model import (ACCENT, INK, MONTHS, MUTED, OUT, PASTIS_COLORS, RULE,
                                 save_png)
from manyup.data.datasets.pastis import CLASS_NAMES
from manyup.data.time import from_days

LANE = {"season": 6.35, "month": 3.3, "date": 0.25}       # lane baselines (y, in month units)
NS = 4


def cell_tokens(o, q: int) -> list[dict]:
    """Every token covering ps4 cell q, with its weight and its 4 x 4 px logits at q."""
    ms = o["images"]["month"]["S2"]
    day0, day1 = int(ms["start"][0]), int(ms["end"][-1])
    t = lambda day: 12 * (day - day0) / (day1 - day0 + 1)  # noqa: E731
    season_of = o["season_of_month"].numpy()
    out = []
    for s in range(NS):
        months = np.nonzero((season_of == s) & o["present"].numpy())[0]
        if not len(months):
            continue
        out.append(dict(level="season", ps=16, k=s, t0=float(months[0]), t1=float(months[-1] + 1),
                        name=f"season {s + 1}", w=float(o["wgt"][q, s]), logits=o["cand"][q, s]))
    mods = o["mods"]
    for m, c in enumerate(o["cid"][q].tolist()):
        if c < 0:
            continue
        k = NS + m
        if int(o["tok_level"][c]) == 0:
            mth = int(o["tok_step"][c])
            out.append(dict(level="month", ps=4, k=k, t0=float(mth), t1=float(mth + 1),
                            name=MONTHS[mth], w=float(o["wgt"][q, k]), logits=o["cand"][q, k],
                            gate=float(o["logw"][c].exp()), month=mth))
        else:
            mod = mods[int(o["tok_mod"][c])]
            d = int(o["tok_step"][c])
            day = int(o["images"]["raw"][mod]["start"][d])
            out.append(dict(level="date", ps=4, k=k, t0=t(day), t1=t(day),
                            name=f"{from_days(day):%d %b}", w=float(o["wgt"][q, k]),
                            logits=o["cand"][q, k], gate=float(o["logw"][c].exp()), mod=mod,
                            date=d, day=day))
    return out


def to_map(cells: torch.Tensor) -> torch.Tensor:
    """(256, C, 16) per-cell 4 x 4 px logits -> (C, 64, 64)"""
    return rearrange(cells, "(y x) c (i j) -> c (y i) (x j)", y=16, i=4)


def season_only(o) -> torch.Tensor:
    """(C, 64, 64): the readout with its softmax restricted to the season tokens."""
    w = o["wgt"][:, :NS]
    w = w / w.sum(1, keepdim=True).clamp(min=1e-12)
    return to_map(torch.einsum("qk,qkcp->qcp", w, o["cand"][:, :NS]))


def rank_cells(o) -> list[tuple[float, int, dict]]:
    """Candidate cells, best first -> (score, q, facts)."""
    lvl, ty, tx = o["tok_level"], o["tok_y"], o["tok_x"]
    n = torch.zeros(2, 16, 16)
    for level in (0, 1):
        s = lvl == level
        n[level].index_put_((ty[s], tx[s]), torch.ones(int(s.sum())), accumulate=True)
    target = o["target"]
    lr, hr = season_only(o).argmax(0), o["logits"].argmax(0)
    out = []
    for y in range(16):
        for x in range(16):
            if not (1 <= n[0, y, x] <= 4 and 1 <= n[1, y, x] <= 6):
                continue
            sl = (slice(4 * y, 4 * y + 4), slice(4 * x, 4 * x + 4))
            lab = target[sl]
            if (lab < 0).any():
                continue
            by, bx = y // 4, x // 4
            bl = (slice(16 * by, 16 * by + 16), slice(16 * bx, 16 * bx + 16))
            blk = target[bl]
            ok = blk >= 0
            n_cell, n_blk = len(lab.unique()), len(blk[ok].unique())
            if n_blk < 2:
                continue
            a_lr, a_hr = (lr[sl] == lab).float().mean(), (hr[sl] == lab).float().mean()
            b_lr = (lr[bl] == blk)[ok].float().mean()
            b_hr = (hr[bl] == blk)[ok].float().mean()
            score = (3 * (a_hr - a_lr) + 1.5 * a_hr + 1.0 * (n_cell >= 2) + 0.5 * (n_blk >= 3)
                     + 1.0 * (b_hr - b_lr))
            out.append((float(score), y * 16 + x, dict(
                cell_classes=n_cell, block_classes=n_blk, cell_acc_season=float(a_lr),
                cell_acc_all=float(a_hr), block_acc_season=float(b_lr), block_acc_all=float(b_hr),
                month_crops=int(n[0, y, x]), date_crops=int(n[1, y, x]))))
    return sorted(out, key=lambda v: -v[0])


def spread(xs, size, lo, hi):
    """Thumbnail centres near xs, at least `size` apart, inside [lo, hi]."""
    order = np.argsort(xs)
    pos = np.array(xs, float)[order]
    for _ in range(200):
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < size:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - size / 2, mid + size / 2
        pos = np.clip(pos, lo + size / 2, hi - size / 2)
    out = np.empty(len(pos))
    out[order] = pos
    return out


def figure(o, q: int, path: Path, facts: dict | None = None) -> dict:
    y, x = divmod(q, 16)
    by, bx = y // 4, x // 4
    cy, cx_ = y % 4, x % 4                              # the cell inside its ps16 block
    toks = cell_tokens(o, q)
    target = o["target"].numpy()
    lab_rgb = PASTIS_COLORS[np.where(target < 0, 19, target)]
    lr = season_only(o).argmax(0).numpy()
    hr = o["logits"].argmax(0).numpy()
    mat = path.parent / (path.name + "_materials")
    FW, FH = 16.0, 8.0
    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    fig.text(0.012, 0.962, "Readout: a label cell is predicted from every token that covers it",
             fontsize=13, weight="bold", color=INK)
    fig.text(0.012, 0.928, "each token predicts per pixel at its own patch size · the cell "
             "(orange) pools them over time in logit space · no spatial neighbours",
             fontsize=9, color=MUTED)

    def frame(ax, color=RULE, lw=0.5):
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color), sp.set_linewidth(lw)

    def square(x0, y0, w):                       # axes of w figure-widths, square in inches
        return fig.add_axes([x0, y0, w, w * FW / FH])

    # ---- left: the tile -- label, season tokens only, all tokens
    ok = target >= 0
    tiles = [("label", lab_rgb, ""),
             ("season tokens only (ps16)", PASTIS_COLORS[lr],
              f"{(lr[ok] == target[ok]).mean():.0%} px correct"),
             ("all tokens (ps16 + ps4)", PASTIS_COLORS[hr],
              f"{(hr[ok] == target[ok]).mean():.0%} px correct")]
    for j, (name, img, note) in enumerate(tiles):
        ax = square(0.012, 0.635 - j * 0.295, 0.118)
        ax.imshow(img, extent=(0, 64, 64, 0), interpolation="nearest")
        for v in range(16, 64, 16):
            ax.axhline(v, color="white", lw=0.5, alpha=0.8)
            ax.axvline(v, color="white", lw=0.5, alpha=0.8)
        ax.add_patch(Rectangle((16 * bx, 16 * by), 16, 16, fill=False, ec=INK, lw=1.4))
        ax.add_patch(Rectangle((4 * x, 4 * y), 4, 4, fill=False, ec=ACCENT, lw=1.4))
        frame(ax)
        ax.set_title(name + (f" · {note}" if note else " · 64 × 64 px tile"), fontsize=7.5,
                     color=INK, loc="left", pad=3)
        save_png(img, mat / f"tile_{['label', 'season_only', 'all_tokens'][j]}.png")

    # ---- centre: the tokens on a time axis
    ax = fig.add_axes([0.155, 0.05, 0.585, 0.85])
    ax.set_xlim(-0.2, 12.2), ax.set_ylim(-1.1, 9.0), ax.set_aspect("equal"), ax.axis("off")
    for m in range(12):
        ax.text(m + 0.5, -0.32, MONTHS[m], ha="center", va="center", fontsize=8.5, color=MUTED)
    for m in range(13):
        ax.plot([m, m], [-0.12, 0.0], color=RULE, lw=0.7)
        ax.plot([m, m], [0, 8.75], color=RULE, lw=0.4, ls=(0, (1, 3)), zorder=0)
    ax.plot([0, 12], [0, 0], color=INK, lw=0.8)

    def class_map(lg):                           # (C, n) logits -> (side, side, 3)
        side = int(round(lg.shape[1] ** 0.5))
        return PASTIS_COLORS[lg.argmax(0).view(side, side).numpy()]

    def thumb(cx, y0, size, img, edge=RULE, lw=0.8, mark=False, px_grid=False):
        ax.imshow(img, extent=(cx - size / 2, cx + size / 2, y0, y0 + size),
                  interpolation="nearest", zorder=3)
        n = img.shape[0]
        if px_grid:                               # the token's own pixels
            for v in range(1, n):
                u = size * v / n
                ax.plot([cx - size / 2 + u] * 2, [y0, y0 + size], color="white", lw=0.3,
                        alpha=0.6, zorder=4)
                ax.plot([cx - size / 2, cx + size / 2], [y0 + u] * 2, color="white", lw=0.3,
                        alpha=0.6, zorder=4)
        ax.add_patch(Rectangle((cx - size / 2, y0), size, size, fill=False, ec=edge, lw=lw,
                               zorder=5))
        if mark:                                  # the cell inside a ps16 token's map
            u = size / 4
            ax.add_patch(Rectangle((cx - size / 2 + cx_ * u, y0 + size - (cy + 1) * u), u, u,
                                   fill=False, ec=ACCENT, lw=1.5, zorder=6))

    wmax = max(t["w"] for t in toks)

    def weight(cx, y0, w, width):
        ax.add_patch(Rectangle((cx - width / 2, y0), width, 0.13, fc="#eeeeee", ec="none",
                               zorder=2))
        ax.add_patch(Rectangle((cx - width / 2, y0), width * min(w / wmax, 1), 0.13, fc=INK,
                               ec="none", zorder=3))
        ax.text(cx, y0 - 0.06, f"w = {w:.2f}" if w >= 0.005 else "w < 0.01", ha="center",
                va="top", fontsize=7, color=INK)

    info = []
    cells = [(by * 4 + i) * 16 + (bx * 4 + j) for i in range(4) for j in range(4)]
    for t in [t for t in toks if t["level"] == "season"]:
        # the token's whole 16 x 16 px map, rebuilt from its 16 cells
        lg = torch.stack([o["cand"][c, t["k"]] for c in cells])                # (16, C, 16)
        full = rearrange(lg, "(i j) c (u v) -> c (i u) (j v)", i=4, u=4)
        img = PASTIS_COLORS[full.argmax(0).numpy()]
        cx = (t["t0"] + t["t1"]) / 2
        yb = LANE["season"]
        ax.add_patch(Rectangle((t["t0"] + 0.06, yb), t["t1"] - t["t0"] - 0.12, 0.1,
                               fc="#8a8a8a", ec="none"))
        thumb(cx, yb + 0.6, 1.6, img, INK, 1.0, mark=True)
        weight(cx, yb + 0.4, t["w"], 1.6)
        save_png(img, mat / f"season_{t['k']}_classmap_16px.png")
        info.append({**{k: v for k, v in t.items() if k != "logits"},
                     "class": CLASS_NAMES[int(t["logits"].mean(1).argmax())]})
    for level, size in (("month", 0.9), ("date", 0.9)):
        ts = [t for t in toks if t["level"] == level]
        cxs = spread([(t["t0"] + t["t1"]) / 2 for t in ts], size + 0.16, 0, 12)
        yb = LANE[level]
        for t, cx in zip(ts, cxs):
            if level == "month":
                ax.add_patch(Rectangle((t["t0"] + 0.06, yb), 0.88, 0.1, fc="#8a8a8a", ec="none"))
                ax.plot([cx, t["t0"] + 0.5], [yb + 0.5, yb + 0.1], color=MUTED, lw=0.6)
            else:
                ax.plot([t["t0"]] * 2, [0, yb], color=INK, lw=1.2)
                ax.plot([cx, t["t0"]], [yb + 0.5, yb], color=MUTED, lw=0.6)
            img = class_map(t["logits"])
            thumb(cx, yb + 1.0, size, img, ACCENT, 1.5, px_grid=True)
            weight(cx, yb + 0.8, t["w"], size)
            ax.text(cx, yb + 1.0 + size + 0.07,
                    t["name"] + (f" · {t['mod']}" if level == "date" else ""),
                    ha="center", va="bottom", fontsize=7, color=MUTED)
            save_png(img, mat / f"{level}_{t['k']}_classmap_4px.png")
            info.append({**{k: v for k, v in t.items() if k != "logits"},
                         "class": CLASS_NAMES[int(t["logits"].mean(1).argmax())]})
    n = {lv: sum(t["level"] == lv for t in toks) for lv in LANE}
    w = {lv: sum(t["w"] for t in toks if t["level"] == lv) for lv in LANE}
    wt = {lv: f"{v:.2f}" if v >= 0.005 else "< 0.01" for lv, v in w.items()}
    heads = {"season": f"Season tokens · patch size 16 · each predicts its 16 × 16 px block for "
                       f"3 months (the cell outlined) · total weight {wt['season']}",
             "month": f"Month tokens · patch size 4 · {n['month']} at this cell, each predicts "
                      f"the cell's 4 × 4 px from one monthly composite · total weight "
                      f"{wt['month']}",
             "date": f"Date tokens · patch size 4 · {n['date']} at this cell, each predicts the "
                     f"cell's 4 × 4 px from one acquisition · total weight {wt['date']}"}
    tops = {"season": LANE["season"] + 2.35, "month": LANE["month"] + 2.3,
            "date": LANE["date"] + 2.3}
    for lv in LANE:
        ax.text(0, tops[lv], heads[lv], fontsize=8.3, color=INK, weight="bold", va="bottom")
    ax.text(0, -0.85, "thumbnail: the per-pixel class map of the token's own logits · bar: its "
            "attention weight w at this cell (softmax over the tokens shown)", fontsize=7.5,
            color=MUTED, va="center")

    # ---- right: the temporal pool, over the block and zoomed on the cell
    out = sum(t["w"] * t["logits"] for t in toks)                      # (C, 16)
    match = torch.allclose(out, torch.from_numpy(hr_logits_cell(o, y, x)), atol=2e-2)
    bl = (slice(16 * by, 16 * by + 16), slice(16 * bx, 16 * bx + 16))
    sl = (slice(4 * y, 4 * y + 4), slice(4 * x, 4 * x + 4))
    lab_c = target[sl]
    x0, wd, gap = 0.762, 0.07, 0.008
    fig.text(x0, 0.875, "Temporal pool: weighted sum of the tokens' logits", fontsize=8.3,
             weight="bold", color=INK)
    cols = [("season tokens\nonly", lr), ("all tokens", hr),
            ("label", np.where(target < 0, 19, target))]
    for j, (name, m) in enumerate(cols):
        a2 = square(x0 + j * (wd + gap), 0.60, wd)
        a2.imshow(PASTIS_COLORS[m[bl]], extent=(0, 16, 16, 0), interpolation="nearest")
        a2.add_patch(Rectangle((4 * cx_, 4 * cy), 4, 4, fill=False, ec=ACCENT, lw=1.5))
        frame(a2, INK, 1.0)
        a2.set_title(name, fontsize=8, color=INK, pad=3)
        save_png(PASTIS_COLORS[m[bl]],
                 mat / f"block_{['season_only', 'all_tokens', 'label'][j]}.png")
        a3 = square(x0 + j * (wd + gap), 0.36, wd)
        a3.imshow(PASTIS_COLORS[m[sl]], extent=(0, 4, 4, 0), interpolation="nearest")
        for v in range(1, 4):
            a3.axhline(v, color="white", lw=0.4, alpha=0.7)
            a3.axvline(v, color="white", lw=0.4, alpha=0.7)
        frame(a3, ACCENT, 1.8)
        if j < 2:
            a3.set_xlabel(f"{int((m[sl] == lab_c).sum())} / 16 px correct", fontsize=7.5,
                          color=INK, labelpad=3)
        save_png(PASTIS_COLORS[m[sl]],
                 mat / f"cell_{['season_only', 'all_tokens', 'label'][j]}.png")
    fig.text(x0, 0.815, "the ps16 block, 16 × 16 px", fontsize=7.5, color=MUTED)
    fig.text(x0, 0.535, "the cell, 4 × 4 px", fontsize=7.5, color=MUTED)
    blk_t = target[bl]
    classes = sorted(set(np.unique(blk_t[blk_t >= 0]).tolist()) | set(np.unique(lr[bl]).tolist())
                     | set(np.unique(hr[bl]).tolist())
                     | {int(t["logits"].mean(1).argmax()) for t in toks})
    fig.legend(handles=[Patch(fc=PASTIS_COLORS[k], ec="0.5", lw=0.4, label=CLASS_NAMES[k])
                        for k in classes], loc="upper left", frameon=False, fontsize=7.5,
               ncol=2 if len(classes) > 5 else 1, handlelength=1.0, labelspacing=0.35,
               columnspacing=0.8, bbox_to_anchor=(x0 - 0.004, 0.27))
    for ext in ("png", "pdf"):
        fig.savefig(path.parent / f"{path.name}.{ext}", dpi=220 if ext == "png" else 300)
    plt.close(fig)
    return {"tile": o["tile"], "cell": [y, x], "file": path.name + ".png",
            "weighted_sum_matches_model_output": bool(match), **(facts or {}),
            "weights": {k: round(v, 4) for k, v in w.items()}, "tokens": info}


def hr_logits_cell(o, y, x) -> np.ndarray:
    return o["logits"].numpy()[:, 4 * y:4 * y + 4, 4 * x:4 * x + 4].reshape(-1, 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiles", nargs="+", default=["40091@r0c0"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=8, help="candidate cells to draw, over all tiles")
    ap.add_argument("--per_block", type=int, default=1, help="at most this many per ps16 block")
    ap.add_argument("--cell", default=None, help="y,x on the 16 x 16 ps4 grid (first tile)")
    a = ap.parse_args()
    data = {t: torch.load(OUT / f"collect_{t.replace('@', '_')}{a.suffix}.pt", weights_only=False)
            for t in a.tiles}
    dest = OUT / "readout"
    dest.mkdir(parents=True, exist_ok=True)
    if a.cell:
        yy, xx = map(int, a.cell.split(","))
        chosen = [(0.0, a.tiles[0], yy * 16 + xx, {})]
    else:
        ranked = sorted(((s, t, q, f) for t, o in data.items() for s, q, f in rank_cells(o)),
                        key=lambda v: -v[0])
        chosen, used = [], {}
        for s, t, q, f in ranked:                 # spread over ps16 blocks
            blk = (t, q // 16 // 4, q % 16 // 4)
            if used.get(blk, 0) >= a.per_block:
                continue
            used[blk] = used.get(blk, 0) + 1
            chosen.append((s, t, q, f))
            if len(chosen) == a.n:
                break
    index = []
    for rank, (s, t, q, f) in enumerate(chosen, 1):
        yy, xx = divmod(q, 16)
        tn = t.replace("@", "_")
        name = f"cell_{tn}_y{yy}x{xx}" if a.cell else f"cand{rank:02d}_{tn}_y{yy}x{xx}"
        rec = figure(data[t], q, dest / name, {"score": round(s, 3), **f})
        index.append(rec)
        print(f"{name}: " + (f"cell {f['cell_acc_season']:.2f} -> {f['cell_acc_all']:.2f}, block "
              f"{f['block_acc_season']:.2f} -> {f['block_acc_all']:.2f}, classes cell/block "
              f"{f['cell_classes']}/{f['block_classes']}, " if f else "")
              + f"weights {rec['weights']}, sum matches model: "
              f"{rec['weighted_sum_matches_model_output']}")
    if not a.cell:
        (dest / "candidates.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
