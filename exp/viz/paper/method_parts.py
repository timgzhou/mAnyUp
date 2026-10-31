"""Paper material, method figure: the swappable parts of the hand-drawn method figure, for one
or more PASTIS test tiles, from the final method (exp.viz.paper.method_scan RUNS).

    source env_setup/env_olmo.sh && python -m exp.viz.paper.method_parts --setup s8m4 \\
        --tiles 30201@r64c0

Writes figures/paper/method/<setup>/<tile>/ (transparent PNGs; stacks drawn diagonally, the
front frame at the bottom left, the rest behind it towards the top right; --order spring
(default) starts at Apr-Jun / Apr and wraps around the year, agri = Oct .. Sep; a fixed step
between frames, 1.5x in the 4-frame season stacks):
    stack_months.png          12 monthly S2 composites (Oct .. Sep)
    stack_seasons.png         4 seasonal S2 composites
    stack_lr_features.png     season tokens of the low-res (crop-free) pass, PCA
    stack_selection.png       per-season selection maps on the grid of crop windows
    stack_hr_tokens.png       the picked windows of each month's composite; unpicked area
                              transparent (the frames behind show through); _active: only the
                              months with picks
    stack_month_features.png  each month's final crop tokens alone (sparse, transparent)
    stack_mixed_features.png  final features: the season slices, then the sparse month slices,
                              one PCA basis; _active: months with picks only
    single/                   every slice as its own PNG; entropy, attention, label, predictions
    parts.json                what each file shows (picks, months, accuracies)
S2 images are MEDIAN composites of the raw acquisitions (nodata frames dropped), for display
only: the model reads the dataset's MEAN composites. With cross_hr the season tokens never read
the crops, so the final season slices ARE the low-res pass output.
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.viz.paper.method_scan import OUT, RUNS, build, run_batch  # noqa: E402
from manyup.data.temporal import All  # noqa: E402
from manyup.mixres.data import Collate, Paired  # noqa: E402

MONTHS = ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep")
SEASONS = ("Oct-Dec", "Jan-Mar", "Apr-Jun", "Jul-Sep")
SEL_BG, SEL_ON = (26, 26, 51), (242, 184, 58)          # the draft's navy / gold masks
PX = 512                                               # one slice, px
STEP = 60                                              # diagonal offset between frames, px
SEASON_STEP = 90                                       # ... in the 4-frame season stacks
# display order of the stacks (front first): "spring" starts at Apr-Jun, the season the method
# crops most, then wraps around the agricultural year; "agri" = Oct .. Sep as the data
ORDERS = {"spring": ([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], [2, 3, 0, 1]),
          "agri": (list(range(12)), [0, 1, 2, 3])}


# ---------------------------------------------------------------- images
def median_composites(raw, axis, fallback):
    """Raw S2 frames (Tr, C, H, W) with dates -> the per-pixel median of the frames inside each
    step of `axis` (start / end days), nodata (< -100) dropped; a step with no frame: fallback."""
    x = raw.data.float().numpy().copy()
    x[x < -100] = np.nan
    day = raw.time.start.numpy()
    out = []
    for k in range(len(axis.start)):
        j = (day >= int(axis.start[k])) & (day <= int(axis.end[k]))
        f = np.nanmedian(x[j], 0) if j.any() else np.full(x.shape[1:], np.nan)
        out.append(np.where(np.isnan(f), fallback[k].numpy(), f))
    return torch.from_numpy(np.stack(out)).float()


def s2_stretch(month_data):
    """One stretch per tile for months and seasons, set by the per-pixel median over the
    months (clouds do not move it): one range for the three bands, its 1st to 99.5th
    percentile, gamma 0.9 (clouds saturate to white)."""
    x = np.median(month_data[:, [2, 1, 0]].numpy(), 0)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5) * 1.15
    return lambda f: np.clip((f[[2, 1, 0]].numpy().transpose(1, 2, 0) - lo) / (hi - lo + 1e-9),
                             0, 1) ** 0.9


class PCA:
    def __init__(self, X):
        X = X.double()
        self.mean = X.mean(0)
        _, _, self.V = torch.pca_lowrank(X - self.mean, q=3, center=False, niter=6)
        P = ((X - self.mean) @ self.V).numpy()
        self.lo, self.hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)

    def __call__(self, f):
        P = ((f.double() - self.mean) @ self.V).numpy()
        return np.clip((P - self.lo) / (self.hi - self.lo), 0, 1)


def to_img(a, px=PX):
    """(h, w, 3|4) float in [0, 1] -> RGBA PIL image px x px (nearest)."""
    a = np.asarray(a, np.float32)
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.ones_like(a[..., :1])], -1)
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGBA")
    return im.resize((px, px), Image.NEAREST)


def stack(frames, path, step=STEP, border=10, line=(150, 150, 150, 255)):
    """Frames drawn back to front along the diagonal, frame 0 in front at the bottom left; each
    with a white border, a thin grey outline and a soft shadow; transparent background. A
    frame's transparent pixels stay transparent (the frames behind show through), and so does
    its shadow there."""
    n = len(frames)
    S = PX + 2 * border
    pad = 24
    W = S + step * (n - 1) + 2 * pad
    canvas = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    for i in reversed(range(n)):
        x0, y0 = pad + step * i, pad + step * (n - 1 - i)
        card = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(card)
        d.rectangle([0, 0, S - 1, S - 1], outline=(255, 255, 255, 255), width=border)
        card.alpha_composite(frames[i], (border, border))
        d.rectangle([0, 0, S - 1, S - 1], outline=line, width=2)
        sh = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        dark = Image.new("RGBA", (S, S), (0, 0, 0, 70))
        sh.paste(dark, (x0 + 6, y0 + 6), card.getchannel("A"))
        canvas = Image.alpha_composite(canvas, sh.filter(ImageFilter.GaussianBlur(6)))
        canvas.alpha_composite(card, (x0, y0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# ---------------------------------------------------------------- one tile
@torch.no_grad()
def tile(model, selector, cfg, task, ds, raw_ds, tid, out_dir, run, order="spring"):
    item = ds[ds.ds.index_of(tid)]
    months, seasons = item
    batch = Collate(task)([item])
    ctx, plan, out, ex = run_batch(model, selector, batch)
    cp = cfg.select.crop_px
    g = 64 // cp
    mm, sm = months.modalities["S2"], seasons.modalities["S2"]
    raw = raw_ds[raw_ds.index_of(tid)].modalities["S2"]
    s2m = median_composites(raw, mm.time, mm.data.float())
    s2s = median_composites(raw, sm.time, sm.data.float())
    rgb = s2_stretch(s2m)
    T, NS = s2m.shape[0], s2s.shape[0]
    season_of = ctx.season[0].cpu()
    valid = ctx.valid[0].cpu()
    py, px, t = (v.cpu() for v in (plan.py, plan.px, plan.t))
    lr_F = ctx.out0.lr_F[0].float().cpu()                           # (NS, h, w, D)
    hr = out.hr.float().cpu()                                        # (N, D) plan order
    pca = PCA(torch.cat([lr_F.reshape(-1, lr_F.shape[-1]), hr]))
    d = out_dir / tid.replace("@", "_")
    one = d / "single"

    def save(name, im):
        one.mkdir(parents=True, exist_ok=True)
        im.save(one / f"{name}.png")
        return im

    month_im = [save(f"month_{m:02d}_{MONTHS[m]}", to_img(rgb(s2m[m]))) for m in range(T)]
    season_im = [save(f"season_{s}_{SEASONS[s]}", to_img(rgb(s2s[s]))) for s in range(NS)]
    lr_im = [save(f"lr_feature_{s}_{SEASONS[s]}", to_img(pca(lr_F[s]))) for s in range(NS)]
    # selection: window (gy, gx) picked for season s (every present month of s cropped there)
    sel = np.zeros((NS, g, g), bool)
    for y_, x_, t_ in zip(py, px, t):
        sel[int(season_of[t_]), int(y_) // cp, int(x_) // cp] = True
    sel_im = []
    for s in range(NS):
        a = np.where(sel[s][..., None], np.array(SEL_ON) / 255, np.array(SEL_BG) / 255)
        sel_im.append(save(f"selection_{s}_{SEASONS[s]}", to_img(a)))
    # sparse HR windows per month, and their final tokens (unpicked: transparent)
    hr_im, mix_im, active = [], [], []
    for m in range(T):
        a = np.zeros((64, 64, 4), np.float32)
        f = np.zeros((g, g, 4), np.float32)
        img = rgb(s2m[m])
        k = (t == m).nonzero()[:, 0]
        for j in k.tolist():
            y_, x_ = int(py[j]), int(px[j])
            a[y_:y_ + cp, x_:x_ + cp, :3] = img[y_:y_ + cp, x_:x_ + cp]
            a[y_:y_ + cp, x_:x_ + cp, 3] = 1
            f[y_ // cp, x_ // cp, :3] = pca(hr[j][None])[0]
            f[y_ // cp, x_ // cp, 3] = 1
        hr_im.append(save(f"hr_tokens_{m:02d}_{MONTHS[m]}", to_img(a)))
        mix_im.append(save(f"month_feature_{m:02d}_{MONTHS[m]}", to_img(f)))
        if len(k):
            active.append(m)
    # extras
    tab = plt.get_cmap("tab20").colors
    pal = np.array([(0, 0, 0)] + [tab[i] for i in range(1, 19)] + [(1, 1, 1)])
    tgt = batch.target[0].long().numpy()
    save("label", to_img(pal[np.where(tgt < 0, 19, tgt)]))
    for name, lg in (("pred_lr_pass", ctx.out0.logits), ("pred_final", out.logits)):
        save(name, to_img(pal[lg[0].argmax(0).cpu().numpy()]))
    ent = ex["ent"][0].float().cpu().numpy()
    save("entropy", to_img(plt.get_cmap("viridis")(ent / max(ent.max(), 1e-6))))
    at_all = ex["attn"][0].cpu().numpy()
    for s in range(NS):
        save(f"attention_{s}_{SEASONS[s]}", to_img(plt.get_cmap("viridis")(
            (at_all[s] - at_all.min()) / (np.ptp(at_all) + 1e-9))))

    mo, so = ORDERS[order]
    M = lambda ims: [ims[m] for m in mo]                                # noqa: E731
    A = lambda ims: [ims[m] for m in mo if m in active]                 # noqa: E731
    S = lambda ims: [ims[s] for s in so]                                # noqa: E731
    stack(M(month_im), d / "stack_months.png")
    stack(S(season_im), d / "stack_seasons.png", SEASON_STEP)
    stack(S(lr_im), d / "stack_lr_features.png", SEASON_STEP)
    stack(S(sel_im), d / "stack_selection.png", SEASON_STEP)
    stack(M(hr_im), d / "stack_hr_tokens.png")
    stack(A(hr_im), d / "stack_hr_tokens_active.png")
    stack(M(mix_im), d / "stack_month_features.png")
    stack(A(mix_im), d / "stack_month_features_active.png")
    stack(S(lr_im) + M(mix_im), d / "stack_mixed_features.png")
    stack(S(lr_im) + A(mix_im), d / "stack_mixed_features_active.png")
    ok = tgt >= 0
    acc = lambda lg: float((lg[0].argmax(0).cpu().numpy()[ok] == tgt[ok]).mean())  # noqa: E731
    info = {"tile": tid, "run": run, "windows": int(sel.sum()), "crops": len(t),
            "windows_per_season": {SEASONS[s]: int(sel[s].sum()) for s in range(NS)},
            "months_with_picks": [MONTHS[m] for m in active],
            "months_present": [MONTHS[m] for m in range(T) if valid[m]],
            "pixel_acc_lr_pass": acc(ctx.out0.logits), "pixel_acc_final": acc(out.logits),
            "pca": "3 principal components of this tile's season tokens + crop tokens",
            "s2": "median composites of the raw acquisitions (display only)",
            "stack_order_front_first": {"months": [MONTHS[m] for m in mo],
                                        "seasons": [SEASONS[s] for s in so]}}
    (d / "parts.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--setup", default="s8m4", choices=tuple(RUNS))
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--order", default="spring", choices=tuple(ORDERS))
    a = ap.parse_args()
    torch.manual_seed(0)
    task = TASKS["pastis"]
    ds = Paired(task, "test")
    raw_ds = task.open("test", {"S2": All()})
    b0 = Collate(task)([ds[0]])
    model, selector, cfg = build(RUNS[a.setup], (b0.ms, b0.ss))
    for tid in a.tiles:
        tile(model, selector, cfg, task, ds, raw_ds, tid, OUT / a.setup, RUNS[a.setup], a.order)


if __name__ == "__main__":
    main()
