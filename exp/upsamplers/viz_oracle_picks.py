"""Where does the label-error oracle put its ps8 crops, vs random locations at the same budget?

For each dataset, one model (the zero-shot LR checkpoint of the oracle round, or any
--readout native_mean | native_set run via --ckpt), three passes over the SAME weights:
    LR      ps16 seasonal tokens only (k = 0)
    oracle  the n locations whose 8 px window has the highest LR loss vs the TRUE label
    random  n random locations
each location a ps8 crop at every present month. Two test samples per dataset: the hardest
(highest mean LR loss) and the median of the first --scan samples with enough labels.

One PNG per sample, results/paper/oracle_picks/<task>_<tag>.png, 3 rows x 4 panels:
    LR      tokenization (RGB + ps16 grid) | label | LR prediction | LR per-pixel loss
    oracle  tokenization (+ ps8 windows)  | window score + picks | prediction | loss change
    random  tokenization (+ ps8 windows)  | LR loss + picks      | prediction | loss change
The loss is what the oracle ranks: cross-entropy (classification) or squared error of the
standardized target (regression); unlabelled pixels are white.

    python -u -m exp.upsamplers.viz_oracle_picks --tasks pastis,biomassters,yieldsat_raw
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp.common.tasks import AGBM_MEAN, AGBM_STD, TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.viz_apt_picks import luma, luma_stretch, slot_label  # noqa: E402

OUT_DIR = Path("results/paper/oracle_picks")
CKPT = {"pastis": "inject_fft_pastis_tiny_none_native_lr0.001_ep50",
        "biomassters": "inject_fft_biomassters_tiny_none_native_lr0.001_ep10",
        "yieldsat_raw": "inject_fft_yieldsat_raw_tiny_none_native_set_lr0.001_ep50"}
P = 8                                   # crop side, px
UNLABELLED = "white"                    # tab20 has a light grey class colour


def build(task, ckpt: str, readout: str, k_frac: float, attn: str = "joint"):
    m = fi.InjectFT(task, "tiny", 0, "oracle", k_frac, "location", readout=readout,
                    crop_px=P, pos="center", attn=attn).cuda()
    sd = torch.load(f"checkpoints/{ckpt}.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")}
    if readout == "native_mean":
        sd = {k: v for k, v in sd.items() if not k.startswith("readout.score.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing {missing[:5]} unexpected {unexpected[:5]}")
    return m.eval()


def label_of(task, target):
    """(64, 64) float label (NaN: unlabelled) in display units, and the standardized one."""
    t = target[0]
    if task.kind == "classification":
        y = t.float()
        y[y < 0] = float("nan")
        return y, y
    y = t[0] if t.ndim == 3 else t                           # YieldSAT: (value, field id)
    if task.name == "biomassters":
        return y * AGBM_STD + AGBM_MEAN, y
    idx = task.index
    return y * idx["target_std"] + idx["target_mean"], y


def to_display(task, logits):
    if task.kind == "classification":
        return logits[0].argmax(0).float()
    z = logits[0, 0]
    if task.name == "biomassters":
        return z * AGBM_STD + AGBM_MEAN
    idx = task.index
    return z * idx["target_std"] + idx["target_mean"]


def pixel_loss(task, logits, y_std):
    """(64, 64) the oracle's per-pixel loss; NaN where unlabelled."""
    if task.kind == "classification":
        t = torch.nan_to_num(y_std, nan=-1).long()[None]
        px = F.cross_entropy(logits.float(), t, ignore_index=-1, reduction="none")[0]
    else:
        px = (logits[0, 0].float() - y_std) ** 2
    return px.masked_fill(~torch.isfinite(y_std), float("nan"))


@torch.no_grad()
def run(m, batch, mode: str, k_frac: float):
    ms, ss, season, present, pairs, extra, target = batch
    m.sel_name = "oracle" if mode != "random" else "random"
    m.k_frac = 0.0 if mode == "lr" else k_frac
    m.last_pick = None
    torch.manual_seed(0)
    logits, _, n = m(ms, ss, season, present, pairs, extra)
    logits = logits.float()
    if mode == "lr":
        return logits, None, None, 0
    mask, scores, _ = m.last_pick                                 # (1, T, H, W)
    cells = mask[0].any(0).nonzero().tolist()                     # [(y, x)] ps4 cells
    if m.loc_px is not None:                                      # --loc_stride anchors
        tl = [tuple(int(m.loc_px[v]) for v in c) for c in cells]
    else:
        tl = [tuple(int(min(max(v * fi.LOOKUP_PS + fi.LOOKUP_PS // 2 - P // 2, 0), fi.TILE - P))
                    for v in c) for c in cells]                   # window top-left (py, px)
    return logits, tl, scores[0, 0].float(), int(n)


def windows(ax, tl, color, lw=1.4):
    for py, px in tl:
        ax.add_patch(mpatches.Rectangle((px - 0.5, py - 0.5), P, P, linewidth=lw,
                                        edgecolor=color, facecolor="none"))


def grid16(ax):
    for v in range(16, 64, 16):
        ax.axhline(v - 0.5, color="white", lw=0.6, alpha=0.8)
        ax.axvline(v - 0.5, color="white", lw=0.6, alpha=0.8)


def window_mean(loss, tl):
    lz = torch.nan_to_num(loss, nan=0.0)
    return float(np.mean([lz[py:py + P, px:px + P].mean().item() for py, px in tl])) if tl else 0.0


def metric(task, logits, y_std):
    ok = torch.isfinite(y_std)
    if task.kind == "classification":
        acc = (logits[0].argmax(0) == torch.nan_to_num(y_std, nan=-1).long())[ok].float().mean()
        return f"acc {acc.item():.3f}"
    return f"RMSE(z) {((logits[0, 0] - y_std)[ok] ** 2).mean().sqrt().item():.3f}"


def render(task, batch, sample_i, res, k_frac, n_loc, tag, ckpt, readout, attn, stride=4):
    ms, ss, season, present, pairs, extra, target = batch
    months = present[0].nonzero(as_tuple=True)[0].tolist()
    t_show = months[len(months) // 2]
    rgb = luma(ms, task, t_show, luma_stretch(ms, task, months, sample_i, "test"))
    y_disp, y_std = label_of(task, target.cuda())
    lr_logits = res["lr"][0]
    lr_loss = pixel_loss(task, lr_logits, y_std)
    labelled = torch.isfinite(y_std).cpu().numpy()

    cls = task.kind == "classification"
    if cls:
        cmap = matplotlib.colormaps["tab20"].resampled(20)
        kw = dict(cmap=cmap, vmin=-0.5, vmax=19.5, interpolation="nearest")
    else:
        lo, hi = np.nanpercentile(y_disp.cpu().numpy(), [2, 98])
        kw = dict(cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
    lmax = float(np.nanpercentile(lr_loss.cpu().numpy(), 99)) or 1.0
    grey = np.where(labelled, np.nan, 1.0)

    def show_map(ax, arr, mask_unlab=True, **k):
        a = arr.detach().cpu().numpy().astype(float)
        if mask_unlab:
            a = np.where(labelled, a, np.nan)
        cm = plt.get_cmap(k.pop("cmap", "viridis")).copy()
        cm.set_bad(UNLABELLED)
        return ax.imshow(a, cmap=cm, **k)

    fig, axes = plt.subplots(3, 4, figsize=(15, 11.4))
    G = ss.timestamps.shape[1]
    T = len(months)

    ax = axes[0, 0]
    ax.imshow(rgb); grid16(ax)
    ax.set_title(f"LR tokens: {G} seasons x 16 ps16 cells\n= {G * 16} tokens/modality group "
                 f"(RGB {slot_label(ms.timestamps[0], t_show)})", fontsize=9)
    ax = axes[0, 1]
    im = show_map(ax, y_disp, **kw)
    ax.set_title("label" + ("" if cls else f" ({'t/ha'})"), fontsize=9)
    if not cls:
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 2]
    show_map(ax, to_display(task, lr_logits), mask_unlab=False, **kw)
    ax.set_title(f"LR prediction (no crops)\n{metric(task, lr_logits, y_std)}", fontsize=9)
    ax = axes[0, 3]
    im = show_map(ax, lr_loss, cmap="magma", vmin=0, vmax=lmax)
    grid16(ax)
    ax.set_title("LR per-pixel loss (what the oracle ranks)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for row, (mode, color) in ((1, ("oracle", "#ff2d2d")), (2, ("random", "#00e5ff"))):
        logits, tl, score, n = res[mode]
        loss = pixel_loss(task, logits, y_std)
        ax = axes[row, 0]
        ax.imshow(rgb); grid16(ax); windows(ax, tl, color)
        ax.set_title(f"{mode}: + {len(tl)} ps8 windows x {T} months\n= {n} HR crops "
                     f"(LR loss in windows {window_mean(lr_loss, tl):.3f})", fontsize=9)
        ax.set_ylabel(mode, fontsize=12)
        ax = axes[row, 1]
        if mode == "oracle":
            sc = score.cpu().numpy()
            cm = plt.get_cmap("magma").copy()
            cm.set_bad(UNLABELLED)                                # not a candidate window
            im = ax.imshow(np.where(np.isfinite(sc), sc, np.nan), cmap=cm,
                           interpolation="nearest", extent=(-0.5, 63.5, 63.5, -0.5))
            ax.set_title("oracle score per candidate window\n(mean LR loss over it, at its "
                         "top-left cell)" if stride != 4 else
                         "oracle score per ps4 cell\n(mean LR loss over its 8 px window)",
                         fontsize=9)
        else:
            im = show_map(ax, lr_loss, cmap="magma", vmin=0, vmax=lmax)
            ax.set_title("LR loss + random picks", fontsize=9)
        windows(ax, tl, color)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[row, 2]
        show_map(ax, to_display(task, logits), mask_unlab=False, **kw)
        windows(ax, tl, color, lw=1.0)
        ax.set_title(f"prediction with {mode} crops\n{metric(task, logits, y_std)}", fontsize=9)
        ax = axes[row, 3]
        d = loss - lr_loss
        dm = float(np.nanpercentile(np.abs(d.cpu().numpy()), 99)) or 1e-3
        im = show_map(ax, d, cmap="RdBu_r", vmin=-dm, vmax=dm)
        windows(ax, tl, "black", lw=1.0)
        inside = window_mean(loss, tl) - window_mean(lr_loss, tl)
        ax.set_title(f"loss change (red = worse)\nmean in windows {inside:+.3f}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{task.name} test sample {sample_i} ({tag}) -- {n_loc} locations "
                 f"(k_frac {k_frac:g}), ps16 seasons + ps8 crops at L0, {readout}, attn {attn}, "
                 f"candidate stride {stride} px\n"
                 f"weights: {ckpt}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    sfx = ("" if attn == "joint" else f"_{attn}") + ("" if stride == 4 else f"_st{stride}")
    out = OUT_DIR / f"{task.name}_n{n_loc}{sfx}_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", default="pastis,biomassters,yieldsat_raw")
    ap.add_argument("--n_loc", type=int, default=16)
    ap.add_argument("--scan", type=int, default=48, help="test samples scanned to choose 2")
    ap.add_argument("--min_labelled", type=float, default=0.3)
    ap.add_argument("--ckpt", default="", help="one run for every task (default: CKPT)")
    ap.add_argument("--readout", default="native_mean", choices=("native_mean", "native_set"))
    ap.add_argument("--attn", default="joint", choices=("joint", "oneway", "overlap"),
                    help="oneway: the lr tokens never read the crops (crops read lr)")
    ap.add_argument("--loc_stride", type=int, default=4, choices=(4, 6, 8))
    a = ap.parse_args()
    k_frac = a.n_loc / 256
    for name in a.tasks.split(","):
        task = TASKS[name]
        ckpt = a.ckpt or CKPT[name]
        m = build(task, ckpt, a.readout, k_frac, a.attn)
        if a.loc_stride != 4:
            m.set_loc_stride(a.loc_stride)
        dl = fi.loader(task, "test", 1, 2, False)
        cands = []                                  # (mean LR loss, i, batch)
        for i, batch in enumerate(dl):
            if i >= a.scan:
                break
            _, y_std = label_of(task, batch[-1].cuda())
            ok = torch.isfinite(y_std)
            if ok.float().mean() < a.min_labelled:
                continue
            lr = run(m, batch, "lr", k_frac)
            cands.append((float(torch.nan_to_num(pixel_loss(task, lr[0], y_std))[ok].mean()),
                          i, batch))
        cands.sort(key=lambda c: -c[0])
        print(f"{name}: {len(cands)} candidates", flush=True)
        for tag, (_, i, batch) in (("hard", cands[0]), ("median", cands[len(cands) // 2])):
            res = {mode: run(m, batch, mode, k_frac) for mode in ("lr", "oracle", "random")}
            render(task, batch, i, res, k_frac, a.n_loc, tag, ckpt, a.readout, a.attn, a.loc_stride)


if __name__ == "__main__":
    main()
