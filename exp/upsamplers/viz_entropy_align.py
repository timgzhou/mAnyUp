"""Is entropy a good label-free proxy for where the LR model's loss is highest -- and entropy of
what: the prediction, the input or the embedded input?

One LR-only model (PASTIS, ps8 seasonal tokens, no crops in training:
inject_fft_pastis_tiny_none_native_set_lrps8), no training here. Per test sample, one column per
season (that season token's own prediction, as the better-half oracle scores seasons) and a
"pooled" column (the model's output, as the oracle scores windows):
    row 1   season composite, S2 RGB               (pooled column: the label)
    row 2   LR prediction                          (season: its own pa2px head map)
    row 3   LR loss vs the label (cross-entropy)   <- the target every row is compared with
    row 4   entropy of the LR prediction           (softmax entropy per pixel)
    row 5   APT entropy of the seasonal input      (every band of the season composite)
    row 6   APT entropy of the embedded input      (OlmoEarth's patch embedding applied per pixel,
                                                    patch size 1, no position / time encodings)
    APT entropy (arXiv:2510.18091, as InjectFT._apt_scores, space term): per LR cell (8 x 8 px,
    non-overlapping), per channel: values min-max scaled within the cell, 32 bins, Shannon
    entropy -sum p log2 p of the cell's 64 values; mean over channels. Pooled column = mean over
    the used seasons. Void (unlabelled) pixels are grey and left out of every comparison.
Alignment, over the ps4 windows (the crop unit; each map averaged over the window, windows
without labels left out):
    where   Spearman r (average ranks for ties) of each row with row 3 of the same column
    when    within each window, the half of its used seasons with the lowest score vs the half
            with the lowest loss (the better-half oracle's temporal rule): share matched (chance
            ~0.5); summary only
Per sample: figures/entropy_align/pastis_s<i>.png; averaged over --agg_samples test samples:
figures/entropy_align/pastis_summary.png (and printed).

    python -u -m exp.upsamplers.viz_entropy_align
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.learned_select import collect  # noqa: E402
from exp.upsamplers.viz_apt_picks import _rgb_bands  # noqa: E402
from manyup.data.datasets.pastis import CLASS_NAMES  # noqa: E402
from manyup.encoders.olmoearth import _to_device  # noqa: E402

OUT = Path("figures/entropy_align")
CKPT = "inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50"
APT_CELL, APT_BINS, CELL = 8, fi.APT_BINS, 4
VOID = "#b4b4b4"
# rows 3-6: key -> (row label, colour-bar label)
KEYS = {"loss": ("LR loss", "cross-entropy (nats)"),
        "ent_pred": ("entropy: prediction", "entropy (nats)"),
        "ent_in": ("APT entropy: input", "bits, mean over bands"),
        "ent_emb": ("APT entropy: embedded input", "bits, mean over channels")}
ROWS = ["S2 RGB / label", "LR prediction"] + [v[0] for v in KEYS.values()]


def load_model(task):
    m = fi.InjectFT(task, "tiny", 0, "oracle_bothalf", 0.5, "location", readout="native_set",
                    lr_patch=8, crop_px=4, pos="center").cuda()
    sd = torch.load(f"checkpoints/{CKPT}.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{CKPT}: missing {missing[:5]} unexpected {unexpected[:5]}")
    m.requires_grad_(False)
    return m.eval()


def apt_entropy(x, cell=APT_CELL, bins=APT_BINS):
    """(N, C, 64, 64) -> (N, 64, 64): APT's per-cell histogram entropy (InjectFT._apt_scores,
    space term) -- per non-overlapping cell x cell block and channel, values min-max scaled within
    the block, `bins` bins, -sum p log2 p; mean over channels; repeated over the block's pixels."""
    N, C, Hp, Wp = x.shape
    h, w = Hp // cell, Wp // cell
    out = torch.zeros(N, h, w, device=x.device)
    for c0 in range(0, C, 32):                                  # channel chunks: memory
        v = x[:, c0:c0 + 32].float()
        blk = v.unfold(2, cell, cell).unfold(3, cell, cell).flatten(-2)   # (N, c, h, w, cell^2)
        lo, hi = blk.amin(-1, keepdim=True), blk.amax(-1, keepdim=True)
        b = ((blk - lo) / (hi - lo + 1e-6) * (bins - 1)).round().long().clamp(0, bins - 1)
        p = F.one_hot(b, bins).sum(-2).float()
        p = p / p.sum(-1, keepdim=True).clamp(min=1e-8)
        out += -(p * (p + 1e-8).log2()).sum(-1).sum(1)
    out = out / C
    return out.repeat_interleave(cell, 1).repeat_interleave(cell, 2)


@torch.no_grad()
def maps_for(m, task, x, y):
    """-> dict of (B, NS+1, 64, 64) maps (column NS = pooled), used (B, NS), rgb (B, NS, 64,64,3)."""
    ms, ss, season, present, pairs, extra = x
    rb = collect(m, x)
    lr_F, season, present = rb["lr_F"], rb["season"], rb["present"]
    B, NS = lr_F.shape[:2]
    used = (F.one_hot(season.clamp(min=0), NS).bool()
            & (present & (season >= 0))[..., None]).any(1)                         # (B, NS)
    w = used.float() / used.float().sum(1, keepdim=True).clamp(min=1)
    # predictions: each season token's own map, and the model's pooled output
    mu = fi.UncHead(m.head.linear, "unc_ent").maps(lr_F, m.lr_ps)[0]               # (B,NS,C,64,64)
    empty = torch.zeros(0, dtype=torch.long, device=lr_F.device)
    none = (lr_F.new_zeros(0, lr_F.shape[-1]), (empty,) * 4)
    pooled = m.readout(m.head.linear, lr_F, season, present, *none).float()         # (B,C,64,64)
    logits = torch.cat([mu, pooled[:, None]], 1)                                   # (B,NS+1,C,..)
    loss = F.cross_entropy(logits.flatten(0, 1), y.long().repeat_interleave(NS + 1, 0),
                           ignore_index=-1, reduction="none").view(B, NS + 1, 64, 64)
    lp = logits.log_softmax(2)
    ent_pred = -(lp.exp() * lp).sum(2)

    def with_pool(e):                     # (B, NS, 64, 64) -> + the mean over the used seasons
        return torch.cat([e, torch.einsum("bs,bshw->bhw", w, e)[:, None]], 1)
    # input: every band of each season composite
    dev_ss = _to_device(ss, lr_F.device, torch.float32)
    fields = [v for k, v in dev_ss.as_dict().items() if torch.is_tensor(v) and v.ndim == 5
              and "mask" not in k and v.is_floating_point()]
    inp = torch.cat(fields, -1).permute(0, 3, 4, 1, 2)                             # (B,G,C,64,64)
    ent_in = with_pool(apt_entropy(inp.flatten(0, 1)).view(B, NS, 64, 64))
    # embedded input: the patch embedding per pixel (patch size 1), no position / time encodings
    enc = m.mix.encoder
    ce = enc.composite_encodings.forward
    enc.composite_encodings.forward = lambda tokens, *a, **k: tokens
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = m.mix.embed(_to_device(ss, lr_F.device, torch.bfloat16), 1).pooled()
    finally:
        enc.composite_encodings.forward = ce
    emb = emb.float().permute(0, 1, 4, 2, 3)                                        # (B,NS,D,64,64)
    ent_emb = with_pool(apt_entropy(emb.flatten(0, 1)).view(B, NS, 64, 64))
    oe, idx = _rgb_bands(task)
    rgb = getattr(dev_ss, oe)[..., idx].permute(0, 3, 1, 2, 4).float()             # (B,G,64,64,3)
    return dict(pred=logits.argmax(2), loss=loss, ent_pred=ent_pred, ent_in=ent_in,
                ent_emb=ent_emb, used=used, rgb=rgb)


def ranks(v):
    """Average ranks (ties share the mean of their positions)."""
    srt, order = v.sort()
    _, inv, cnt = torch.unique_consecutive(srt, return_inverse=True, return_counts=True)
    end = cnt.cumsum(0).float()
    avg = end - (cnt.float() - 1) / 2                         # 1-based mean position per group
    r = torch.empty_like(v, dtype=torch.float)
    r[order] = avg[inv]
    return r


def spearman(a, b, ok):
    a, b = a[ok], b[ok]
    if a.numel() < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return torch.corrcoef(torch.stack([ranks(a), ranks(b)]))[0, 1].item()


def window_maps(mp, y):
    """-> (ok (B, 16, 16), {key: (B, NS+1, 16, 16) window means}); loss over labelled pixels."""
    lab = (y >= 0).float()[:, None]
    nl = F.avg_pool2d(lab, CELL)
    win = {"loss": F.avg_pool2d(mp["loss"] * lab, CELL) / nl.clamp(min=1e-6)}
    for k in KEYS:
        if k != "loss":
            win[k] = F.avg_pool2d(mp[k], CELL)
    return (nl > 0)[:, 0], win


def window_r(mp, y):
    """{key: (B, NS+1) Spearman r vs the loss over the labelled ps4 windows}."""
    ok, win = window_maps(mp, y)
    return {k: torch.tensor([[spearman(win[k][b, c].flatten(), win["loss"][b, c].flatten(),
                                       ok[b].flatten()) for c in range(win[k].shape[1])]
                             for b in range(win[k].shape[0])]) for k in KEYS}


def season_agree(mp, y):
    """{key: (hit, total)}: per labelled ps4 window, the lowest-score half of the used seasons
    vs the lowest-loss half."""
    NS = mp["used"].shape[1]
    ok, win = window_maps(mp, y)
    u = mp["used"][:, :, None, None] & ok[:, None]                                   # (B,NS,16,16)
    half = (mp["used"].sum(1)[:, None, None, None] // 2).clamp(min=1)

    def low_half(v):
        r = v[:, :NS].masked_fill(~u, float("inf")).argsort(1).argsort(1)
        return (r < half) & u
    ref = low_half(win["loss"])
    return {k: (float((ref & low_half(win[k])).sum()), float(ref.sum()))
            for k in KEYS if k != "loss"}


def render(mp, y, r, b, idx, ts):
    used = [s for s in range(mp["used"].shape[1]) if bool(mp["used"][b, s])]
    cols = used + ["pooled"]
    fig, axes = plt.subplots(len(ROWS), len(cols), figsize=(3.3 * len(cols) + 2.2, 17.2))
    pal = list(matplotlib.colormaps["tab20"].colors)
    pal[14], pal[15] = "#1f2a6b", "#e6c229"       # tab20's two greys -> grey means only void
    cls = matplotlib.colors.ListedColormap(pal)
    cls.set_bad(VOID)
    ckw = dict(cmap=cls, vmin=-0.5, vmax=19.5, interpolation="nearest")
    yl = y[b].cpu().numpy().astype(float)
    yl[yl < 0] = np.nan
    void = np.isnan(yl)
    rgb = mp["rgb"][b].cpu().numpy()
    lo, hi = np.nanpercentile(rgb[used], [2, 98])
    seen = set(np.unique(yl[~void]).astype(int))
    # one colour scale per row (shared by the seasons and the pooled column)
    rng_ = {}
    for k in KEYS:
        vals = mp[k][b].float().cpu().numpy()
        if k == "loss":
            vals = np.where(void[None], np.nan, vals)
        rng_[k] = (np.nanpercentile(vals, 1), np.nanpercentile(vals, 99))
    for j, c in enumerate(cols):
        ci = mp["used"].shape[1] if c == "pooled" else c
        name = "pooled (model output)" if c == "pooled" else f"season {c} [{ts[c]}]"
        for i in range(len(ROWS)):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                if c == "pooled":
                    ax.imshow(yl, **ckw)
                    ax.set_title(f"{name}\nlabel (grey: void, ignored)", fontsize=9)
                else:
                    ax.imshow(np.clip((rgb[c] - lo) / (hi - lo), 0, 1) ** 0.8)
                    ax.set_title(f"{name}\nS2 RGB", fontsize=9)
                continue
            if i == 1:
                pr = mp["pred"][b, ci].cpu().numpy()
                seen |= set(np.unique(pr).astype(int))
                ax.imshow(pr, **ckw)
                a = (pr == np.nan_to_num(yl, nan=-1))[~void].mean()
                ax.set_title(f"prediction, acc {a:.2f}", fontsize=9)
                continue
            key = list(KEYS)[i - 2]
            v = mp[key][b, ci].float().cpu().numpy()
            if key == "loss":
                v = np.where(void, np.nan, v)
            cm = plt.get_cmap("magma" if key == "loss" else "viridis").copy()
            cm.set_bad(VOID)
            im = ax.imshow(v, cmap=cm, vmin=rng_[key][0], vmax=rng_[key][1],
                           interpolation="nearest")
            if key == "loss":
                ax.set_title("LR loss (grey: void)", fontsize=9)
            else:
                ax.set_title(f"Spearman r vs loss {r[key][b, ci]:+.2f}", fontsize=9)
            if c == "pooled":
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cb.set_label(KEYS[key][1], fontsize=8)
                cb.ax.tick_params(labelsize=7)
    for i, lab in enumerate(ROWS):
        axes[i, 0].set_ylabel(lab, fontsize=10)
    handles = [mpatches.Patch(color=cls(k), label=CLASS_NAMES[k]) for k in sorted(seen)]
    handles.append(mpatches.Patch(color=VOID, label="void (ignored)"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.999, 0.94), fontsize=8,
               title="classes (rows 1-2)", title_fontsize=8, frameon=False)
    fig.suptitle(f"PASTIS test sample {idx}: does entropy find the LR model's loss? (LR-only ps8 "
                 f"seasons, {CKPT})\nSpearman r over ps4 windows of labelled pixels; one colour "
                 f"scale per row", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1 - 2.2 / (3.3 * len(cols) + 2.2), 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"pastis_s{idx}.png"
    fig.savefig(p, dpi=95)
    plt.close(fig)
    print(f"wrote {p}", flush=True)


def summary(agg, sa, n):
    keys = [k for k in KEYS if k != "loss"]
    labels = [KEYS[k][0] for k in keys]
    xs = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.bar(xs - 0.2, [agg[k][0] for k in keys], 0.4, label="per season (vs that season's loss)",
           color="#2a78d6")
    ax.bar(xs + 0.2, [agg[k][1] for k in keys], 0.4, label="pooled (vs the model output's loss)",
           color="#e8710a")
    for j, k in enumerate(keys):
        for d, v in ((-0.2, agg[k][0]), (0.2, agg[k][1])):
            ax.text(j + d, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(min(0, min(min(v) for v in agg.values()) - 0.05), 1.0)
    ax.set_ylabel("mean Spearman r vs the LR loss")
    ax.set_title("WHERE: ranking of ps4 windows", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[1]
    ax.bar(xs, [sa[k] for k in keys], color="#7b1fa2", width=0.5)
    for j, k in enumerate(keys):
        ax.text(j, sa[k], f"{sa[k]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#8c8b86", ls="--", lw=1, label="chance")
    ax.set_ylim(0.3, 0.85)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("share of the lowest-loss half matched")
    ax.set_title("WHEN: a window's lowest-score half of seasons\nvs its lowest-loss half", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"PASTIS, {n} test samples, LR-only ps8-season model: label-free scores vs the "
                 f"LR loss", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pastis_summary.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT / 'pastis_summary.png'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--agg_samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    task = TASKS["pastis"]
    m = load_model(task)
    dl = fi.loader(task, "test", 32, 8, False)
    seas = {k: [] for k in KEYS}
    pool = {k: [] for k in KEYS}
    agree = {k: [0.0, 0.0] for k in KEYS if k != "loss"}
    n = 0
    for *x, t in dl:
        if n >= a.agg_samples:
            break
        y = t.cuda().long()
        mp = maps_for(m, task, x, y)
        r = window_r(mp, y)
        NS = mp["used"].shape[1]
        keep = (y >= 0).float().mean((1, 2)).cpu() >= 0.3
        for k in KEYS:
            u = mp["used"].cpu() & keep[:, None]
            seas[k] += r[k][:, :NS][u].tolist()
            pool[k] += r[k][keep, NS].tolist()
        for k, (h, tot) in season_agree(mp, y).items():
            agree[k][0] += h
            agree[k][1] += tot
        n += len(y)
    agg = {k: (float(np.nanmean(seas[k])), float(np.nanmean(pool[k]))) for k in KEYS}
    sa = {k: h / max(tot, 1) for k, (h, tot) in agree.items()}
    print(f"vs LR loss over ps4 windows, {n} test samples (>= 30% labelled):")
    for k in KEYS:
        print(f"  {KEYS[k][0]:28s} Spearman r per season {agg[k][0]:+.3f} | pooled "
              f"{agg[k][1]:+.3f}" + ("" if k == "loss" else f" | season choice {sa[k]:.3f}"))
    summary(agg, sa, n)
    dl1 = fi.loader(task, "test", 1, 4, False)
    rng = random.Random(a.seed)
    want = sorted(rng.sample(range(len(dl1.dataset)), 4 * a.samples))
    done = 0
    for i, (*x, t) in enumerate(dl1):
        if done >= a.samples or i > want[-1]:
            break
        if i not in want:
            continue
        y = t.cuda().long()
        if (y >= 0).float().mean() < 0.3:
            continue
        mp = maps_for(m, task, x, y)
        r = window_r(mp, y)
        tsm = x[1].timestamps[0]                                # season composites' (d, m, y)
        ts = [f"{int(tsm[s, 2]) % 100:02d}-{int(tsm[s, 1]) + 1:02d}" for s in range(tsm.shape[0])]
        render(mp, y, r, 0, i, ts)
        done += 1


if __name__ == "__main__":
    main()
