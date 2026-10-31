"""WHEN, as viz_entropy_align asks WHERE: at each crop window, does the LR model's per-season
prediction entropy rank the seasons the way that season's own loss does?

Same LR-only models, maps and windows as exp.upsamplers.viz_entropy_align (each season token's own
pa2px head map; void pixels left out), one per setup:
    s8m4   ps8 seasons, ps4 windows   inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50
    s16m8  ps16 seasons, ps8 windows  inject_fft_pastis_tiny_none_native_set_lr0.001_ep50
Per labelled window with >= 3 used seasons, over its used seasons:
    r        Spearman r of the score with the loss (chance 0)
    top-1    the lowest-score season is the lowest-loss season (chance 1 / n seasons)
    half     the lowest-score half of the seasons vs the lowest-loss half: share matched (chance
             ~0.5; = viz_entropy_align's "when")
Each also with the tile's season effect split off, for prediction entropy:
    tile entropy   every window ranks the seasons by the tile-mean entropy (a per-tile season
                   choice; label-free)
    tile loss      ... by the tile-mean loss (the best per-tile season choice; uses labels)
    residual       entropy and loss each minus their tile mean per season: does entropy know
                   where a season is worse than it is on the rest of the tile?
Per sample: figures/entropy_align/time_<setup>_s<i>.png (rows: S2 RGB, window loss, window
entropy, each window's loss rank of the season, its entropy rank; last column: the label, the
lowest-loss and lowest-entropy season per window, r per window); over --agg_samples test
samples: figures/entropy_align/time_<setup>_summary.png (and printed).

    python -u -m exp.upsamplers.viz_entropy_time --setup s8m4
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import random  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.viz_entropy_align import KEYS, OUT, VOID, maps_for  # noqa: E402

SETUPS = {"s8m4": ("inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50", 8, 4),
          "s16m8": ("inject_fft_pastis_tiny_none_native_set_lr0.001_ep50", 16, 8)}
SCORES = [k for k in KEYS if k != "loss"]
MIN_SEASONS = 3
SEASON_COLOURS = ["#2a78d6", "#e8710a", "#2e9e44", "#c62828", "#7b1fa2", "#8d6e63", "#00838f",
                  "#f9a825"]


def load_model(task, setup):
    ckpt, lr_patch, cell = SETUPS[setup]
    m = fi.InjectFT(task, "tiny", 0, "oracle_bothalf", 0.5, "location", readout="native_set",
                    lr_patch=lr_patch, crop_px=cell, pos="center").cuda()
    sd = torch.load(f"checkpoints/{ckpt}.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing {missing[:5]} unexpected {unexpected[:5]}")
    m.requires_grad_(False)
    return m.eval()


def window_maps(mp, y, cell):
    """-> ok (B, h, w) windows with labels, {key: (B, NS, h, w)} per-season window means."""
    NS = mp["used"].shape[1]
    lab = (y >= 0).float()[:, None]
    nl = F.avg_pool2d(lab, cell)
    win = {"loss": (F.avg_pool2d(mp["loss"][:, :NS] * lab, cell) / nl.clamp(min=1e-6))}
    for k in SCORES:
        win[k] = F.avg_pool2d(mp[k][:, :NS], cell)
    return (nl > 0)[:, 0], win


def season_ranks(v, u):
    """0-based rank of each used season within its window (dim 1); unused seasons rank last."""
    return v.masked_fill(~u, float("inf")).argsort(1).argsort(1).float()


def when_stats(score, loss, u):
    """Per window, over its used seasons (u (B, NS, h, w)): Spearman r, top-1 hit, the share of
    the lowest-loss half matched. -> three (B, h, w) maps; nan where < MIN_SEASONS seasons."""
    n = u.sum(1).float()
    rs, rl = season_ranks(score, u), season_ranks(loss, u)
    m = ((n - 1) / 2)[:, None]
    cov = ((rs - m) * (rl - m) * u).sum(1)
    var = (n * (n * n - 1) / 12).clamp(min=1e-6)
    r = cov / var
    top1 = ((rs == 0) & (rl == 0)).any(1).float()
    half = (n // 2).clamp(min=1)[:, None]
    low_s, low_l = (rs < half) & u, (rl < half) & u
    hm = (low_s & low_l).sum(1).float() / half[:, 0]
    bad = n < MIN_SEASONS
    return [x.masked_fill(bad, float("nan")) for x in (r, top1, hm)]


def tile_mean(v, u):
    """(B, NS, h, w) -> each season's mean over the tile's labelled windows, broadcast back."""
    s = (v * u).sum((2, 3)) / u.sum((2, 3)).clamp(min=1)
    return s[:, :, None, None].expand_as(v)


def comparisons(mp, y, cell):
    """{name: (score, loss)} window maps ranked against each other; plus ok, u."""
    ok, win = window_maps(mp, y, cell)
    u = mp["used"][:, :, None, None] & ok[:, None]
    loss, ent = win["loss"], win["ent_pred"]
    cmp = {k: (win[k], loss) for k in SCORES}
    cmp["tile entropy"] = (tile_mean(ent, u), loss)
    cmp["tile loss"] = (tile_mean(loss, u), loss)
    cmp["residual"] = (ent - tile_mean(ent, u), loss - tile_mean(loss, u))
    return cmp, win, ok, u


def render(mp, win, ok, u, y, b, idx, ts, setup):
    cell = SETUPS[setup][2]
    used = [s for s in range(mp["used"].shape[1]) if bool(mp["used"][b, s])]
    r, top1, _ = when_stats(win["ent_pred"], win["loss"], u)
    rl = season_ranks(win["loss"], u)[b].cpu().numpy() + 1
    re = season_ranks(win["ent_pred"], u)[b].cpu().numpy() + 1
    okb = ok[b].cpu().numpy()
    rows = ["S2 RGB", "window LR loss", "window entropy", "loss rank (1 = lowest)",
            "entropy rank (1 = lowest)"]
    cols = len(used) + 1
    fig, axes = plt.subplots(len(rows), cols, figsize=(3.2 * cols + 1.2, 15.5))
    rgb = mp["rgb"][b].cpu().numpy()
    lo, hi = np.nanpercentile(rgb[used], [2, 98])
    rng_ = {}
    for k in ("loss", "ent_pred"):
        v = win[k][b, used].cpu().numpy()[:, okb]
        rng_[k] = (np.nanpercentile(v, 1), np.nanpercentile(v, 99))
    n_used = len(used)
    rank_cm = plt.get_cmap("RdYlGn_r", n_used)
    for j, s in enumerate(used):
        for i in range(len(rows)):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, j].imshow(np.clip((rgb[s] - lo) / (hi - lo), 0, 1) ** 0.8)
        axes[0, j].set_title(f"season {s} [{ts[s]}]", fontsize=10,
                             color=SEASON_COLOURS[s % len(SEASON_COLOURS)])
        for i, k, cm in ((1, "loss", "magma"), (2, "ent_pred", "viridis")):
            v = np.where(okb, win[k][b, s].cpu().numpy(), np.nan)
            c = plt.get_cmap(cm).copy(); c.set_bad(VOID)
            im = axes[i, j].imshow(v, cmap=c, vmin=rng_[k][0], vmax=rng_[k][1],
                                   interpolation="nearest")
            axes[i, j].set_title(f"tile mean {np.nanmean(v):.2f}", fontsize=9)
            if j == n_used - 1:
                fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
        for i, rk in ((3, rl), (4, re)):
            v = np.where(okb, rk[s], np.nan)
            c = rank_cm.copy(); c.set_bad(VOID)
            im = axes[i, j].imshow(v, cmap=c, vmin=0.5, vmax=n_used + 0.5, interpolation="nearest")
            if i == 4:
                same = (rl[s] == re[s])[okb].mean()
                axes[i, j].set_title(f"same rank as loss: {same:.0%}", fontsize=9)
            if j == n_used - 1:
                cb = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.03,
                                  ticks=range(1, n_used + 1))
                cb.ax.tick_params(labelsize=7)
    # last column: label, best season per window by loss / entropy, r per window
    ax = axes[:, -1]
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    yl = y[b].cpu().numpy().astype(float)
    yl[yl < 0] = np.nan
    pal = list(matplotlib.colormaps["tab20"].colors)
    pal[14], pal[15] = "#1f2a6b", "#e6c229"
    cls = matplotlib.colors.ListedColormap(pal); cls.set_bad(VOID)
    ax[0].imshow(yl, cmap=cls, vmin=-0.5, vmax=19.5, interpolation="nearest")
    ax[0].set_title("label (grey: void)", fontsize=9)
    scm = matplotlib.colors.ListedColormap(SEASON_COLOURS); scm.set_bad(VOID)
    best_l = np.where(okb, np.argmin(np.where(rl == 1, 0, 1), 0), np.nan)
    best_e = np.where(okb, np.argmin(np.where(re == 1, 0, 1), 0), np.nan)
    for a, v, t in ((ax[1], best_l, "lowest-loss season"),
                    (ax[2], best_e, "lowest-entropy season")):
        a.imshow(v, cmap=scm, vmin=-0.5, vmax=len(SEASON_COLOURS) - 0.5, interpolation="nearest")
        a.set_title(t + " (colour = season)", fontsize=9)
    t1 = top1[b].cpu().numpy()
    ax[2].set_title(f"lowest-entropy season\nsame as lowest-loss: {np.nanmean(t1):.0%} "
                    f"(chance {1 / n_used:.0%})", fontsize=9)
    rb = r[b].cpu().numpy()
    c = plt.get_cmap("RdBu").copy(); c.set_bad(VOID)
    im = ax[3].imshow(rb, cmap=c, vmin=-1, vmax=1, interpolation="nearest")
    ax[3].set_title(f"Spearman r over seasons per window\nmean {np.nanmean(rb):+.2f}", fontsize=9)
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    ax[4].hist(rb[~np.isnan(rb)], bins=np.linspace(-1, 1, 2 * n_used + 2), color="#2a78d6")
    ax[4].axvline(0, color="black", lw=0.6)
    ax[4].set_xticks([-1, 0, 1]); ax[4].tick_params(labelsize=7)
    ax[4].set_title("r per window", fontsize=9)
    for i, lab in enumerate(rows):
        axes[i, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle(f"PASTIS test sample {idx}: does entropy find WHEN (which season) at each "
                 f"ps{cell} window? ({setup}, LR-only ps{SETUPS[setup][1]} seasons)\nper-season "
                 f"head maps averaged per window; ranks within each window over its seasons",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"time_{setup}_s{idx}.png"
    fig.savefig(p, dpi=95)
    plt.close(fig)
    print(f"wrote {p}", flush=True)


LABELS = {"ent_pred": "entropy:\nprediction", "ent_in": "APT entropy:\ninput",
          "ent_emb": "APT entropy:\nembedded", "tile entropy": "tile-mean\nentropy",
          "tile loss": "tile-mean loss\n(labels)", "residual": "residual\n(entropy - tile)"}


def summary(agg, rhist, nseas, n, nwin, setup):
    names = list(LABELS)
    xs = np.arange(len(names))
    chance_top1 = float(np.mean(1 / np.asarray(nseas)))
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]))
    colours = ["#2a78d6"] * 3 + ["#8c8b86", "#e8710a", "#7b1fa2"]
    for ax, j, title, chance in ((axes[0], 0, "Spearman r over seasons", 0.0),
                                 (axes[1], 1, "top-1: lowest-score season =\nlowest-loss season",
                                  chance_top1),
                                 (axes[2], 2, "lowest-score half of seasons:\nshare of the "
                                  "lowest-loss half matched", 0.5)):
        v = [agg[k][j] for k in names]
        ax.bar(xs, v, color=colours, width=0.6)
        for x, y_ in zip(xs, v):
            ax.text(x, y_, f"{y_:.2f}", ha="center", va="bottom" if y_ >= 0 else "top", fontsize=8)
        ax.axhline(chance, color="black", ls="--", lw=1, label=f"chance {chance:.2f}")
        ax.set_xticks(xs); ax.set_xticklabels([LABELS[k] for k in names], fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        lo = min(min(v), chance)
        ax.set_ylim(min(0, lo - 0.05), 1.0)
    axes[0].set_ylabel("mean over windows")
    ax = axes[3]
    bins = np.linspace(-1.1, 1.1, 12)      # 4 seasons: r moves in steps of 0.2
    for k, c in (("ent_pred", "#2a78d6"), ("residual", "#7b1fa2"), ("tile loss", "#e8710a")):
        ax.hist(rhist[k], bins=bins, histtype="step", lw=1.6, color=c, density=True,
                label=LABELS[k].replace("\n", " "))
    ax.hist(rhist["null"], bins=bins, histtype="stepfilled", color="#d0d0d0", density=True,
            label="null (seasons shuffled)")
    ax.set_title("r per window", fontsize=10)
    ax.set_xlabel("Spearman r over the window's seasons")
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"PASTIS {setup}: WHEN -- per window, ranking its seasons by score vs by that "
                 f"season's LR loss ({n} test samples, {nwin} windows with >= {MIN_SEASONS} seasons, "
                 f"{np.mean(nseas):.1f} seasons on average)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = OUT / f"time_{setup}_summary.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"wrote {p}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--setup", choices=list(SETUPS), default="s8m4")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--agg_samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cell = SETUPS[a.setup][2]
    task = TASKS["pastis"]
    m = load_model(task, a.setup)
    dl = fi.loader(task, "test", 32, 8, False)
    vals = {k: [[], [], []] for k in LABELS}
    rhist = {k: [] for k in ("ent_pred", "residual", "tile loss", "null")}
    nseas = []
    n = 0
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    for *x, t in dl:
        if n >= a.agg_samples:
            break
        y = t.cuda().long()
        mp = maps_for(m, task, x, y)
        keep = ((y >= 0).float().mean((1, 2)) >= 0.3)[:, None, None]
        cmp, win, ok, u = comparisons(mp, y, cell)
        for k, (s, l_) in cmp.items():
            st = when_stats(s, l_, u)
            for j in range(3):
                v = st[j][keep.expand_as(st[j])]
                vals[k][j] += v[~v.isnan()].tolist()
            if k in rhist:
                v = st[0][keep.expand_as(st[0])]
                rhist[k] += v[~v.isnan()].tolist()
        # null: the entropy shuffled over each window's seasons
        noise = torch.rand(win["loss"].shape, device="cuda", generator=g)
        st = when_stats(noise, win["loss"], u)[0]
        v = st[keep.expand_as(st)]
        rhist["null"] += v[~v.isnan()].tolist()
        ns = u.sum(1)[keep.expand_as(ok) & ok]
        nseas += ns[ns >= MIN_SEASONS].tolist()
        n += len(y)
    agg = {k: tuple(float(np.mean(v[j])) for j in range(3)) for k, v in vals.items()}
    print(f"WHEN, {a.setup}, {n} test samples (>= 30% labelled), {len(nseas)} ps{cell} windows "
          f"with >= {MIN_SEASONS} seasons ({np.mean(nseas):.2f} on average; top-1 chance "
          f"{np.mean(1 / np.asarray(nseas)):.3f}):")
    for k in LABELS:
        print(f"  {LABELS[k].replace(chr(10), ' '):28s} r {agg[k][0]:+.3f} | top-1 {agg[k][1]:.3f}"
              f" | half {agg[k][2]:.3f}")
    print(f"  {'null (shuffled)':28s} r {np.mean(rhist['null']):+.3f}")
    summary(agg, rhist, nseas, n, len(nseas), a.setup)
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
        _, win, ok, u = comparisons(mp, y, cell)
        tsm = x[1].timestamps[0]
        ts = [f"{int(tsm[s, 2]) % 100:02d}-{int(tsm[s, 1]) + 1:02d}" for s in range(tsm.shape[0])]
        render(mp, win, ok, u, y, 0, i, ts, a.setup)
        done += 1


if __name__ == "__main__":
    main()
