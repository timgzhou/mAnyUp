"""How does the s8m4m2 oracle choose its crops, and how does the learned selector (arm A) differ?

Models (PASTIS, ps8 seasons + ps4 month crops, n1 = 128 windows' worth):
    oracle   better-half oracle + a second layer of ps2 month crops (--level2 m2, n2 = 256),
             picks from the TRUE label: layer 1 = the highest-loss ps4 windows, each its
             lower-loss half of seasons; layer 2 = the highest-loss ps2 cells under layer 1,
             each its lower-loss half of the cropped months
    learned  arm A: warm from the better-half ckpt, fine-tuned with the frozen linear rank head's
             picks (no labels); layer 1 only
    entropy  unc_ent: the same, picks from the entropy of the model's own per-season predictions
             (no labels, no training): the highest-entropy windows of the mean season
             prediction, each its lowest-entropy half of seasons

1. Per-sample figures (figures/select_compare/pastis_<tag>.png), the samples where the oracle
   gains most over the learned model and a median one:
       row 0   RGB | label | oracle prediction | learned prediction | oracle LR loss (its
               spatial score) | learned location score
       row 1   per season: the oracle model's own per-season loss map (its temporal score) +
               layer-1 windows (red) and layer-2 ps2 cells (yellow) cropped in that season
       row 2   per season: the learned season score + the learned picks (cyan), the oracle's
               layer-1 windows of that season outlined thin red for reference
2. Aggregate statistics over --stat_batches test batches (figures/select_compare/pastis_stats.png,
   printed too): for the (season, window) pairs each picks
       boundary share     windows holding >1 labelled class (vs all windows)
       background share   pixels of class 0 in the picked windows (vs all)
       loss percentile    the window's crop-free label-loss percentile within its sample
       brightness rank    the picked season's S2 visible brightness rank among the window's used
                          seasons (0 = darkest, 1 = brightest; clouds are bright)
       season share       picks per season index (each sample's seasons in time order)
       layer 2            boundary share of the oracle's ps2 cells vs all ps2 cells under layer 1

    python -u -m exp.upsamplers.viz_select_compare
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

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.learned_select import build, collect, run_name  # noqa: E402
from exp.upsamplers.pick_matrix import load  # noqa: E402
from exp.upsamplers.viz_apt_picks import _rgb_bands, luma, luma_stretch, slot_label  # noqa: E402

OUT = Path("figures/select_compare")
TASK, SETUP, N1 = "pastis", "s8m4", 128
O_CK = ("inject_fft_pastis_tiny_oracle_bothalf_L0_k0.5_loc_native_set_poscenter_l2m2n256_warm"
        "_lrps8_lr0.001_ep50")
P1, P2 = 4, 2                                     # layer-1 / layer-2 crop px
RED, YEL, CYAN, GREEN = "#ff2d2d", "#ffd400", "#00e5ff", "#39d353"


def load_models():
    o = build(TASK, SETUP, N1)                     # the better-half n128 model, then the l2m2 one
    o.set_level2("m2", 256)
    sd = torch.load(f"checkpoints/{O_CK}.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")}
    missing, unexpected = o.load_state_dict(sd, strict=False)
    if unexpected or any(not k.startswith("rank_head.") for k in missing):
        raise SystemExit(f"{O_CK}: missing {missing[:5]} unexpected {unexpected[:5]}")
    o.rank_head = None
    o.requires_grad_(False)
    o.eval()
    orig = o._level2_picks

    def capture(*args, **kw):                      # keep the layer-2 mask of the last forward
        out = orig(*args, **kw)
        o._l2_mask = out[0]
        return out
    o._level2_picks = capture
    a, a_name = load(TASK, SETUP, N1, "linear")
    a.score_src = "learned"
    e, e_name = load(TASK, SETUP, N1, "unc_ent")   # entropy picks (its own head, no training)
    e.score_src = "learned"
    return o, a, a_name, e, e_name


@torch.no_grad()
def passes(o, a, e, x):
    """-> dict of tensors for one batch: predictions, picks and the three models' scores."""
    o.l2 = None                                    # crop-free collection: no layer 2
    rbo = collect(o, x)
    o.l2 = "m2"
    a.score_src = "label"
    rba = collect(a, x)
    a.score_src = "learned"
    l_sc, l_loc = a.rank_scores(rba["lr_F"], rba["season"], rba["present"])
    o.last_pick, o._l2_mask = None, None
    torch.manual_seed(0)
    lo = o(*x)[0].float()
    l1o = o.last_pick[0]
    l2o = o._l2_mask
    a.last_pick = None
    la = a(*x)[0].float()
    l1a = a.last_pick[0]
    rbe = collect(e, x)
    e_sc, e_loc = e.rank_scores(rbe["lr_F"], rbe["season"], rbe["present"])
    NS = rbe["lr_F"].shape[1]
    used = (F.one_hot(rbe["season"].clamp(min=0), NS).bool()
            & (rbe["present"] & (rbe["season"] >= 0))[..., None]).any(1)
    e_own, e_tot = e.rank_head.uncertainty(e.rank_head.maps(rbe["lr_F"], e.lr_ps)[0], None, used)
    e.last_pick = None
    le = e(*x)[0].float()
    l1e = e.last_pick[0]
    maps = fi.UncHead(o.head.linear, "unc_ent").maps(rbo["lr_F"], o.lr_ps)[0]   # (B, NS, C, 64, 64)
    return dict(lo=lo, la=la, l1o=l1o, l2o=l2o, l1a=l1a, o_sc=rbo["sc"], o_loc=rbo["loc"],
                a_loc_label=rba["loc"], l_sc=l_sc, l_loc=l_loc, maps=maps,
                le=le, l1e=l1e, e_loc_label=rbe["loc"], e_sc=e_sc, e_own=e_own, e_tot=e_tot,
                season=rbo["season"], present=rbo["present"])


def per_season(mask, season, present, NS):
    """(B, T, H, W) month mask -> (B, NS, H, W): cropped in any present month of the season."""
    oh = (F.one_hot(season.clamp(min=0), NS).bool() & (present & (season >= 0))[..., None])
    return torch.einsum("bthw,bts->bshw", mask.float(), oh.float()) > 0


def acc(logits, y):
    ok = y >= 0
    return float((logits.argmax(1) == y)[ok].float().mean()) if ok.any() else float("nan")


def boxes(ax, m, size, color, lw):
    for yy, xx in m.nonzero().tolist():
        ax.add_patch(mpatches.Rectangle((xx * size - 0.5, yy * size - 0.5), size, size,
                                        linewidth=lw, edgecolor=color, facecolor="none"))


def render(task, x, y, r, i, tag, ms_rgb):
    ms = x[0]
    season, present = r["season"][i], r["present"][i]
    NS = r["maps"].shape[1]
    used = [s for s in range(NS) if bool(((season == s) & present).any())]
    months = present.nonzero(as_tuple=True)[0].tolist()
    yl = y[i].cpu().numpy().astype(float)
    yl[yl < 0] = np.nan
    cmap = matplotlib.colormaps["tab20"].resampled(20)
    kw = dict(cmap=cmap, vmin=-0.5, vmax=19.5, interpolation="nearest")
    ncol = max(8, len(used))
    fig, axes = plt.subplots(4, ncol, figsize=(3.1 * ncol, 13.4))
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    o1 = per_season(r["l1o"][i:i + 1], season[None], present[None], NS)[0]   # (NS, 16, 16)
    o2 = per_season(r["l2o"][i:i + 1], season[None], present[None], NS)[0]   # (NS, 32, 32)
    a1 = per_season(r["l1a"][i:i + 1], season[None], present[None], NS)[0]
    e1 = per_season(r["l1e"][i:i + 1], season[None], present[None], NS)[0]
    t_mid = months[len(months) // 2]
    ax = axes[0, 0]; ax.imshow(ms_rgb(t_mid)); ax.set_frame_on(True)
    ax.set_title(f"RGB {slot_label(ms.timestamps[i], t_mid)}", fontsize=9)
    ax = axes[0, 1]; ax.imshow(yl, **kw); ax.set_title("label", fontsize=9)
    for c, (k, name) in enumerate((("lo", "oracle s8m4m2"), ("la", "learned (arm A)"),
                                   ("le", "entropy (unc_ent)"))):
        ax = axes[0, 2 + c]
        ax.imshow(r[k][i].argmax(0).cpu().numpy(), **kw)
        ax.set_title(f"{name} prediction\nacc {acc(r[k][i:i + 1], y[i:i + 1]):.3f}", fontsize=9)
    ax = axes[0, 5]
    im = ax.imshow(r["o_loc"][i].float().cpu().numpy(), cmap="magma", interpolation="nearest",
                   extent=(-0.5, 63.5, 63.5, -0.5))
    boxes(ax, o1.any(0), P1, RED, 0.9)
    ax.set_title("oracle spatial score: crop-free\nlabel loss per ps4 window (+ its picks)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 6]
    im = ax.imshow(r["l_loc"][i].float().cpu().numpy(), cmap="viridis", interpolation="nearest",
                   extent=(-0.5, 63.5, 63.5, -0.5))
    boxes(ax, a1.any(0), P1, CYAN, 0.9)
    ax.set_title("learned spatial score\n(+ its picks)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 7]
    im = ax.imshow(r["e_tot"][i].float().cpu().numpy(), cmap="cividis", interpolation="nearest")
    boxes(ax, e1.any(0), P1, GREEN, 0.9)
    ax.set_title("entropy spatial score: entropy of\nthe mean season prediction (+ picks)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    yt = torch.nan_to_num(torch.tensor(yl), nan=-1).long().to(r["maps"].device)
    for c, s in enumerate(used):
        ms_of = [t for t in months if int(season[t]) == s]
        lab = ", ".join(slot_label(ms.timestamps[i], t)[2:] for t in ms_of)
        loss = F.cross_entropy(r["maps"][i, s][None].float(), yt[None], ignore_index=-1,
                               reduction="none")[0].cpu().numpy()
        loss = np.where(np.isnan(yl), np.nan, loss)
        ax = axes[1, c]; ax.set_frame_on(True)
        cm = plt.get_cmap("magma").copy(); cm.set_bad("white")
        ax.imshow(loss, cmap=cm, vmin=0, vmax=np.nanpercentile(loss, 99) or 1)
        boxes(ax, o1[s], P1, RED, 1.0)
        boxes(ax, o2[s], P2, YEL, 0.7)
        ax.set_title(f"oracle, season {s} [{lab}]\nown loss | {int(o1[s].sum())} windows, "
                     f"{int(o2[s].sum())} ps2", fontsize=8)
        ax = axes[2, c]; ax.set_frame_on(True)
        sc = r["l_sc"][i, s].float().cpu().numpy()
        ax.imshow(np.where(np.isfinite(sc), sc, np.nan), cmap="viridis", interpolation="nearest",
                  extent=(-0.5, 63.5, 63.5, -0.5))
        boxes(ax, o1[s], P1, RED, 0.5)
        boxes(ax, a1[s], P1, CYAN, 1.1)
        both = int((o1[s] & a1[s]).sum())
        ax.set_title(f"learned, season {s}: season score\n{int(a1[s].sum())} windows "
                     f"({both} = oracle's)", fontsize=8)
        ax = axes[3, c]; ax.set_frame_on(True)
        ax.imshow(r["e_own"][i, s].float().cpu().numpy(), cmap="cividis", interpolation="nearest")
        boxes(ax, o1[s], P1, RED, 0.5)
        boxes(ax, e1[s], P1, GREEN, 1.1)
        both = int((o1[s] & e1[s]).sum())
        ax.set_title(f"entropy, season {s}: own entropy\n{int(e1[s].sum())} windows "
                     f"({both} = oracle's)", fontsize=8)
    axes[1, 0].set_ylabel("oracle (label)", fontsize=11)
    axes[2, 0].set_ylabel("learned (arm A)", fontsize=11)
    axes[3, 0].set_ylabel("entropy (unc_ent)", fontsize=11)
    fig.suptitle(f"PASTIS test sample ({tag}): s8m4m2 oracle (red: layer-1 ps4 windows, yellow: "
                 f"layer-2 ps2 cells) vs learned selector (cyan, arm A) vs entropy selector (green)\n"
                 f"season maps: low = picked (the better half); the oracle reads the TRUE label, "
                 f"the other two do not", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"pastis_{tag}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"wrote {p}", flush=True)


def multi_class(y, size):
    """(B, 64, 64) labels (-1 unlabelled) -> (B, 64/size, 64/size) bool: >1 class in the cell."""
    oh = F.one_hot(y.clamp(min=0), 20).permute(0, 3, 1, 2).float() * (y >= 0)[:, None]
    return (F.max_pool2d(oh, size) > 0).sum(1) > 1


@torch.no_grad()
def stats(o, a, e, dl, batches, task):
    oe_name, rgb_idx = _rgb_bands(task)
    acc_ = {k: [] for k in ("o", "a", "e")}
    out = {k: {"bound": [], "bg": [], "pct": [], "bright": [], "season": None}
           for k in ("oracle", "learned", "entropy")}
    agree = {"learned": [], "entropy": []}
    base = {"bound": [], "bg": []}
    l2b = {"picked": [], "under_l1": []}
    for bi, (*x, target) in enumerate(dl):
        if bi >= batches:
            break
        y = target.cuda().long()
        r = passes(o, a, e, x)
        season, present = r["season"], r["present"]
        NS = r["maps"].shape[1]
        valid = present & (season >= 0)
        used = (F.one_hot(season.clamp(min=0), NS).bool() & valid[..., None]).any(1)   # (B, NS)
        mc = multi_class(y, P1)                                                         # (B,16,16)
        bg = F.avg_pool2d(((y == 0) & (y >= 0)).float()[:, None], P1)[:, 0]
        base["bound"].append(mc.float().mean().item())
        base["bg"].append(bg.mean().item())
        # season brightness per (season, window): mean visible reflectance over its months
        s2 = getattr(x[0], oe_name)[..., rgb_idx].float().cuda().mean(-1)               # (B,64,64,T)
        s2 = F.avg_pool2d(s2.permute(0, 3, 1, 2), P1)                                   # (B,T,16,16)
        oh = (F.one_hot(season.clamp(min=0), NS).float() * valid[..., None].float())    # (B,T,NS)
        br = torch.einsum("bthw,bts->bshw", s2, oh) / oh.sum(1)[:, :, None, None].clamp(min=1)
        br = br.masked_fill(~used[:, :, None, None], float("nan"))
        nb = used.sum(1).float()[:, None, None, None]
        brank = (br[:, :, None] > br[:, None]).sum(2).float() / (nb - 1).clamp(min=1)  # (B,NS,H,W)
        o_ps = per_season(r["l1o"], season, present, NS)
        for k, mask, loss in (("oracle", r["l1o"], r["o_loc"]), ("learned", r["l1a"], r["a_loc_label"]),
                              ("entropy", r["l1e"], r["e_loc_label"])):
            ps = per_season(mask, season, present, NS)                                  # (B,NS,16,16)
            win = ps.any(1)
            out[k]["bound"].append(mc[win].float().mean().item())
            out[k]["bg"].append(bg[win].mean().item())
            lf = loss.float().flatten(1)
            pct = (lf[:, :, None] > lf[:, None]).float().mean(2).view_as(loss)          # (B,16,16)
            out[k]["pct"].append(pct[win].mean().item())
            out[k]["bright"].append(brank[ps & used[:, :, None, None]].mean().item())
            cnt = ps.sum((2, 3)).float().sum(0).cpu()
            out[k]["season"] = cnt if out[k]["season"] is None else out[k]["season"] + cnt
            if k != "oracle":           # share of the oracle's (season, window) pairs also picked
                agree[k].append(float((ps & o_ps).sum() / o_ps.sum().clamp(min=1)))
        mc2 = multi_class(y, P2)                                                        # (B,32,32)
        l2 = r["l2o"].any(1)
        under = F.interpolate(r["l1o"].any(1)[:, None].float(), scale_factor=2)[:, 0] > 0
        l2b["picked"].append(mc2[l2].float().mean().item())
        l2b["under_l1"].append(mc2[under].float().mean().item())
        acc_["o"].append(acc(r["lo"], y)); acc_["a"].append(acc(r["la"], y))
        acc_["e"].append(acc(r["le"], y))
    m = lambda v: float(np.nanmean(v))  # noqa: E731
    res = {k: {s: m(v[s]) for s in ("bound", "bg", "pct", "bright")} for k, v in out.items()}
    res["all windows"] = {"bound": m(base["bound"]), "bg": m(base["bg"]), "pct": 0.5,
                          "bright": 0.5}
    seas = {k: (v["season"] / v["season"].sum()).numpy() for k, v in out.items()}
    print(f"pixel acc: oracle {m(acc_['o']):.3f}  learned {m(acc_['a']):.3f}  entropy "
          f"{m(acc_['e']):.3f}")
    print(f"share of the oracle's (season, window) picks also picked: learned "
          f"{m(agree['learned']):.3f}  entropy {m(agree['entropy']):.3f}")
    for k, v in res.items():
        print(f"{k:12s} boundary {v['bound']:.3f} | background {v['bg']:.3f} | loss pct "
              f"{v['pct']:.3f} | brightness rank {v['bright']:.3f}")
    print("season share  oracle", np.round(seas["oracle"], 3), " learned",
          np.round(seas["learned"], 3), " entropy", np.round(seas["entropy"], 3))
    print(f"layer 2: boundary share of picked ps2 cells {m(l2b['picked']):.3f} vs all ps2 cells "
          f"under layer 1 {m(l2b['under_l1']):.3f}")
    fig, axes = plt.subplots(1, 6, figsize=(24, 3.9))
    names = ["oracle", "learned", "entropy", "all windows"]
    cols = [RED, CYAN, GREEN, "#8c8b86"]
    for ax, (key, title) in zip(axes[:4], (("bound", "windows with >1 class"),
                                           ("bg", "background (class 0) pixels"),
                                           ("pct", "crop-free loss percentile\nwithin sample"),
                                           ("bright", "picked season's brightness rank\n"
                                                      "in its window (0 dark, 1 bright)"))):
        vals = [res[n][key] for n in names]
        ax.bar(names, vals, color=cols)
        for j, v in enumerate(vals):
            ax.text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
    ax = axes[4]
    w = 0.27
    idx = np.arange(len(seas["oracle"]))
    for j, (k, c) in enumerate((("oracle", RED), ("learned", CYAN), ("entropy", GREEN))):
        ax.bar(idx + (j - 1) * w, seas[k], w, color=c, label=k)
    ax.set_xticks(idx); ax.set_xlabel("season index (time order)")
    ax.set_title(f"share of (season, window) picks\nlayer 2 ps2 cells on boundaries: "
                 f"{m(l2b['picked']):.2f} vs {m(l2b['under_l1']):.2f} under layer 1", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[5]
    vals = [m(agree["learned"]), m(agree["entropy"])]
    ax.bar(["learned", "entropy"], vals, color=[CYAN, GREEN])
    for j, v in enumerate(vals):
        ax.text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="#8c8b86", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_title("share of the oracle's (season, window)\npicks also picked (dashed: chance)",
                 fontsize=10)
    fig.suptitle(f"PASTIS test, {batches} batches: what the s8m4m2 oracle picks vs the learned "
                 f"selector (arm A) and the entropy selector (unc_ent), layer 1 (pixel acc oracle "
                 f"{m(acc_['o']):.3f}, learned {m(acc_['a']):.3f}, entropy {m(acc_['e']):.3f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pastis_stats.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT / 'pastis_stats.png'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan", type=int, default=64, help="test samples scanned for the figures")
    ap.add_argument("--stat_batches", type=int, default=20)
    ap.add_argument("--samples", type=int, default=3)
    a_ = ap.parse_args()
    task = TASKS[TASK]
    o, a, a_name, e, e_name = load_models()
    print(f"oracle {O_CK}\nlearned {a_name}\nentropy {e_name}", flush=True)
    dl = fi.loader(task, "test", 1, 4, False)
    gaps = []
    for i, (*x, target) in enumerate(dl):
        if i >= a_.scan:
            break
        y = target.cuda().long()
        if (y >= 0).float().mean() < 0.3:
            continue
        r = passes(o, a, e, x)
        gaps.append((acc(r["lo"], y) - acc(r["la"], y), i))
    gaps.sort(reverse=True)
    pick = {f"gap{j}": gaps[j][1] for j in range(a_.samples - 1)}
    pick["median"] = gaps[len(gaps) // 2][1]
    for tag, idx in pick.items():
        for i, (*x, target) in enumerate(dl):
            if i == idx:
                break
        y = target.cuda().long()
        r = passes(o, a, e, x)
        months = r["present"][0].nonzero(as_tuple=True)[0].tolist()
        st = luma_stretch(x[0], task, months, idx, "test")
        render(task, x, y, r, 0, f"{tag}_s{idx}", lambda t: luma(x[0], task, t, st))
    stats(o, a, e, fi.loader(task, "test", 32, 8, False), a_.stat_batches, task)


if __name__ == "__main__":
    main()
