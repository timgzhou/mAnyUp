"""Does a few noisy channels drag down the input's APT entropy? Channel by channel, on the same
LR-only model, cells and ps4 windows as exp.upsamplers.viz_entropy_align:
    APT entropy   per 8 px LR cell: the channel's 64 values min-max scaled within the cell, 32
                  bins, Shannon entropy (bits)
    cell std      the channel's plain standard deviation within the cell (contrast, which APT's
                  per-cell min-max discards) -- a reference
For each of the 14 input channels (S2 bands + S1 VV / VH of the season composite):
    where   mean Spearman r with the LR loss over labelled ps4 windows, per season (vs that
            season's own loss) and pooled (the channel's mean over used seasons vs the model
            output's loss)
    when    within each window, the channel's lowest-score half of seasons = the lowest-loss half
            (chance ~ 0.5)
Plus the all-channel mean (= the summary's input row) and a held-out subset: channels ranked by
pooled r on the first half of the samples, the top 3 averaged and scored on the second half.
-> figures/entropy_align/pastis_input_channels.png, printed table.

    python -u -m exp.upsamplers.entropy_channels
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from olmoearth_pretrain.data.constants import Modality  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.viz_entropy_align import (APT_CELL, CELL, apt_entropy, load_model,  # noqa: E402
                                              maps_for, spearman)
from manyup.encoders.olmoearth import _to_device  # noqa: E402

OUT = Path("figures/entropy_align")


def channels(ss, device):
    """(B, G, C, 64, 64) every band of the season composites + their names (as maps_for)."""
    dev_ss = _to_device(ss, device, torch.float32)
    xs, names = [], []
    for k, v in dev_ss.as_dict().items():
        if torch.is_tensor(v) and v.ndim == 5 and "mask" not in k and v.is_floating_point():
            xs.append(v)
            names += [f"{'S2' if 's2' in k or 'sentinel2' in k else 'S1'} {b}"
                      for b in Modality.get(k).band_order]
    return torch.cat(xs, -1).permute(0, 3, 4, 1, 2), names


def cell_std(x, cell=APT_CELL):
    """(N, 1, 64, 64) -> (N, 64, 64): the std within each non-overlapping cell, repeated."""
    blk = x[:, 0].unfold(1, cell, cell).unfold(2, cell, cell).flatten(-2)
    s = blk.float().std(-1)
    return s.repeat_interleave(cell, 1).repeat_interleave(cell, 2)


def scores(score, loss, used, ok):
    """score (B, NS, 64, 64), loss (B, NS+1, 64, 64) -> per-sample lists: per-season r, pooled r,
    and season-choice (hit, total)."""
    B, NS = score.shape[:2]
    lab = ok.float()[:, None]
    nl = F.avg_pool2d(lab, CELL)
    okw = (nl > 0)[:, 0]
    Lw = F.avg_pool2d(loss * lab, CELL) / nl.clamp(min=1e-6)                   # (B, NS+1, 16, 16)
    w = used.float() / used.float().sum(1, keepdim=True).clamp(min=1)
    Sw = F.avg_pool2d(score, CELL)                                              # (B, NS, 16, 16)
    Pw = torch.einsum("bs,bshw->bhw", w, Sw)
    per_s = [[spearman(Sw[b, s].flatten(), Lw[b, s].flatten(), okw[b].flatten())
              for s in range(NS) if used[b, s]] for b in range(B)]
    pooled = [spearman(Pw[b].flatten(), Lw[b, NS].flatten(), okw[b].flatten()) for b in range(B)]
    u = used[:, :, None, None] & okw[:, None]
    half = (used.sum(1)[:, None, None, None] // 2).clamp(min=1)

    def low_half(v):
        r = v.masked_fill(~u, float("inf")).argsort(1).argsort(1)
        return (r < half) & u
    ref = low_half(Lw[:, :NS])
    hit = (ref & low_half(Sw)).flatten(1).sum(1).float()
    tot = ref.flatten(1).sum(1).float()
    return per_s, pooled, list(zip(hit.tolist(), tot.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agg_samples", type=int, default=256)
    a = ap.parse_args()
    task = TASKS["pastis"]
    m = load_model(task)
    dl = fi.loader(task, "test", 32, 8, False)
    res = {}            # (measure, channel) -> {"season": [per-sample mean], "pooled": [], "when": []}
    names = None
    n = 0
    for *x, t in dl:
        if n >= a.agg_samples:
            break
        y = t.cuda().long()
        mp = maps_for(m, task, x, y)
        keep = ((y >= 0).float().mean((1, 2)) >= 0.3).cpu()
        inp, names = channels(x[1], y.device)                                   # (B,NS,C,64,64)
        B, NS, C = inp.shape[:3]
        flat = inp.flatten(0, 1)
        maps = {}
        for c in range(C):
            maps[("APT entropy", c)] = apt_entropy(flat[:, c:c + 1]).view(B, NS, 64, 64)
            maps[("cell std", c)] = cell_std(flat[:, c:c + 1]).view(B, NS, 64, 64)
        maps[("APT entropy", "all")] = mp["ent_in"][:, :NS]
        maps[("cell std", "all")] = torch.stack([maps[("cell std", c)] for c in range(C)]).mean(0)
        for key, sc in maps.items():
            per_s, pooled, when = scores(sc, mp["loss"], mp["used"], y >= 0)
            d = res.setdefault(key, {"season": [], "pooled": [], "when": []})
            for b in range(B):
                if keep[b]:
                    d["season"].append(float(np.nanmean(per_s[b])) if per_s[b] else np.nan)
                    d["pooled"].append(pooled[b])
                    d["when"].append(when[b])
        n += B
    half = len(next(iter(res.values()))["pooled"]) // 2
    stat = {}
    for key, d in res.items():
        h = np.array(d["when"])
        stat[key] = (np.nanmean(d["season"]), np.nanmean(d["pooled"]),
                     h[:, 0].sum() / max(h[:, 1].sum(), 1), np.nanstd(d["pooled"]))
    # held-out subset: rank channels by pooled r on the first half, average the top 3 on the second
    out_sub = {}
    for meas in ("APT entropy", "cell std"):
        rank = sorted(range(len(names)), key=lambda c: -np.nanmean(res[(meas, c)]["pooled"][:half]))
        top = rank[:3]
        print(f"{meas}: top 3 channels on the first half: {[names[c] for c in top]}")
        out_sub[meas] = (top, np.nanmean([np.nanmean(res[(meas, c)]["pooled"][half:]) for c in top]),
                         np.nanmean(res[(meas, "all")]["pooled"][half:]))
    print(f"\n{n} test samples (>= 30% labelled: {len(res[('APT entropy', 0)]['pooled'])}); "
          "Spearman r vs the LR loss over ps4 windows")
    print(f"{'channel':10s} | {'APT entropy: season  pooled (sd)  when':40s} | "
          f"{'cell std: season  pooled (sd)  when':36s}")
    for c in list(range(len(names))) + ["all"]:
        lab = names[c] if c != "all" else "ALL (mean)"
        e, s = stat[("APT entropy", c)], stat[("cell std", c)]
        print(f"{lab:10s} | {e[0]:+.3f}  {e[1]:+.3f} ({e[3]:.2f})  {e[2]:.3f}{'':14s} | "
              f"{s[0]:+.3f}  {s[1]:+.3f} ({s[3]:.2f})  {s[2]:.3f}")
    for meas, (top, sub, allc) in out_sub.items():
        print(f"held-out second half, {meas}: mean of top-3 channels' pooled r {sub:+.3f} vs "
              f"all-channel mean {allc:+.3f}")
    # figure: per channel, pooled and per-season r for both measures + when
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), gridspec_kw=dict(width_ratios=(1, 1, 0.8)))
    labels = names + ["ALL"]
    xs = np.arange(len(labels))
    for ax, meas, col in ((axes[0], "APT entropy", "#2a78d6"), (axes[1], "cell std", "#e8710a")):
        ks = [(meas, c) for c in range(len(names))] + [(meas, "all")]
        ax.bar(xs - 0.2, [stat[k][0] for k in ks], 0.4, color=col, alpha=0.55, label="per season")
        ax.bar(xs + 0.2, [stat[k][1] for k in ks], 0.4, color=col, label="pooled")
        for j, k in enumerate(ks):
            ax.text(j + 0.2, stat[k][1], f"{stat[k][1]:.2f}", ha="center",
                    va="bottom" if stat[k][1] >= 0 else "top", fontsize=6.5)
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(len(names) - 0.5, color="#8c8b86", lw=0.8, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylim(-0.25, 0.6)
        ax.set_ylabel("mean Spearman r vs LR loss")
        top, sub, allc = out_sub[meas]
        ax.set_title(f"{meas} per input channel (WHERE)\nheld-out top-3 "
                     f"({', '.join(names[c] for c in top)}): {sub:+.2f} vs all {allc:+.2f}",
                     fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    for j, (meas, col) in enumerate((("APT entropy", "#2a78d6"), ("cell std", "#e8710a"))):
        ks = [(meas, c) for c in range(len(names))] + [(meas, "all")]
        ax.bar(xs + (j - 0.5) * 0.4, [stat[k][2] for k in ks], 0.4, color=col, label=meas)
    ax.axhline(0.5, color="#8c8b86", ls="--", lw=1)
    ax.set_ylim(0.35, 0.65)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_title("WHEN: lowest-score half of seasons\n= lowest-loss half (dashed: chance)", fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle(f"PASTIS, {n} test samples, LR-only ps8-season model: input channels one at a "
                 f"time vs the LR loss (prediction entropy for reference: .67 per season, .78 pooled, "
                 f".72 when)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pastis_input_channels.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT / 'pastis_input_channels.png'}")


if __name__ == "__main__":
    main()
