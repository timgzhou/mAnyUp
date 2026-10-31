"""Paper material, method figure: run the final method (s16m8: entropy space at the global
threshold + readout-attention time + cross_hr, schedule B) on every PASTIS test tile and rank the
tiles for the method figure: picks concentrated in one half of the tile (one half easy, one
hard), a moderate number of picks, picks spread over seasons, few cloudy months.

    source env_setup/env_olmo.sh && python -m exp.viz.paper.method_scan [--setup s16m8|s8m4]

-> figures/paper/method/scan_<setup>.pt (per-tile statistics) and scan_top_<setup>.png (contact sheet of the
best-ranked tiles without S2 nodata frames: monthly S2, picks per season, entropy, label, predictions)
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
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from manyup.mixres import MixedResModel  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import Context, build_selector, spatial_uncertainty  # noqa: E402

OUT = Path("figures/paper/method")
RUNS = {
    "s16m8": "mixres_pastis_ps16_quadg-skip-qs0.25-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c8_seed0_a11df7da",
    "s8m4": "mixres_pastis_ps8_quadg-skip-qs0.25-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c4_seed0_fbe9cd48",
    # q_s 1/2: half of the windows above the global threshold
    "s16m8_qs0.5": "mixres_pastis_ps16_quadg-skip-qs0.5-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c8_seed0_f0ff2d8e",
    "s8m4_qs0.5": "mixres_pastis_ps8_quadg-skip-qs0.5-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c4_seed0_a736def1"}


def clean(ds, tid) -> bool:
    """No S2 nodata fill (-10000 frames) in the tile's monthly MEAN composites."""
    m = ds[ds.ds.index_of(tid)][0]
    return float(m.modalities["S2"].data[:, :3].amin()) > -100


def build(run: str, example):
    """A finished mixres run: model + selector with their best weights (val threshold table)."""
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    model = MixedResModel(TASKS[cfg.task], cfg.model, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    selector = build_selector(cfg.select, model, example)
    nn.ModuleList(selector.modules()).load_state_dict(ck["selector"])
    for m in selector.modules():
        m.eval()
    return model, selector, cfg


@torch.no_grad()
def run_batch(model, selector, batch):
    """-> the selection and both passes of one batch (inference, pass 1 reused)."""
    ctx = Context.from_batch(model, batch)
    sel = selector(ctx)
    plan = sel.plan[0]
    out = model(batch.ms, batch.ss, batch.season, batch.present, sel.plan, None, None,
                ctx.seasons, ctx.lr_enc)
    u_s, u_t, top, r = selector.uncertainties(ctx, rank=True)
    ent = spatial_uncertainty(ctx)
    attn = ctx.model.readout0._score(ctx.out0.lr_F.float())                # (B, NS, 4, 4)
    return ctx, plan, out, dict(ent=ent, u_t=u_t, top=top, attn=attn)


def halves(P):
    """(g, g) pick counts -> (largest share of picks in one half, which half)."""
    n, k = P.sum(), P.shape[0] // 2
    if n == 0:
        return 0.0, ""
    h = {"top": P[:k].sum(), "bottom": P[k:].sum(), "left": P[:, :k].sum(),
         "right": P[:, k:].sum()}
    k = max(h, key=h.get)
    return float(h[k] / n), k


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--setup", default="s16m8", choices=tuple(RUNS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="batches (0 = all)")
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    torch.manual_seed(0)
    task = TASKS["pastis"]
    dl = loader(task, a.split, 16, 8, False)
    ids = dl.dataset.ds
    b0 = next(iter(dl))
    run = RUNS[a.setup]
    model, selector, cfg = build(run, (b0.ms, b0.ss))
    cp = cfg.select.crop_px
    g = 64 // cp
    rows, i0 = [], 0
    for bi, batch in enumerate(dl):
        if a.limit and bi >= a.limit:
            break
        ctx, plan, out, ex = run_batch(model, selector, batch)
        B = ctx.B
        season = ctx.season
        tgt = batch.target.cuda().long()
        p0, p1 = ctx.out0.logits.argmax(1), out.logits.argmax(1)
        s2 = batch.ms.sentinel2_l2a.float()                            # (B, H, W, T, C)
        rgb = s2[..., [2, 1, 0]].mean(-1)                              # normalized brightness
        for b in range(B):
            k = plan.b == b
            gy, gx, t = plan.py[k] // cp, plan.px[k] // cp, plan.t[k]
            win = torch.zeros(g, g, dtype=torch.long)                  # 1 + season, 0 = none
            win[gy.cpu(), gx.cpu()] = 1 + season[b, t].cpu()
            P = (win > 0).float().numpy()
            share, half = halves(P)
            ok = tgt[b] >= 0
            acc0 = float((p0[b][ok] == tgt[b][ok]).float().mean()) if ok.any() else float("nan")
            acc1 = float((p1[b][ok] == tgt[b][ok]).float().mean()) if ok.any() else float("nan")
            seasons_used = len(set(win[win > 0].tolist()))
            present = ctx.valid[b].cpu()
            cloud = (rgb[b] > 0.7).float().mean((0, 1)).cpu()          # (T,) bright share
            rows.append(dict(
                id=ids.sample_id(i0 + b), n_win=int(P.sum()), n_crops=int(k.sum()),
                share=share, half=half, seasons=seasons_used, acc0=acc0, acc1=acc1,
                present=present, cloud=cloud, win=win,
                ent=F.avg_pool2d(ex["ent"][b][None, None].float(), cp)[0, 0].cpu(),
                target=tgt[b].cpu().to(torch.int8), pred0=p0[b].cpu().to(torch.int8),
                pred1=p1[b].cpu().to(torch.int8),
                rgb=torch.stack([s2[b, ..., t_, [2, 1, 0]] for t_ in range(s2.shape[3])]
                                ).half().cpu()))
        i0 += B
        if bi % 20 == 0:
            print(f"batch {bi}/{len(dl)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"run": run, "rows": rows}, OUT / f"scan_{a.setup}.pt")

    # ---- rank: concentrated picks, moderate count, several seasons, clear months, all present
    def score(r):
        if not (0.12 <= r["n_win"] / g ** 2 <= 0.38) or not bool(r["present"].all()):
            return -1.0
        clear = 1.0 - float(r["cloud"].mean())
        return (r["share"] + 0.5 * clear + 0.1 * (r["seasons"] - 1)
                + 0.5 * (r["acc1"] - r["acc0"]))
    order = [i for i in sorted(range(len(rows)), key=lambda i: -score(rows[i]))[:3 * a.top]
             if clean(dl.dataset, rows[i]["id"])][:a.top]
    print("rank id n_win crops share half seasons cloud acc0 acc1")
    for j, i in enumerate(order):
        r = rows[i]
        print(f"{j:2d} {r['id']:14s} {r['n_win']:3d} {r['n_crops']:3d} {r['share']:.2f} "
              f"{r['half']:6s} {r['seasons']} {float(r['cloud'].mean()):.2f} "
              f"{r['acc0']:.3f} {r['acc1']:.3f}")
    contact(rows, order, OUT / f"scan_top_{a.setup}.png")


def contact(rows, order, path):
    """One row per tile: 12 months (S2, per-tile stretch), picks per season, entropy, label,
    pass-1 and final prediction."""
    tab = plt.get_cmap("tab20").colors
    pal = np.array([(0, 0, 0)] + [tab[i] for i in range(1, 19)] + [(1, 1, 1)])
    cols = 12 + 4 + 4
    fig, axs = plt.subplots(len(order), cols, figsize=(cols * 0.9, len(order) * 0.95))
    for j, i in enumerate(order):
        r = rows[i]
        x = r["rgb"].float().numpy()                                   # (T, H, W, 3)
        lo, hi = np.percentile(x, [2, 98])
        for t in range(12):
            axs[j, t].imshow(np.clip((x[t] - lo) / (hi - lo), 0, 1))
        for s in range(4):
            axs[j, 12 + s].imshow((r["win"] == s + 1).numpy(), cmap="magma", vmin=0, vmax=1.3)
        axs[j, 16].imshow(r["ent"].numpy(), cmap="viridis")
        for c, k in enumerate(("target", "pred0", "pred1")):
            v = r[k].long().numpy()
            axs[j, 17 + c].imshow(pal[np.where(v < 0, 19, v)])
        axs[j, 0].set_ylabel(f"{j} {r['id']}", fontsize=5, rotation=0, ha="right")
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for c, t in enumerate([f"m{t}" for t in range(12)] + [f"S{s}" for s in range(4)]
                          + ["entropy", "label", "pass 1", "final"]):
        axs[0, c].set_title(t, fontsize=6)
    fig.subplots_adjust(0.08, 0.01, 0.99, 0.98, 0.04, 0.04)
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
