"""Can the label oracle's selection quantity -- the per-pixel loss of the model's crop-free pass
-- be predicted from the low-resolution prediction, guided by the high-resolution input?
(manyup.mixres.lossup). No selection training: on a FROZEN co-trained space-then-time model
(entropy + learned, one season; mixres_spacetime.sh), train LossUp (mode stats / bilinear /
manyup) on the train split, then on the test split compare how each score ranks the TRUE loss:

  space  (pooled prediction, the spatial oracle's map): per sample over candidate windows,
         spearman(score, loss) and capture@n = true loss of the score's top-n windows / of the
         loss's own top-n (n = a quarter of the candidates)
  time   (each season's map at the windows that tile the tile): per location over its used
         seasons, spearman(score, loss), and the share of random's regret the score's lowest
         season saves (regret = picked season's loss - the location's lowest)

Scores: entropy (space: highest picked; time: lowest), the run's learned season scorer (time
only), LossUp's predicted loss (window mean of exp(pred) - 1; space: highest; time: lowest).
-> rows of results/pastis/lossup_probe.csv.

    python -u -m exp.mixres.lossup_probe --setup s16m8 --mode manyup [--epochs 10]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.lossup import LossUp  # noqa: E402
from manyup.mixres.residual import window_pixels  # noqa: E402
from manyup.mixres.selectors import (Context, Entropy, LabelLoss, SpaceTime,  # noqa: E402
                                     build_selector, pixel_label_loss, season_maps)
from manyup.mixres.sign import season_spearman  # noqa: E402

RUNS = {  # the frozen co-trained models (entropy + learned, one season, a middle n)
    "s16m8": "mixres_pastis_ps16_entropy-top+season_residual-every_location-one-resoraclexattn_"
             "conv-noneprior_n16_c8s8_warm_seed0",
    "s8m4": "mixres_pastis_ps8_entropy-top+season_residual-every_location-one-resoraclexattn_"
            "conv-noneprior_n64_c4_warm_seed0",
}


def load(prefix, nw):
    run = next(p.stem for p in Path("checkpoints").glob(prefix + "_*.pt")
               if not p.stem.endswith("_state"))
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    task = TASKS[cfg.task]
    model = MixedResModel(task, cfg.model).cuda().eval()
    model.requires_grad_(False)
    load_legacy(model, run)
    b0 = next(iter(loader(task, "val", 8, nw, False)))
    sel = build_selector(cfg.select, model, (b0.ms, b0.ss))
    assert isinstance(sel, SpaceTime)
    nets = nn.ModuleList(sel.modules())
    nets.load_state_dict(ck["selector"])
    nets.eval()
    return run, cfg, task, model, sel


def composites(ms, season, valid, NS):
    """-> (B, NS + 1, C, 64, 64): each season's mean over its present months, then all months."""
    pix = window_pixels(ms, 1)                                        # (B, 64, 64, T, C)
    w = (F.one_hot(season.clamp(min=0), NS).bool() & valid[..., None]).float()   # (B, T, NS)
    w = torch.cat([w, valid.float()[..., None]], -1)                  # + all months
    c = torch.einsum("bxytc,btk->bkcxy", pix, w)
    return c / w.sum(1)[:, :, None, None, None].clamp(min=1)


def maps(model, ctx, norm):
    """-> logits (B K, n_cls, 64, 64), tokens (B K, D, gh, gw), guide (B K, C, 64, 64), valid
    (B K,), K = NS + 1 (the pooled prediction last)."""
    out0 = ctx.out0
    B, NS = ctx.B, ctx.NS
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_log = season_maps(model, out0.lr_F)
        s_log = s_log.float().view(B, NS, *s_log.shape[1:])
        logits = torch.cat([s_log, out0.logits.float()[:, None]], 1).flatten(0, 1)
        tok = out0.lr_F.float().permute(0, 1, 4, 2, 3)                # (B, NS, D, gh, gw)
        u = ctx.used.float()[:, :, None, None, None]
        pooled = (tok * u).sum(1, keepdim=True) / u.sum(1, keepdim=True).clamp(min=1)
        tok = torch.cat([tok, pooled], 1).flatten(0, 1)
        g = composites(ctx.ms, ctx.season, ctx.valid, NS)
        g = ((g - norm[0][:, None, None]) / norm[1][:, None, None]).flatten(0, 1)
        valid = torch.cat([ctx.used, torch.ones_like(ctx.used[:, :1])], 1).flatten()
    return logits, tok, g, valid


def guide_norm(task, nw, batches=20):
    s = s2 = n = 0
    for i, b in enumerate(loader(task, "train", 32, nw, True)):
        if i >= batches:
            break
        x = window_pixels(b.ms, 1).flatten(0, 3).float()
        s, s2, n = s + x.sum(0), s2 + (x ** 2).sum(0), n + x.shape[0]
    mu = s / n
    return mu.cuda(), (s2 / n - mu ** 2).clamp(min=1e-8).sqrt().cuda()


def rank(x):
    return x.argsort(-1).argsort(-1).float()


def spearman_rows(a, b):
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / ((ra ** 2).sum(-1) * (rb ** 2).sum(-1)).sqrt().clamp(min=1e-9)


@torch.no_grad()
def evaluate(task, model, sel, net, norm, nw, bs, limit=0):
    geo = sel.geo
    cell = geo.tiling()
    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(v)
    for i, b in enumerate(loader(task, "test", bs, nw, False)):
        if limit and i >= limit:
            break
        ctx = Context.from_batch(model, b)
        B, NS = ctx.B, ctx.NS
        lab = LabelLoss()(ctx, geo, ("window", "season"))
        ent = Entropy()(ctx, geo, ("window", "season"))
        scores = {"entropy": (ent.window.float(), ent.season.float())}
        if net is not None:
            logits, tok, g, _ = maps(model, ctx, norm)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(logits, tok, g).float()
            p = geo.window_mean(p.clamp(max=20).expm1()).view(B, NS + 1, 16, 16)
            scores["lossup"] = (p[:, NS], p[:, :NS])
        # season scores: lower = keep (entropy, predicted loss, the learned r)
        scores["learned"] = (None, sel.scorer(ctx, geo, ("season",)).season.float())
        # space: candidates of the pooled loss map
        Lw = lab.window.float().flatten(1)
        cand = torch.isfinite(Lw[0]).nonzero()[:, 0]
        Lw = Lw[:, cand]
        nt = max(1, len(cand) // 4)
        best = Lw.topk(nt, 1).values.sum(1)
        # time: the tiling cells, locations with >= 2 used seasons
        c = cell.to(Lw.device)
        Ls = lab.season.float()[:, :, c[:, None], c[None, :]]
        used = ctx.used
        u = used[:, :, None, None].expand_as(Ls)
        ok2 = u.sum(1) >= 2
        Lu = Ls.masked_fill(~u, float("nan"))
        lo = Lu.nan_to_num(float("inf")).amin(1)
        reg_r = (Lu.nanmean(1) - lo)[ok2]
        add("random/time_regret", reg_r.sum().item())
        for name, (W, S) in scores.items():
            if W is not None:
                Wc = W.flatten(1)[:, cand]
                add(f"{name}/space_spearman", spearman_rows(Wc, Lw).mean().item())
                pick = Wc.topk(nt, 1).indices
                add(f"{name}/space_capture", (Lw.gather(1, pick).sum(1) / best).mean().item())
            Sc = S[:, :, c[:, None], c[None, :]]
            sp = season_spearman(Sc, Ls, used)
            add(f"{name}/time_spearman", sp[torch.isfinite(sp)].mean().item())
            pk = Sc.masked_fill(~u, float("inf")).argmin(1, keepdim=True)
            add(f"{name}/time_regret", (Ls.gather(1, pk)[:, 0] - lo)[ok2].sum().item())
    out = {}
    for k, v in acc.items():
        if k.endswith("time_regret"):
            continue
        out[k] = sum(v) / len(v)
    rr = sum(acc["random/time_regret"])
    for name in ("entropy", "learned", "lossup"):
        if f"{name}/time_regret" in acc:
            out[f"{name}/time_regret_saved"] = 1 - sum(acc[f"{name}/time_regret"]) / rr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=tuple(RUNS))
    ap.add_argument("--mode", required=True, choices=("stats", "bilinear", "manyup", "none"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(0)
    run, cfg, task, model, sel = load(RUNS[a.setup], a.num_workers)
    norm = guide_norm(task, a.num_workers)
    net = None
    if a.mode != "none":
        b0 = next(iter(loader(task, "val", 4, a.num_workers, False)))
        ctx0 = Context.from_batch(model, b0)
        lg, tk, g, _ = maps(model, ctx0, norm)
        net = LossUp(tk.shape[1], lg.shape[1], g.shape[1], a.mode).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
        print(f"{run}: LossUp {a.mode}, {sum(p.numel() for p in net.parameters()):,} params",
              flush=True)
        for ep in range(a.epochs):
            net.train()
            t0, tot, n = time.time(), 0.0, 0
            for i, b in enumerate(loader(task, "train", a.batch_size, a.num_workers, True)):
                if a.limit_batches and i >= a.limit_batches:
                    break
                ctx = Context.from_batch(model, b)
                logits, tok, g, valid = maps(model, ctx, norm)
                tgt = b.target.cuda().repeat_interleave(ctx.NS + 1, 0)
                with torch.no_grad():
                    y = pixel_label_loss(logits, tgt).log1p()
                    m = (tgt != -1) & valid[:, None, None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = net(logits, tok, g).float()
                loss = F.smooth_l1_loss(p[m], y[m])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot, n = tot + loss.item(), n + 1
            print(f"epoch {ep + 1}/{a.epochs} | loss {tot / max(n, 1):.4f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        net.eval()
    res = evaluate(task, model, sel, net, norm, a.num_workers, a.batch_size, a.limit_batches)
    for k in sorted(res):
        print(f"  {k:32s} {res[k]:.4f}")
    if not a.limit_batches:
        for name in ("entropy", "learned", "lossup"):
            row = {k.split("/")[1]: round(v, 5) for k, v in res.items() if k.startswith(name + "/")}
            if row:
                _append(Path("results/pastis/lossup_probe.csv"), {
                    "job": os.environ.get("SLURM_JOB_ID", ""), "setup": a.setup, "run": run,
                    "mode": a.mode, "method": name, "epochs": a.epochs,
                    "space_spearman": row.get("space_spearman", ""),
                    "space_capture": row.get("space_capture", ""),
                    "time_spearman": row.get("time_spearman", ""),
                    "time_regret_saved": row.get("time_regret_saved", "")})


if __name__ == "__main__":
    main()
