"""Can the crop-free pass (pass 1) stop early? Probe on a trained Quadrant checkpoint (the green
arm: global spatial threshold + readout attention, space crop, rank score): pass 1 exits after
block L and the selector reads that pass instead of the full one (L = 12). Exit designs:

    raw      the final norm, the model's own readout and head on block L's tokens (no training)
    label    an exit head trained on block L's tokens (backbone frozen) with the task loss vs
             the label: final norm, a residual MLP adapter (zero init: starts as raw), and a
             copy of the readout + head; its entropy and readout attention are what the
             selector reads
    distill  the same head trained label-free to match the full pass's two signals: KL of the
             per-pixel class distribution and KL of the per-cell softmax of the readout
             attention score over the used seasons

Each is compared with the full pass:

    rho_s        Spearman correlation of the crop-window entropies, within each test tile, mean
    win_recall   share of the full pass's high-entropy windows the layer-L pass also picks; its
                 global threshold recalibrated on the train split (the same share q_s of the
                 layer-L entropies, as training does), so windows/sample may differ
    win_ratio    windows picked at L / at 12
    season_top1  share of picked windows (the full pass's) where both passes attend most to
                 the same season (rank score; cells with >= 2 used seasons)
    crop_recall  share of the full pass's crops (sample, month, window) in the layer-L plan
    miou, gmacs  test mIoU with pass 2 run on the layer-L plan (the trained model, no retraining);
                 GMACs/sample = pass 1 up to block L with its exit head + pass 2 (traced)
-> rows of results/<task>/mixres_early_exit.csv (exit heads: checkpoints/<run>_exit.pt).

    python -u -m exp.mixres.early_exit_probe --runs <checkpoint name> ... [--layers 2 3 4 6 8 12]
        [--designs raw label distill] [--exit_epochs 2]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import copy  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from exp.bench.flop_trace import append_row, count_macs  # noqa: E402
from exp.mixres.compute_breakdown import load  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.model import Out  # noqa: E402
from manyup.mixres.selectors import Context, Quadrant, QuadrantState, pixel_entropy  # noqa: E402


class ExitHead(nn.Module):
    """Block L's tokens -> the crop-free pass's Out: final norm (frozen, the model's), residual
    MLP adapter (zero-init output), a trainable copy of the readout and head."""

    def __init__(self, model):
        super().__init__()
        D = model.mix.encoder.embedding_size
        self.adapt = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        nn.init.zeros_(self.adapt[-1].weight)
        nn.init.zeros_(self.adapt[-1].bias)
        self.readout, self.head = copy.deepcopy(model.readout), copy.deepcopy(model.head)

    def forward(self, model, lr, xL, season, present) -> Out:
        x = model.mix.encoder.norm(xL)
        x = x + self.adapt(x)
        lr_F = lr.pooled(x).float()
        z = torch.zeros(0, dtype=torch.long, device=lr_F.device)
        hr_F = lr_F.new_zeros(0, lr_F.shape[-1])
        logits = self.readout(self.head.linear, lr_F, season, present, hr_F, (z,) * 4)
        return Out(logits.float(), lr_F, hr_F, lr)


@contextmanager
def readout_of(model, head):
    """The selector reads model.readout._score: point it at the exit head's readout."""
    keep = model.readout
    if head is not None:
        model.readout = head.readout
    try:
        yield
    finally:
        model.readout = keep


def season_logp(model, readout, out, ctx):
    """(B, NS, gh, gw) log-softmax of the readout attention score over the used seasons."""
    s = readout._score(out.lr_F).float()
    months = (F.one_hot(ctx.season.clamp(min=0), ctx.NS).float()
              * ctx.valid.float()[..., None]).sum(1)
    return s.masked_fill((months == 0)[:, :, None, None], -1e4).log_softmax(1)


def train_exits(model, dl, layers, designs, epochs, limit=0):
    """{(design, L): ExitHead} trained on the frozen model, all in one pass per batch."""
    heads = nn.ModuleDict({f"{d}_{L}": ExitHead(model) for d in designs for L in layers}).cuda()
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3)
    model.eval()
    t0 = time.time()
    for ep in range(epochs):
        tot, n = {k: 0.0 for k in heads}, 0
        for i, batch in enumerate(dl):
            if limit and i >= limit:
                break
            ctx = Context.from_batch(model, batch)
            target = batch.target.cuda().long()
            with torch.no_grad():
                outs = early_outs(model, batch, [len(model.mix.encoder.blocks)], keep_trace=True)
                full, trace = outs["full"], outs["trace"]
                p_full = full.logits.float().log_softmax(1)
                s_full = season_logp(model, model.readout, full, ctx)
            loss = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for k, h in heads.items():
                    d, L = k.split("_")
                    o = h(model, full.seasons, trace[int(L)], ctx.season, ctx.present)
                    if d == "label":
                        l_k = F.cross_entropy(o.logits, target, ignore_index=-1)
                    else:
                        lp = o.logits.log_softmax(1)
                        l_k = (F.kl_div(lp, p_full, log_target=True, reduction="none").sum(1)
                               .mean()
                               + F.kl_div(season_logp(model, h.readout, o, ctx), s_full,
                                          log_target=True, reduction="none").sum(1).mean())
                    tot[k] += float(l_k.detach())
                    loss = loss + l_k
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n += 1
        print(f"exit heads epoch {ep + 1}/{epochs} ({(time.time() - t0) / 60:.1f} min): "
              + " ".join(f"{k} {v / n:.3f}" for k, v in tot.items()), flush=True)
    heads.eval()
    return heads


@torch.no_grad()
def early_outs(model, batch, layers, heads=None, keep_trace=False):
    """-> {(design, L): Out of the crop-free pass exiting after block L} (raw: the model's own
    norm + readout + head; trained designs: their ExitHead), "full" = the full pass."""
    ms, ss, season, present = model._inputs(batch.ms, batch.ss, batch.season, batch.present)
    mix = model.mix
    nb = len(mix.encoder.blocks)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = model.tokenize_seasons(ss)
        trace, x = mix._lr_trace(lr.x, lr.seqlens, 0, nb)
        trace = trace + [x]
        outs = {}
        for L in layers:
            lr_F = lr.pooled(mix.encoder.norm(trace[L])).float()
            hr_F = lr_F.new_zeros(0, lr_F.shape[-1])
            outs[("raw", L)] = Out(model.decode(lr_F, hr_F, [], season, present).float(), lr_F,
                                   hr_F, lr)
        for k, h in (heads or {}).items():
            d, L = k.split("_")
            outs[(d, int(L))] = h(model, lr, trace[int(L)], season, present)
    outs["full"] = outs[("raw", nb)] if ("raw", nb) in outs else None
    if keep_trace:
        lr_F = lr.pooled(mix.encoder.norm(x)).float()
        outs["full"] = Out(model.decode(lr_F, lr_F.new_zeros(0, lr_F.shape[-1]), [], season,
                                        present).float(), lr_F, lr_F.new_zeros(0, 1), lr)
        outs["trace"] = trace
    return outs


def window_entropy(out, crop_px):
    return F.avg_pool2d(pixel_entropy(out.logits.float())[:, None], crop_px)[:, 0]


def spearman(a, b):
    """Per row of (B, N) -> (B,)."""
    ra, rb = (v.argsort(1).argsort(1).float() for v in (a, b))
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp(min=1e-9)


def crop_keys(plan, T):
    return set(((((plan.b * T + plan.t) * 64 + plan.py) * 64 + plan.px)).tolist())


def variants(layers, designs, nb):
    return [(d, L) for d in designs for L in layers if not (d != "raw" and L == nb)]


@torch.no_grad()
def calibrate(model, sel, dl, layers, designs, heads, batches):
    """Each variant's QuadrantState tables from the train split (quantiles of u_s, u_t)."""
    nb = len(model.mix.encoder.blocks)
    vs = variants(layers, designs, nb)
    us, ut = {v: [] for v in vs}, {v: [] for v in vs}
    for i, batch in enumerate(dl):
        if i >= batches:
            break
        outs = early_outs(model, batch, layers, heads)
        for v in vs:
            ctx = Context.from_batch(model, batch)
            ctx._out0 = outs[v]
            with readout_of(model, heads[f"{v[0]}_{v[1]}"] if v[0] != "raw" else None):
                _, u_t, _ = sel.uncertainties(ctx)
            us[v].append(window_entropy(outs[v], sel.geo.crop_px).flatten().cpu())
            ut[v].append(u_t.flatten().cpu())
    lv = torch.linspace(0, 1, QuadrantState.LEVELS)
    q = lambda x: torch.quantile(x[torch.randperm(len(x))[:2_000_000]].float(), lv).cuda()  # noqa: E731
    return {v: (q(torch.cat(us[v])), q(torch.cat(ut[v]))) for v in vs}


@torch.no_grad()
def probe(model, sel, task, dl, layers, designs, heads, tabs, stored):
    nb = len(model.mix.encoder.blocks)
    vs = variants(layers, designs, nb)
    vs = [("raw", nb)] + [v for v in vs if v != ("raw", nb)]     # the reference first
    acc = {v: dict(rho=0.0, n=0, wr=0, wn=0, w12=0, st=0, sn=0, cr=0, cn=0, crops=0, p2=0.0)
           for v in vs}
    meters = {v: task.meter() for v in vs}
    pass1_cost = {}
    for i, batch in enumerate(dl):
        outs = early_outs(model, batch, layers, heads)
        T = batch.present.shape[1]
        B = batch.target.shape[0]
        ref = {}
        for v in vs:
            head = heads[f"{v[0]}_{v[1]}"] if v[0] != "raw" else None
            ctx = Context.from_batch(model, batch)
            ctx._out0 = outs[v]
            sel.state.tab_s, sel.state.tab_t = stored if v == ("raw", nb) else tabs[v]
            with readout_of(model, head):
                s = sel(ctx)
                _, _, _, r = sel.uncertainties(ctx, rank=True)
            u_s = window_entropy(outs[v], sel.geo.crop_px)
            hs = u_s >= sel.state.threshold(sel.state.tab_s, sel.cfg.q_s)
            plan = s.plan[0]
            keys = crop_keys(plan, T)
            if v == ("raw", nb):
                ref = dict(u_s=u_s, hs=hs, top=r.argmax(1), used=torch.isfinite(r).sum(1) >= 2,
                           keys=keys)
            a = acc[v]
            a["rho"] += float(spearman(u_s.flatten(1), ref["u_s"].flatten(1)).sum()); a["n"] += B
            a["wr"] += int((hs & ref["hs"]).sum()); a["wn"] += int(hs.sum())
            a["w12"] += int(ref["hs"].sum())
            f = sel.lr_patch // sel.geo.crop_px                 # windows -> their lr cell
            cell = lambda m: m.repeat_interleave(f, -2).repeat_interleave(f, -1)  # noqa: E731
            m = ref["hs"] & cell(ref["used"])
            a["st"] += int((cell(r.argmax(1) == ref["top"]) & m).sum()); a["sn"] += int(m.sum())
            a["cr"] += len(keys & ref["keys"]); a["cn"] += len(ref["keys"])
            a["crops"] += len(plan)
            with count_macs() as c:                             # pass 2 (season tokens reused)
                logits = model(batch.ms, batch.ss, batch.season, batch.present, s.plan, s.logw,
                               batch.dates, outs[v].seasons).logits
            a["p2"] += c.total
            meters[v].update(logits, batch.target.cuda())
        if i == 0:                                              # pass 1 to block L, traced once
            ms, ss, season, present = model._inputs(batch.ms, batch.ss, batch.season,
                                                    batch.present)
            for v in vs:
                d, L = v
                with count_macs() as c, torch.autocast("cuda", dtype=torch.bfloat16):
                    lr = model.tokenize_seasons(ss)
                    xL = model.mix.blocks(lr.x, lr.seqlens, 0, L)
                    if d == "raw":
                        lr_F = lr.pooled(model.mix.encoder.norm(xL)).float()
                        model.decode(lr_F, lr_F.new_zeros(0, lr_F.shape[-1]), [], season,
                                     present)
                    else:
                        heads[f"{d}_{L}"](model, lr, xL, season, present)
                pass1_cost[v] = c.total / B
    rows = []
    for v in vs:
        a, n = acc[v], acc[v]["n"]
        res = meters[v].compute()
        rows.append({"design": v[0], "layer": v[1], "rho_s": round(a["rho"] / n, 4),
                     "win_recall": round(a["wr"] / max(a["w12"], 1), 4),
                     "win_ratio": round(a["wn"] / max(a["w12"], 1), 4),
                     "season_top1": round(a["st"] / max(a["sn"], 1), 4),
                     "crop_recall": round(a["cr"] / max(a["cn"], 1), 4),
                     "crops_per_sample": round(a["crops"] / n, 1),
                     "test_miou": round(float(res["miou"]), 5),
                     "pass1_gmacs": round(pass1_cost[v] / 1e9, 3),
                     "gmacs": round((pass1_cost[v] + a["p2"] / n) / 1e9, 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 3, 4, 6, 8, 10, 12])
    ap.add_argument("--designs", nargs="+", default=["raw", "label", "distill"],
                    choices=["raw", "label", "distill"])
    ap.add_argument("--exit_epochs", type=int, default=2)
    ap.add_argument("--calib_batches", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="test / exit-training batches (0: all)")
    a = ap.parse_args()
    for run in a.runs:
        cfg, task, model, sel = load(run, 8)
        if not isinstance(sel, Quadrant):
            raise SystemExit(f"{run}: not a Quadrant run")
        nb = len(model.mix.encoder.blocks)
        layers = sorted(set(a.layers) | {nb})
        bs = task.batch_size[16]
        trained = [d for d in a.designs if d != "raw"]
        heads = {}
        if trained:
            heads = train_exits(model, loader(task, "train", bs, 8, True),
                                [L for L in layers if L < nb], trained, a.exit_epochs, a.limit)
            if not a.limit:
                torch.save(heads.state_dict(), f"checkpoints/{run}_exit.pt")
        stored = (sel.state.tab_s.clone(), sel.state.tab_t.clone())
        designs = sorted(set(a.designs) | {"raw"}, key=["raw", "label", "distill"].index)
        tabs = calibrate(model, sel, loader(task, "train", bs, 8, True), layers, designs, heads,
                         a.calib_batches)
        dl = loader(task, "test", bs, 8, False)
        if a.limit:
            dl = [b for i, b in zip(range(a.limit), dl)]
        for row in probe(model, sel, task, dl, layers, designs, heads, tabs, stored):
            row = {"run": run, **row}
            print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
            if not a.limit:
                append_row(Path(f"results/{task.name}/mixres_early_exit_heads.csv"), row)


if __name__ == "__main__":
    main()
