"""Asymmetric-duo spatial uncertainty (after "Asymmetric Duos", a big model + a cheap sidekick,
logits combined with temperature weights fitted on val). Evaluation-only on a trained global-
mode Quadrant checkpoint (the green arm): no new model.

    big        the crop-free pass's pa2px prediction z_px (B, C, 64, 64), as now
    sidekick   pa2pa + bilinear: z_px averaged over each season token's footprint (= a linear
               pa2pa head, the pa2px head being linear in the token), bilinearly upsampled
    duo        z = a z_px + b z_side, (a, b) fitted by NLL on the val split

Spatial uncertainties compared (each window's mean; global threshold = the val split's (1 - q_s)
quantile, as exp.mixres.train does at test):
    entropy        H(softmax z_px)                 the method (reference)
    side_entropy   H(softmax z_side)
    duo_entropy    H(softmax z_duo)
    disagree       JS(softmax z_px, softmax z_side)
-> per uncertainty: Spearman vs entropy (crop windows, within tile), test mIoU, crops and GMACs
with the run's own pass 2 (trained on entropy picks), rows of
results/<task>/mixres_duo_probe.csv.

    python -u -m exp.mixres.duo_probe --runs <checkpoint name> ...
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp.bench.flop_trace import append_row  # noqa: E402
from exp.mixres.compute_breakdown import load  # noqa: E402
from exp.mixres.threshold_split import evaluate  # noqa: E402
from exp.mixres.train import val_threshold  # noqa: E402
from manyup.mixres import selectors  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import Context, Quadrant, pixel_entropy  # noqa: E402

REAL = selectors.spatial_uncertainty


def sidekick(z, P):
    """(B, C, 64, 64) pa2px logits -> pa2pa (mean over each P x P token footprint), bilinear."""
    return F.interpolate(F.avg_pool2d(z, P), size=z.shape[-2:], mode="bilinear",
                         align_corners=False)


def js(z1, z2):
    p, q = z1.softmax(1), z2.softmax(1)
    m = (p + q) / 2
    kl = lambda a: (a * (a.clamp(min=1e-8).log() - m.clamp(min=1e-8).log())).sum(1)  # noqa: E731
    return (kl(p) + kl(q)) / 2


@torch.no_grad()
def fit_duo(model, dl, P, batches=40, pixels=2_000_000):
    """(a, b) of the duo by NLL on val pixels (labelled ones)."""
    zs, zk, ys = [], [], []
    for i, batch in enumerate(dl):
        if i >= batches:
            break
        z = Context.from_batch(model, batch).out0.logits.float()
        y = batch.target.cuda().long()
        ok = y != -1
        zs.append(z.permute(0, 2, 3, 1)[ok].cpu())
        zk.append(sidekick(z, P).permute(0, 2, 3, 1)[ok].cpu())
        ys.append(y[ok].cpu())
    zs, zk, ys = torch.cat(zs), torch.cat(zk), torch.cat(ys)
    idx = torch.randperm(len(ys))[:pixels]
    zs, zk, ys = zs[idx].cuda(), zk[idx].cuda(), ys[idx].cuda()
    w = torch.tensor([1.0, 0.0], device="cuda", requires_grad=True)
    with torch.enable_grad():
        opt = torch.optim.LBFGS([w], lr=0.5, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(w[0] * zs + w[1] * zk, ys)
            loss.backward()
            return loss
        opt.step(closure)
    nll = {k: float(F.cross_entropy(v, ys)) for k, v in
           (("px", zs), ("side", zk), ("duo", w[0].detach() * zs + w[1].detach() * zk))}
    return w.detach().tolist(), nll


def make_unc(kind, P, ab):
    def unc(ctx, k="entropy"):
        if ctx.model.task.kind != "classification":
            return REAL(ctx, k)
        z = ctx.out0.logits.float()
        if kind == "entropy":
            return pixel_entropy(z)
        zk = sidekick(z, P)
        if kind == "side_entropy":
            return pixel_entropy(zk)
        if kind == "duo_entropy":
            return pixel_entropy(ab[0] * z + ab[1] * zk)
        return js(z, zk)
    return unc


@torch.no_grad()
def spearman_vs_entropy(model, sel, dl, kinds, P, ab, batches=20):
    out = {k: 0.0 for k in kinds}
    n = 0
    rank = lambda v: v.argsort(1).argsort(1).float()                         # noqa: E731
    for i, batch in enumerate(dl):
        if i >= batches:
            break
        ctx = Context.from_batch(model, batch)
        pool = lambda u: sel.pool(u, sel.geo.crop_px).flatten(1)            # noqa: E731
        ref = rank(pool(make_unc("entropy", P, ab)(ctx)))
        ref = ref - ref.mean(1, keepdim=True)
        for k in kinds:
            r = rank(pool(make_unc(k, P, ab)(ctx)))
            r = r - r.mean(1, keepdim=True)
            out[k] += float(((r * ref).sum(1) / (r.norm(dim=1) * ref.norm(dim=1)).clamp(min=1e-9))
                            .sum())
        n += batch.target.shape[0]
    return {k: v / n for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+",
                    default=["entropy", "side_entropy", "duo_entropy", "disagree"])
    a = ap.parse_args()
    for run in a.runs:
        cfg, task, model, sel = load(run, 8)
        if not isinstance(sel, Quadrant) or sel.cfg.mode != "global" or sel.cfg.space != "crop":
            raise SystemExit(f"{run}: not a global-mode, crop-space Quadrant run")
        bs, P = task.batch_size[16], cfg.model.lr_patch
        dl = lambda s: loader(task, s, bs, 8, False)                         # noqa: E731
        ab, nll = fit_duo(model, dl("val"), P)
        print(f"{run}: duo (a, b) = {ab[0]:.3f}, {ab[1]:.3f} | val NLL px {nll['px']:.4f} "
              f"side {nll['side']:.4f} duo {nll['duo']:.4f}", flush=True)
        rho = spearman_vs_entropy(model, sel, dl("test"), a.kinds, P, ab)
        for k in a.kinds:
            selectors.spatial_uncertainty = make_unc(k, P, ab)
            try:
                val_threshold(model, sel, dl("val"))
                miou, crops, gmacs = evaluate(model, sel, task, dl("test"))
            finally:
                selectors.spatial_uncertainty = REAL
            row = {"run": run, "q_s": sel.cfg.q_s, "unc": k, "a": round(ab[0], 4),
                   "b": round(ab[1], 4), "nll_px": round(nll["px"], 4),
                   "nll_side": round(nll["side"], 4), "nll_duo": round(nll["duo"], 4),
                   "rho_vs_entropy": round(rho[k], 4), "test_miou": round(miou, 5),
                   "crops_per_sample": round(crops, 1), "gmacs": round(gmacs, 3)}
            print(" | ".join(f"{k2} {v}" for k2, v in row.items()), flush=True)
            append_row(Path(f"results/{task.name}/mixres_duo_probe.csv"), row)


if __name__ == "__main__":
    main()
