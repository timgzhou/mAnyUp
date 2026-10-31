"""How strict must the space-then-time selection be? Evaluation-only ablation on a trained
SpaceTime checkpoint (scripts/slurm/mixres_spacetime.sh): replace the top-k picks by sampling
without replacement, P ~ exp(z / tau) (selectors.SpaceTime.tau_space / tau_time; z the
z-scored entropy over the sample's windows / minus the z-scored learned season score over the
location's used seasons; tau 0 = the trained top-k, inf = uniform), in space only, time only
or both, and measure on the test split
    test mIoU with the sampled picks (same model, same budget)
    loc_overlap    share of the deterministic pick's locations the sample also picked
    month_overlap  at locations both picked, share of the deterministic months also picked
    crop_overlap   share of the deterministic crops (location, month) also picked
-> rows of results/pastis/mixres_sampling.csv.

    python -u -m exp.mixres.sample_eval --run <checkpoint name> [--taus 0.5 1 2 inf] [--seed 0]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from exp.mixres.train import n_crops  # noqa: E402
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import Context, SpaceTime, build_selector  # noqa: E402


def load(run, bs, nw):
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    task = TASKS[cfg.task]
    model = MixedResModel(task, cfg.model).cuda().eval()
    load_legacy(model, run)
    b0 = next(iter(loader(task, "val", bs, nw, False)))
    sel = build_selector(cfg.select, model, (b0.ms, b0.ss))
    if not isinstance(sel, SpaceTime):
        raise SystemExit(f"{run}: not a space-then-time run")
    nets = nn.ModuleList(sel.modules())
    nets.load_state_dict(ck["selector"])
    nets.eval()
    return cfg, task, model, sel


@torch.no_grad()
def evaluate(cfg, task, model, sel, dl, tau_s, tau_t, seed):
    torch.manual_seed(seed)
    meter, crops, n = task.meter(), 0, 0
    hit = {"loc": [0, 0], "month": [0, 0], "crop": [0, 0]}
    for batch in dl:
        ctx = Context.from_batch(model, batch)
        sel.tau_space, sel.tau_time = 0.0, 0.0
        det = sel(ctx).mask                                             # (B, T, 16, 16)
        sel.tau_space, sel.tau_time = tau_s, tau_t
        s = sel(ctx)
        m = s.mask
        ld, ls = det.any(1), m.any(1)
        both = ld & ls
        hit["loc"][0] += int((ld & ls).sum()); hit["loc"][1] += int(ld.sum())
        dm = det & both[:, None]
        hit["month"][0] += int((dm & m).sum()); hit["month"][1] += int(dm.sum())
        hit["crop"][0] += int((det & m).sum()); hit["crop"][1] += int(det.sum())
        out = model(batch.ms, batch.ss, batch.season, batch.present, s.plan, s.logw,
                    batch.dates, ctx.seasons)
        meter.update(out.logits, batch.target.cuda())
        crops, n = crops + n_crops(s.plan), n + batch.target.shape[0]
    res = meter.compute()
    return res, crops / n, {k: a / max(b, 1) for k, (a, b) in hit.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 1.0, 2.0, float("inf")])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    cfg, task, model, sel = load(a.run, 32, a.num_workers)
    bs = cfg.train.batch_size or task.batch_size[16]
    dl = loader(task, "test", bs, a.num_workers, False)
    conds = [(0.0, 0.0)] + [(t, 0.0) for t in a.taus] + [(0.0, t) for t in a.taus] \
        + [(t, t) for t in a.taus]
    s = cfg.select
    for tau_s, tau_t in conds:
        res, crops, ov = evaluate(cfg, task, model, sel, dl, tau_s, tau_t, a.seed)
        print(f"tau space {tau_s:g} time {tau_t:g} | test mIoU {res['miou']:.4f} | "
              f"{crops:.0f} crops/sample | overlap loc {ov['loc']:.3f} month {ov['month']:.3f} "
              f"crop {ov['crop']:.3f}", flush=True)
        _append(Path("results/pastis/mixres_sampling.csv"), {
            "job": os.environ.get("SLURM_JOB_ID", ""), "run": a.run,
            "lr_patch": cfg.model.lr_patch, "n": s.n, "keep": s.keep, "tau_space": tau_s,
            "tau_time": tau_t, "seed": a.seed, "test_miou": round(res["miou"], 5),
            "crops_per_sample": round(crops, 1), "loc_overlap": round(ov["loc"], 4),
            "month_overlap": round(ov["month"], 4), "crop_overlap": round(ov["crop"], 4)})


if __name__ == "__main__":
    main()
