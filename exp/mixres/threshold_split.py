"""Which split should the Quadrant spatial threshold come from? Evaluation-only, on a trained
global-mode Quadrant checkpoint (the green arm): the stored threshold is the EMA of the training
batches' entropy quantiles, and a model that fits its training set is more certain there, so the
threshold may sit too low and the test set gets more crops than q_s asks for.

Per split (train: the first --train_batches shuffled batches; val, test: all), the crop-window
entropies of the crop-free pass:
    share_<split>   share of windows above the stored threshold (target: q_s)
    tau_<split>     that split's own (1 - q_s) quantile
Then the test split is evaluated with each threshold source (stored = the trained run; val = the
val split's quantile table; test = an oracle budget match, reference only):
    miou_<src>, crops_<src> (per sample), gmacs_<src> (both passes, traced on every test batch)
-> rows of results/<task>/mixres_threshold_split.csv.

    python -u -m exp.mixres.threshold_split --runs <checkpoint name> ...
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp.bench.flop_trace import append_row, count_macs  # noqa: E402
from exp.mixres.compute_breakdown import load  # noqa: E402
from exp.mixres.train import n_crops, step  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import Context, Quadrant, QuadrantState, pixel_entropy  # noqa: E402


@torch.no_grad()
def window_entropies(model, sel, dl, limit=0):
    out = []
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        ctx = Context.from_batch(model, batch)
        out.append(F.avg_pool2d(pixel_entropy(ctx.out0.logits.float())[:, None],
                                sel.geo.crop_px)[:, 0].flatten().cpu())
    return torch.cat(out)


def table(u):
    lv = torch.linspace(0, 1, QuadrantState.LEVELS)
    return torch.quantile(u[torch.randperm(len(u))[:4_000_000]].float(), lv).cuda()


@torch.no_grad()
def evaluate(model, sel, task, dl):
    meter, crops, n, macs = task.meter(), 0, 0, 0.0
    for batch in dl:
        with count_macs() as c:
            logits, _, nc = step(model, sel, batch)
        macs += c.total
        meter.update(logits, batch.target.cuda())
        crops, n = crops + nc, n + batch.target.shape[0]
    return float(meter.compute()["miou"]), crops / n, macs / n / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--train_batches", type=int, default=60)
    a = ap.parse_args()
    for run in a.runs:
        cfg, task, model, sel = load(run, 8)
        if not isinstance(sel, Quadrant) or sel.cfg.mode != "global":
            raise SystemExit(f"{run}: not a global-mode Quadrant run")
        bs, q = task.batch_size[16], sel.cfg.q_s
        stored = sel.state.tab_s.clone()
        tau = float(sel.state.threshold(stored, q))
        row = {"run": run, "q_s": q, "tau_stored": round(tau, 4)}
        tabs = {"stored": stored}
        for split, dl, lim in (("train", loader(task, "train", bs, 8, True), a.train_batches),
                               ("val", loader(task, "val", bs, 8, False), 0),
                               ("test", loader(task, "test", bs, 8, False), 0)):
            u = window_entropies(model, sel, dl, lim)
            row[f"share_{split}"] = round(float((u >= tau).float().mean()), 4)
            row[f"tau_{split}"] = round(float(torch.quantile(u[:4_000_000].float(), 1 - q)), 4)
            if split != "train":
                tabs[split] = table(u)
        for src, tab in tabs.items():
            sel.state.tab_s.copy_(tab)
            miou, crops, gmacs = evaluate(model, sel, task, loader(task, "test", bs, 8, False))
            row.update({f"miou_{src}": round(miou, 5), f"crops_{src}": round(crops, 1),
                        f"gmacs_{src}": round(gmacs, 3)})
        sel.state.tab_s.copy_(stored)
        print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
        append_row(Path(f"results/{task.name}/mixres_threshold_split.csv"), row)


if __name__ == "__main__":
    main()
