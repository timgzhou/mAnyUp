"""Where does a trained mixed-resolution arm spend its inference compute? GMACs/sample of each
phase of exp.mixres.train.step, traced separately (exp.bench.flop_trace.count_macs):

    pass1.embed    the season composites -> tokens (the uniform pass; reused by pass 2)
    pass1.encode   the transformer blocks on the season tokens alone
    pass1.decode   the readout + head -> the crop-free map (what the selector reads)
    selector       everything the selector adds on top of pass 1 (entropy, attention score, ...)
    pass2.embed    the crops -> tokens
    pass2.encode   the blocks on season + crop tokens, in one sequence
    pass2.decode   the readout + head over season and crop tokens -> the final map

each split into attention (FlashAttention, analytic) and the rest (linear / conv / einsum), with
the tokens per block and the crops/sample. Also: the share of samples with no crop (their
pass 2 = a rerun of pass 1). -> rows of results/<task>/mixres_compute_breakdown.csv.

    python -u -m exp.mixres.compute_breakdown --runs <checkpoint name> ... [--batches 20]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.bench.flop_trace import append_row, count_macs  # noqa: E402
from exp.common.tasks import TASKS  # noqa: E402
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.model import Out  # noqa: E402
from manyup.mixres.selectors import Context, build_selector  # noqa: E402

PHASES = ("pass1.embed", "pass1.encode", "pass1.decode", "selector", "pass2.embed",
          "pass2.encode", "pass2.decode")


def load(run, nw):
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    task = TASKS[cfg.task]
    bs = task.batch_size[16]
    model = MixedResModel(task, cfg.model).cuda().eval()
    load_legacy(model, run)
    b0 = next(iter(loader(task, "val", bs, nw, False)))
    sel = build_selector(cfg.select, model, (b0.ms, b0.ss))
    nets = nn.ModuleList(sel.modules())
    nets.load_state_dict(ck["selector"])
    nets.eval()
    return cfg, task, model, sel


@torch.no_grad()
def breakdown(model, sel, dl, batches):
    """-> {phase: [MACs, attention MACs, query tokens over blocks]}, samples, crops, no-crop
    samples. Mirrors model.uniform + step at inference, one counter per phase."""
    acc = {p: [0.0, 0.0, 0] for p in PHASES}

    def run(phase, f):
        with count_macs() as c:
            r = f()
        a = acc[phase]
        a[0] += c.total; a[1] += c.attn; a[2] += c.token_blocks
        return r

    n = crops = empty = 0
    for i, batch in enumerate(dl):
        if i >= batches:
            break
        ctx = Context.from_batch(model, batch)
        ms, ss, season, present = model._inputs(batch.ms, batch.ss, batch.season, batch.present)
        NS = ss.timestamps.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):            # = model.uniform
            lr = run("pass1.embed", lambda: model.tokenize_seasons(ss))
            lr_F, hr_F = run("pass1.encode", lambda: model.encode(lr, [], NS))
            logits = run("pass1.decode",
                         lambda: model.decode(lr_F, hr_F, [], season, present).float())
        ctx._out0 = Out(logits, lr_F, hr_F, lr)
        s = run("selector", lambda: sel(ctx))
        plans, logw = model.split(s.plan, s.logw)
        cr = run("pass2.embed", lambda: [(model.tokenize_crops(ms, p, season, lr, batch.dates), p)
                                         for p in plans])
        F2 = run("pass2.encode", lambda: model.encode(lr, cr, NS))
        run("pass2.decode", lambda: model.decode(*F2, plans, season, present, logw))
        B = batch.target.shape[0]
        n += B
        crops += sum(len(p) for p in plans)
        has = torch.zeros(B, dtype=torch.bool, device=model.device)
        for p in plans:
            has[p.b] = True
        empty += int((~has).sum())
    return acc, n, crops, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    for run in a.runs:
        cfg, task, model, sel = load(run, 8)
        dl = loader(task, a.split, task.batch_size[16], 8, False)
        acc, n, crops, empty = breakdown(model, sel, dl, a.batches)
        nb = len(model.mix.encoder.blocks)
        row = {"run": run, "split": a.split, "samples": n,
               "crops_per_sample": round(crops / n, 2), "no_crop_share": round(empty / n, 4)}
        for p, (t, at, tok) in acc.items():
            row[f"{p}_gmacs"] = round(t / n / 1e9, 4)
            row[f"{p}_attn"] = round(at / n / 1e9, 4)
            row[f"{p}_tokens"] = round(tok / n / nb, 1)
        row["total_gmacs"] = round(sum(v[0] for v in acc.values()) / n / 1e9, 4)
        print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
        append_row(Path(f"results/{task.name}/mixres_compute_breakdown.csv"), row)


if __name__ == "__main__":
    main()
