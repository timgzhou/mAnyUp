"""Fine-tune a mixed-resolution OlmoEarth model with a crop selector (manyup.mixres).

    python -u -m exp.mixres.train task=pastis select.scorer=entropy select.allocator=halves \\
        select.keep=all select.n=8 train.init_ckpt=<location oracle> train.max_hours=2.5
    python -u -m exp.mixres.train --config run.json train.seed=1

Each step: the selector reads the batch (and the model's own uniform pass) and returns a plan;
the model encodes the season tokens with the plan's crops; loss = task loss + the selector's
auxiliary loss. Resumable per epoch (checkpoints/<name>_state.pt; train.max_hours for chains
of short jobs). The best-val epoch is tested -> checkpoints/<name>.pt (config + weights) and a
row of results/<task>/mixres.csv. Training schedule as exp.upsamplers.finetune_inject: a warm
start trains its loaded weights at lr x 0.1 and new modules at lr; a cold start freezes the
backbone for the first 20% of the epochs; ReduceLROnPlateau on the val metric.
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import (FREEZE_EPOCH_FRACTION, SCHEDULER_COOLDOWN,  # noqa: E402
                                 SCHEDULER_FACTOR, SCHEDULER_PATIENCE, UNFREEZE_LR_FACTOR,
                                 _append, _brief)
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, apply_overrides, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import Context, build_selector  # noqa: E402


def n_crops(plan) -> int:
    return sum(len(p) for p in plan) if isinstance(plan, list) else len(plan)


def step(model, selector, batch):
    """One batch -> (logits, auxiliary loss or None, crops)."""
    ctx = Context.from_batch(model, batch)
    sel = selector(ctx)
    reuse = not model.training and not torch.is_grad_enabled()     # inference: embed once
    out = model(batch.ms, batch.ss, batch.season, batch.present, sel.plan, sel.logw,
                batch.dates, ctx.seasons if reuse else None)
    return out.logits, sel.aux, n_crops(sel.plan)


@torch.no_grad()
def evaluate(model, selector, task, dl, limit=0):
    model.eval()
    for m in selector.modules():
        m.eval()
    meter, crops, n = task.meter(), 0, 0
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        logits, _, nc = step(model, selector, batch)
        meter.update(logits, batch.target.cuda())
        crops, n = crops + nc, n + batch.target.shape[0]
    return meter.compute(), crops / max(n, 1)


@torch.no_grad()
def measure_compute(model, selector, dl, batches: int) -> dict:
    """Inference cost per sample over the first `batches` batches of dl, everything the arm runs
    at test time (the selector's passes, the crops, the decoder): exp.bench.flop_trace."""
    from exp.bench.flop_trace import count_macs
    model.eval()
    for m in selector.modules():
        m.eval()
    n_samples = crops = 0
    with count_macs() as count:
        for i, batch in enumerate(dl):
            if i >= batches:
                break
            crops += step(model, selector, batch)[2]
            n_samples += batch.target.shape[0]
    n_blocks = len(model.mix.encoder.blocks)
    return {"gmacs": round(count.total / n_samples / 1e9, 3),
            "gmacs_attention": round(count.attn / n_samples / 1e9, 3),
            "tokens_per_block": round(count.token_blocks / n_samples / n_blocks, 1),
            "longest_sequence": count.max_seq, "compute_samples": n_samples,
            "compute_crops_per_sample": round(crops / n_samples, 1)}


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def run(cfg: RunConfig) -> dict:
    task = TASKS[cfg.task]
    cfg.check(task.kind)
    tc = cfg.train
    name = cfg.name()
    smoke = "_smoke" if tc.limit_batches else ""
    state_path = Path(f"checkpoints/{name}{smoke}_state.pt")
    if state_path.exists() and torch.load(state_path, weights_only=False).get("done"):
        print(f"{state_path} says this run is finished; nothing to do")
        return {}
    torch.manual_seed(tc.seed)
    model = MixedResModel(task, cfg.model).cuda()
    if tc.init_ckpt:
        unused = load_legacy(model, tc.init_ckpt)
        print(f"warm start from {tc.init_ckpt} (unused keys: {len(unused)})")
    bs = tc.batch_size or task.batch_size[16]
    dates = cfg.select.level2 is not None and cfg.select.level2.source == "dates"
    dl = {s: loader(task, s, bs, tc.num_workers, s == "train", dates)
          for s in ("train", "val", "test")}
    b0 = next(iter(dl["val"]))
    example = (b0.ms, b0.ss)
    selector = build_selector(cfg.select, model, example)
    nets = nn.ModuleList(selector.modules())
    print(f"run {name}\n  {json.dumps(cfg.to_dict())}", flush=True)

    better = task.meter().higher_is_better
    if tc.init_ckpt:        # loaded weights at the fine-tune rate, new modules at the full rate
        groups = [{"params": list(model.parameters()), "lr": tc.lr * UNFREEZE_LR_FACTOR}]
        if len(nets):
            groups.append({"params": list(nets.parameters()), "lr": tc.lr})
        opt = torch.optim.AdamW(groups)
    else:
        opt = torch.optim.AdamW(list(model.parameters()) + list(nets.parameters()), lr=tc.lr)
    sched = ReduceLROnPlateau(opt, mode="max" if better else "min", factor=SCHEDULER_FACTOR,
                              patience=SCHEDULER_PATIENCE, cooldown=SCHEDULER_COOLDOWN)
    freeze_epochs = 0 if tc.init_ckpt else math.ceil(FREEZE_EPOCH_FRACTION * tc.epochs)
    loss_fn = task.loss().cuda()
    backbone = model.mix.encoder
    res = cfg.select.residual

    def state():
        return {"model": model.state_dict(), "selector": nets.state_dict()}

    start, hours_before, epoch_s = 0, 0.0, None
    best, best_epoch, best_val, best_state = None, -1, None, None
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]["model"])
        nets.load_state_dict(st["model"]["selector"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, hours_before, epoch_s = st["epoch"], st["hours"], st["epoch_s"]
        best, best_epoch, best_val, best_state = (st["best"], st["best_epoch"], st["best_val"],
                                                  st["best_state"])
        print(f"resumed from {state_path} after epoch {start}")
    t_start = time.time()

    for epoch in range(start, tc.epochs):
        if tc.max_hours and epoch_s and (time.time() - t_start + 1.1 * epoch_s) / 3600 > tc.max_hours:
            print(f"STOPPED before epoch {epoch + 1} (train.max_hours); rerun to resume")
            return {}
        e0 = time.time()
        torch.manual_seed(tc.seed + epoch)
        tuning = epoch >= freeze_epochs
        backbone.requires_grad_(tuning)
        if res is not None and res.train == "both":      # the oracle weight 1 -> 0
            selector.scorer.lam = max(0.0, 1.0 - epoch / max(1, round(res.decay * tc.epochs)))
        if epoch == freeze_epochs and not tc.init_ckpt:
            for g in opt.param_groups:
                g["lr"] = tc.lr * UNFREEZE_LR_FACTOR
            print(f"epoch {epoch + 1}: backbone unfrozen, lr -> {tc.lr * UNFREEZE_LR_FACTOR:g}")
        model.train()
        backbone.train(tuning)
        nets.train()
        loss_sum, n, crops, samples = 0.0, 0, 0, 0
        for i, batch in enumerate(dl["train"]):
            if tc.limit_batches and i >= tc.limit_batches:
                break
            target = batch.target.cuda()
            logits, aux, nc = step(model, selector, batch)
            loss = loss_fn(logits, target)
            if aux is not None:
                loss = loss + aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum, n = loss_sum + loss.item(), n + 1
            crops, samples = crops + nc, samples + target.shape[0]
            if i % 100 == 0:
                extra = "" if aux is None else f" | selector aux loss {aux.item():.4f}"
                print(f"  step {i}/{len(dl['train'])} loss {loss.item():.4f}{extra}", flush=True)
        train_s = time.time() - e0
        val, _ = evaluate(model, selector, task, dl["val"], tc.limit_batches)
        sched.step(val["primary"])
        print(f"epoch {epoch + 1}/{tc.epochs} | {samples / train_s:.1f} samples/s "
              f"({train_s / 60:.1f} min) | {crops / max(samples, 1):.0f} crops/sample | "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB\n"
              f"   train_loss {loss_sum / max(n, 1):.4f} | val {_brief(val)}", flush=True)
        if best is None or (val["primary"] > best if better else val["primary"] < best):
            best, best_epoch, best_val = val["primary"], epoch + 1, val
            best_state = {k: {p: t.detach().cpu().clone() for p, t in v.items()}
                          for k, v in state().items()}
        epoch_s = time.time() - e0
        tmp = state_path.with_suffix(".tmp")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "hours": hours_before + (time.time() - t_start) / 3600,
                    "epoch_s": epoch_s, "best": best, "best_epoch": best_epoch,
                    "best_val": best_val, "best_state": best_state}, tmp)
        tmp.replace(state_path)

    if best_state is None:
        best_val, _ = evaluate(model, selector, task, dl["val"], tc.limit_batches)
        best, best_epoch = best_val["primary"], tc.epochs
        best_state = state()
    model.load_state_dict(best_state["model"])
    nets.load_state_dict(best_state["selector"])
    test, test_crops = evaluate(model, selector, task, dl["test"], tc.limit_batches)
    hours = hours_before + (time.time() - t_start) / 3600
    cost = measure_compute(model, selector, dl["val"], tc.compute_batches)
    print(f"TEST {name} (best val epoch {best_epoch}, val {best:.4f}): {_brief(test)} | "
          f"{test_crops:.0f} crops/sample | {cost['gmacs']:.2f} GMACs/sample "
          f"({cost['tokens_per_block']:.0f} tokens/block)", flush=True)
    if not tc.limit_batches:
        torch.save({"config": cfg.to_dict(), "model": best_state["model"],
                    "selector": best_state["selector"]}, f"checkpoints/{name}.pt")
    _append(Path(f"results/{cfg.task}/mixres{smoke}.csv"), {
        "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, **flat(cfg.to_dict()),
        "batch_size": bs, "best_epoch": best_epoch, "hours": round(hours, 2),
        "test_crops_per_sample": round(test_crops, 1), **cost,
        **{f"val_{k}": round(v, 5) for k, v in best_val.items()},
        **{f"test_{k}": round(v, 5) for k, v in test.items()}})
    torch.save({"done": True}, state_path)
    return {"val": best_val, "test": test, "crops": test_crops, "name": name, "cost": cost}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="", help="a RunConfig JSON (as saved in a checkpoint)")
    ap.add_argument("overrides", nargs="*", help="dotted key=value, e.g. select.n=16")
    a = ap.parse_args(argv)
    cfg = RunConfig()
    if a.config:
        with open(a.config) as f:
            cfg = from_dict(RunConfig, json.load(f))
    return run(apply_overrides(cfg, a.overrides))


if __name__ == "__main__":
    main()
