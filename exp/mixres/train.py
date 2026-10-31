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
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import (FREEZE_EPOCH_FRACTION, SCHEDULER_COOLDOWN,  # noqa: E402
                                 SCHEDULER_FACTOR, SCHEDULER_PATIENCE, UNFREEZE_LR_FACTOR,
                                 _append, _brief)
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, apply_overrides, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.selectors import (Context, Quadrant, SeasonTemperature,  # noqa: E402
                                     build_selector, pixel_entropy, season_maps)
from manyup.mixres.steps import StepThreshold, build_step_selector  # noqa: E402


def n_crops(plan) -> int:
    return sum(len(p) for p in plan) if isinstance(plan, list) else len(plan)


def step(model, selector, batch, season_loss: float = 0.0, detach: bool = False):
    """One batch -> (logits, auxiliary loss or None, crops). season_loss: + that weight x the
    cross-entropy of every used season token's own map (training only)."""
    ctx = Context.from_batch(model, batch)
    sel = selector(ctx)
    reuse = not model.training and not torch.is_grad_enabled()     # inference: embed once
    if reuse and ctx.seasons is not None and n_crops(sel.plan) == 0:
        return ctx.out0.logits, sel.aux, 0      # no crops: the crop-free pass IS the output
    out = model(batch.ms, batch.ss, batch.season, batch.present, sel.plan, sel.logw,
                batch.dates, ctx.seasons if reuse else None, ctx.lr_enc if reuse else None)
    aux = sel.aux
    if model.exit is not None and model.training:      # the exit head tracks the full pass
        ex = model.exit_loss(batch.ms, batch.ss, batch.season, batch.present)
        aux = ex if aux is None else aux + ex
    if season_loss and model.training:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = season_maps(model, out.lr_F.detach() if detach else out.lr_F)   # (B NS, C, 64, 64)
        t = batch.target.cuda().long().repeat_interleave(ctx.NS, 0)
        ce = F.cross_entropy(m.float(), t, ignore_index=-1, reduction="none")
        ok = (t != -1) & ctx.used.flatten()[:, None, None]
        sl = season_loss * ce[ok].mean()
        aux = sl if aux is None else aux + sl
    return out.logits, aux, n_crops(sel.plan)


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
def fit_temperatures(model, selector, dl, limit=0):
    """select.entropy_T_fit=val: fit every offline SeasonTemperature of the selector by NLL on
    the val split's season maps (the model's crop-free pass), before the epoch's val score."""
    temps = [m for m in selector.modules() if isinstance(m, SeasonTemperature) and not m.online]
    if not temps:
        return
    model.eval()
    maps = []
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        ctx = Context.from_batch(model, batch)
        lr_F = ctx.out0.lr_F
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = season_maps(model, lr_F)
        maps.append((m.view(ctx.B, ctx.NS, *m.shape[1:]).half(), batch.target.cuda(), ctx.used))
    for t in temps:
        t.fit(maps)
    print(f"   season T (val fit): {[round(v, 3) for v in temps[0].logT.exp().tolist()]}",
          flush=True)


@torch.no_grad()
def val_threshold(model, selector, dl, limit=0) -> bool:
    """A global-mode Quadrant: replace its spatial threshold table (the training batches' EMA)
    by the val split's quantiles of the same uncertainty, so the test set gets ~q_s of its
    windows (the model is more certain on its training set: the training table sits low and
    over-crops held-out tiles; exp.mixres.threshold_split). -> whether it applied."""
    if not isinstance(selector, (Quadrant, StepThreshold)) or selector.cfg.mode != "global":
        return False
    model.eval()
    u = []
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        ctx = Context.from_batch(model, batch)
        u.append(selector.threshold_values(ctx).float().cpu())
    u = torch.cat(u)
    u = u[torch.randperm(len(u))[:4_000_000]]
    lv = torch.linspace(0, 1, selector.state.LEVELS)
    selector.state.tab_s.copy_(torch.quantile(u, lv).to(selector.state.tab_s.device))
    return True


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


def set_attn(model, attn: str):
    """Switch who attends to whom in the crop pass (the weights are shared by every mode)."""
    model.mix.attn = attn
    model.cfg = replace(model.cfg, attn=attn)


def oracle_selector(cfg: RunConfig, model, example):
    """The selector schedule's stage 1: the location oracle (the legacy oracle_L0_k*_loc runs)
    at stage1_n windows of the crop grid, every present month."""
    s = replace(cfg.select, scorer="label", allocator="top", n=cfg.train.stage1_n,
                loc_stride=cfg.select.crop_px, quadrant=None, time_scorer=None,
                time_allocator=None, residual=None, level2=None, route_alpha=None)
    return build_selector(s, model, example)


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
    if tc.init_exit:
        f, key = tc.init_exit.rsplit(":", 1)
        sd = torch.load(f"checkpoints/{f}.pt", map_location="cpu", weights_only=False)
        model.exit.load_state_dict({k[len(key) + 1:]: v for k, v in sd.items()
                                    if k.startswith(key + ".")})
        print(f"exit head from {tc.init_exit}")
    bs = tc.batch_size or task.batch_size[16]
    dates = cfg.select.level2 is not None and cfg.select.level2.source == "dates"
    dl = {s: loader(task, s, bs, tc.num_workers, s == "train", dates,
                    tc.epoch_samples if s == "train" else 0)
          for s in ("train", "val", "test")}
    b0 = next(iter(dl["val"]))
    example = (b0.ms, b0.ss)
    build = build_step_selector if task.per_step else build_selector
    selector = build(cfg.select, model, example)
    nets = nn.ModuleList(selector.modules())
    if tc.init_selector:
        nets.load_state_dict(torch.load(f"checkpoints/{tc.init_selector}.pt", map_location="cpu",
                                        weights_only=False)["selector"])
        print(f"selector networks from {tc.init_selector}")
    print(f"run {name}\n  {json.dumps(cfg.to_dict())}", flush=True)

    better = task.meter().higher_is_better
    if tc.freeze_model:     # only the selector's networks learn; the model stays as loaded
        model.requires_grad_(False)
        if tc.epochs and not len(nets):
            raise ValueError("train.freeze_model with epochs > 0: the selector has nothing to train")
    S1 = tc.stage1_epochs           # selector schedule: epochs [0, S1) the location oracle
    oracle = oracle_selector(cfg, model, example) if S1 else None
    stage_attn = lambda e: tc.stage1_attn if e < S1 else cfg.model.attn       # noqa: E731

    def make_opt(warm: bool):
        if tc.freeze_model:
            opt = torch.optim.AdamW(list(nets.parameters()) or [torch.zeros(1)],
                                    lr=tc.lr)
        elif warm:          # loaded weights at the fine-tune rate, new modules at the full rate
            groups = [{"params": list(model.parameters()), "lr": tc.lr * UNFREEZE_LR_FACTOR}]
            if any(True for _ in nets.parameters()):
                groups.append({"params": list(nets.parameters()), "lr": tc.lr})
            opt = torch.optim.AdamW(groups)
        else:
            opt = torch.optim.AdamW(list(model.parameters()) + list(nets.parameters()),
                                    lr=tc.lr)
        return opt, ReduceLROnPlateau(opt, mode="max" if better else "min",
                                      factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
                                      cooldown=SCHEDULER_COOLDOWN)

    freeze_epochs = (0 if tc.init_ckpt
                     else math.ceil(FREEZE_EPOCH_FRACTION * (S1 or tc.epochs)))
    loss_fn = task.loss().cuda()
    backbone = model.mix.encoder
    res = cfg.select.residual

    def state():
        return {"model": model.state_dict(), "selector": nets.state_dict()}

    start, hours_before, epoch_s = 0, 0.0, None
    best, best_epoch, best_val, best_state = None, -1, None, None
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        start = st["epoch"]
    opt, sched = make_opt(bool(tc.init_ckpt) or bool(S1 and start > S1))   # (= S1: the loop restarts it)
    set_attn(model, stage_attn(start))
    if state_path.exists():
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
        if S1 and epoch == S1:      # stage 2: the configured selector, a warm restart
            opt, sched = make_opt(True)
            set_attn(model, cfg.model.attn)
            print(f"epoch {epoch + 1}: stage 2 (selector {cfg.select.scorer}"
                  f"{' quadrant' if cfg.select.quadrant else ''}, attn {cfg.model.attn}), "
                  f"optimizer restarted, lr {tc.lr * UNFREEZE_LR_FACTOR:g}", flush=True)
        active = oracle if S1 and epoch < S1 else selector
        e0 = time.time()
        torch.manual_seed(tc.seed + epoch)
        tuning = epoch >= freeze_epochs and not tc.freeze_model
        backbone.requires_grad_(tuning)
        if res is not None and res.train == "both":      # the oracle weight 1 -> 0
            selector.scorer.lam = max(0.0, 1.0 - epoch / max(1, round(res.decay * tc.epochs)))
        if tc.oracle_mix:           # the oracle's share of the picks, p -> 0 (or constant)
            span = round(tc.oracle_anneal * tc.epochs)
            selector.oracle_unit = tc.oracle_mix
            selector.oracle_p = (tc.oracle_p * max(0.0, 1.0 - epoch / span) if span
                                 else tc.oracle_p)
            print(f"epoch {epoch + 1}: oracle mix {tc.oracle_mix} p {selector.oracle_p:.3f}",
                  flush=True)
        if epoch == freeze_epochs and not tc.init_ckpt:
            for g in opt.param_groups:
                g["lr"] = tc.lr * UNFREEZE_LR_FACTOR
            print(f"epoch {epoch + 1}: backbone unfrozen, lr -> {tc.lr * UNFREEZE_LR_FACTOR:g}")
        model.train(not tc.freeze_model)
        backbone.train(tuning)
        nets.train()
        loss_sum, n, crops, samples = 0.0, 0, 0, 0
        for i, batch in enumerate(dl["train"]):
            if tc.limit_batches and i >= tc.limit_batches:
                break
            target = batch.target.cuda()
            if tc.freeze_model:     # the selector's own loss only: no crop pass needed
                sel = active(Context.from_batch(model, batch))
                if sel.aux is None:
                    raise ValueError("train.freeze_model: the selector gives no training loss")
                loss, aux, nc = sel.aux, sel.aux, n_crops(sel.plan)
            else:
                logits, aux, nc = step(model, active, batch, tc.season_loss, tc.season_loss_detach)
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
                flip = getattr(active.scorer, "last_flip", None)
                extra += "" if flip is None else f" | flipped {flip:.1%} of locations"
                q = getattr(active, "last", None)
                extra += "" if not q else " | " + " ".join(f"{k} {v:.3f}" for k, v in q.items())
                print(f"  step {i}/{len(dl['train'])} loss {loss.item():.4f}{extra}", flush=True)
        train_s = time.time() - e0
        fit_temperatures(model, active, dl["val"], tc.limit_batches)
        val, _ = evaluate(model, active, task, dl["val"], tc.limit_batches)
        sched.step(val["primary"])
        print(f"epoch {epoch + 1}/{tc.epochs} | {samples / train_s:.1f} samples/s "
              f"({train_s / 60:.1f} min) | {crops / max(samples, 1):.0f} crops/sample | "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB\n"
              f"   train_loss {loss_sum / max(n, 1):.4f} | val {_brief(val)}", flush=True)
        if epoch < S1:              # stage 1 picks crops with the label: never the result
            pass
        elif best is None or (val["primary"] > best if better else val["primary"] < best):
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
    thr = "val" if val_threshold(model, selector, dl["val"], tc.limit_batches) else ""
    if thr:
        best_state["selector"] = {k: v.detach().cpu().clone() for k, v in nets.state_dict().items()}
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
        **({"test_threshold": thr} if thr else {}),
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
