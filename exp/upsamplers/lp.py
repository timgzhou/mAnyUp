"""Linear probe on upsampled frozen OlmoEarth features, for a task in exp/common/tasks.py.

    cached ps16 tile64 features (series) --FeatureUpsampler, guided by the task's own data-->
    ps4 token grid (16 x 16) --> the task's pa2px head (p = 4; time mean, or post - pre)

The head is the one the OlmoEarth LP uses on real ps4 features, so ps16 LP (no upsampling)
and ps4ts64 LP bracket every upsampler. Same protocol as exp.olmoearth.train's LP: one head
per learning rate on a shared forward pass, AdamW + ReduceLROnPlateau, best validation epoch
tested; one CSV row per learning rate in results/<task>/upsampler_lp.csv
(upsampler_lp_shared.csv for --shared_probe).

--shared_probe RUN (UPA / UPMA): no training. Each learning rate's best head from the
bilinear LP run RUN is applied to this upsampler's features on the test set only -- valid
because these upsamplers output convex combinations of the input features (the same space
bilinear outputs), and needed because they optimize per image (~0.8 s each on an L40S).
The test set can be split over jobs: --shard i/n evaluates the i-th of n contiguous parts and
saves its meters; --merge n adds the n parts' meters and writes the CSV rows.

    python -u -m exp.upsamplers.lp --task pastis --upsampler anyup --model_size tiny \
        --lrs 1e-2,5e-3,1e-3,1e-4,3e-5 --epochs 50
"""
import argparse
import copy
import functools
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from exp.common.heads import Pa2PxHead
from exp.common.tasks import TASKS
from exp.olmoearth.train import Arm, _append, _brief
from exp.upsamplers.methods import METHODS, SHARED_PROBE, build, roles
from exp.upsamplers.task_data import OUT_GRID, UpsamplerData, out_geometry


def run_name(a) -> str:
    ck = f"_{Path(a.ckpt).stem}" if a.ckpt else ""
    k = f"_k{a.k}" if a.k is not None else ""
    bn = "_bn" if a.head_norm else ""
    return f"up_{a.task}_{a.model_size}_{a.upsampler}{ck}{k}_lp{bn}_ep{a.epochs}"


@torch.no_grad()
def upsample(up, batch):
    side = {r: f for r, f in batch.feats.items() if r != "lr"}
    return up(batch.feats["lr"], batch.guidance, out_geometry(batch), **side)


@torch.no_grad()
def evaluate(up, heads, task, dl, limit, raw=False):
    """Metrics per head, or its meter (raw=True; mergeable across shards)."""
    for h in heads:
        h.eval()
    meters = [task.meter() for _ in heads]
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        f = upsample(up, batch.to("cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for h, m in zip(heads, meters):
                m.update(h(f.tokens, f.frame_valid).float(), batch.target)
    return meters if raw else [m.compute() for m in meters]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--upsampler", required=True, choices=METHODS)
    ap.add_argument("--ckpt", default=None, help="mAnyUp / timAnyUp checkpoint")
    ap.add_argument("--lrs", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--model_size", default="tiny")
    ap.add_argument("--batch_size", type=int, default=None, help="default: the task's ps16 one")
    ap.add_argument("--shared_probe", default=None,
                    help="test-only with the heads of this bilinear LP run (UPA / UPMA)")
    ap.add_argument("--shard", default=None, help="i/n: --shared_probe on part i of n")
    ap.add_argument("--merge", type=int, default=0, help="--shared_probe: merge n shards")
    ap.add_argument("--fit_steps", type=int, default=50, help="UPA / UPMA per-image fit")
    ap.add_argument("--k", type=int, default=None,
                    help="timAnyUp lookup budget (default: trained; 0 = no lookups)")
    ap.add_argument("--head_norm", action="store_true",
                    help="standardize features before the linear head (Pa2PxHead norm)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()

    task = TASKS[a.task]
    lrs = [float(x) for x in a.lrs.split(",")]
    if a.upsampler in SHARED_PROBE and not a.shared_probe:
        raise SystemExit(f"{a.upsampler} optimizes per image: pass --shared_probe <bilinear run>")
    bs = a.batch_size or task.batch_size[16]
    name = run_name(a) + ("_shared" if a.shared_probe else "")
    smoke = "_smoke" if a.limit_batches else ""
    # shared-probe rows have no validation columns: their own CSV keeps both well-formed
    csv_path = Path(f"results/{a.task}/upsampler_lp{'_shared' if a.shared_probe else ''}"
                    f"{smoke}.csv")
    torch.manual_seed(a.seed)
    if a.k is not None and a.upsampler != "timanyup":
        raise SystemExit("--k is timAnyUp's lookup budget")
    up = build(a.upsampler, task, a.ckpt, a.fit_steps, a.k).cuda().eval()
    print(f"Run: {name} | {up.key} | {up.time_matching} | lrs {lrs} | batch {bs}")
    splits = ("test",) if a.shared_probe else ("train", "val", "test")
    sets = {s: UpsamplerData(task, s, a.model_size, roles(a.upsampler),
                             guidance=up.guidance is not None) for s in splits}
    for s in sets.values():
        if s.ds is not None:
            up.check(s.ds)
    D = sets["test"].embed_dim
    part = dict(sets)
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        N = len(sets["test"])
        part["test"] = torch.utils.data.Subset(sets["test"], range(N * i // n, N * (i + 1) // n))
    dl = {s: DataLoader(part[s], batch_size=bs, shuffle=s == "train", num_workers=a.num_workers,
                        collate_fn=d.collate, pin_memory=True, drop_last=s == "train",
                        persistent_workers=a.num_workers > 0) for s, d in sets.items()}
    p = 64 // OUT_GRID
    better = task.meter().higher_is_better
    arms = []
    for lr in ([] if a.shared_probe else lrs):
        head = Pa2PxHead(task.reduction, D, task.num_outputs, p, norm=a.head_norm).cuda()
        arms.append(Arm(lr, head, list(head.parameters()), task, better))
    t_start = time.time()

    if a.shared_probe:
        shard_file = lambda i, n: Path(f"checkpoints/{name}_shard{i}of{n}.pt")  # noqa: E731
        if a.merge:
            parts = [torch.load(shard_file(i, a.merge), weights_only=False)
                     for i in range(a.merge)]
            meters = [functools.reduce(lambda x, y: x.merge(y), ms) for ms in zip(*parts)]
        else:
            heads = []
            for lr in lrs:
                ck = torch.load(f"checkpoints/{a.shared_probe}_lr{lr:g}.pt",
                                map_location="cpu", weights_only=False)
                heads.append(Pa2PxHead(task.reduction, D, task.num_outputs, p,
                                       norm=a.head_norm).cuda())
                heads[-1].load_state_dict(ck["head"])
            meters = evaluate(up, heads, task, dl["test"], a.limit_batches, raw=True)
            if a.shard:
                torch.save(meters, shard_file(i, n))
                print(f"saved shard {a.shard} ({len(part['test'])} samples)")
                return
        rows = []
        for lr, m in zip(lrs, meters):
            test = m.compute()
            print(f"TEST lr {lr:g} (head of {a.shared_probe}): {_brief(test)}")
            rows.append((lr, None, None, test))
    else:
        loss_fn = task.loss().cuda()
        for epoch in range(a.epochs):
            torch.manual_seed(a.seed + epoch)
            for arm in arms:
                arm.head.train()
            t0, n = time.time(), 0
            for i, batch in enumerate(dl["train"]):
                if a.limit_batches and i >= a.limit_batches:
                    break
                f = upsample(up, batch.to("cuda"))
                for arm in arms:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = loss_fn(arm.head(f.tokens, f.frame_valid).float(), batch.target)
                    arm.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    arm.opt.step()
                    arm.loss_sum, arm.loss_n = arm.loss_sum + loss.item(), arm.loss_n + 1
                n += len(batch.ids)
            train_s = time.time() - t0
            vals = evaluate(up, [arm.head for arm in arms], task, dl["val"], a.limit_batches)
            print(f"epoch {epoch + 1}/{a.epochs} | {n / train_s:.1f} samples/s "
                  f"({train_s / 60:.1f} min) | eval {(time.time() - t0 - train_s) / 60:.1f} min")
            for arm, v in zip(arms, vals):
                loss = arm.loss_sum / max(arm.loss_n, 1)
                arm.loss_sum, arm.loss_n = 0.0, 0
                arm.sched.step(v["primary"])
                print(f"   lr {arm.lr:g}: train_loss {loss:.4f} | val {_brief(v)}")
                if math.isfinite(loss) and arm.improved(v["primary"]):
                    arm.best, arm.best_epoch, arm.best_val = v["primary"], epoch + 1, v
                    arm.state = {"head": copy.deepcopy(arm.head.state_dict())}
        rows = []
        os.makedirs("checkpoints", exist_ok=True)
        for arm in arms:
            if arm.state is None:
                print(f"lr {arm.lr:g}: no finite validation result, nothing to test")
                continue
            arm.head.load_state_dict(arm.state["head"])
            test = evaluate(up, [arm.head], task, dl["test"], a.limit_batches)[0]
            print(f"TEST lr {arm.lr:g} (best val epoch {arm.best_epoch}, val {arm.best:.4f}): "
                  f"{_brief(test)}")
            if not a.limit_batches:
                torch.save({"args": vars(a), "lr": arm.lr, **arm.state},
                           f"checkpoints/{name}_lr{arm.lr:g}.pt")
            rows.append((arm.lr, arm.best_epoch, arm.best_val, test))

    hours = round((time.time() - t_start) / 3600, 2)
    for lr, best_epoch, best_val, test in rows:
        _append(csv_path, {
            "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "task": a.task,
            "model_size": a.model_size, "upsampler": a.upsampler, "key": up.key,
            "time_matching": up.time_matching, "space": up.output_space,
            "ckpt": a.ckpt or "", "probe": a.shared_probe or "trained", "lr": lr,
            "head_norm": a.head_norm,
            "epochs": a.epochs, "batch_size": bs, "best_epoch": best_epoch or "",
            "hours": hours,
            **{f"val_{k}": round(x, 5) for k, x in (best_val or {}).items()},
            **{f"test_{k}": round(x, 5) for k, x in test.items()}})
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
