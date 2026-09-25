"""Linear probe (LP) or full fine-tuning (FFT) of OlmoEarth on a task in exp/common/tasks.py.

Model: OlmoEarthEncoder (per-timestep tokens, `--temporal_mode` series | per_timestep, windows
of `--tile_size` px, `--patch_size`) -> the task's pa2px head (time mean, or post - pre).

  lp   encoder frozen. One head per learning rate in --lrs, all trained on the SAME frozen
       forward pass (each with its own optimizer and schedule), so an LR sweep costs one job.
  fft  one learning rate. OlmoEarth's fine-tuning protocol (olmoearth_pretrain.evals.finetune):
       backbone frozen for the first FREEZE_EPOCH_FRACTION of epochs, then everything trains
       at lr * UNFREEZE_LR_FACTOR.
Both: AdamW, ReduceLROnPlateau on the validation metric, the best validation epoch is kept
and evaluated on test. One CSV row per learning rate in results/<task>/olmoearth.csv.

Resumable: the full training state is saved after every epoch (checkpoints/<run>_state.pt) and
a rerun of the same command continues from it (or exits at once if the run is finished).
With --max_hours the job stops cleanly before an epoch that would not fit, so long runs go
as a chain of shorter jobs (scripts/slurm/sweep.sh).

`--from_cache per_timestep|reduced` (lp only) reads the frozen features from the matching
exp.olmoearth.extract cache instead of running the encoder: the same probe, without the
forward pass. A reduced cache already holds what the head reads, so its head reduces nothing.

    python -u -m exp.olmoearth.train --task pastis --temporal_mode series --patch_size 4 \
        --tile_size 64 --mode lp --lrs 1e-3,1e-4,3e-5 --epochs 50
"""
import os
# fine-tuning allocates many differently sized activations; without this, fragmentation
# alone can exhaust a 48 GB card at a batch that fits
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from exp.common import olmo_bootstrap  # noqa: E402,F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.heads import Pa2PxHead  # noqa: E402
from exp.common.paths import FEATURES  # noqa: E402
from exp.olmoearth.extract import STORES, cache_name  # noqa: E402
from manyup.data.feature_cache import FeatureCache  # noqa: E402
from exp.common.tasks import TASKS, Task  # noqa: E402
from manyup.data import mtmm_collate  # noqa: E402
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput  # noqa: E402

# OlmoEarth's fine-tuning constants (olmoearth_pretrain/evals/finetune/constants.py, 0.1.2)
FREEZE_EPOCH_FRACTION = 0.2
UNFREEZE_LR_FACTOR = 0.1
SCHEDULER_FACTOR, SCHEDULER_PATIENCE, SCHEDULER_COOLDOWN = 0.2, 2, 10


class Collate:
    """In the loader workers: collate, pack into OlmoEarth's input, and make the targets."""

    def __init__(self, task: Task, inputs: OlmoEarthInput):
        self.task, self.inputs = task, inputs

    def __call__(self, items):
        batch = mtmm_collate(items)
        sample, frame_valid = self.inputs.to_olmoearth(batch)
        return sample, frame_valid, self.task.target(batch)


def loader(task, inputs, split, batch_size, num_workers, shuffle):
    return DataLoader(task.dataset(split), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=Collate(task, inputs),
                      persistent_workers=num_workers > 0, drop_last=shuffle)


class CachedTokens(torch.utils.data.Dataset):
    """(tokens, frame_valid, target) per sample of an exp.olmoearth.extract cache."""

    def __init__(self, root, name, split):
        self.cache = FeatureCache(root, name, split)
        self.reduced = self.cache.meta["store"] == "reduced"
        self.target = torch.load(self.cache.split_dir / "target.pt")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        tok = torch.load(self.cache.split_dir / f"{i}.pt")
        fv = torch.ones(1, dtype=torch.bool) if self.reduced else self.cache.frame_valid[i]
        t = self.target[i]
        return tok, fv, (t.long() if t.dtype == torch.int16 else t)


class FromCache(torch.nn.Module):
    """Stands in for the frozen encoder when the tokens come from a cache."""

    def forward(self, tokens, temporal_mode):
        return tokens.cuda(non_blocking=True)


def run_name(a) -> str:
    src = f"_cache{a.from_cache}" if a.from_cache else ""
    return (f"oe_{a.task}_{a.model_size}_{a.temporal_mode}_ps{a.patch_size}ts{a.tile_size}"
            f"_{a.mode}{src}{'_bn' if a.head_norm else ''}_ep{a.epochs}")


class Arm:
    """One head (and, for FFT, the encoder) with its optimizer, schedule and best state."""

    def __init__(self, lr, head, params, task, better):
        self.lr, self.head, self.better = lr, head, better
        self.opt = torch.optim.AdamW(params, lr=lr)
        self.sched = ReduceLROnPlateau(self.opt, mode="max" if better else "min",
                                       factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
                                       cooldown=SCHEDULER_COOLDOWN)
        self.best, self.best_epoch, self.best_val, self.state = None, -1, None, None
        self.loss_sum, self.loss_n = 0.0, 0

    def improved(self, v) -> bool:
        return self.best is None or (v > self.best if self.better else v < self.best)


@torch.no_grad()
def evaluate(encoder, heads, task, dl, mode, limit):
    encoder.eval()
    for h in heads:
        h.eval()
    meters = [task.meter() for _ in heads]
    for i, (sample, fv, target) in enumerate(dl):
        if limit and i >= limit:
            break
        fv, target = fv.cuda(), target.cuda()
        tokens = encoder(sample, mode)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for h, m in zip(heads, meters):
                m.update(h(tokens, fv).float(), target)
    return [m.compute() for m in meters]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--temporal_mode", required=True, choices=OlmoEarthEncoder.TEMPORAL_MODES)
    ap.add_argument("--patch_size", type=int, required=True)
    ap.add_argument("--tile_size", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=("lp", "fft"))
    ap.add_argument("--lrs", required=True, help="comma-separated; fft takes exactly one")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    ap.add_argument("--batch_size", type=int, default=None,
                    help="default: the task's batch size for this patch size")
    ap.add_argument("--micro_batch", type=int, default=None,
                    help="fft: forward/backward in chunks of this many samples, accumulating "
                         "gradients over the batch (same update, less memory)")
    ap.add_argument("--from_cache", default=None, choices=STORES,
                    help="lp: read features from this exp.olmoearth.extract cache")
    ap.add_argument("--head_norm", action="store_true",
                    help="standardize features before the linear head (Pa2PxHead norm)")
    ap.add_argument("--features_root", default=str(FEATURES))
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_batches", type=int, default=0,
                    help="cap train/eval batches per epoch (smoke tests, throughput); "
                         "results then go to a _smoke CSV")
    ap.add_argument("--results_csv", default=None)
    ap.add_argument("--max_hours", type=float, default=None,
                    help="stop (state saved) before an epoch that would end past this")
    a = ap.parse_args()

    task = TASKS[a.task]
    lrs = [float(x) for x in a.lrs.split(",")]
    if a.mode == "fft" and len(lrs) != 1:
        raise SystemExit("fft trains the encoder, so one learning rate per job")
    bs = a.batch_size or task.batch_size[a.patch_size]
    name = run_name(a)
    smoke = "_smoke" if a.limit_batches else ""
    csv_path = Path(a.results_csv or f"results/{a.task}/olmoearth{smoke}.csv")
    print(f"Run: {name} | lrs {lrs} | batch {bs}")
    torch.manual_seed(a.seed)

    reduction = task.reduction
    if a.from_cache:
        if a.mode != "lp":
            raise SystemExit("--from_cache is for lp: fft needs the encoder")
        cname = cache_name(a.task, a.model_size, a.temporal_mode, a.patch_size, a.tile_size,
                           a.from_cache)
        sets = {s: CachedTokens(a.features_root, cname, s) for s in ("train", "val", "test")}
        dl = {s: DataLoader(d, batch_size=bs, shuffle=s == "train", num_workers=a.num_workers,
                            pin_memory=True, persistent_workers=a.num_workers > 0,
                            drop_last=s == "train") for s, d in sets.items()}
        encoder, embed_dim = FromCache(), sets["train"].cache.meta["embed_dim"]
        if sets["train"].reduced:
            reduction = "time_mean"        # over the one stored step: the identity
        print(f"Features from cache {cname}")
    else:
        inputs = OlmoEarthInput(task.olmoearth.modalities, task.olmoearth.impute)
        encoder = OlmoEarthEncoder(MODEL_SIZE_TO_ID[a.model_size], inputs, a.patch_size,
                                   a.tile_size).cuda()
        encoder.check(task.dataset("train"), a.temporal_mode)
        dl = {s: loader(task, inputs, s, bs, a.num_workers, s == "train")
              for s in ("train", "val", "test")}
        embed_dim = encoder.encoder.embedding_size
    better = task.meter().higher_is_better
    state_path = Path(f"checkpoints/{name}_lr{'-'.join(f'{x:g}' for x in lrs)}{smoke}_state.pt")
    if state_path.exists() and torch.load(state_path, weights_only=False).get("done"):
        print(f"{state_path} says this run is finished; nothing to do")
        return

    arms = []
    for lr in lrs:
        head = Pa2PxHead(reduction, embed_dim, task.num_outputs, a.patch_size,
                         norm=a.head_norm).cuda()
        # fft: the encoder is in the (single) optimizer from the start; it has no gradients,
        # so AdamW leaves it alone, until it is unfrozen
        params = list(head.parameters()) + (list(encoder.parameters()) if a.mode == "fft" else [])
        arms.append(Arm(lr, head, params, task, better))
    freeze_epochs = math.ceil(FREEZE_EPOCH_FRACTION * a.epochs) if a.mode == "fft" else a.epochs
    loss_fn = task.loss().cuda()
    heads = [arm.head for arm in arms]
    start_epoch, hours_before, epoch_s = 0, 0.0, None
    if state_path.exists():
        start_epoch, hours_before, epoch_s = _load_state(state_path, encoder, arms, a.mode)
        print(f"Resumed from {state_path} after epoch {start_epoch}")
    t_start = time.time()

    for epoch in range(start_epoch, a.epochs):
        if a.max_hours and epoch_s and (time.time() - t_start + 1.1 * epoch_s) / 3600 > a.max_hours:
            print(f"STOPPED before epoch {epoch + 1}: it would not fit in --max_hours "
                  f"{a.max_hours}; rerun the same command to resume")
            return
        e0 = time.time()
        torch.manual_seed(a.seed + epoch)          # same shuffle order with or without resuming
        tuning = a.mode == "fft" and epoch >= freeze_epochs
        encoder.requires_grad_(tuning)
        if a.mode == "fft" and epoch == freeze_epochs:
            arms[0].opt.param_groups[0]["lr"] = lrs[0] * UNFREEZE_LR_FACTOR
            print(f"epoch {epoch + 1}: backbone unfrozen, lr -> {lrs[0] * UNFREEZE_LR_FACTOR:g}")
        encoder.train(tuning)
        for h in heads:
            h.train()
        t0, n = time.time(), 0
        for i, (sample, fv, target) in enumerate(dl["train"]):
            if a.limit_batches and i >= a.limit_batches:
                break
            fv, target = fv.cuda(), target.cuda()
            B = target.shape[0]
            if tuning:                     # fft, one arm: accumulate over micro-batches
                arm = arms[0]
                arm.opt.zero_grad(set_to_none=True)
                total = 0.0
                for lo in range(0, B, a.micro_batch or B):
                    hi = min(lo + (a.micro_batch or B), B)
                    tokens = encoder(_rows(sample, lo, hi), a.temporal_mode)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = loss_fn(arm.head(tokens, fv[lo:hi]).float(), target[lo:hi])
                    (loss * (hi - lo) / B).backward()
                    total += loss.item() * (hi - lo) / B
                arm.opt.step()
                arm.loss_sum, arm.loss_n = arm.loss_sum + total, arm.loss_n + 1
            else:                          # frozen encoder: one forward feeds every head
                with torch.no_grad():
                    tokens = encoder(sample, a.temporal_mode)
                for arm in arms:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = loss_fn(arm.head(tokens, fv).float(), target)
                    arm.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    arm.opt.step()
                    arm.loss_sum, arm.loss_n = arm.loss_sum + loss.item(), arm.loss_n + 1
            n += B
        train_s = time.time() - t0

        vals = evaluate(encoder, heads, task, dl["val"], a.temporal_mode, a.limit_batches)
        print(f"epoch {epoch + 1}/{a.epochs} | {n / train_s:.1f} samples/s train "
              f"({train_s / 60:.1f} min) | eval {(time.time() - t0 - train_s) / 60:.1f} min | "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        for arm, v in zip(arms, vals):
            loss = arm.loss_sum / max(arm.loss_n, 1)
            arm.loss_sum, arm.loss_n = 0.0, 0
            arm.sched.step(v["primary"])
            print(f"   lr {arm.lr:g}: train_loss {loss:.4f} | val {_brief(v)}")
            if not math.isfinite(loss):
                print(f"   lr {arm.lr:g}: loss is not finite")
            if arm.improved(v["primary"]):
                arm.best, arm.best_epoch, arm.best_val = v["primary"], epoch + 1, v
                arm.state = {"head": copy.deepcopy(arm.head.state_dict())}
                if a.mode == "fft":
                    arm.state["encoder"] = {k: t.detach().cpu().clone()
                                            for k, t in encoder.encoder.state_dict().items()}
        epoch_s = time.time() - e0
        hours = hours_before + (time.time() - t_start) / 3600
        _save_state(state_path, epoch + 1, hours, epoch_s, encoder, arms, a.mode)

    # test every arm at its best validation epoch
    elapsed = (hours_before * 3600) + time.time() - t_start
    os.makedirs("checkpoints", exist_ok=True)
    for arm in arms:
        if arm.state is None:
            print(f"lr {arm.lr:g}: no finite validation result, nothing to test")
            continue
        arm.head.load_state_dict(arm.state["head"])
        if "encoder" in arm.state:
            encoder.encoder.load_state_dict(arm.state["encoder"])
        test = evaluate(encoder, [arm.head], task, dl["test"], a.temporal_mode,
                        a.limit_batches)[0]
        print(f"TEST lr {arm.lr:g} (best val epoch {arm.best_epoch}, val {arm.best:.4f}): "
              f"{_brief(test)}")
        if not a.limit_batches:
            torch.save({"args": vars(a), "lr": arm.lr, **arm.state},
                       f"checkpoints/{name}_lr{arm.lr:g}.pt")
        row = {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "task": a.task,
               "model_size": a.model_size, "temporal_mode": a.temporal_mode,
               "patch_size": a.patch_size, "tile_size": a.tile_size, "mode": a.mode,
               "lr": arm.lr, "epochs": a.epochs, "batch_size": bs, "head_norm": a.head_norm,
               "best_epoch": arm.best_epoch, "hours": round(elapsed / 3600, 2),
               **{f"val_{k}": round(x, 5) for k, x in arm.best_val.items()},
               **{f"test_{k}": round(x, 5) for k, x in test.items()}}
        _append(csv_path, row)
    print(f"Wrote {csv_path}")
    torch.save({"done": True}, state_path)          # the full state is no longer needed


def _save_state(path, epochs_done, hours, epoch_s, encoder, arms, mode):
    """Everything needed to continue after `epochs_done` epochs; written atomically."""
    state = {"epochs_done": epochs_done, "hours": hours, "epoch_s": epoch_s, "arms": [
        {"head": a.head.state_dict(), "opt": a.opt.state_dict(), "sched": a.sched.state_dict(),
         "best": a.best, "best_epoch": a.best_epoch, "best_val": a.best_val, "state": a.state}
        for a in arms]}
    if mode == "fft":
        state["encoder"] = encoder.encoder.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _load_state(path, encoder, arms, mode):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if mode == "fft":
        encoder.encoder.load_state_dict(state["encoder"])
    for arm, s in zip(arms, state["arms"], strict=True):
        arm.head.load_state_dict(s["head"])
        arm.opt.load_state_dict(s["opt"])
        arm.sched.load_state_dict(s["sched"])
        arm.best, arm.best_epoch, arm.best_val = s["best"], s["best_epoch"], s["best_val"]
        arm.state = s["state"]
    return state["epochs_done"], state["hours"], state["epoch_s"]


def _rows(sample, lo, hi):
    """Samples lo:hi of a packed OlmoEarth batch (every field is batch-first)."""
    return sample._replace(**{k: v[lo:hi] for k, v in sample.as_dict().items() if v is not None})


def _brief(metrics: dict) -> str:
    """Metrics for the log; per-class IoUs only when there are a few (all go to the CSV)."""
    per_class = [k for k in metrics if k.startswith("iou_")]
    keep = [k for k in metrics if k != "primary" and (len(per_class) <= 3 or k not in per_class)]
    return f"primary {metrics['primary']:.4f} | " + " ".join(f"{k} {metrics[k]:.4f}" for k in keep)


def _append(path: Path, row: dict) -> None:
    """Append a row; a row with columns the file lacks widens its header (older rows get
    empty cells), so a new field never misaligns the CSV. Locked: jobs append concurrently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, fields = [], []
        if path.exists():
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                fields, rows = list(r.fieldnames or []), list(r)
        extra = [k for k in row if k not in fields]
        if not extra:
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            return
        fields += extra
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows + [row])
        tmp.replace(path)


if __name__ == "__main__":
    main()
