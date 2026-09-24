"""Train the UTAE baseline on a task in exp/common/tasks.py (data: Task.utae_dataset).

Several modalities are early-fused: the others are resampled onto the first one's dates
(Task.utae.modalities order) and concatenated by channel, one UTAE (the best PASTIS setup in
results/pastis/utae_pastis.csv). Output conv sized to the task (classes, or 1 for regression);
loss and metrics are the task's, as for OlmoEarth. AdamW at a constant learning rate, the
best validation epoch is kept and evaluated on test; one CSV row in results/<task>/utae.csv.

    python -u -m exp.utae.train --task biomassters --lr 1e-3 --epochs 100
"""
import argparse
import copy
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from exp.common.tasks import TASKS, Task
from exp.utae.fusion import build_model
from exp.utae.inputs import UTAEInput
from manyup.data import mtmm_collate


class Collate:
    def __init__(self, task: Task, pack: UTAEInput):
        self.task, self.pack = task, pack

    def __call__(self, items):
        batch = mtmm_collate(items)
        data, dates = self.pack(batch)
        return data, dates, self.task.target(batch)


def loader(task, split, batch_size, num_workers, shuffle):
    u = task.utae
    pack = UTAEInput(list(u.modalities), split, norm=u.norm, reference_day=u.reference_day)
    return DataLoader(task.utae_dataset(split), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=Collate(task, pack),
                      persistent_workers=num_workers > 0, drop_last=shuffle)


@torch.no_grad()
def evaluate(model, task, dl, limit):
    model.eval()
    meter = task.meter()
    for i, (data, dates, target) in enumerate(dl):
        if limit and i >= limit:
            break
        out = model({k: v.cuda() for k, v in data.items()}, {k: v.cuda() for k, v in dates.items()})
        meter.update(out.float(), target.cuda())
    return meter.compute()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=None, help="default: the task's")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_batches", type=int, default=0,
                    help="cap train/eval batches per epoch; results then go to a _smoke CSV")
    a = ap.parse_args()

    task = TASKS[a.task]
    mods = list(task.utae.modalities)
    bs = a.batch_size or task.utae.batch_size
    name = f"utae_{a.task}_{'_'.join(m.lower() for m in mods)}_lr{a.lr:g}_ep{a.epochs}"
    csv_path = Path(f"results/{a.task}/utae{'_smoke' if a.limit_batches else ''}.csv")
    print(f"Run: {name} | batch {bs}")
    torch.manual_seed(a.seed)

    dl = {s: loader(task, s, bs, a.num_workers, s == "train") for s in ("train", "val", "test")}
    spec = dl["train"].dataset.SPEC
    channels = {m: len(spec.modalities[m].bands) for m in mods}
    model = build_model(mods, "early", task.num_outputs, channels, primary=mods[0]).cuda()
    print(f"UTAE {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"inputs {channels}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loss_fn = task.loss().cuda()
    better = task.meter().higher_is_better
    best, best_epoch, best_val, best_state = None, -1, None, None
    t_start = time.time()

    for epoch in range(a.epochs):
        model.train()
        t0, losses = time.time(), []
        for i, (data, dates, target) in enumerate(dl["train"]):
            if a.limit_batches and i >= a.limit_batches:
                break
            out = model({k: v.cuda() for k, v in data.items()},
                        {k: v.cuda() for k, v in dates.items()})
            loss = loss_fn(out.float(), target.cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_s = time.time() - t0
        v = evaluate(model, task, dl["val"], a.limit_batches)
        print(f"epoch {epoch + 1}/{a.epochs} | train_loss {sum(losses) / max(len(losses), 1):.4f}"
              f" | {train_s / 60:.1f} min | val primary {v['primary']:.4f}")
        if best is None or (v["primary"] > best if better else v["primary"] < best):
            best, best_epoch, best_val = v["primary"], epoch + 1, v
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test = evaluate(model, task, dl["test"], a.limit_batches)
    print(f"TEST (best val epoch {best_epoch}, val {best:.4f}): "
          + " ".join(f"{k} {x:.4f}" for k, x in test.items() if not k.startswith("iou_")))
    if not a.limit_batches:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"args": vars(a), "model": best_state}, f"checkpoints/{name}.pt")
    row = {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "task": a.task,
           "modalities": "+".join(mods), "lr": a.lr, "epochs": a.epochs, "batch_size": bs,
           "best_epoch": best_epoch, "hours": round((time.time() - t_start) / 3600, 2),
           **{f"val_{k}": round(x, 5) for k, x in best_val.items()},
           **{f"test_{k}": round(x, 5) for k, x in test.items()}}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
