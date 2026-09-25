"""Train mAnyUp or timAnyUp on a task's cached features (exp/upsamplers/task_data.py).

    lr      ps16 tile64 series        (T, 4, 4, D)    input
    hr      ps4 tile4 per_timestep    (T, 16, 16, D)  timAnyUp's side feature
    target  ps4 tile64 series         (T, 16, 16, D)  what the output should match
    guidance: the task's full guidance stack (Task.guidance) from its own samples

mAnyUp follows the task's time handling (exp/upsamplers/methods.py): time-pooled features,
guidance and target where the head averages over time, else every timestep on its own.
manyup_t is mAnyUp per timestep on every task: timAnyUp without the high-res lookup.
Loss: Cosine_MSE (AnyUp's), as legacy exp/upsamplers/train_manyup.py with down_reg 0, no
projector, window 1.0, transform depth 0 (the setup of the checkpoints in use).

timAnyUp is always per timestep, with the three losses of legacy train_timanyup.py:
    recon      Cosine_MSE(project(blend(upsample, transform(hr), top-k mask)), target)
    query      BCE(query logits, rank of the projected upsample's per-location error)
    transform  Cosine_MSE(transform(hr), upsample) at the lowest-error locations
(both auxiliary targets detached). The lookup budget k is a fraction of the T x 16 x 16
locations (legacy: 512 of 3072 = 1/6), so it means the same for T = 12 and T = 2.

AdamW with linear warmup + cosine decay per step, bf16 autocast. The checkpoint with the best
validation reconstruction loss is kept (checkpoints/upsamplers/<run>.pt, loadable with
MAnyUp / TimAnyUp.from_checkpoint); resumable per epoch, --max_hours stops cleanly.

    python -u -m exp.upsamplers.train --task pastis --method manyup --epochs 30
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exp.common.tasks import TASKS
from exp.olmoearth.train import _append
from exp.upsamplers.common import warmup_cosine
from exp.upsamplers.methods import pooled
from exp.upsamplers.task_data import UpsamplerData, out_geometry
from manyup.data.features import time_mean
from manyup.data.guidance import GuidanceSpec, spec_to_dict
from manyup.loss import Cosine_MSE, cosmse_map
from manyup.mAnyUp import mAnyUp
from manyup.timAnyUp import TimAnyUp as TimModel
from manyup.timAnyUp import blend, random_mask, rank_normalize, topk_mask, vectors_at
from manyup.upsamplers import MAnyUp, TimAnyUp

cos_mse = Cosine_MSE()


def flat(x):
    """(B, T, H, W, D) tokens -> (B * T, D, H, W) maps, the layout the losses read."""
    return x.flatten(0, 1).permute(0, 3, 1, 2).float()


def manyup_losses(up, batch):
    lr, tgt = batch.feats["lr"], batch.feats["target"]
    out = up(lr, batch.guidance, out_geometry(batch)).tokens               # (B, T', 16, 16, D)
    if up.time_matching == "pooled":
        want = time_mean(tgt.tokens.float(), tgt.frame_valid)[:, None]
        src = time_mean(lr.tokens.float(), lr.frame_valid)[:, None]
    else:
        want, src = tgt.tokens, lr.tokens
    bil = F.interpolate(flat(src), size=want.shape[2:4], mode="bilinear", align_corners=False)
    loss = cos_mse(flat(out), flat(want))["total"]
    with torch.no_grad():
        c_bil = cos_mse(bil, flat(want))["total"]
    return loss, {"recon": loss, "c_bilinear": c_bil}


def timanyup_losses(up, batch, a):
    """Legacy train_timanyup._forward_losses on the wrapper's model and guidance."""
    m = up.model
    lr, hr, tgt = batch.feats["lr"], batch.feats["hr"], batch.feats["target"]
    lrhc = lr.tokens.float().permute(0, 1, 4, 2, 3).contiguous()           # (B, T, D, h, w)
    hrlc = hr.tokens.float().permute(0, 1, 4, 2, 3).contiguous()
    hrhc = tgt.tokens.float().permute(0, 1, 4, 2, 3).contiguous()
    guide = up.build_guidance(batch.guidance, lr).to(lrhc.device)          # (B, T, C, H, W)
    size = tuple(hrhc.shape[-2:])
    lrhc_up = m.upsample(lrhc, guide, size)
    hrlc_t = m.transform(hrlc)
    scores = m.query(lrhc, size, lrhc_up=lrhc_up)
    fl = lambda x: x.flatten(0, 1).float()                                  # noqa: E731
    up_f, hr_f = fl(lrhc_up), fl(hrhc)

    mask = topk_mask(scores.detach().float(), up.k)
    l_recon = cos_mse(fl(m.project(blend(lrhc_up, hrlc_t, mask))), hr_f)["total"]

    up_proj_f = fl(m.project(lrhc_up))
    err = cosmse_map(up_proj_f, hr_f).reshape(scores.shape).detach()
    l_query = F.binary_cross_entropy_with_logits(scores.float(), rank_normalize(err))

    B = err.shape[0]
    flat_err = err.reshape(B, -1)
    n_fit = max(1, int(flat_err.shape[1] * a.transform_frac))
    fit = torch.zeros_like(flat_err, dtype=torch.bool).scatter_(
        1, flat_err.argsort(dim=1)[:, :n_fit], True)
    fit = fit.reshape(err.shape)
    l_transform = cos_mse(vectors_at(hrlc_t, fit)[..., None, None],
                          vectors_at(lrhc_up.detach(), fit)[..., None, None])["total"]
    loss = l_recon + a.lambda_query * l_query + a.lambda_transform * l_transform
    with torch.no_grad():
        bil = F.interpolate(fl(lrhc), size=size, mode="bilinear", align_corners=False)
        rfused = blend(lrhc_up, hrlc_t, random_mask(scores.float(), up.k))
        logs = {"recon": l_recon, "query": l_query, "transform": l_transform,
                "c_upsample": cos_mse(up_proj_f, hr_f)["total"],
                "c_random": cos_mse(fl(m.project(rfused)), hr_f)["total"],
                "c_bilinear": cos_mse(bil, hr_f)["total"]}
    return loss, logs


def build(a, task, D, T):
    if a.method in ("manyup", "manyup_t"):
        per_t = a.method == "manyup_t" or not pooled(task)
        spec = GuidanceSpec(task.guidance, None if per_t else "mean")
        model = mAnyUp(input_dim=spec.channels, qk_dim=a.qk_dim, feat_dim=D,
                       transform_depth=0, window_ratio=a.window_ratio)
        return MAnyUp(model, spec, output_space="input", name=a.method), spec
    spec = GuidanceSpec(task.guidance, None)
    model = TimModel(input_dim=spec.channels, qk_dim=a.qk_dim, feat_dim=D, num_frames=T,
                     transform_depth=0, window_ratio=a.window_ratio, query_input=a.query_input)
    k = max(1, round(a.k_frac * T * 16 * 16))
    return TimAnyUp(model, spec, k), spec


def checkpoint(a, up, spec, D, T, epoch, val):
    """In the format MAnyUp / TimAnyUp.from_checkpoint read."""
    ck = {"model": up.model.state_dict(), "guidance": spec_to_dict(spec), "input_dim":
          spec.channels, "feat_dim": D, "qk_dim": a.qk_dim, "window_ratio": a.window_ratio,
          "transform_depth": 0, "epoch": epoch, "val": val,
          "args": {**vars(a), "k": getattr(up, "k", None)}}
    if a.method != "timanyup":
        ck.update(arch="manyup", proj_head=None)
    else:
        ck.update(num_frames=T, query_input=a.query_input, k=up.k)
    return ck


@torch.no_grad()
def evaluate(up, dl, losses, limit):
    up.eval()
    agg, n = {}, 0
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, logs = losses(batch.to("cuda"))
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    up.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--method", required=True, choices=("manyup", "manyup_t", "timanyup"))
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--model_size", default="tiny")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--qk_dim", type=int, default=128)
    ap.add_argument("--window_ratio", type=float, default=1.0)
    ap.add_argument("--k_frac", type=float, default=1 / 6, help="timAnyUp lookup budget")
    ap.add_argument("--transform_frac", type=float, default=0.25)
    ap.add_argument("--query_input", default="bilinear", choices=("bilinear", "upsampled"))
    ap.add_argument("--lambda_query", type=float, default=1.0)
    ap.add_argument("--lambda_transform", type=float, default=1.0)
    ap.add_argument("--val_batches", type=int, default=200, help="per epoch; 0 = all")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=None)
    a = ap.parse_args()

    task = TASKS[a.task]
    roles = ("lr", "hr", "target") if a.method == "timanyup" else ("lr", "target")
    sets = {s: UpsamplerData(task, s, a.model_size, roles) for s in ("train", "val")}
    D = sets["train"].embed_dim
    T = sets["train"][0][1]["lr"].shape[0]
    up, spec = build(a, task, D, T)
    up = up.cuda().train()
    up.check(sets["train"].ds)
    name = f"{a.method}_{a.task}_{a.model_size}_ep{a.epochs}"
    smoke = "_smoke" if a.limit_batches else ""
    out = Path(f"checkpoints/upsamplers/{name}{smoke}.pt")
    state_path = out.with_name(out.stem + "_state.pt")
    print(f"Run: {name} | {up.key} | {up.time_matching} | T {T} D {D}"
          + (f" | k {up.k}" if a.method == "timanyup" else ""))
    dl = {s: DataLoader(d, batch_size=a.batch_size, shuffle=s == "train", drop_last=s == "train",
                        num_workers=a.num_workers, collate_fn=d.collate, pin_memory=True,
                        persistent_workers=a.num_workers > 0) for s, d in sets.items()}
    if a.method != "timanyup":
        losses = lambda b: manyup_losses(up, b)                           # noqa: E731
    else:
        losses = lambda b: timanyup_losses(up, b, a)                      # noqa: E731
    steps = a.epochs * (a.limit_batches or len(dl["train"]))
    opt = torch.optim.AdamW(up.parameters(), lr=a.lr)
    sched = warmup_cosine(opt, steps, a.warmup_frac, a.lr, a.lr_min)
    start, best, epoch_s = 0, math.inf, None
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st.get("done"):
            print(f"{state_path} says this run is finished; nothing to do")
            return
        up.model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, epoch_s = st["epoch"], st["best"], st["epoch_s"]
        print(f"Resumed after epoch {start}")
    t_start = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start, a.epochs):
        if a.max_hours and epoch_s and (time.time() - t_start + 1.1 * epoch_s) / 3600 > a.max_hours:
            print(f"STOPPED before epoch {epoch + 1} (--max_hours); rerun to resume")
            return
        e0 = time.time()
        torch.manual_seed(a.seed + epoch)
        agg, n = {}, 0
        for i, batch in enumerate(dl["train"]):
            if a.limit_batches and i >= a.limit_batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logs = losses(batch.to("cuda"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
            if i % 200 == 0:
                print(f"  step {i}/{len(dl['train'])} loss {loss.item():.4f}", flush=True)
        train_s = time.time() - e0
        val = evaluate(up, dl["val"], losses, a.limit_batches or a.val_batches)
        print(f"epoch {epoch + 1}/{a.epochs} | {n * a.batch_size / train_s:.1f} samples/s | "
              "train " + " ".join(f"{k} {v / max(n, 1):.4f}" for k, v in agg.items())
              + " | val " + " ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
        if val["recon"] < best:
            best = val["recon"]
            torch.save(checkpoint(a, up, spec, D, T, epoch + 1, val), out)
            print(f"  best val recon {best:.4f} -> {out}")
        epoch_s = time.time() - e0
        tmp = state_path.with_suffix(".tmp")
        torch.save({"model": up.model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch + 1, "best": best,
                    "epoch_s": epoch_s}, tmp)
        tmp.replace(state_path)

    ck = torch.load(out, map_location="cpu", weights_only=False)
    _append(Path(f"results/{a.task}/upsampler_train{smoke}.csv"), {
        "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "task": a.task,
        "model_size": a.model_size, "method": a.method, "key": up.key,
        "time_matching": up.time_matching, "epochs": a.epochs, "best_epoch": ck["epoch"],
        "k": getattr(up, "k", ""), **{f"val_{k}": round(v, 5) for k, v in ck["val"].items()},
        "ckpt": str(out)})
    torch.save({"done": True}, state_path)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
