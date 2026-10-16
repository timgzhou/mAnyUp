"""Train seasonal timAnyUp's fuser and learned lookup selectors (manyup/seasonal.py).

    lr_season  seasonal ps16 tile64 cache (exp.olmoearth.extract --group task.season)
    hr         ps4 tile4 per_timestep cache: the lookup, dense for now (every location cached)
    target     ps4 tile64 series cache, one step per month: what the output should match
    upsampler  a trained per-timestep mAnyUp (manyup_t; GEOID: manyup), FROZEN; it upsamples
               every month from its season's tokens with the month's own guidance

One loop, three losses, none feeding another:
    fuser      Cosine_MSE(up + g(up, hr), target) on EVERY present location (the fuser is per
               location, so dense training fits it for any selection)
    selectors  BCE(logits, rank of gain) per learned selector, gain = err_up - err_fused with
               the current fuser, detached: top-k by gain minimizes reconstruction exactly

Validation (every epoch) and the final test report reconstruction -- mean cosmse over present
locations, the fused vector at the selected ones and the upsample elsewhere -- at budgets
k = f x T x 16 x 16 for every selector:
    random, kl_mean (untrained heuristic), attn, content (learned),
    oracle_gain (top-k by true gain: the ceiling), oracle_err (top-k by true err_up: what the
    old timAnyUp query target approximates), plus none (k = 0) and all (fused everywhere).
The checkpoint with the best validation reconstruction of `attn` at the middle budget is kept;
its test table is appended to results/<task>/seasonal_selector.csv.

    source env_setup/env_olmo.sh
    python -u -m exp.upsamplers.train_seasonal --task pastis --epochs 10 \\
        --upsampler_ckpt checkpoints/upsamplers/manyup_t_pastis_tiny_ep20.pt
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
from exp.upsamplers.task_data import OUT_GRID, UpsamplerData
from manyup.data.guidance import build_guidance_batch, guidance_present
from manyup.loss import Cosine_MSE, cosmse_map
from manyup.seasonal import (AttentionSelector, ContentSelector, KLMeanSelector, RandomSelector,
                             ResidualFuser, SeasonalContext, season_index, season_mean, select,
                             upsample_with_attention)
from manyup.timAnyUp import rank_normalize
from manyup.upsamplers import MAnyUp

cos_mse = Cosine_MSE()
ROLES = ("lr_season", "hr", "target")


def forward(batch, upsampler, device):
    """-> SeasonalContext, hr (B,T,D,H,W), target (B,T,D,H,W). Only present months are
    upsampled: guidance of their own dates, a season with data."""
    lrs, hr, tgt, guid = (batch.feats["lr_season"], batch.feats["hr"], batch.feats["target"],
                          batch.guidance)
    spec, model = upsampler.guidance, upsampler.model
    # dates and validity on the CPU, where the guidance batch lives
    t0, t1 = tgt.t_start.cpu(), tgt.t_end.cpu()
    season = season_index(lrs.t_start.cpu(), lrs.t_end.cpu(), t0, t1)          # (B, T)
    present = (tgt.frame_valid.cpu() & (season >= 0) & guidance_present(guid, spec, t0, t1)
               & lrs.frame_valid.cpu().gather(1, season.clamp(min=0)))
    guide = build_guidance_batch(guid, spec, match="nearest", f_start=t0,
                                 f_end=t1).to(device)                           # (B,T,C,H,W)
    B, T = present.shape
    G = lrs.tokens.shape[1]
    size = (OUT_GRID, OUT_GRID)
    lr = lrs.tokens.to(device).float().permute(0, 1, 4, 2, 3)                   # (B,G,D,h,w)
    feats = lr[torch.arange(B, device=device)[:, None], season.clamp(min=0).to(device)]
    D, h, w = feats.shape[2:]
    sel = present.flatten().to(device)
    with torch.no_grad():
        o, a = upsample_with_attention(model, feats.flatten(0, 1)[sel],
                                       guide.flatten(0, 1)[sel], size)
    up = torch.zeros(B * T, D, *size, device=device)                 # float32, not autocast's
    up[sel] = o.float()
    attn = torch.zeros(B * T, a.shape[1], h * w, device=device)
    attn[sel] = a.float()
    up, attn = up.reshape(B, T, D, *size), attn.reshape(B, T, -1, h * w)
    present = present.to(device)
    ctx = SeasonalContext(up, attn, season_mean(attn, season.to(device), present, G), present)
    perm = lambda f: f.tokens.to(device).float().permute(0, 1, 4, 2, 3)          # noqa: E731
    return ctx, perm(hr), perm(tgt)


def err_maps(ctx, fused, want):
    """cosmse per location (B,T,H,W) of the upsample and the fused map; 0 off present months
    (their upsample is all-zero, which the min-max part of cosmse would turn into NaN)."""
    fl = lambda x: x.flatten(0, 1).float()                                       # noqa: E731
    shape = ctx.up.shape[:2] + ctx.grid
    m = ctx.present[:, :, None, None]
    e_up = cosmse_map(fl(ctx.up), fl(want)).reshape(shape).nan_to_num()
    e_f = cosmse_map(fl(fused), fl(want)).reshape(shape).nan_to_num()
    return e_up * m, e_f * m


def losses(batch, upsampler, fuser, selectors, device):
    ctx, hr, want = forward(batch, upsampler, device)
    fused = fuser(ctx.up, hr)
    keep = ctx.present.flatten()
    l_recon = cos_mse(fused.flatten(0, 1)[keep], want.flatten(0, 1)[keep])["total"]
    with torch.no_grad():
        e_up, e_f = err_maps(ctx, fused.detach(), want)
        where = ctx.present[:, :, None, None].expand_as(e_up)
        rank = rank_normalize(e_up - e_f, where)
    logs = {"recon": l_recon.detach()}
    loss = l_recon
    for name, s in selectors.items():
        l = F.binary_cross_entropy_with_logits(s(ctx).float()[where], rank[where])
        logs[f"sel_{name}"] = l.detach()
        loss = loss + l
    return loss, logs


@torch.no_grad()
def evaluate(dl, upsampler, fuser, selectors, device, k_fracs, limit=0):
    """{(selector, frac): mean cosmse over present locations}, plus ("none", 0) / ("all", 1)."""
    fuser.eval(), [s.eval() for s in selectors.values()]
    fixed = {"random": RandomSelector(), "kl_mean": KLMeanSelector()}
    sums, n = {}, 0
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx, hr, want = forward(batch, upsampler, device)
            fused = fuser(ctx.up, hr)
            scores = {**{k: s(ctx) for k, s in fixed.items()},
                      **{k: s(ctx) for k, s in selectors.items()}}
        e_up, e_f = err_maps(ctx, fused.float(), want)
        scores.update(oracle_gain=e_up - e_f, oracle_err=e_up)
        T, (H, W) = ctx.up.shape[1], ctx.grid
        add = lambda key, e: sums.__setitem__(key, sums.get(key, 0.0) + float(e.sum()))  # noqa
        add(("none", 0.0), e_up)
        add(("all", 1.0), e_f)
        for f in k_fracs:
            k = max(1, round(f * T * H * W))
            for name, sc in scores.items():
                m = select(sc, ctx.present, k)
                add((name, f), torch.where(m, e_f, e_up))
        n += int(ctx.present.sum()) * H * W
    fuser.train(), [s.train() for s in selectors.values()]
    return {k: v / max(n, 1) for k, v in sums.items()}


def table(res, k_fracs) -> str:
    names = [k for k in dict.fromkeys(n for n, _ in res) if k not in ("none", "all")]
    lines = [f"{'selector':>12} " + " ".join(f"{'k=' + format(f, 'g'):>9}" for f in k_fracs),
             f"{'none (k=0)':>12} " + " ".join(f"{res[('none', 0.0)]:9.4f}" for _ in k_fracs)]
    lines += [f"{n:>12} " + " ".join(f"{res[(n, f)]:9.4f}" for f in k_fracs) for n in names]
    lines.append(f"{'all (dense)':>12} " + " ".join(f"{res[('all', 1.0)]:9.4f}" for _ in k_fracs))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--upsampler_ckpt", required=True, help="trained per-timestep mAnyUp")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--model_size", default="tiny")
    ap.add_argument("--fuser_hidden", type=int, default=0, help="0: linear 2D -> D")
    ap.add_argument("--selector_hidden", type=int, default=64)
    ap.add_argument("--k_fracs", default="0.05,0.1667,0.33")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--val_batches", type=int, default=200, help="per epoch; 0 = all")
    ap.add_argument("--test_batches", type=int, default=0, help="0 = all")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    task, device = TASKS[a.task], torch.device("cuda")
    k_fracs = [float(x) for x in a.k_fracs.split(",")]
    k_mid = k_fracs[len(k_fracs) // 2]
    upsampler = MAnyUp.from_checkpoint(a.upsampler_ckpt).to(device)      # frozen, eval
    if upsampler.time_matching == "pooled":
        raise SystemExit(f"{a.upsampler_ckpt} is time-pooled; need a per-timestep mAnyUp")
    sets = {s: UpsamplerData(task, s, a.model_size, ROLES) for s in ("train", "val", "test")}
    D = sets["train"].embed_dim
    fuser = ResidualFuser(D, a.fuser_hidden).to(device)
    selectors = torch.nn.ModuleDict({
        "attn": AttentionSelector(a.selector_hidden),
        "content": ContentSelector(D, a.selector_hidden)}).to(device)
    params = list(fuser.parameters()) + list(selectors.parameters())
    name = (f"seasonal_{a.task}_{a.model_size}_ep{a.epochs}"
            + (f"_fh{a.fuser_hidden}" if a.fuser_hidden else ""))
    smoke = "_smoke" if a.limit_batches else ""
    out = Path(f"checkpoints/upsamplers/{name}{smoke}.pt")
    state_path = out.with_name(out.stem + "_state.pt")
    print(f"Run: {name} | upsampler {upsampler.key} | D {D} | fuser "
          f"{sum(p.numel() for p in fuser.parameters())} params | selectors "
          + " ".join(f"{k} {sum(p.numel() for p in s.parameters())}" for k, s in selectors.items()))
    dl = {s: DataLoader(d, batch_size=a.batch_size, shuffle=s == "train", drop_last=s == "train",
                        num_workers=a.num_workers, collate_fn=d.collate, pin_memory=True,
                        persistent_workers=a.num_workers > 0) for s, d in sets.items()}
    steps = a.epochs * (a.limit_batches or len(dl["train"]))
    opt = torch.optim.AdamW(params, lr=a.lr)
    sched = warmup_cosine(opt, steps, a.warmup_frac, a.lr, a.lr_min)
    start, best, epoch_s = 0, math.inf, None
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st.get("done"):
            print(f"{state_path} says this run is finished; nothing to do")
            return
        fuser.load_state_dict(st["fuser"])
        selectors.load_state_dict(st["selectors"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, epoch_s = st["epoch"], st["best"], st["epoch_s"]
        print(f"Resumed after epoch {start}")
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

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
                loss, logs = losses(batch.to(device), upsampler, fuser, selectors, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
            if i % 200 == 0:
                print(f"  step {i}/{len(dl['train'])} loss {loss.item():.4f}", flush=True)
        val = evaluate(dl["val"], upsampler, fuser, selectors, device, k_fracs,
                       a.limit_batches or a.val_batches)
        print(f"epoch {epoch + 1}/{a.epochs} | {n * a.batch_size / (time.time() - e0):.1f} "
              "samples/s | train " + " ".join(f"{k} {v / max(n, 1):.4f}" for k, v in agg.items())
              + f"\n{table(val, k_fracs)}", flush=True)
        score = val[("attn", k_mid)]
        if score < best:
            best = score
            torch.save({"fuser": fuser.state_dict(), "selectors": selectors.state_dict(),
                        "feat_dim": D, "fuser_hidden": a.fuser_hidden,
                        "selector_hidden": a.selector_hidden,
                        "upsampler_ckpt": a.upsampler_ckpt, "epoch": epoch + 1,
                        "val": {f"{k}@{f:g}": v for (k, f), v in val.items()},
                        "args": vars(a)}, out)
            print(f"  best val recon (attn @ k={k_mid:g}) {best:.4f} -> {out}")
        epoch_s = time.time() - e0
        tmp = state_path.with_suffix(".tmp")
        torch.save({"fuser": fuser.state_dict(), "selectors": selectors.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch + 1,
                    "best": best, "epoch_s": epoch_s}, tmp)
        tmp.replace(state_path)

    ck = torch.load(out, map_location="cpu", weights_only=False)
    fuser.load_state_dict(ck["fuser"])
    selectors.load_state_dict(ck["selectors"])
    test = evaluate(dl["test"], upsampler, fuser, selectors, device, k_fracs,
                    a.limit_batches or a.test_batches)
    print(f"TEST (epoch {ck['epoch']}), mean cosmse over present locations:\n"
          f"{table(test, k_fracs)}")
    csv = Path(f"results/{a.task}/seasonal_selector{smoke}.csv")
    for (sel, f), v in test.items():
        _append(csv, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "task": a.task,
                      "model_size": a.model_size, "upsampler_ckpt": a.upsampler_ckpt,
                      "fuser_hidden": a.fuser_hidden, "best_epoch": ck["epoch"],
                      "selector": sel, "k_frac": f, "test_recon": round(v, 5)})
    print(f"Wrote {csv}")
    torch.save({"done": True}, state_path)


if __name__ == "__main__":
    main()
