"""Does a learned innovation predictor pick BETTER crops than random? A test-time-only test:
the same trained weights (random (month, cell) picks, --readout native_set), the same budget,
only the picks change.

Per test batch:
    pre-look   innovation_diag.prelook: the crop-free pass (seasons through all blocks), each
               month's raw ps4 embedding, pixel variance -> the fit_innovation predictor ->
               a score per (month, ps4 cell)
    picks      the top k slots per sample (k = k_frac x 3072) by
        random            uniform (several seeds)
        pred              the predicted score
        pred_spread       spread first: every cell's best month (by score) before any cell's
                          second, then third, ... (best-scored within each tier)
        pred_soft<l>      standardized score - l x (rank of the month inside its cell)
        innovation        the TRUE innovation (an all-slot crop pass, test labels not used):
                          the ceiling of what the predictor is trained to find
        innovation_spread the same, spread first
        gain / gain_spread  the TRUE per-slot loss drop (uses the LABELS: a ceiling, never a
                          method): loss over the slot's pixels of its season token's logits
                          minus of its crop's logits (crops from the all-slot pass)
        <x>_within        random's SPATIAL layout kept (how many crops each cell gets, from a
                          random draw), score x only chooses WHICH MONTHS inside each cell
    -> the task metric per mode, appended to results/<task>/innovation_picks.csv

    python -m exp.upsamplers.eval_innovation_picks --task biomassters \
        --run inject_fft_biomassters_tiny_random_L0_k0.167_native_set_lr0.001_ep10 \
        --predictor checkpoints/innov_pred_biomassters_k0.167.pt --k_frac 0.16666666666666666
"""
import os

from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims)

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.fit_innovation import Predictor, features  # noqa: E402
from exp.upsamplers.innovation_diag import innovation, prelook  # noqa: E402
from manyup.encoders.olmoearth import _to_device  # noqa: E402


def month_rank(score, present):
    """(B, T, H, W) -> rank of each month inside its cell by score (0 = best); absent months
    rank last."""
    s = score.masked_fill(~present[:, :, None, None], float("-inf"))
    return s.argsort(1, descending=True).argsort(1).float()


def within(score, present, k):
    """Keep a random draw's per-cell crop counts; fill each cell with its top-scored months."""
    B, T, H, W = score.shape
    where = present[:, :, None, None].expand_as(score)
    rnd = torch.rand(B, T, H, W, device=score.device).masked_fill(~where, -1.0)
    thr = rnd.flatten(1).topk(k, 1).values[:, -1][:, None, None, None]
    counts = ((rnd >= thr) & where).sum(1, keepdim=True)                     # (B,1,H,W)
    chosen = (month_rank(score, present) < counts) & where
    return chosen.float() + 1e-3 * torch.rand_like(score)                   # top-k = chosen


def spread(score, present, lam=None):
    """Spread-first (lam None): tier = month rank inside the cell dominates, score breaks ties;
    soft: per-sample standardized score - lam x month rank."""
    where = present[:, :, None, None].expand_as(score)
    r = month_rank(score, present)
    if lam is None:
        z = torch.sigmoid(score.float())                               # (0, 1): tie-break only
        return -r + 0.5 * z
    B = score.shape[0]
    v = score.float().masked_fill(~where, float("nan")).reshape(B, -1)
    mu = v.nanmean(1)[:, None, None, None]
    sd = (v - v.nanmean(1, keepdim=True)).pow(2).nanmean(1).sqrt()[:, None, None, None]
    return (score - mu) / sd.clamp(min=1e-6) - lam * r


@torch.no_grad()
def main():
    olmo_bootstrap.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", required=True, help="checkpoints/<run>.pt: random, native_set")
    ap.add_argument("--predictor", required=True, help="fit_innovation --save output")
    ap.add_argument("--k_frac", type=float, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3, help="random-pick repeats")
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV rows")
    a = ap.parse_args()
    task = TASKS[a.task]
    m = fi.InjectFT(task, "tiny", 0, "random", a.k_frac, "slot", readout="native_set").cuda().eval()
    m.load_state_dict(torch.load(f"checkpoints/{a.run}.pt", map_location="cpu",
                                 weights_only=False)["model"])
    pk = torch.load(a.predictor, map_location="cpu", weights_only=False)
    pred = Predictor(pk["fdim"], temporal=pk["temporal"]).cuda().eval()
    pred.load_state_dict(pk["state"])
    modes = ([f"random{s}" for s in range(a.seeds)] + ["pred", "pred_spread", "pred_soft0.5",
             "pred_soft1", "innovation", "innovation_spread", "gain", "gain_spread",
             "pred_within", "innovation_within", "gain_within"])
    meters = {k: task.meter() for k in modes}
    dl = fi.loader(task, a.split, a.batch_size, 8, False)
    for i, batch in enumerate(dl):
        if a.limit_batches and i >= a.limit_batches:
            break
        ms, ss, season, present, pairs, extra, tgt = batch
        dev = "cuda"
        msd, ssd = _to_device(ms, dev, torch.bfloat16), _to_device(ss, dev, torch.bfloat16)
        sd, pd = season.to(dev), present.to(dev)
        lr, pre = prelook(m, msd, ssd, sd, pd)
        B, T = pd.shape
        x, _, valid = features(pre, torch.arange(B), dev)
        score = pred(x, valid).view(B, T, 16, 16)
        inn, (b, t, y, xx), L_crop, L_seas = innovation(m, lr, pre["F0"].float(), msd, sd, pd,
                                                        extra["frac"].to(dev))
        gain = torch.zeros_like(inn)                  # label-aware per-slot loss drop
        tg = tgt.to(dev)
        tg = tg[:, 0] if tg.ndim == 4 else tg                                  # (B, 64, 64)
        lab = rearrange(tg, "B (Y u) (X w) -> B Y X (u w)", u=4, w=4)[b, y, xx]
        if task.kind == "classification":
            ok = lab >= 0
            ce = lambda L: F.cross_entropy(L, lab.long().clamp(min=0), reduction="none") * ok  # noqa: E731
        else:
            ok = torch.isfinite(lab.float()) & (lab.float() >= 0)
            ce = lambda L: (L[:, 0] - lab.float().nan_to_num(0.0)) ** 2 * ok                   # noqa: E731
        gain[b, t, y, xx] = (ce(L_seas) - ce(L_crop)).sum(1) / ok.sum(1).clamp(min=1)
        picks = {"pred": score, "pred_spread": spread(score, pd),
                 "pred_soft0.5": spread(score, pd, 0.5), "pred_soft1": spread(score, pd, 1.0),
                 "innovation": inn, "innovation_spread": spread(inn.log1p(), pd),
                 "gain": gain, "gain_spread": spread(gain / gain.abs().amax().clamp(min=1e-6), pd)}
        k = round(a.k_frac * T * 16 * 16)
        for name in ("pred", "innovation", "gain"):
            torch.manual_seed(7 + i)
            picks[f"{name}_within"] = within(picks[name], pd, k)
        for mode in modes:
            if mode.startswith("random"):
                torch.manual_seed(1000 * int(mode[6:]) + i)
                m.fixed_scores = None
            else:
                m.fixed_scores = picks[mode]
            logits, _, nc = m(ms, ss, season, present, pairs, extra)
            meters[mode].update(logits, tgt.cuda())
        m.fixed_scores = None
    rand = [meters[f"random{s}"].compute()["primary"] for s in range(a.seeds)]
    print(f"random mean {sum(rand) / len(rand):.4f} (seeds {[round(r, 4) for r in rand]})")
    for mode in modes:
        res = meters[mode].compute()
        print(f"{mode:18s} {res['primary']:.4f}", flush=True)
        if not a.limit_batches:
            _append(Path(f"results/{a.task}/innovation_picks.csv"), {
                "job": os.environ.get("SLURM_JOB_ID", ""), "run": a.run,
                "predictor": a.predictor, "k_frac": a.k_frac, "split": a.split, "pick": mode,
                **{f"{a.split}_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
