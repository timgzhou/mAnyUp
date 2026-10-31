"""Learn the better-half seasons oracle's picks without labels: a linear rank head on the final
season tokens of a FROZEN oracle_bothalf checkpoint (everything else fixed), then the same
weights tested with each way of scoring the (season, window) pairs.

Rank head (InjectFT.rank_scores): Linear(D -> 2 r r), r = season patch / crop size, so one pair of
scalars per candidate window under each season token:
    location channel   summed over the sample's used seasons -> the window's score; trained to
                       rank windows like the LR-only model's pooled loss vs the label
    season channel     per season; trained to rank a window's seasons like their per-season
                       loss (the lowest half is picked, as oracle_bothalf does)
Loss: ListNet (softmax cross-entropy) over each sample's windows and each window's seasons,
targets z-scored per list. Targets are the frozen checkpoint's own losses on the train split
(--fit_split train), or on the val split (--fit_split val: the model errs on held-out tiles as it
will on test, not as on the tiles it was fitted to): the even val batches fit the head, the odd
ones pick its epoch (head file and CSV rows tagged fitval).

Test-time scoring of the same weights (--score_src), one row each in
results/<task>/learned_select.csv:
    label        the oracle itself (labels at test time): should reproduce the run's own test
    season_sum   locations by the SUM of per-season label losses (does one scalar suffice?)
    learned      the rank head (no labels)
    entropy      predictive entropy instead of loss (no labels, no training; classification)
    season_random   random (season, window) pairs at the same budget (lower bound)
Also logged: the share of the oracle's (season, window) picks that each scoring recovers.

    python -m exp.upsamplers.learned_select --task pastis --setup s16m8 --n 16
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402

EPOCHS = {"pastis": 50, "biomassters": 10, "yieldsat_raw": 50}
SETUPS = {  # setup -> (season patch, crop px, loc_stride or None, run tag)
    "s16m8": (16, 8, 8, "_cpx8_st8_poscenter_lr"),
    "s8m4": (8, 4, None, "_poscenter_lrps8_lr"),
}


def run_name(task, setup, n):
    return (f"inject_fft_{task}_tiny_oracle_bothalf_L0_k{n / 256:.3g}_loc_native_set"
            f"{SETUPS[setup][3]}0.001_ep{EPOCHS[task]}")


def build(task, setup, n, arch="linear"):
    lp, cp, stride, _ = SETUPS[setup]
    m = fi.InjectFT(TASKS[task], "tiny", 0, "oracle_bothalf", n / 256, "location",
                    readout="native_set", lr_patch=lp, crop_px=cp, pos="center").cuda()
    if stride:
        m.set_loc_stride(stride)
    sd = torch.load(f"checkpoints/{run_name(task, setup, n)}.pt", map_location="cpu",
                    weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")}
    m.load_state_dict(sd)
    m.requires_grad_(False)
    D = m.backbone().embedding_size
    m.rank_head = fi.make_rank_head(arch, D, 2 * (lp // cp) ** 2, m.head.linear).cuda()
    return m.eval()


def listnet(pred, target, valid, beta):
    """Mean over lists of the softmax cross-entropy between ranks; pred / target / valid (L, N);
    lists with < 2 valid entries or constant targets are skipped."""
    t = target.masked_fill(~valid, 0.0)
    cnt = valid.sum(1, keepdim=True).clamp(min=1)
    mu = t.sum(1, keepdim=True) / cnt
    sd = (((t - mu) ** 2) * valid).sum(1, keepdim=True).div(cnt).sqrt()
    keep = (valid.sum(1) >= 2) & (sd[:, 0] > 1e-6)
    if not keep.any():
        return pred.sum() * 0
    z = ((t - mu) / sd.clamp(min=1e-6) * beta).masked_fill(~valid, float("-inf"))
    lp = pred.masked_fill(~valid, float("-inf")).log_softmax(1)
    p = z.softmax(1)
    return -(p * lp.masked_fill(~valid, 0.0)).sum(1)[keep].mean()


def collect(m, x):
    """One crop-free pass through the frozen model -> its features and label targets."""
    m.rank_collect = True
    with torch.no_grad():
        m(*x)
    m.rank_collect = False
    return m._rank_batch


def rank_loss(m, rb, beta):
    lr_F = rb["lr_F"]
    sc_t, loc_t = rb["sc"], rb["loc"]
    sc_p, loc_p = m.rank_scores(lr_F, rb["season"], rb["present"], rb.get("e"))
    cand = torch.isfinite(loc_p)                                       # (B, H, W) candidates
    B, NS = sc_t.shape[:2]
    l_loc = listnet(loc_p.masked_fill(~cand, 0.0).flatten(1), loc_t.flatten(1),
                    cand.flatten(1), beta)
    valid = present_seasons(rb)[:, :, None, None] & cand[:, None]      # (B, NS, H, W)
    l_sea = listnet(sc_p.masked_fill(~valid, 0.0).permute(0, 2, 3, 1).reshape(-1, NS),
                    sc_t.permute(0, 2, 3, 1).reshape(-1, NS),
                    valid.permute(0, 2, 3, 1).reshape(-1, NS), beta)
    return l_loc, l_sea


def unc_nll(m, rb):
    """--arch unc_nll / unc_both: the per-season proper scoring rule of the uncertainty head's
    maps vs the label -- cross-entropy (classification) or Gaussian NLL with the predicted
    log-variance (regression) -- over labelled pixels of each sample's used seasons."""
    used = present_seasons(rb)
    _, _, mu, lv = m.unc_scores(rb["lr_F"], used)
    tgt = rb["tgt"].to(mu.device)
    B, NS = mu.shape[:2]
    if lv is None:
        t = tgt.long()
        nll = F.cross_entropy(mu.flatten(0, 1), t.repeat_interleave(NS, 0), ignore_index=-1,
                              reduction="none").view(B, NS, *t.shape[-2:])
        lab = (t != -1).float()[:, None]
    else:
        y = (tgt[:, 0] if tgt.ndim == 4 else tgt).float()[:, None]           # (B, 1, 64, 64)
        lab = torch.isfinite(y).float()
        nll = 0.5 * (lv + (mu[:, :, 0] - y.nan_to_num(0.0)) ** 2 * (-lv).exp())
    w = lab * used.float()[:, :, None, None]
    return (nll * w).sum() / w.sum().clamp(min=1)


@torch.no_grad()
def init_logvar(m, rb):
    """Regression: start the log-variance at the log mean squared residual of a batch."""
    head = m.rank_head
    if getattr(head, "lv", None) is None:
        return
    used = present_seasons(rb)
    _, _, mu, _ = m.unc_scores(rb["lr_F"], used)
    tgt = rb["tgt"].to(mu.device)
    y = (tgt[:, 0] if tgt.ndim == 4 else tgt).float()[:, None]
    r2 = (mu[:, :, 0] - y) ** 2
    r2 = r2[torch.isfinite(r2) & used[:, :, None, None].expand_as(r2)]
    head.lv.bias.fill_(float(r2.mean().clamp(min=1e-6).log()))


def present_seasons(rb):
    season, present = rb["season"], rb["present"]
    NS = rb["lr_F"].shape[1]
    valid = present & (season >= 0)
    return (F.one_hot(season.clamp(min=0), NS).bool() & valid[..., None]).any(1)


def picks(m, rb, sc, loc):
    """(B, T, H, W) crop mask of the better-half rule for these scores."""
    B = sc.shape[0]
    return m._season_pick(sc, rb["season"], rb["present"], B, rb["H"], rb["n"], None, rb["G"],
                          None, loc)


@torch.no_grad()
def recall(m, dl, limit):
    """Share of the oracle's (month, cell) picks that the learned picks recover."""
    hit = tot = 0.0
    for i, (*x, _) in enumerate(dl):
        if i >= limit:
            break
        rb = collect(m, x)
        ref = picks(m, rb, rb["sc"], rb["loc"])
        sc_p, loc_p = m.rank_scores(rb["lr_F"], rb["season"], rb["present"], rb.get("e"))
        got = picks(m, rb, sc_p, loc_p)
        hit += float((ref & got).sum())
        tot += float(ref.sum())
    return hit / max(tot, 1.0)


@torch.no_grad()
def pick_recall(m, dl, limit, src):
    """Same share for a fixed scoring source (season_sum / entropy / season_random)."""
    hit = tot = 0.0
    for i, (*x, _) in enumerate(dl):
        if i >= limit:
            break
        rb = collect(m, x)
        ref = picks(m, rb, rb["sc"], rb["loc"])
        m.score_src, m.last_pick = src, None
        if src == "season_random":
            m.sel_name = "season_random"
        torch.manual_seed(i)
        m(*x)
        got = m.last_pick[0].to(ref.device)
        m.score_src, m.sel_name = "label", "oracle_bothalf"
        hit += float((ref & got).sum())
        tot += float(ref.sum())
    return hit / max(tot, 1.0)


class Every:
    """Every other batch of a loader (offset 0 or 1): one half of a split, never held in memory."""

    def __init__(self, dl, offset):
        self.dl, self.offset = dl, offset

    def __iter__(self):
        for i, b in enumerate(self.dl):
            if i % 2 == self.offset:
                yield b

    def __len__(self):
        return (len(self.dl) + 1 - self.offset) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPOCHS))
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--n", type=int, required=True, help="the checkpoint's budget (locations)")
    ap.add_argument("--arch", default="linear", choices=fi.RANK_ARCHS,
                    help="rank head: linear per token, conv (3 x 3 + hidden layer), attention "
                         "(sattn / mattn / smattn), or uncertainty of the per-season predictions "
                         "(unc_ent / unc0 untrained; unc_nll / unc_rank / unc_both trained)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=2.0, help="ListNet target sharpness")
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV, no save")
    ap.add_argument("--fit_split", default="train", choices=["train", "val"])
    a = ap.parse_args()
    task = TASKS[a.task]
    m = build(a.task, a.setup, a.n, a.arch)
    name = run_name(a.task, a.setup, a.n)
    fitval = a.fit_split == "val"
    head_path = Path(f"checkpoints/{name}_rankhead{'' if a.arch == 'linear' else '_' + a.arch}"
                     f"{'_fitval' if fitval else ''}.pt")
    dl = {s: fi.loader(task, s, a.batch_size, a.num_workers, s == "train") for s in ("train", "val", "test")}
    vb = a.limit_batches or a.val_batches
    fit_dl, pick_dl = dl["train"], dl["val"]
    if fitval:                          # fixed halves of the (unshuffled) val split, streamed
        fit_dl, pick_dl = Every(dl["val"], 0), Every(dl["val"], 1)
        vb = a.limit_batches or len(pick_dl)

    if head_path.exists() and not a.limit_batches:
        m.rank_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        print(f"loaded {head_path}")
    elif a.arch in ("unc_ent", "unc0"):    # no parameters: the model's own head
        if not a.limit_batches:
            torch.save({}, head_path)
    else:
        opt = torch.optim.Adam(m.rank_head.parameters(), lr=a.lr)
        best, best_state, step, t0 = -1.0, None, 0, time.time()
        for ep in range(a.epochs):
            m.rank_head.train()
            for i, (*x, _) in enumerate(fit_dl):
                if (a.limit_batches and i >= a.limit_batches) or step >= a.max_steps:
                    break
                rb = collect(m, x)
                if step == 0 and a.arch in ("unc_nll", "unc_both"):
                    init_logvar(m, rb)
                if a.arch == "unc_nll":
                    l_loc, l_sea = unc_nll(m, rb), torch.zeros(())
                else:
                    l_loc, l_sea = rank_loss(m, rb, a.beta)
                    if a.arch == "unc_both":
                        l_loc = l_loc + unc_nll(m, rb)
                opt.zero_grad()
                (l_loc + l_sea).backward()
                opt.step()
                step += 1
                if step % 200 == 0:
                    print(f"  step {step} loc(+nll) {l_loc.item():.4f} season {l_sea.item():.4f}",
                          flush=True)
            m.rank_head.eval()
            r = recall(m, pick_dl, vb)
            print(f"epoch {ep + 1}: val pick recall {r:.3f} ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
            if r > best:
                best, best_state = r, {k: v.detach().cpu().clone()
                                       for k, v in m.rank_head.state_dict().items()}
            if step >= a.max_steps:
                break
        m.rank_head.load_state_dict(best_state)
        if not a.limit_batches:
            torch.save(best_state, head_path)

    m.rank_head.eval()
    srcs = ["label", "season_sum", "learned", "season_random"] if a.arch == "linear" else ["learned"]
    if task.kind == "classification" and a.arch == "linear":
        srcs.insert(3, "entropy")
    out_csv = Path(f"results/{a.task}/learned_select.csv")
    done = set()
    if out_csv.exists() and not a.limit_batches:      # a chained rerun: skip logged modes
        import csv
        with open(out_csv) as fh:
            done = {r["score_src"] for r in csv.DictReader(fh)
                    if r["run"] == name and (r.get("arch") or "linear") == a.arch
                    and (r.get("fit_split") or "train") == a.fit_split}
    for src in srcs:
        if src in done:
            print(f"TEST {src}: already in {out_csv}")
            continue
        m.score_src = "label" if src == "season_random" else src
        m.sel_name = "season_random" if src == "season_random" else "oracle_bothalf"
        torch.manual_seed(0)
        res, crops = fi.evaluate(m, task, dl["test"], a.limit_batches)
        m.score_src, m.sel_name = "label", "oracle_bothalf"
        rec = (1.0 if src == "label" else
               recall(m, dl["test"], vb) if src == "learned" else
               pick_recall(m, dl["test"], vb, src))
        print(f"TEST {src}: primary {res['primary']:.4f} | pick recall {rec:.3f} | "
              f"{crops:.0f} crops/sample", flush=True)
        if a.limit_batches:
            continue
        _append(out_csv, {
            "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "setup": a.setup, "n": a.n,
            "score_src": src, "arch": a.arch, "fit_split": a.fit_split, "pick_recall": round(rec, 4), "crops_per_sample": round(crops, 1),
            **{f"test_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
