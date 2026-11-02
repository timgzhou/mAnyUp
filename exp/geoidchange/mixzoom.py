"""2b, one model: months (pool) -> the zoomed months' single dates.

Pass 1: one frame per calendar month of the window (cfg pool) at ps16, attending to each other;
its pair head scores every month transition. Pass 2: every date of the two chosen months (+ the
date before each run, for its reference) as ps16 tokens, placed at their real day on the same
time axis (days / 6), reading the month tokens and each other (attn cross_hr); the pair head on
consecutive dates gives the date scores and maps. Train: the months = the post's month and the
one before + a random earlier pair (negatives); labels as harness. Eval: top 2 months by score.

    python -m exp.geoidchange.mixzoom name=z pool=min ps=16 topk=2
"""
import json
import logging
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset

from exp.geoidchange.harness import (BEFORE, CE_W, DB_FLOOR, MODEL_SIZE_TO_ID, ROOT, Chips, INP, IoU,
                                     OlmoEarthEncoder, date_metrics, eval_ids, loss_fn, month_key,
                                     pool_months, to_sample)
from manyup.encoders.olmoearth import crop_patches
from manyup.inject import MixedResOE

logging.disable(logging.INFO)


def date_set(days, months):
    """dates of `months` + the date before each -> date indices, and which pairs (t-1, t) count."""
    keep = np.array([month_key(d) in months for d in days])
    keep[0] = False
    sel = sorted({t for t in np.nonzero(keep)[0]} | {t - 1 for t in np.nonzero(keep)[0]})
    pairs = [(sel.index(t - 1), sel.index(t)) for t in np.nonzero(keep)[0]]
    return np.array(sel, dtype=int), pairs


class Tr(Dataset):
    def __init__(self, c, n, pool, seed=0):
        self.c, self.pool = c, pool
        self.ids = np.random.default_rng(seed).permutation(len(c.recs))[:n]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        c, i = self.c, int(self.ids[k])
        rng = np.random.default_rng()
        r = c.recs[i]
        y = c.label(i)
        w = c.good(i, c.window(i, rng))
        if len(w) < 3:
            w = np.array([0, 0, 0])
        a, ok = c.frames(i, w)
        days = r["day"][w]
        f, md, mo = pool_months(a, ok, days, self.pool)
        if len(f) < 2:
            f, md, mo = np.repeat(f, 2, 0), np.repeat(md, 2), np.repeat(mo, 2)
        pm = month_key(r["post"])
        mlab = np.full(len(f) - 1, -1)
        mlab[mo[1:] < month_key(r["event"] - BEFORE)] = 0
        mlab[mo[1:] > pm] = 0
        mlab[mo[1:] == pm] = 1
        months = {pm, int(rng.choice(mo))}      # the post month + any other, as a top-2 sees
        sel, pairs = date_set(days, months)
        if not len(sel):
            sel, pairs = np.array([0, min(1, len(w) - 1)]), [(0, 1)]
        dlab = []
        for p, q in pairs:
            d = days[sel[q]]
            dlab.append(1 if d == r["post"] else
                        0 if d < r["event"] - BEFORE or d > r["post"] else -1)
        return f, md, a[sel], days[sel], np.array(pairs), np.array(dlab), mlab, y


def pad(items, key_x, key_d):
    B, T = len(items), max(len(it[key_d]) for it in items)
    x = np.zeros((B, T, 2, 64, 64), np.float32)
    d = np.zeros((B, T), np.int64)
    v = np.zeros((B, T), bool)
    for b, it in enumerate(items):
        n = len(it[key_d])
        x[b, :n], d[b, :n], v[b, :n] = it[key_x], it[key_d], True
        d[b, n:] = it[key_d][-1]
    return x, d, v


def collate(items):
    mx, md, mv = pad(items, 0, 1)
    dx, dd, dv = pad(items, 2, 3)
    B = len(items)
    P = max(len(it[4]) for it in items)
    pr = np.full((B, P, 2), -1)
    dl = np.full((B, P), -1)
    Q = max(len(it[6]) for it in items)
    ml = np.full((B, Q), -1)
    for b, it in enumerate(items):
        pr[b, :len(it[4])], dl[b, :len(it[5])] = it[4], it[5]
        ml[b, :len(it[6])] = it[6]
    return (to_sample(mx, md, mv), torch.from_numpy(md), to_sample(dx, dd, dv), torch.from_numpy(dd),
            torch.from_numpy(dv), torch.from_numpy(pr), torch.from_numpy(dl), torch.from_numpy(ml),
            torch.from_numpy(np.stack([it[7] for it in items])))


class Zoom(nn.Module):
    def __init__(self, ps=16, month_slot=False):
        super().__init__()
        enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID["tiny"], INP, 16, 64, name="tiny").encoder
        self.mix = MixedResOE(enc, 0, "center", ps, attn="cross_hr")
        self.ps, self.month_slot = ps, month_slot
        self.head = nn.Linear(enc.embedding_size, 2 * ps * ps)
        nn.init.zeros_(self.head.weight)

    def lr(self, s, days):
        s = s._replace(**{k: v.cuda(non_blocking=True) for k, v in s._asdict().items()
                          if v is not None})
        t = (days - days[:, :1]).float().cuda() / 6
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr = self.mix.embed(s, self.ps, times=t)
            enc = self.mix.lr_trace(lr)
        return lr, enc, lr.pooled(enc[1]).float()

    def tomap(self, d):
        return rearrange(self.head(d), "b q h w (c i j) -> b q c (h i) (w j)", c=2, i=self.ps, j=self.ps)

    def month_logits(self, f):
        return self.tomap(f[:, 1:] - f[:, :-1])

    def dates(self, lr, enc, sd, dd, dv, t0):
        """(B, D) dates -> tokens (B, D, h, w, D_e)."""
        sd = sd._replace(**{k: v.cuda(non_blocking=True) for k, v in sd._asdict().items()
                            if v is not None})
        ps, g = self.ps, 64 // self.ps
        dv = dv.cuda()
        b, t = dv.nonzero(as_tuple=True)
        n = len(b)
        cy, cx = torch.meshgrid(torch.arange(g, device=b.device), torch.arange(g, device=b.device),
                                indexing="ij")
        bb = b.repeat_interleave(g * g)
        tt = t.repeat_interleave(g * g)
        py = cy.flatten().repeat(n) * ps
        px = cx.flatten().repeat(n) * ps
        slot = ((dd.cuda()[bb, tt] - t0.cuda()[bb]).float()) / 6
        if self.month_slot:                 # the old encoding: a date at its month's slot
            slot = (slot * 6 / 30.44).floor() * 30.44 / 6
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crop_patches(sd, bb, tt, py, px, ps, pixels=True), ps, slot=slot,
                                py=py, px=px)
            _, (h,) = self.mix.tail_multi(lr, [(hr, bb)], enc)
        x = hr.pooled(h).float()[:, 0, 0, 0]
        B, D = dv.shape
        out = x.new_zeros(B, D, g, g, x.shape[-1])
        out[bb, tt, py // ps, px // ps] = x
        return out

    def pair_logits(self, f, pairs):
        p = pairs.cuda().clamp(min=0)
        bi = torch.arange(len(p), device=p.device)[:, None]
        return self.tomap(f[bi, p[..., 1]] - f[bi, p[..., 0]])


@torch.no_grad()
def report(net, split, pool, topk, n=1500, month_w=None):
    c = Chips(split)
    net.eval()
    rows, seg, mrows = [], IoU(), []
    npairs, ndates, nm = 0, 0, 0
    for i in eval_ids(c, n):
        r = c.recs[i]
        w = c.good(i, c.window(i))
        if len(w) < 2:
            continue
        a, ok = c.frames(i, w)
        days = r["day"][w]
        f, md, mo = pool_months(a, ok, days, pool)
        y = c.label(i)
        if len(f) < 2:
            continue
        s = to_sample(f[None], md[None], np.ones((1, len(f)), bool))
        lr, enc, F_ = net.lr(s, torch.from_numpy(md[None]))
        ms = net.month_logits(F_)[0].softmax(1)[:, 1].mean((1, 2)).cpu().numpy()
        top = {int(mo[t + 1]) for t in np.argsort(-ms)[:topk]}
        sel, pairs = date_set(days, top)
        if not len(pairs):
            sel, pairs = np.arange(len(days)), [(t - 1, t) for t in range(1, len(days))]
        dt_ = torch.from_numpy(days[sel][None])
        sd = to_sample(a[sel][None], days[sel][None], np.ones((1, len(sel)), bool))
        tok = net.dates(lr, enc, sd, dt_, torch.ones(1, len(sel), dtype=torch.bool),
                        torch.from_numpy(md[:1]))
        m = net.pair_logits(tok, torch.tensor(pairs)[None])[0].softmax(1)[:, 1]
        sc = m.mean((1, 2)).cpu().numpy()
        t = int(sc.argmax())
        rows.append((days[sel[pairs[t][1]]], r["post"], r["event"]))
        seg.add((m[t] > .5).cpu().numpy(), y)
        npairs += len(pairs); ndates += len(sel); nm += len(f)
    res = date_metrics(rows)
    res.update(iou_pred=round(seg.get(), 4), date_pairs=round(npairs / len(rows), 1),
               dates=round(ndates / len(rows), 1), months=round(nm / len(rows), 1))
    return res


def main(argv):
    kw = dict(a.split("=", 1) for a in argv)
    pool, ps, seed = kw.get("pool", "min"), int(kw.get("ps", 16)), int(kw.get("seed", 0))
    ep, bs = int(kw.get("epochs", 6)), int(kw.get("bs", 24))
    torch.manual_seed(seed)
    net = Zoom(ps, kw.get("slot") == "month").cuda()
    dl = DataLoader(Tr(Chips("train"), int(kw.get("n", 16000)), pool, seed), batch_size=bs,
                    shuffle=True, num_workers=7, collate_fn=collate, drop_last=True,
                    persistent_workers=True, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    t0 = time.time()
    for e in range(ep):
        net.mix.encoder.requires_grad_(e >= 1)
        if e == 1:
            for g in opt.param_groups:
                g["lr"] = 1e-4
        net.train()
        ls, n = 0.0, 0
        for ms, md, sd, dd, dv, pr, dl_, ml, y in dl:
            lr, enc, f = net.lr(ms, md)
            lm = loss_fn(net.month_logits(f), ml, y, 1.0)
            tok = net.dates(lr, enc, sd, dd, dv, md[:, 0])
            ld = loss_fn(net.pair_logits(tok, pr), dl_, y, 1.0)
            loss = lm + ld
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ls, n = ls + float(loss), n + 1
        print(f"[{kw.get('name')}] epoch {e + 1}/{ep} loss {ls / n:.4f} {time.time() - t0:.0f}s",
              flush=True)
    res = {"name": kw.get("name")}
    for k in (1, 2):
        res[f"top{k}"] = report(net, kw.get("split", "val"), pool, k, int(kw.get("eval_n", 800)))
    print(json.dumps(res), flush=True)
    open(f"{ROOT}/mixzoom.jsonl", "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
