"""Smoke-test harness for GEOID change: one model (OlmoEarth tiny + MixedResOE, pair-diff head),
sequences of frames, per-pair labels, one evaluation (where: flood IoU; when: date error).

Sequence modes (cfg.mode):
  bi       GEOID's own (pre, post) GRD pair, label on it
  pair     RTC: two frames. Train = (ref, post) with the label + (neg_p) a consecutive pair
           entirely before the event with an all-zero map. Eval = every consecutive pair, alone
  series   RTC: all dates of the window in one sequence, the same pair labels (ignored pairs
           after post). Eval = the window's sequence
  month    as series with one frame per calendar month (cfg.pool: mean / median / min / max)
  ctx      pair + the K dates before it in one sequence (only the last pair decoded)
"""
import datetime as dt
import hashlib
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from torch import nn
from torch.utils.data import DataLoader, Dataset

from exp.common import olmo_bootstrap
olmo_bootstrap.apply()
from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from manyup.data.datasets.geoidflood import DB_FLOOR  # noqa: E402
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput, _timestamps  # noqa: E402
from manyup.inject import MixedResOE  # noqa: E402

ROOT = "/tmp/geoidchange"
NODATA = -32768
CE_W = (0.5459, 1.4541)
INP = OlmoEarthInput({"S1": "sentinel1"})
BAND = INP._band_index("S1", ("VV", "VH"))
WINDOW, MARGIN, BEFORE = 183, 12, 14


@dataclass
class Cfg:
    name: str = "run"
    mode: str = "pair"
    ps: int = 16
    pos: str = "prev"       # the positive's reference: prev date | preev (last date <= event - 6)
    neg_p: float = 0.5      # pair mode: share of sequences that are negatives
    neg_w: float = 1.0
    time: str = "index"     # index: OlmoEarth's 0, 1, ..; days: days since first frame / 6
    pool: str = "mean"
    ctx: int = 0
    comp: int = 0           # pair: the reference frame = mean dB of the K dates up to it
    neg_after: int = 0      # 1: pairs after the post date are negatives too
    ref: str = "prev"       # head on (late - prev frame) | (late - mean of all earlier frames)
    epochs: int = 3
    lr: float = 1e-3
    bs: int = 64
    train_n: int = 16000
    eval_n: int = 1500
    seed: int = 0
    freeze: int = 1         # head-only epochs
    series_bs: int = 16


def month_key(d):
    x = dt.date(1970, 1, 1) + dt.timedelta(int(d))
    return x.year * 12 + x.month


class Chips:
    def __init__(self, split):
        self.split = split
        self.recs = json.load(open(f"{ROOT}/{split}.json"))
        self.X = np.memmap(f"{ROOT}/{split}.x", dtype=np.int16, mode="r").reshape(-1, 2, 64, 64)
        self.Y = np.load(f"{ROOT}/{split}.y.npy", mmap_mode="r")
        self.G = np.load(f"{ROOT}/{split}.geoid.npy", mmap_mode="r")
        for r in self.recs:
            r["day"] = np.array(r["day"])

    def label(self, i):
        y = np.asarray(self.Y[i]).astype(np.int64)
        t = (y == 2).astype(np.int64)
        t[y == 255] = -1
        if self.split == "val" and self.recs[i]["chip"].startswith("EMSR342"):
            t[:] = -1
        return t

    def window(self, i, rng=None):
        r = self.recs[i]
        lo = r["post"] + MARGIN - WINDOW + 1
        hi = max(min(r["event"], r["post"]) - BEFORE, lo)
        if rng is not None:
            s = int(rng.integers(lo, hi + 1))
        else:
            u = int(hashlib.md5(r["chip"].encode()).hexdigest()[:8], 16) / 16 ** 8
            s = lo + int(u * (hi - lo + 1))
        return np.nonzero((r["day"] >= s) & (r["day"] < s + WINDOW))[0]

    def frames(self, i, idx):
        """dates idx -> (T, 2, 64, 64) dB, (T, 64, 64) valid. No-data pixels: the frame's median."""
        off = self.recs[i]["off"]
        a = np.stack([self.X[off + j] for j in idx]).astype(np.float32)
        ok = a != NODATA
        a = a / 100
        for t in range(len(a)):
            for b in range(2):
                bad = ~ok[t, b]
                if bad.any():
                    a[t, b][bad] = np.median(a[t, b][~bad]) if (~bad).any() else DB_FLOOR
        return a, ok.all(1)

    def good(self, i, idx):
        """Of dates idx, those with >= 90% valid pixels."""
        off = self.recs[i]["off"]
        return np.array([j for j in idx if (self.X[off + j, 0] != NODATA).mean() >= 0.9], dtype=int)


def pool_months(a, ok, days, pool):
    """(T, 2, H, W) -> one frame per calendar month: (M, 2, H, W), mid days (M,), months (M,)."""
    m = np.array([month_key(d) for d in days])
    fr, ds, ms = [], [], []
    f = {"mean": np.nanmean, "median": np.nanmedian, "min": np.nanmin, "max": np.nanmax}[pool]
    x = np.where(np.repeat(ok[:, None], 2, 1), a, np.nan)
    for u in np.unique(m):
        s = x[m == u]
        fr.append(np.nan_to_num(f(s, 0), nan=float(np.nanmedian(s)) if np.isfinite(s).any()
                                else DB_FLOOR))
        ds.append(int(days[m == u].mean())); ms.append(u)
    return np.stack(fr), np.array(ds), np.array(ms)


class Train(Dataset):
    def __init__(self, chips, cfg):
        self.c, self.cfg = chips, cfg
        rng = np.random.default_rng(cfg.seed)
        self.ids = rng.permutation(len(chips.recs))[:cfg.train_n]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        c, cfg, i = self.c, self.cfg, int(self.ids[k])
        rng = np.random.default_rng()
        r = c.recs[i]
        y = c.label(i)
        if cfg.mode == "bi":
            g = np.asarray(c.G[i], dtype=np.float32)
            ok = g != NODATA
            g = np.where(ok, g / 100, DB_FLOOR)
            return g, np.array([r["pre"], r["post"]]), np.array([[0, 1]]), np.array([1]), y
        w = c.good(i, c.window(i, rng))
        if len(w) < 2:                      # nothing usable: an ignored pair
            w = np.array([0, 0])
            a, _ = c.frames(i, w)
            return a, r["day"][w], np.array([[0, 1]]), np.array([-1]), y
        days = r["day"][w]
        pi = np.nonzero(days == r["post"])[0]
        if cfg.mode in ("series", "month"):
            a, ok = c.frames(i, w)
            lab = np.full(len(w) - 1, -1)
            lab[days[1:] < r["event"] - BEFORE] = 0
            if len(pi) and pi[0] > 0:
                lab[pi[0] - 1] = 1
            pairs = np.stack([np.arange(len(w) - 1), np.arange(1, len(w))], 1)
            if cfg.mode == "month":
                f, md, mo = pool_months(a, ok, days, cfg.pool)
                pm = month_key(r["post"])
                evm = month_key(r["event"] - BEFORE)
                pairs = np.stack([np.arange(len(f) - 1), np.arange(1, len(f))], 1)
                lab = np.full(len(f) - 1, -1)
                lab[mo[1:] < evm] = 0
                lab[mo[1:] == pm] = 1
                return f, md, pairs, lab, y
            return a, days, pairs, lab, y
        # pair: one positive or one negative; ctx: the K dates before it joined
        neg = np.nonzero((days[1:] < r["event"] - BEFORE)
                         | (cfg.neg_after and days[1:] > r["post"]))[0]
        if len(pi) and pi[0] > 0 and (len(neg) == 0 or rng.random() >= cfg.neg_p):
            b = pi[0]
            ai = b - 1
            if cfg.pos == "preev":
                pre = np.nonzero(days <= r["event"] - 6)[0]
                ai = pre[-1] if len(pre) else b - 1
            lab = 1
        elif len(neg):
            ai = int(rng.choice(neg)); b = ai + 1; lab = 0
        else:
            ai, b, lab = 0, min(1, len(w) - 1), -1
        if cfg.comp:
            idx = list(range(max(0, ai - cfg.comp + 1), ai + 1)) + [b]
            a, ok = c.frames(i, w[idx])
            return (np.stack([a[:-1].mean(0), a[-1]]), days[[ai, b]], np.array([[0, 1]]),
                    np.array([lab]), y)
        idx = list(range(max(0, ai - cfg.ctx), ai + 1)) + [b]
        a, ok = c.frames(i, w[idx])
        T = len(idx)
        return a, days[idx], np.array([[T - 2, T - 1]]), np.array([lab]), y


def collate(items):
    B, T = len(items), max(len(it[1]) for it in items)
    P = max(len(it[2]) for it in items)
    x = np.full((B, T, 2, 64, 64), 0.0, np.float32)
    days = np.zeros((B, T), np.int64)
    valid = np.zeros((B, T), bool)
    pairs = np.full((B, P, 2), -1)
    lab = np.full((B, P), -1)
    y = np.stack([it[4] for it in items])
    for b, (a, d, p, l, _) in enumerate(items):
        x[b, :len(d)], days[b, :len(d)], valid[b, :len(d)] = a, d, True
        days[b, len(d):] = d[-1]
        pairs[b, :len(p)], lab[b, :len(p)] = p, l
    return to_sample(x, days, valid), torch.from_numpy(days), torch.from_numpy(valid), \
        torch.from_numpy(pairs), torch.from_numpy(lab), torch.from_numpy(y)


def to_sample(x, days, valid):
    arr = np.transpose(x[:, :, BAND], (0, 3, 4, 1, 2))                    # (B, H, W, T, C)
    s1 = torch.from_numpy(INP.normalizer.normalize(INP.olmo["S1"], arr)).float()
    mask = torch.full_like(s1, MaskValue.ONLINE_ENCODER.value)
    b, t = np.nonzero(~valid)
    mask[b, :, :, t] = MaskValue.MISSING.value
    return MaskedOlmoEarthSample(sentinel1=s1, sentinel1_mask=mask,
                                 timestamps=_timestamps(torch.from_numpy(days)))


class Net(nn.Module):
    def __init__(self, ps, time_mode="index", ref="prev"):
        super().__init__()
        enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID["tiny"], INP, 16, 64, name="tiny").encoder
        self.mix = MixedResOE(enc, 0, "center", ps)
        self.ps, self.time_mode, self.ref = ps, time_mode, ref
        self.head = nn.Linear(enc.embedding_size, 2 * ps * ps)
        nn.init.zeros_(self.head.weight)
        self.head.bias.data[0::2] = 0.0

    def encode(self, s, days, valid):
        s = s._replace(**{k: v.cuda(non_blocking=True) for k, v in s._asdict().items()
                          if v is not None})
        times = None
        if self.time_mode == "days":
            times = (days - days[:, :1]).float().cuda() / 6
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = self.mix.embed(s, self.ps, times=times)
            x, _ = self.mix.tail_multi(p, [])
        return p.pooled(x).float()                                          # (B, T, h, w, D)

    def pairs(self, f, pairs, valid=None):
        """(B, T, h, w, D), (B, P, 2) -> (B, P, 2, 64, 64) logits (pad pairs: garbage)."""
        B = f.shape[0]
        p = pairs.cuda().clamp(min=0)
        bi = torch.arange(B, device=f.device)[:, None]
        d = f[bi, p[..., 1]] - f[bi, p[..., 0]]
        if self.ref == "mean":              # (one pair per sequence: ctx modes)
            T = f.shape[1]
            w = (torch.arange(T, device=f.device)[None] < p[:, 0, 1, None]).float()
            if valid is not None:
                w = w * valid.cuda().float()
            w = w[:, :, None, None, None]
            ref = (f * w).sum(1) / w.sum(1).clamp(min=1)
            d = f[bi, p[..., 1]] - ref[:, None]
        return rearrange(self.head(d), "b q h w (c i j) -> b q c (h i) (w j)",
                         c=2, i=self.ps, j=self.ps)


def loss_fn(logits, lab, y, neg_w):
    """logits (B, P, 2, 64, 64); lab (B, P) 1 the label, 0 no flood, -1 ignore."""
    B, P = lab.shape
    w = torch.tensor(CE_W, device=logits.device)
    y = y.cuda()
    lab = lab.cuda()
    tot, n = 0.0, 0
    bpos, ppos = (lab == 1).nonzero(as_tuple=True)
    if len(bpos):
        tot = tot + F.cross_entropy(logits[bpos, ppos], y[bpos], weight=w, ignore_index=-1)
        n += 1
    bneg, pneg = (lab == 0).nonzero(as_tuple=True)
    if len(bneg):
        t = torch.where(y[bneg] < 0, y[bneg], torch.zeros_like(y[bneg]))
        tot = tot + neg_w * F.cross_entropy(logits[bneg, pneg], t, weight=w, ignore_index=-1)
        n += 1
    return tot


def train(cfg: Cfg):
    torch.manual_seed(cfg.seed)
    net = Net(cfg.ps, cfg.time, cfg.ref).cuda()
    ds = Train(Chips("train"), cfg)
    bs = cfg.series_bs if cfg.mode in ("series",) else cfg.bs
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=7, collate_fn=collate,
                    drop_last=True, persistent_workers=True, pin_memory=True)
    enc = net.mix.encoder
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    t0 = time.time()
    for e in range(cfg.epochs):
        tune = e >= cfg.freeze
        enc.requires_grad_(tune)
        if e == cfg.freeze:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * 0.1
        net.train()
        ls, n = 0.0, 0
        for s, days, valid, pairs, lab, y in dl:
            f = net.encode(s, days, valid)
            logits = net.pairs(f, pairs, valid)
            loss = loss_fn(logits, lab, y, cfg.neg_w)
            if not torch.is_tensor(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ls, n = ls + loss.item(), n + 1
        print(f"[{cfg.name}] epoch {e + 1}/{cfg.epochs} loss {ls / max(n, 1):.4f} "
              f"{time.time() - t0:.0f}s", flush=True)
    return net


class IoU:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, p, y):
        ok = y >= 0
        self.tp += int(((p == 1) & (y == 1) & ok).sum())
        self.fp += int(((p == 1) & (y == 0) & ok).sum())
        self.fn += int(((p == 0) & (y == 1) & ok).sum())

    def get(self):
        return self.tp / max(self.tp + self.fp + self.fn, 1)


def eval_ids(c, n):
    ids = np.random.default_rng(1).permutation(len(c.recs))
    if c.split == "val":
        ids = [i for i in ids if not c.recs[i]["chip"].startswith("EMSR342")]
    return [int(i) for i in ids[:n]]


def date_metrics(rows):
    """rows: (pred day, post, event) -> the when metrics."""
    e = np.array([r[0] - r[1] for r in rows], dtype=float)
    a = np.abs(e)
    on = np.array([r[2] - 12 <= r[0] <= r[1] + 12 for r in rows])
    return {"hit0": round(float((a == 0).mean()), 3), "hit6": round(float((a <= 6).mean()), 3),
            "hit12": round(float((a <= 12).mean()), 3), "hit24": round(float((a <= 24).mean()), 3),
            "med_abs": float(np.median(a)), "in_event": round(float(on.mean()), 3)}


@torch.no_grad()
def eval_bi(net, split, n=4000):
    """GEOID's pair: flood IoU."""
    c = Chips(split)
    net.eval()
    io = IoU()
    ids = eval_ids(c, n)
    for k in range(0, len(ids), 128):
        b = ids[k:k + 128]
        g = np.stack([np.asarray(c.G[i], np.float32) for i in b])
        g = np.where(g != NODATA, g / 100, DB_FLOOR)
        days = np.array([[c.recs[i]["pre"], c.recs[i]["post"]] for i in b])
        s = to_sample(g, days, np.ones(days.shape, bool))
        f = net.encode(s, torch.from_numpy(days), None)
        pairs = torch.tensor([[[0, 1]]] * len(b))
        p = net.pairs(f, pairs)[:, 0].argmax(1).cpu().numpy()
        io.add(p, np.stack([c.label(i) for i in b]))
    return io.get()


@torch.no_grad()
def score_pairs(net, cfg, c, ids, chunk=256):
    """-> [(chip index, later-frame days, score per pair, valid-pixel maps (P, 64, 64) bool or
    None)] running every consecutive pair of the window as the mode says."""
    net.eval()
    out = []
    if cfg.mode == "series":
        for i in ids:
            w = c.good(i, c.window(i))
            if len(w) < 2:
                continue
            a, ok = c.frames(i, w)
            days = c.recs[i]["day"][w]
            s = to_sample(a[None], days[None], np.ones((1, len(w)), bool))
            f = net.encode(s, torch.from_numpy(days[None]), None)
            pairs = torch.stack([torch.arange(len(w) - 1), torch.arange(1, len(w))], 1)[None]
            out.append((i, days, net.pairs(f, pairs)[0].softmax(1)[:, 1].half().cpu()))
        return out
    for i in ids:
        w = c.good(i, c.window(i))
        if len(w) < 2:
            continue
        days = c.recs[i]["day"][w]
        a, ok = c.frames(i, w)
        if cfg.mode == "month":
            a, days, _ = pool_months(a, ok, days, cfg.pool)
            if len(a) < 2:
                continue
        T = len(a)
        maps = []
        for k in range(1, T, chunk):
            idx = list(range(k, min(k + chunk, T)))
            ctx = cfg.ctx
            L = 2 if cfg.comp else ctx + 2
            x = np.zeros((len(idx), L, 2, 64, 64), np.float32)
            d = np.zeros((len(idx), L), np.int64)
            v = np.zeros((len(idx), L), bool)
            pr = []
            for n, b in enumerate(idx):
                if cfg.comp:
                    x[n, 0] = a[max(0, b - cfg.comp):b].mean(0)
                    x[n, 1], d[n], v[n] = a[b], days[[b - 1, b]], True
                    pr.append([[0, 1]])
                    continue
                sel = list(range(max(0, b - 1 - ctx), b)) + [b]
                x[n, :len(sel)], d[n, :len(sel)], v[n, :len(sel)] = a[sel], days[sel], True
                d[n, len(sel):] = days[sel[-1]]
                pr.append([[len(sel) - 2, len(sel) - 1]])
            f = net.encode(to_sample(x, d, v), torch.from_numpy(d), None)
            maps.append(net.pairs(f, torch.tensor(pr), torch.from_numpy(v))[:, 0]
                        .softmax(1)[:, 1].half().cpu())
        out.append((i, days, torch.cat(maps)))
    return out


def report(net, cfg, split="test", scored=None):
    c = Chips(split)
    ids = eval_ids(c, cfg.eval_n)
    t0 = time.time()
    scored = scored or score_pairs(net, cfg, c, ids)
    dt_ = time.time() - t0
    rows, seg, post_seg = [], IoU(), IoU()
    pairs_n = 0
    for i, days, maps in scored:
        r = c.recs[i]
        y = c.label(i)
        sc = maps.float().mean((1, 2)).numpy()
        pairs_n += len(sc)
        t = int(sc.argmax())
        rows.append((days[t + 1], r["post"], r["event"]))
        seg.add((maps[t] > .5).numpy(), y)
        if cfg.mode == "month":
            mk = np.array([month_key(d) for d in days[1:]])
            q = np.nonzero(mk == month_key(r["post"]))[0]
        else:
            q = np.nonzero(days[1:] == r["post"])[0]
        q = int(q[0]) if len(q) else int(np.abs(days[1:] - r["post"]).argmin())
        post_seg.add((maps[q] > .5).numpy(), y)
    m = date_metrics(rows)
    m.update(iou_pred=round(seg.get(), 4), iou_post=round(post_seg.get(), 4),
             chips=len(rows), pairs_per_chip=round(pairs_n / max(len(rows), 1), 1),
             eval_s=round(dt_, 1))
    return m, scored


def log(cfg, res):
    with open(f"{ROOT}/results.jsonl", "a") as f:
        f.write(json.dumps({"cfg": asdict(cfg), **res}) + "\n")
    print(json.dumps({"name": cfg.name, **res}), flush=True)
