"""Part 1: spatial-only mixed resolution on GEOID's (pre, post) pair.

Pass 1 encodes both dates at ps_lr, head(post - pre) gives the change map. The most uncertain
windows (entropy of that map; random / label oracle as controls) get a crop token on BOTH dates;
crops read the season tokens and the other crops (attn cross_hr), the crop pair's difference goes
through the same head (area-pooled to the crop), and replaces the coarse logits there.

    python -m exp.geoidchange.mixbi name=m sel=entropy q=0.25 crop=8
"""
import logging
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from exp.geoidchange.harness import (CE_W, DB_FLOOR, MODEL_SIZE_TO_ID, NODATA, Chips, Cfg, INP, IoU,
                                     OlmoEarthEncoder, Train, collate, eval_ids, to_sample)
from manyup.encoders.olmoearth import crop_patches
from manyup.inject import MixedResOE

logging.disable(logging.INFO)


class MixBi(nn.Module):
    def __init__(self, lr_ps=16, crop=8):
        super().__init__()
        enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID["tiny"], INP, 16, 64, name="tiny").encoder
        self.mix = MixedResOE(enc, 0, "center", lr_ps, attn="cross_hr")
        self.lr_ps, self.crop = lr_ps, crop
        self.head = nn.Linear(enc.embedding_size, 2 * 16 * 16)
        nn.init.zeros_(self.head.weight)

    def tok2map(self, d, ps):
        """(..., D) -> (N, 2, ps, ps)."""
        m = self.head(d).view(-1, 2, 16, 16)
        return F.avg_pool2d(m, 16 // ps) if ps < 16 else m

    def forward(self, s, sel="entropy", q=0.25, y=None):
        s = s._replace(**{k: v.cuda(non_blocking=True) for k, v in s._asdict().items()
                          if v is not None})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr = self.mix.embed(s, self.lr_ps)
            enc = self.mix.lr_trace(lr)
        f = lr.pooled(enc[1]).float()                                      # (B, 2, h, w, D)
        B, _, h, w, _ = f.shape
        P = self.lr_ps
        m0 = self.tok2map((f[:, 1] - f[:, 0]).reshape(B * h * w, -1), P)
        m0 = m0.view(B, h, w, 2, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, 2, h * P, w * P)
        if q <= 0:
            return m0, 0, m0
        cp = self.crop
        with torch.no_grad():
            lp = m0.float().log_softmax(1)
            u = -(lp.exp() * lp).sum(1)
            if sel == "random":
                u = torch.rand_like(u)
            elif sel == "oracle":
                u = F.cross_entropy(m0.float(), y.cuda(), ignore_index=-1, reduction="none")
            elif sel == "grid":
                u = (torch.arange(64, device=u.device)[None] // 3 % 2 == 0).float()
                u = (u[..., None] * u[:, None] + torch.rand_like(u[:, :1]) * 1e-3)
            win = F.avg_pool2d(u[:, None], cp)[:, 0]                        # (B, W, W)
            k = max(1, round(q * win[0].numel()))
            idx = win.flatten(1).topk(k, 1).indices                        # (B, k)
        b = torch.arange(B, device=idx.device).repeat_interleave(k)
        py, px = (idx.flatten() // win.shape[-1]) * cp, (idx.flatten() % win.shape[-1]) * cp
        n = len(b)
        bb, yy, xx = b.repeat(2), py.repeat(2), px.repeat(2)
        tt = torch.cat([torch.zeros(n, dtype=torch.long, device=b.device),
                        torch.ones(n, dtype=torch.long, device=b.device)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crop_patches(s, bb, tt, yy, xx, cp, pixels=True), cp,
                                slot=tt, py=yy, px=xx)
            _, (h_x,) = self.mix.tail_multi(lr, [(hr, bb)], enc)
        hr_f = hr.pooled(h_x).float()[:, 0, 0, 0]
        cm = self.tok2map(hr_f[n:] - hr_f[:n], cp)                          # (n, 2, cp, cp)
        ar = torch.arange(cp, device=b.device)
        r = py[:, None, None] + ar[None, :, None]
        c = px[:, None, None] + ar[None, None, :]
        out = m0.permute(0, 2, 3, 1).clone()
        out[b[:, None, None], r, c] = cm.permute(0, 2, 3, 1).to(out.dtype)
        return out.permute(0, 3, 1, 2), n * 2, m0


@torch.no_grad()
def evaluate(net, split, sel, q, n=4000):
    c = Chips(split)
    net.eval()
    io, ioa = IoU(), IoU()
    ids = eval_ids(c, n)
    t, cnt = 0, 0
    for k in range(0, len(ids), 128):
        b = ids[k:k + 128]
        g = np.stack([np.asarray(c.G[i], np.float32) for i in b])
        g = np.where(g != NODATA, g / 100, DB_FLOOR)
        days = np.array([[c.recs[i]["pre"], c.recs[i]["post"]] for i in b])
        y = np.stack([c.label(i) for i in b])
        out, nc, m0 = net(to_sample(g, days, np.ones(days.shape, bool)), sel, q,
                          torch.from_numpy(y))
        io.add(out.argmax(1).cpu().numpy(), y)
        ioa.add(m0.argmax(1).cpu().numpy(), y)
        t, cnt = t + nc, cnt + len(b)
    return round(io.get(), 4), round(ioa.get(), 4), round(t / cnt, 1)


def main(argv):
    kw = dict(a.split("=", 1) for a in argv)
    sel, q, crop = kw.get("sel", "entropy"), float(kw.get("q", .25)), int(kw.get("crop", 8))
    lr_ps, ep = int(kw.get("lr_ps", 16)), int(kw.get("epochs", 6))
    start = int(kw.get("start", 2))          # crops from this epoch (the first: head only)
    tq = float(kw.get("tq", q))
    cfg = Cfg(mode="bi", ps=lr_ps, bs=64)
    torch.manual_seed(0)
    net = MixBi(lr_ps, crop).cuda()
    dl = DataLoader(Train(Chips("train"), cfg), batch_size=64, shuffle=True, num_workers=7,
                    collate_fn=collate, drop_last=True, persistent_workers=True, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    w = torch.tensor(CE_W).cuda()
    t0 = time.time()
    for e in range(ep):
        net.mix.encoder.requires_grad_(e >= 1)
        if e == 1:
            for g in opt.param_groups:
                g["lr"] = 1e-4
        net.train()
        ls, n = 0.0, 0
        for s, *_, y in dl:
            out, _, m0 = net(s, sel, tq if e >= start else 0.0, y)
            loss = F.cross_entropy(out.float(), y.cuda(), weight=w, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ls, n = ls + loss.item(), n + 1
        print(f"epoch {e + 1}/{ep} loss {ls / n:.4f} {time.time() - t0:.0f}s", flush=True)
    va, va0, nc = evaluate(net, "val", sel, q)
    te, te0, _ = evaluate(net, "test", sel, q)
    import json
    res = dict(name=kw.get("name"), sel=sel, q=q, crop=crop, lr_ps=lr_ps, val=va, val_lr=va0,
               test=te, test_lr=te0, crops=nc)
    print(json.dumps(res), flush=True)
    open("/tmp/geoidchange/mixbi.jsonl", "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
