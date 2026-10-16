"""Is innovation LEARNABLE from what a selector can see before it looks? Fits a small predictor
on the per-slot dumps of exp.upsamplers.innovation_diag (--dump) and reports how much of the
true innovation its top-q slots capture on held-out samples (random: q; oracle: the diag's
top-q share).

Per (month t, ps4 cell p) slot, the predictor sees only pre-look inputs:
    e_t(p), e_t(p) - season mean of e(p)   raw ps4 patch embedding of the month (D each)
    F0[:, p16]                             crop-free contextual season tokens at p's ps16
                                           cell, ALL seasons (NS x D) -- "all seasons' features"
    s_logits[:, p]                         crop-free native logits of every season over p (NS x C)
    onehots                                t's season (NS), month index (T), p's place in its
                                           ps16 cell (16); log pixel variance of the month in p
-> Linear -> [--temporal: self-attention over the months of the cell] -> MLP -> score.
Loss: listwise, per sample: cross-entropy between softmax(scores over present slots) and the
innovation shares inn / sum(inn) -- the ranking top-q selection needs, not its scale.

    python -m exp.upsamplers.fit_innovation --fit cache/innov/pastis_val \
        --test cache/innov/pastis_test [--temporal] [--epochs 30]
Prints capture@0.1/0.2/0.3 (predictor, oracle, random) and the within-cell month choice.
"""
import argparse
import glob
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

QS = (0.1, 0.2, 0.3)


def load(dirname):
    """All dumped batches, concatenated on the sample axis."""
    parts = [torch.load(f, weights_only=False) for f in sorted(glob.glob(f"{dirname}/batch*.pt"))]
    if not parts:
        raise SystemExit(f"no dumps in {dirname}")
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0] if parts[0][k] is not None}


def pool_prelook(d, f: int = 2):
    """innovation_diag.prelook inputs on the ps(4 f) grid: each coarser cell's raw embedding,
    pixel variance and season logits = the mean over its f x f ps4 cells (F0 is per ps16 cell
    already). For a joint selector choosing among ps4 and ps8 crops."""
    out = dict(d)
    e = d["e"].float()                                                            # (B,T,H,W,D)
    B, T, H, W, D = e.shape
    out["e"] = e.reshape(B, T, H // f, f, W // f, f, D).mean((3, 5)).to(d["e"].dtype)
    pv = d["pixel_var"].float()
    out["pixel_var"] = pv.reshape(B, T, H // f, f, W // f, f).mean((3, 5))
    sl = d["s_logits"].float()                                                    # (B,G,H,W,C)
    G, C = sl.shape[1], sl.shape[-1]
    out["s_logits"] = sl.reshape(B, G, H // f, f, W // f, f, C).mean((3, 5)).to(d["s_logits"].dtype)
    out.pop("inn", None)
    return out


def features(d, idx, dev):
    """Samples idx -> x (n, T, Q, Fdim) pre-look features, target (n, T, Q), valid (n, T, Q).
    Without "inn" (test-time selection): target zeros, valid = present."""
    e = d["e"][idx].to(dev).float()                                   # (n,T,16,16,D)
    n, T, H, W, D = e.shape
    season, present = d["season"][idx].to(dev), d["present"][idx].to(dev)
    G = d["F0"].shape[1]
    w = F.one_hot(season.clamp(min=0), G).float() * present[..., None].float()   # (n,T,G)
    e_s = torch.einsum("ntg,nthwd->nghwd", w, e) / w.sum(1).clamp(min=1)[:, :, None, None, None]
    e_bar = e_s[torch.arange(n, device=dev)[:, None], season.clamp(min=0)]         # (n,T,16,16,D)
    F0 = d["F0"][idx].to(dev).float()                                             # (n,G,4,4,D)
    f = H // F0.shape[2]                                                          # 4 (ps4) or 2 (ps8)
    F0 = F0.repeat_interleave(f, 2).repeat_interleave(f, 3)                       # (n,G,H,W,D)
    F0 = F0.permute(0, 2, 3, 1, 4).reshape(n, 1, H, W, G * D).expand(n, T, H, W, G * D)
    parts = [e, e - e_bar, F0]
    if "s_logits" in d:               # absent: a selector that reads no season logits
        sl = d["s_logits"][idx].to(dev).float()                                   # (n,G,16,16,C)
        C = sl.shape[-1]
        parts.append(sl.permute(0, 2, 3, 1, 4).reshape(n, 1, H, W, G * C).expand(n, T, H, W, G * C))
    oh_s = F.one_hot(season.clamp(min=0), G).float()[:, :, None, None].expand(n, T, H, W, G)
    oh_t = torch.eye(T, device=dev)[None, :, None, None].expand(n, T, H, W, T)
    sub = torch.arange(f * f, device=dev).view(f, f).repeat(4, 4)                 # place in ps16 cell
    oh_p = F.one_hot(sub, 16).float()[None, None].expand(n, T, H, W, 16)
    parts += [oh_s, oh_t, oh_p]
    if "pixel_var" in d:              # absent: no hand-made pixel statistic
        parts.append(d["pixel_var"][idx].to(dev).float().clamp(min=0).log1p()[..., None])
    x = torch.cat(parts, -1)
    if "inn" in d:
        tgt = d["inn"][idx].to(dev).float()
        valid = present[:, :, None, None].expand(n, T, H, W) & (tgt > 0)
    else:
        tgt = torch.zeros(n, T, H, W, device=dev)
        valid = present[:, :, None, None].expand(n, T, H, W)
    return x.reshape(n, T, H * W, -1), tgt.reshape(n, T, H * W), valid.reshape(n, T, H * W)


class Predictor(nn.Module):
    def __init__(self, fdim, hidden=256, temporal=False):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, hidden), nn.GELU())
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True) if temporal else None
        self.n1 = nn.LayerNorm(hidden) if temporal else None
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x, valid):
        """x (n, T, Q, F), valid (n, T, Q) -> scores (n, T, Q)."""
        h = self.inp(x)
        if self.attn is not None:                    # the months of each cell see each other
            n, T, Q, Hd = h.shape
            hc = h.permute(0, 2, 1, 3).reshape(n * Q, T, Hd)
            vc = valid.permute(0, 2, 1).reshape(n * Q, T)
            pad = ~vc
            pad[pad.all(1)] = False                  # keep empty cells finite (masked later)
            a, _ = self.attn(self.n1(hc), self.n1(hc), self.n1(hc), key_padding_mask=pad)
            h = (hc + a).reshape(n, Q, T, Hd).permute(0, 2, 1, 3)
        return self.out(h)[..., 0]


def listwise_loss(score, tgt, valid):
    n = score.shape[0]
    s = score.masked_fill(~valid, float("-inf")).reshape(n, -1)
    p = (tgt * valid).reshape(n, -1)
    p = p / p.sum(1, keepdim=True).clamp(min=1e-12)
    ok = valid.reshape(n, -1).any(1)
    return -(p * torch.log_softmax(s, 1).nan_to_num(0.0, neginf=0.0)).sum(1)[ok].mean()


def capture(score, tgt, valid):
    """Per sample share of innovation inside the top-q slots by score / by the truth, and the
    within-cell month choice (predicted best month == true best month)."""
    res = {f"pred@{q}": [] for q in QS} | {f"oracle@{q}": [] for q in QS} | {"month_hit": [],
                                                                           "month_chance": []}
    for i in range(score.shape[0]):
        v = valid[i]
        if v.sum() < 10:
            continue
        s, t = score[i][v], tgt[i][v]
        tot = t.sum().clamp(min=1e-12)
        for q in QS:
            k = max(1, round(q * len(t)))
            res[f"pred@{q}"].append(float(t[s.argsort(descending=True)[:k]].sum() / tot))
            res[f"oracle@{q}"].append(float(t.sort(descending=True).values[:k].sum() / tot))
        sc = score[i].masked_fill(~valid[i], float("-inf"))            # (T, Q)
        tc = tgt[i].masked_fill(~valid[i], float("-inf"))
        cells = valid[i].any(0)
        res["month_hit"].append(float((sc.argmax(0) == tc.argmax(0))[cells].float().mean()))
        res["month_chance"].append(float((1 / valid[i].sum(0)[cells].float()).mean()))
    return {k: sum(v) / max(len(v), 1) for k, v in res.items()}


def evaluate(model, d, dev, bs=16):
    model.eval()
    agg, n = {}, 0
    with torch.no_grad():
        for s in range(0, len(d["present"]), bs):
            idx = torch.arange(s, min(s + bs, len(d["present"])))
            x, tgt, valid = features(d, idx, dev)
            r = capture(model(x, valid), tgt, valid)
            for k, v in r.items():
                agg[k] = agg.get(k, 0.0) + v * len(idx)
            n += len(idx)
    return {k: v / n for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="dump dir to fit on (90%) + early-stop on (10%)")
    ap.add_argument("--test", required=True, help="held-out dump dir")
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    ap.add_argument("--save", default="", help="save the predictor (for eval_innovation_picks)")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    fit, test = load(a.fit), load(a.test)
    N = len(fit["present"])
    perm = torch.randperm(N)
    tr, va = perm[: int(0.9 * N)], perm[int(0.9 * N):]
    fdim = features(fit, tr[:1], dev)[0].shape[-1]
    model = Predictor(fdim, temporal=a.temporal).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sub = lambda d, idx: {k: v[idx] for k, v in d.items()}              # noqa: E731
    va_d = sub(fit, va)
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        order = tr[torch.randperm(len(tr))]
        tot = 0.0
        for s in range(0, len(order), a.bs):
            x, tgt, valid = features(fit, order[s:s + a.bs], dev)
            loss = listwise_loss(model(x, valid), tgt, valid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(order[s:s + a.bs])
        r = evaluate(model, va_d, dev)
        print(f"epoch {ep + 1} loss {tot / len(order):.4f} | held-out fit split "
              f"capture@0.2 {r['pred@0.2']:.3f} (oracle {r['oracle@0.2']:.3f})", flush=True)
        if r["pred@0.2"] > best:
            best = r["pred@0.2"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "fdim": fdim, "temporal": a.temporal}, a.save)
    r = evaluate(model, test, dev)
    r |= {"random@" + str(q): q for q in QS}
    r |= {"fit": a.fit, "test": a.test, "temporal": a.temporal, "n_fit": N,
          "n_test": len(test["present"])}
    print("TEST", json.dumps(r, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
