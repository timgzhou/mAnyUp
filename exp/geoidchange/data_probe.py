"""No model: when do labelled flood pixels darken, and which month pooling keeps it?"""
import datetime as dt
import json
import sys

import numpy as np

ROOT = "/tmp/geoidchange"
NODATA = -32768


def month_of(day):
    d = dt.date(1970, 1, 1) + dt.timedelta(int(day))
    return d.year * 12 + d.month


def load(split):
    recs = json.load(open(f"{ROOT}/{split}.json"))
    x = np.memmap(f"{ROOT}/{split}.x", dtype=np.int16, mode="r").reshape(-1, 2, 64, 64)
    return recs, x, np.load(f"{ROOT}/{split}.y.npy")


def auc(pos, neg):
    v = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(v)) + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(split):
    recs, X, Y = load(split)
    rng = np.random.default_rng(0)
    offs, offs_ev, gap, is_post, skip = [], [], [], [], 0
    pools = {}
    for r, y in zip(recs, Y):
        if r["chip"].startswith("EMSR342"):
            continue
        flood, land = y == 2, y == 0
        if flood.sum() < 64 or land.sum() < 64:
            skip += 1
            continue
        day = np.array(r["day"])
        a = X[r["off"]:r["off"] + len(day)].astype(np.float32)
        ok = a != NODATA
        a = np.where(ok, a / 100, np.nan)
        # within [post - 60, post + 12]: relative VV of flood vs land per date
        sel = np.nonzero((day >= r["post"] - 90) & (day <= r["post"] + 30))[0]
        if len(sel) < 4 or r["post"] not in day:
            skip += 1
            continue
        v = np.nanmedian(a[sel, 0][:, flood], 1) - np.nanmedian(a[sel, 0][:, land], 1)
        dv = v[1:] - v[:-1]
        if not np.isfinite(dv).any():
            skip += 1
            continue
        i = int(np.nanargmin(dv)) + 1
        offs.append(day[sel[i]] - r["post"])
        offs_ev.append(day[sel[i]] - r["event"])
        is_post.append(day[sel[i]] == r["post"])
        gap.append(r["post"] - r["event"])
        # pooling: month of post vs previous month, per pixel
        m = np.array([month_of(d) for d in day])
        mp = month_of(r["post"])
        now, prev = a[m == mp], a[m == mp - 1]
        pi = int(np.nonzero(day == r["post"])[0][0])
        if len(prev) == 0 or pi == 0:
            continue
        fi = rng.choice(np.nonzero(flood.ravel())[0], 64)
        li = rng.choice(np.nonzero(land.ravel())[0], 64)
        chg = {}
        for name, f in (("mean", np.nanmean), ("median", np.nanmedian), ("min", np.nanmin),
                        ("max", np.nanmax)):
            chg[f"month {name}"] = f(now, 0) - f(prev, 0)
            chg[f"level {name}"] = f(now, 0)
        chg["date post-prevdate"] = a[pi] - a[pi - 1]
        chg["date post-prevmonthmean"] = a[pi] - np.nanmean(prev, 0)
        chg["date post-prevmonthmin"] = a[pi] - np.nanmin(prev, 0)
        chg["level post date"] = a[pi]
        for k, c in chg.items():
            for b in (0, 1):
                f_, l_ = c[b].ravel()[fi], c[b].ravel()[li]
                pools.setdefault((k, "VV VH".split()[b]), [[], [], 0, 0])
                p = pools[(k, "VV VH".split()[b])]
                p[0].append(f_); p[1].append(l_)
    offs, offs_ev = np.array(offs), np.array(offs_ev)
    print(f"{split}: {len(offs)} chips, skipped {skip}")
    print("biggest flood-vs-land drop is the post date itself:", np.mean(is_post).round(3))
    print("offset of biggest drop from post (d) pct 5/25/50/75/95:", np.percentile(offs, [5, 25, 50, 75, 95]))
    print("from CEMS event date:", np.percentile(offs_ev, [5, 25, 50, 75, 95]))
    print("|offs| <= 0/6/12/24 d:", [round(float(np.mean(np.abs(offs) <= d)), 3) for d in (0, 6, 12, 24)])
    print("post - event (d):", np.percentile(gap, [5, 25, 50, 75, 95]))
    for k, (f_, l_, *_) in sorted(pools.items()):
        f_, l_ = np.concatenate(f_), np.concatenate(l_)
        ok = np.isfinite(f_)
        ok2 = np.isfinite(l_)
        print(f"{k[0]:28s}{k[1]} AUC(flood lower) {1 - auc(f_[ok], l_[ok2]):.3f}  "
              f"med flood {np.median(f_[ok]):+.2f} land {np.median(l_[ok2]):+.2f}")


main(sys.argv[1] if len(sys.argv) > 1 else "val")
