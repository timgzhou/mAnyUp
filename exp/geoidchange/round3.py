"""Zoom (2b) with saved per-pair scores: a monthly model picks months, a date model runs only
inside them. Also monthly-pool sweep. Trains the two models and scores val, then composes."""
import json
import logging
import sys

import numpy as np

from exp.geoidchange.harness import Cfg, Chips, IoU, date_metrics, eval_ids, log, month_key, report, score_pairs, train

logging.disable(logging.INFO)


def compose(month_scored, date_scored, c, k):
    """Restrict the date model's argmax to the dates of the top-k month transitions."""
    ms = {i: (d, m) for i, d, m in month_scored}
    rows, seg, pairs = [], IoU(), 0
    for i, days, maps in date_scored:
        r = c.recs[i]
        sc = maps.float().mean((1, 2)).numpy()
        keep = np.ones(len(sc), bool)
        if i in ms:
            d, m = ms[i]
            s = m.float().mean((1, 2)).numpy()
            top = {month_key(d[t + 1]) for t in np.argsort(-s)[:k]}
            top |= {x - 1 for x in top} if False else set()
            keep = np.array([month_key(x) in top for x in days[1:]])
            if not keep.any():
                keep[:] = True
        pairs += int(keep.sum())
        t = int(np.where(keep, sc, -1).argmax())
        rows.append((days[t + 1], r["post"], r["event"]))
        seg.add((maps[t] > .5).numpy(), c.label(i))
    m = date_metrics(rows)
    m.update(iou_pred=round(seg.get(), 4), date_pairs=round(pairs / len(rows), 1))
    return m


def main():
    kw = dict(a.split("=", 1) for a in sys.argv[1:])
    split = kw.get("split", "val")
    c = Chips(split)
    dcfg = Cfg(name="d", mode="pair", ps=16, epochs=6)
    dn = train(dcfg)
    dres, dsc = report(dn, dcfg, split)
    print("date", json.dumps(dres), flush=True)
    for pool in kw.get("pools", "mean,min,median,max").split(","):
        mc = Cfg(name=f"m_{pool}", mode="month", pool=pool, ps=16, epochs=6, time="days")
        mn = train(mc)
        mres, ms = report(mn, mc, split)
        res = {"month": mres}
        for k in (1, 2, 3):
            res[f"zoom{k}"] = compose(ms, dsc, c, k)
        log(mc, res)


main()
