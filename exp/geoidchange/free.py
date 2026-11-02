"""Model-free when: score each consecutive pair by dB drop; the anchors + block-level pooling."""
import logging
import numpy as np
from exp.geoidchange.harness import Chips, date_metrics, eval_ids, pool_months, month_key
logging.disable(logging.INFO)
import sys
c = Chips(sys.argv[1] if len(sys.argv) > 1 else "val")
ids = eval_ids(c, 1500)
rows = {k: [] for k in ("random", "mean_drop", "frac_drop3", "drop_vs_ref3", "latest")}
rng = np.random.default_rng(0)
for i in ids:
    r = c.recs[i]
    w = c.good(i, c.window(i))
    if len(w) < 2:
        continue
    a, ok = c.frames(i, w)
    d = c.recs[i]["day"][w]
    s = a.mean(1)                           # (T, 64, 64) VV+VH dB
    dif = s[1:] - s[:-1]
    sb = np.stack([s[max(0, t - 3):t].mean(0) for t in range(1, len(s))])
    sc = {"random": rng.random(len(dif)), "mean_drop": -dif.mean((1, 2)),
          "frac_drop3": (dif < -3).mean((1, 2)),
          "drop_vs_ref3": ((s[1:] - sb) < -3).mean((1, 2)),
          "latest": np.arange(len(dif))}
    for k, v in sc.items():
        rows[k].append((d[int(np.argmax(v)) + 1], r["post"], r["event"]))
for k, v in rows.items():
    print(f"{k:14s}", date_metrics(v), flush=True)

