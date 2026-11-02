"""Subset cache for the change-protocol smoke tests: GEOID chips with their whole S1 year (PC RTC,
centi-dB), GEOID's own pre / post (GRD, dB x 100), label and dates, in one flat file per split.

    python -m exp.geoidchange.cache --out /tmp/geoidchange --train 16000 --val 4000 --test 6000
"""
import argparse
import csv
import datetime as dt
import json
import random
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from manyup.data.datasets.geoidflood import DEFAULT_ROOT, ChipSelection, to_db
from manyup.data.datasets.geoidflood_series import PACKED, _meta

ROOT, S = Path(DEFAULT_ROOT), 64
_DATE = re.compile(r"(\d{8})")


def days(s):
    return (dt.datetime.strptime(s, "%Y%m%d").date() - dt.date(1970, 1, 1)).days


def one(a):
    chip, files, ev = a
    tile, rc = chip.split("@r")
    row, col = map(int, rc.split("c"))
    p = PACKED / tile.rsplit("-", 1)[0] / f"{tile}.npy"
    m = _meta(str(p.with_name(p.stem + "_meta.npz")))
    x = np.load(p, mmap_mode="r")[m["chip_row"][(row, col)]]
    w = Window(col, row, S, S)
    with rasterio.open(files["label"]) as r:
        y = r.read(1, window=w).astype(np.uint8)
    pp, ds = [], []
    for k in ("pre", "post"):
        with rasterio.open(files[k]) as r:
            db, ok = to_db(r.read(window=w))
        pp.append(np.where(np.asarray(ok), (np.asarray(db) * 100).round(), -32768).astype(np.int16))
        ds.append(days(_DATE.search(Path(files[k]).name).group(1)))
    return dict(chip=chip, x=np.asarray(x), y=y, geoid=np.stack(pp), day=m["day"].astype(np.int32),
                orbit=m["orbit_filled"].astype(np.int32), post=int(m["post_day"]),
                event=ev, pre=ds[0], post2=ds[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/geoidchange")
    ap.add_argument("--train", type=int, default=16000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--test", type=int, default=6000)
    a = ap.parse_args()
    ev = {}
    with open(ROOT / "data_tiles_s256_st128.csv") as f:
        for r in csv.DictReader(f):
            ev[r["label_id"][:-len("_label")]] = (dt.date.fromisoformat(r["event_date"][:10])
                                                  - dt.date(1970, 1, 1)).days
    idx = json.loads((ROOT / f"chip_index_{ChipSelection(64).key}.json").read_text())
    out = Path(a.out)
    rng = random.Random(0)
    for split, n in (("val", a.val), ("test", a.test), ("train", a.train)):
        ids = [c for c in idx["splits"][split]
               if (PACKED / c.split("@")[0].rsplit("-", 1)[0] / f"{c.split('@')[0]}.npy").exists()]
        ids = sorted(rng.sample(ids, min(n, len(ids))))
        jobs = [(c, idx["tiles"][c.split("@")[0]], ev[c.split("@")[0]]) for c in ids]
        off, recs = 0, []
        with open(out / f"{split}.x", "wb") as fx, Pool(8) as pool:
            ys, gs = [], []
            for i, r in enumerate(pool.imap(one, jobs, chunksize=16)):
                fx.write(r["x"].tobytes())
                ys.append(r["y"]); gs.append(r["geoid"])
                recs.append(dict(chip=r["chip"], off=off, A=len(r["day"]), post=r["post"],
                                 event=r["event"], pre=r["pre"], day=r["day"].tolist(),
                                 orbit=r["orbit"].tolist()))
                off += len(r["day"])
                if i % 1000 == 0:
                    print(split, i, len(jobs), flush=True)
        np.save(out / f"{split}.y.npy", np.stack(ys))
        np.save(out / f"{split}.geoid.npy", np.stack(gs))
        (out / f"{split}.json").write_text(json.dumps(recs))
        print("done", split, off, flush=True)
