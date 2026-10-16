"""Repack the fetched GEOID S1 series (exp.geoidflood.fetch_s1_series) chip-major, for reading.

A fetched tile file holds (A, 2, N, S, S): reading one chip's series would load the whole
tile. Packed, per tile:

    <out>/<event>/<tile>.npy        (N, A, 2, S, S) int16 centi-dB, NODATA = -32768 (memmap)
    <out>/<event>/<tile>_meta.npz   chips (N, 2), day (A,) int days since 1970-01-01,
                                    time (A,) ISO str, rel_orbit (A,), orbit (A,),
                                    source (A,) "pc" | "geoid", post_day, pre_day,
                                    post_idx, pre_idx (index in the packed axis, -1 = none)

Where Planetary Computer lacks GEOID's post (or pre, if within the window) scene, GEOID's own
scene is inserted at
its date (source "geoid"; Sentinel Hub sigma0, 1-2 px geolocation offset), so every series
contains the acquisition its labels were made from. Resumable (a packed tile is skipped).

    python -u -m exp.geoidflood.pack_s1_series --shard 0 --num_shards 8
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from exp.geoidflood.fetch_s1_series import HALF_WINDOW, NODATA, OUT as FETCHED
from manyup.data.time import to_days

PACKED = Path("data/GEOID-Flood-series-packed")


def day_of(iso: str) -> int:
    return to_days(dt.date.fromisoformat(iso[:10]))


def pack(src: Path, dst: Path) -> dict:
    d = np.load(src)
    data = d["data"]                                            # (A, 2, N, S, S)
    time = [str(t) for t in d["time"]]
    rel = list(d["rel_orbit"])
    orbit = [str(o) for o in d["orbit"]]
    source = ["pc"] * len(time)
    extra = []
    for kind in ("post", "pre"):
        iso = str(d[f"{kind}_time"])
        post_iso = str(d["post_time"])
        inside = iso and abs(day_of(iso) - day_of(post_iso)) <= HALF_WINDOW
        if inside and int(d[f"{kind}_idx"]) < 0:        # missing on PC, inside the window
            g = d[f"geoid_{kind}"]                              # (2, N, S, S)
            if (g != NODATA).any():
                extra.append((iso, g, kind))
    for iso, g, kind in extra:
        time.append(iso + "Z")
        rel.append(-1)
        orbit.append("")
        source.append("geoid")
        data = np.concatenate([data, g[None]], 0)
    order = np.argsort(np.array(time), kind="stable")
    data = data[order]
    time = [time[i] for i in order]
    rel = np.array(rel)[order]
    orbit = np.array(orbit)[order]
    source = np.array(source)[order]

    def idx_of(iso):
        if not iso:
            return -1
        t = dt.datetime.fromisoformat(iso)
        for j, s in enumerate(time):
            if abs(dt.datetime.fromisoformat(s[:19]) - t) <= dt.timedelta(minutes=2):
                return j
        return -1

    post, pre = str(d["post_time"]), str(d["pre_time"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(data.transpose(2, 0, 1, 3, 4))   # (N, A, 2, S, S)
    tmp = dst.with_name(dst.stem + ".tmp.npy")
    np.save(tmp, arr)
    np.savez(dst.with_name(dst.stem + "_meta.npz"), chips=d["chips"],
             day=np.array([day_of(s) for s in time], np.int32), time=np.array(time),
             rel_orbit=rel.astype(np.int32), orbit=orbit, source=source,
             post_day=np.array(day_of(post)), pre_day=np.array(day_of(pre) if pre else -1),
             post_idx=np.array(idx_of(post)), pre_idx=np.array(idx_of(pre)))
    tmp.replace(dst)                                            # the .npy last: it marks done
    return {"acq": len(time), "inserted": len(extra)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--src", default=str(FETCHED))
    ap.add_argument("--out", default=str(PACKED))
    a = ap.parse_args()
    files = sorted(Path(a.src).glob("*/*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")][a.shard::a.num_shards]
    done = inserted = 0
    for f in files:
        dst = Path(a.out) / f.parent.name / (f.stem + ".npy")
        if dst.exists():
            continue
        r = pack(f, dst)
        done, inserted = done + 1, inserted + r["inserted"]
    print(f"shard {a.shard}: packed {done} tiles, inserted {inserted} GEOID scenes", flush=True)


if __name__ == "__main__":
    main()
