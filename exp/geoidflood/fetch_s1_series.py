"""Fetch a year of Sentinel-1 around every GEOID-Flood event, for the chips we train on.

Source: Microsoft Planetary Computer `sentinel-1-rtc` (terrain-flattened gamma0, 10 m) -- the
product OlmoEarth was pretrained on (its rslearn config reads Sentinel1RTC from Planetary
Computer, IW, every orbit, in dB). GEOID-Flood itself pulled the SAME acquisitions (CEMS's own S1
scene when there is one) from Sentinel Hub; checked on a sample of tiles, Planetary Computer has
those scenes, geolocated 1-2 px (10-20 m) apart from GEOID's rasters (correlation 0.83-0.92
after the shift, 0.92-0.99 over 16 x 16 blocks).

Per tile that has at least one chip of the chip index (c64: 7,403 tiles), every acquisition
within +-HALF_WINDOW days of the tile's GEOID post-event scene ON THE RELATIVE ORBITS OF GEOID'S
post and pre scenes (the viewing geometries GEOID used; all orbits would be 2-4x the volume, and
mixed geometries add backscatter changes that are not on the ground), read onto the tile's own
UTM grid (bilinear) and cut to the chips. A GEOID scene's orbit is found among the Planetary
Computer passes by its repeat: same time of day (+-3 min), a multiple of 6 days apart (S1A and
S1B share the 12-day orbit grid, 6 days out of phase).

    <out>/<event>/<tile>.npz
        chips      (N, 2) int   (row, col) pixel offsets of the chips in the tile
        data       (A, 2, N, S, S) int16   VV, VH in centi-dB (round(100 * dB)); NODATA if none
        time       (A,) str    acquisition start (ISO, UTC)
        orbit      (A,) str    "ascending" / "descending"
        rel_orbit  (A,) int
        platform   (A,) str
        post_time, pre_time    GEOID's scene times; post_idx / pre_idx their index in time
                               (-1: Planetary Computer lacks it, or it is outside the window)
        geoid_post, geoid_pre  (2, N, S, S) int16 centi-dB: GEOID's own s1grd scenes on the
                               chips (Sentinel Hub sigma0, 1-2 px geolocation offset), for
                               tiles whose scene Planetary Computer lacks
        orbits_kept            relative orbits stored (GEOID post's, pre's)

An acquisition is one (platform, absolute orbit, date): the scenes (frames) of one pass that
cover the tile are mosaicked, first valid pixel wins. Resumable: a tile whose file exists is
skipped. Shard over tiles for parallel jobs:

    python -u -m exp.geoidflood.fetch_s1_series --shard 3 --num_shards 32 --threads 16
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                  GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="6",
                  GDAL_HTTP_RETRY_DELAY="2", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from rasterio.windows import Window  # noqa: E402

ROOT = Path("data/GEOID-Flood-full/geoid-flood")
OUT = Path("data/GEOID-Flood-series")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
COLLECTION = "sentinel-1-rtc"
HALF_WINDOW = 183          # days each side of the post-event scene: 12 months stored
NODATA = -32768
SCENE_TOL = dt.timedelta(minutes=2)
_TS = re.compile(r"_(pre|post)_(\d{8}T\d{6})")


class Token:
    """Planetary Computer SAS token (anonymous), refreshed well before it expires."""

    def __init__(self):
        self.lock, self.value, self.t = threading.Lock(), None, 0.0

    def get(self) -> str:
        with self.lock:
            if self.value is None or time.time() - self.t > 20 * 60:
                self.value = _json(TOKEN_URL)["token"]
                self.t = time.time()
            return self.value


def _json(url, body=None, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode() if body else None,
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def search(bbox, t0: dt.datetime, t1: dt.datetime) -> list[dict]:
    body = {"collections": [COLLECTION], "bbox": bbox, "limit": 1000,
            "datetime": f"{t0.isoformat()}Z/{t1.isoformat()}Z"}
    items, url = [], STAC
    while True:
        r = _json(url, body)
        items += r["features"]
        nxt = [link for link in r.get("links", []) if link.get("rel") == "next"]
        if not nxt:
            return items
        url, body = nxt[0]["href"], nxt[0].get("body", body)


def tiles_with_chips(index_path: Path) -> dict[str, list[tuple[int, int]]]:
    idx = json.loads(index_path.read_text())
    chips = defaultdict(set)
    for ids in idx["splits"].values():
        for c in ids:
            tile, rc = c.split("@")
            r, col = re.fullmatch(r"r(\d+)c(\d+)", rc).groups()
            chips[tile].add((int(r), int(col)))
    return {t: sorted(v) for t, v in sorted(chips.items())}


def scene_time(tile: str, kind: str) -> dt.datetime | None:
    event = tile.rsplit("-", 1)[0]
    hits = sorted((ROOT / event / "s1grd").glob(f"{tile}_s1grd_{kind}_*.tif"))
    if not hits:
        return None
    return dt.datetime.strptime(_TS.search(hits[0].name).group(2), "%Y%m%dT%H%M%S")


def read_acquisition(items, grid, win: Window, token: Token) -> np.ndarray:
    """Mosaic of one pass's scenes on the tile grid window -> (2, h, w) float32 linear, NaN
    where no scene has data."""
    crs, transform, W, H = grid
    out = np.full((2, int(win.height), int(win.width)), np.nan, np.float32)
    for it in items:
        for b, band in enumerate(("vv", "vh")):
            if band not in it["assets"]:        # single-pol scene (VV only): band stays NODATA
                continue
            href = it["assets"][band]["href"] + "?" + token.get()
            for attempt in range(4):
                try:
                    with rasterio.open(href) as src, WarpedVRT(
                            src, crs=crs, transform=transform, width=W, height=H,
                            resampling=Resampling.bilinear, src_nodata=src.nodata,
                            nodata=np.nan, dtype="float32") as v:
                        x = v.read(1, window=win)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
            fill = np.isnan(out[b])
            out[b][fill] = x[fill]
    return out


def same_orbit(t: dt.datetime, t_of: dict, orbit_of: dict) -> set:
    """Relative orbit(s) of an acquisition at time t, from the passes that repeat it."""
    out = set()
    for k, tk in t_of.items():
        tod = abs(((tk - t).total_seconds() + 43200) % 86400 - 43200)       # time-of-day gap
        days = round((tk - t).total_seconds() / 86400)
        if tod < 180 and days % 6 == 0:
            out.add(orbit_of[k])
    return out


def geoid_chips(tile: str, kind: str, chips, size: int) -> np.ndarray:
    event = tile.rsplit("-", 1)[0]
    hits = sorted((ROOT / event / "s1grd").glob(f"{tile}_s1grd_{kind}_*.tif"))
    if not hits:
        return np.full((2, len(chips), size, size), NODATA, np.int16)
    with rasterio.open(hits[0]) as g:
        x = np.stack([g.read(window=Window(c, r, size, size)) for r, c in chips], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10 * np.log10(np.where(np.isfinite(x) & (x > 0), x, np.nan))
    return np.where(np.isfinite(db), np.clip(np.round(db * 100), -32767, 32767),
                    NODATA).astype(np.int16)


def fetch_tile(tile: str, chips: list[tuple[int, int]], size: int, token: Token,
               threads: int) -> dict:
    post, pre = scene_time(tile, "post"), scene_time(tile, "pre")
    event = tile.rsplit("-", 1)[0]
    ref = sorted((ROOT / event / "s1grd").glob(f"{tile}_s1grd_post_*.tif"))[0]
    with rasterio.open(ref) as g:
        grid = (g.crs, g.transform, g.width, g.height)
        bbox = list(transform_bounds(g.crs, "EPSG:4326", *g.bounds))
    rows = [r for r, _ in chips]
    cols = [c for _, c in chips]
    win = Window(min(cols), min(rows), max(cols) + size - min(cols), max(rows) + size - min(rows))
    items = search(bbox, post - dt.timedelta(days=HALF_WINDOW), post + dt.timedelta(days=HALF_WINDOW))
    passes = defaultdict(list)            # one pass = platform + absolute orbit (+ date)
    for it in items:
        p = it["properties"]
        key = (p.get("platform"), p.get("sat:absolute_orbit"), p["datetime"][:10])
        passes[key].append(it)
    t_of = {k: dt.datetime.fromisoformat(min(i["properties"]["datetime"] for i in v)
                                         .replace("Z", "+00:00")).replace(tzinfo=None)
            for k, v in passes.items()}
    orbit_of = {k: passes[k][0]["properties"].get("sat:relative_orbit") for k in passes}
    kept = sorted({o for t in (post, pre) if t is not None for o in same_orbit(t, t_of, orbit_of)})
    keys = sorted((k for k in passes if orbit_of[k] in kept), key=t_of.get)

    def one(k):
        x = read_acquisition(passes[k], grid, win, token)
        cut = np.stack([x[:, r - min(rows):r - min(rows) + size, c - min(cols):c - min(cols) + size]
                        for r, c in chips], 1)                                  # (2, N, S, S)
        with np.errstate(invalid="ignore", divide="ignore"):
            db = 10 * np.log10(cut)
        return np.where(np.isfinite(db), np.clip(np.round(db * 100), -32767, 32767),
                        NODATA).astype(np.int16)

    with ThreadPoolExecutor(threads) as ex:
        data = list(ex.map(one, keys))
    times = [min(i["properties"]["datetime"] for i in passes[k]) for k in keys]
    first = [passes[k][0]["properties"] for k in keys]
    keep = [i for i, d in enumerate(data) if (d != NODATA).any()]          # drop empty passes

    def index_of(t):
        if t is None:
            return -1
        for j, i in enumerate(keep):
            ti = dt.datetime.fromisoformat(times[i].replace("Z", "+00:00")).replace(tzinfo=None)
            if abs(ti - t) <= SCENE_TOL:
                return j
        return -1

    return dict(
        chips=np.array(chips, np.int32),
        data=np.stack([data[i] for i in keep]) if keep else
        np.zeros((0, 2, len(chips), size, size), np.int16),
        time=np.array([times[i] for i in keep]),
        orbit=np.array([first[i].get("sat:orbit_state", "") for i in keep]),
        rel_orbit=np.array([first[i].get("sat:relative_orbit", -1) for i in keep], np.int32),
        platform=np.array([first[i].get("platform", "") for i in keep]),
        post_time=np.array(post.isoformat()), pre_time=np.array(pre.isoformat() if pre else ""),
        post_idx=np.array(index_of(post)), pre_idx=np.array(index_of(pre)),
        geoid_post=geoid_chips(tile, "post", chips, size),
        geoid_pre=geoid_chips(tile, "pre", chips, size),
        orbits_kept=np.array(kept, np.int32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", default=str(ROOT / "chip_index_c64_mapped0.5_class2.json"))
    ap.add_argument("--chip", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="tiles (smoke test)")
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    tiles = list(tiles_with_chips(Path(a.index)).items())[a.shard::a.num_shards]
    if a.limit:
        tiles = tiles[:a.limit]
    token, t_start, done, failed = Token(), time.time(), 0, 0
    out_root = Path(a.out)
    print(f"shard {a.shard}/{a.num_shards}: {len(tiles)} tiles", flush=True)
    for tile, chips in tiles:
        path = out_root / tile.rsplit("-", 1)[0] / f"{tile}.npz"
        if path.exists():
            continue
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            print(f"STOPPED (--max_hours) after {done} tiles; rerun to resume", flush=True)
            return
        t0 = time.time()
        try:
            res = fetch_tile(tile, chips, a.chip, token, a.threads)
        except Exception as e:                  # one bad tile must not stop the shard
            failed += 1
            print(f"FAILED {tile}: {type(e).__name__}: {e}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **res)
        tmp.replace(path)
        done += 1
        print(f"{tile}: {len(chips)} chips, {len(res['time'])} acquisitions, post_idx "
              f"{int(res['post_idx'])}, pre_idx {int(res['pre_idx'])}, "
              f"{res['data'].nbytes / 2**20:.0f} MB, {time.time() - t0:.0f}s", flush=True)
    print(f"shard {a.shard} done: {done} fetched, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
