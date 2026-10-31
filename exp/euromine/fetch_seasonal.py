"""Fetch a seasonal Sentinel-2 L2A series for every EuroMineNet site: one cloud-masked median
composite per calendar quarter (Q1 Jan-Mar ... Q4 Oct-Dec), 2015 Q3 (the archive starts
2015-07-04) to 2024 Q4 = 38 quarters, on the site's release pixel grid (EPSG:4326, the release
transform), so every composite lines up with the release labels pixel for pixel.

Source: Microsoft Planetary Computer `sentinel-2-l2a` (COGs; the product OlmoEarth was pretrained
on). The release images themselves are top-of-atmosphere single-summer-date mosaics (best L1C
matches in August), so year t's label sits in Q3 of year t (a Q3 composite correlates 0.81-0.96
per band with the release image of its year, Q1 0.14-0.41; Czechia_4 2020).

Per (site, quarter): every scene intersecting the site with eo:cloud_cover < MAX_CLOUD, at most
MAX_SCENES of them (least cloudy first); a pixel observation counts when the scene's SCL is
vegetation / bare soil / water / unclassified / snow (4, 5, 6, 7, 11); the composite is the
per-pixel median over those (the +1000 BOA offset of processing baseline >= 04.00 removed), 0
where no clear observation exists. Reflectance bands are read bilinear, SCL nearest, in row
blocks (bounded memory on the 2000 x 4500 px sites).

    <out>/<site>/<year>Q<q>.npz   s2 (10, H, W) uint16 (B02 B03 B04 B05 B06 B07 B08 B8A B11 B12)
                                  n_clear (H, W) uint8, scenes (S,) str, n_scenes int

Resumable (an existing file is skipped); units shard over jobs:
    python -u -m exp.euromine.fetch_seasonal --shard 3 --num_shards 32 --threads 16
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                  GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="8",
                  GDAL_HTTP_RETRY_DELAY="3", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402

from manyup.data.datasets.euromine import BANDS, PACKED, ROOT  # noqa: E402

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
OUT = ROOT / "seasonal"
QUARTERS = [(2015, 3), (2015, 4)] + [(y, q) for y in range(2016, 2025) for q in (1, 2, 3, 4)]
CLEAR_SCL = (4, 5, 6, 7, 11)
MAX_CLOUD, MAX_SCENES = 80, 24
ROWS = 256                                  # row block height


def _post(url, body, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


class Token:
    """The collection's SAS token, refreshed before it expires (~1 h)."""

    def __init__(self):
        self.tok, self.t = None, 0.0

    def __call__(self) -> str:
        if self.tok is None or time.time() - self.t > 1800:
            self.tok = json.load(urllib.request.urlopen(TOKEN, timeout=60))["token"]
            self.t = time.time()
        return self.tok


def quarter_bounds(y: int, q: int) -> str:
    m0 = 3 * (q - 1) + 1
    end = f"{y + 1}-01-01" if q == 4 else f"{y}-{m0 + 3:02d}-01"
    return f"{y}-{m0:02d}-01T00:00:00Z/{end}T00:00:00Z"


def search(bbox, y, q) -> list[dict]:
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": quarter_bounds(y, q),
            "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}}, "limit": 500}
    feats = _post(STAC, body)["features"]
    feats.sort(key=lambda f: f["properties"]["eo:cloud_cover"])
    return feats[:MAX_SCENES]


def read_scene(it, tok, grid, r0, h):
    """One scene on rows [r0, r0 + h) of the site grid -> (10, h, W) float32 (nan: not clear)."""
    crs, tf, W = grid
    sub = Affine(tf.a, tf.b, tf.c + tf.b * r0, tf.d, tf.e, tf.f + tf.e * r0)
    def warp(asset, resampling):
        with rasterio.open(it["assets"][asset]["href"] + "?" + tok) as src, \
             WarpedVRT(src, crs=crs, transform=sub, width=W, height=h, resampling=resampling,
                       src_nodata=0, nodata=0) as v:
            return v.read(1)
    scl = warp("SCL", Resampling.nearest)
    clear = np.isin(scl, CLEAR_SCL)
    if not clear.any():
        return None
    off = 1000.0 if float(it["properties"].get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    x = np.stack([warp(b, Resampling.bilinear).astype(np.float32) for b in BANDS])
    x = np.clip(x - off, 0, None)
    x[:, ~clear | (x == 0).all(0)] = np.nan
    return x


def composite(site: str, f: dict, y: int, q: int, tok: Token, threads: int) -> dict:
    H, W = f["shape"]
    a, b, x0, d, e, y0 = f["transform"]
    bbox = [x0, y0 + e * H, x0 + a * W, y0]
    items = search(bbox, y, q)
    grid = (f["crs"], Affine(a, b, x0, d, e, y0), W)
    s2 = np.zeros((len(BANDS), H, W), np.uint16)
    n = np.zeros((H, W), np.uint8)
    with ThreadPoolExecutor(threads) as ex:
        for r0 in range(0, H, ROWS):
            h = min(ROWS, H - r0)
            t = tok()
            obs = [x for x in ex.map(lambda it: read_scene(it, t, grid, r0, h), items)
                   if x is not None]
            if not obs:
                continue
            st = np.stack(obs)                                          # (S, 10, h, W)
            ok = np.isfinite(st[:, 0])
            n[r0:r0 + h] = ok.sum(0)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # pixels never clear
                med = np.nanmedian(st, 0)
            s2[:, r0:r0 + h] = np.nan_to_num(np.round(med), nan=0).clip(0, 65535).astype(np.uint16)
    return {"s2": s2, "n_clear": n, "scenes": np.array([it["id"] for it in items]),
            "n_scenes": len(items)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sites", nargs="*", help="only these sites (default: all)")
    a = ap.parse_args()
    sites = json.loads((PACKED / "index.json").read_text())["sites"]
    units = [(s, y, q) for s in sorted(sites) if not a.sites or s in a.sites for y, q in QUARTERS]
    mine = units[a.shard::a.num_shards]
    tok = Token()
    done = 0
    for site, y, q in mine:
        path = OUT / site / f"{y}Q{q}.npz"
        if path.exists():
            continue
        t0 = time.time()
        out = composite(site, sites[site], y, q, tok, a.threads)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **out)
        tmp.replace(path)
        done += 1
        cov = float((out["n_clear"] > 0).mean())
        print(f"{site} {y}Q{q}: {out['n_scenes']} scenes, clear coverage {cov:.3f}, "
              f"median clear obs {int(np.median(out['n_clear']))}, {time.time() - t0:.0f} s",
              flush=True)
    print(f"shard {a.shard}/{a.num_shards}: {done} new of {len(mine)} units")


if __name__ == "__main__":
    main()
