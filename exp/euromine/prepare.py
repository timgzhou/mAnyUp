"""Pack the EuroMineNet release (data/EuroMineNet/raw/EuroMineNet/<site>/{image,label}/Year<y>.tif)
into the arrays manyup.data.datasets.euromine reads:

    python -u -m exp.euromine.prepare pack [--workers 16]
    python -u -m exp.euromine.prepare verify          # every tile vs the GeoTIFFs
    python -u -m exp.euromine.prepare pack_seasonal   # fetch_seasonal's quarters -> one array per site

image Year<y>.tif: 10 bands int16 (B2 B3 B4 B5 B6 B7 B8 B8A B11 B12), EPSG:4326 at ~10 m;
label Year<y>.tif: uint8 {0, 255}, same pixel grid, no georeference (the image's is used).
A site's ten years share one shape except where the release is off by a pixel or two at the
edge; then every year is cut to the common top-left extent (recorded in index.json as
"trimmed"). Negative reflectances (none expected) are clipped to 0 and counted.

Splits: the release's datasets/train_val_test_splits.py lists (SPLITS below; its train list
names Germany_13 twice, and Greece_3 is in no list, so it is left out as in the release).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

from manyup.data.datasets.euromine import (BANDS, PACKED, ROOT, SEASONAL, TILE, YEARS,
                                           tile_origins)

RAW = ROOT / "raw" / "EuroMineNet"
SPLITS = {
    "train": ["Austria_1", "Bulgaria_10", "Bulgaria_11", "Bulgaria_13", "Bulgaria_14",
              "Bulgaria_15", "Bulgaria_16", "Bulgaria_17", "Bulgaria_18", "Bulgaria_2",
              "Bulgaria_4", "Bulgaria_5", "Bulgaria_6", "Bulgaria_7", "Bulgaria_9", "Czechia_1",
              "Czechia_3", "Finland_1", "Finland_2", "Finland_4", "Germany_1", "Germany_10",
              "Germany_11", "Germany_12", "Germany_13", "Germany_14", "Germany_15",
              "Germany_16", "Germany_3", "Germany_5", "Germany_7", "Germany_8", "Greece_1",
              "Greece_11", "Greece_12", "Greece_13", "Greece_2", "Greece_4", "Greece_5",
              "Greece_6", "Greece_7", "Greece_8", "Greece_9", "Hungary_1", "Hungary_2",
              "Hungary_5", "Hungary_6", "Italy_1", "Italy_3", "Poland_1", "Poland_10",
              "Poland_11", "Poland_12", "Poland_13", "Poland_14", "Poland_17", "Poland_2",
              "Poland_3", "Poland_4", "Poland_5", "Poland_6", "Poland_7", "Poland_8", "Poland_9",
              "Portuga_2", "Portuga_3", "Portuga_4", "Portuga_5", "Portuga_6", "Romania_1",
              "Romania_11", "Romania_12", "Romania_2", "Romania_3", "Romania_5", "Romania_7",
              "Romania_8", "Romania_9", "Slovakia_2", "Spain_1", "Spain_10", "Spain_11",
              "Spain_12", "Spain_13", "Spain_14", "Spain_15", "Spain_16", "Spain_17", "Spain_18",
              "Spain_19", "Spain_2", "Spain_20", "Spain_3", "Spain_5", "Spain_6", "Spain_7",
              "Spain_8", "Spain_9", "Sweden_5", "Sweden_8", "Sweden_9"],
    "val": ["Bulgaria_12", "Bulgaria_3", "Czechia_2", "Finland_7", "Germany_6", "Germany_9",
            "Greece_10", "Hungary_4", "Hungary_7", "Poland_18", "Portuga_1", "Romania_4",
            "Sweden_2", "Sweden_3"],
    "test": ["Bulgaria_1", "Bulgaria_8", "Czechia_4", "Finland_3", "Finland_5", "Finland_6",
             "Germany_2", "Germany_4", "Hungary_8", "Italy_2", "Poland_15", "Romania_10",
             "Romania_6", "Slovakia_1", "Sweden_1", "Sweden_4", "Sweden_7"],
}
RELEASE_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")


def split_of() -> dict[str, str]:
    out = {}
    for s, sites in SPLITS.items():
        for site in sites:
            if out.get(site, s) != s:
                raise ValueError(f"{site} in two splits")
            out[site] = s
    return out


def read_site(site: str):
    """-> (s2 (T, 10, H, W) int16, labels (T, H, W) uint8 {0, 255}, crs, transform, trimmed)."""
    imgs, labs, crs, tf = [], [], None, None
    for y in YEARS:
        with rasterio.open(RAW / site / "image" / f"Year{y}.tif") as r:
            if tuple(r.descriptions) != RELEASE_BANDS:
                raise ValueError(f"{site}/{y}: bands {r.descriptions}")
            if crs is None:
                crs, tf = str(r.crs), tuple(r.transform)[:6]
            elif (str(r.crs), tuple(r.transform)[:6]) != (crs, tf):
                raise ValueError(f"{site}/{y}: grid differs from {YEARS[0]}")
            imgs.append(r.read())
        with rasterio.open(RAW / site / "label" / f"Year{y}.tif") as r:
            labs.append(r.read(1))
    shapes = {a.shape[-2:] for a in imgs + labs}
    H, W = min(s[0] for s in shapes), min(s[1] for s in shapes)
    trimmed = len(shapes) > 1
    if trimmed and max(max(s[0] for s in shapes) - H, max(s[1] for s in shapes) - W) > 2:
        raise ValueError(f"{site}: shapes {sorted(shapes)} differ by more than 2 px")
    s2 = np.stack([a[:, :H, :W] for a in imgs])
    lab = np.stack([a[:H, :W] for a in labs])
    if not set(np.unique(lab).tolist()) <= {0, 255}:
        raise ValueError(f"{site}: label values {np.unique(lab)}")
    return s2, lab, crs, tf, trimmed


def pack_site(site: str) -> dict:
    s2, lab, crs, tf, trimmed = read_site(site)
    neg = int((s2 < 0).sum())
    out = PACKED
    np.save(out / "s2" / f"{site}.npy", np.clip(s2, 0, None).astype(np.uint16))
    np.save(out / "label" / f"{site}.npy", (lab > 0).astype(np.uint8))
    H, W = lab.shape[1:]
    m = lab > 0
    return {"shape": [H, W], "crs": crs, "transform": list(tf), "trimmed": trimmed,
            "negative_values": neg, "tiles": [list(o) for o in tile_origins(H, W)],
            "mine_frac": [round(float(v), 5) for v in m.mean((1, 2))],
            "changed_frac": round(float((m != m[:1]).any(0).mean()), 5)}


def pack(workers: int):
    for d in ("s2", "label"):
        (PACKED / d).mkdir(parents=True, exist_ok=True)
    splits = split_of()
    sites = sorted(splits)
    with ProcessPoolExecutor(workers) as ex:
        info = dict(zip(sites, ex.map(pack_site, sites)))
    for s in sites:
        info[s]["split"] = splits[s]
    idx = {"years": list(YEARS), "bands": list(BANDS), "tile": TILE, "sites": info,
           "left_out": sorted(p.name for p in RAW.iterdir() if p.name not in splits)}
    (PACKED / "index.json").write_text(json.dumps(idx, indent=1))
    for sp in SPLITS:
        ss = [s for s in sites if splits[s] == sp]
        print(f"{sp}: {len(ss)} sites, {sum(len(info[s]['tiles']) for s in ss)} tiles")
    print("trimmed:", [s for s in sites if info[s]["trimmed"]],
          "| negative values:", sum(info[s]["negative_values"] for s in sites),
          "| left out:", idx["left_out"])


def verify_site(site: str) -> int:
    """Every tile of the packed site against the GeoTIFFs; -> tiles checked."""
    from manyup.data.datasets.euromine import decode_years, encode_years
    import torch
    s2, lab, *_ = read_site(site)
    ps2 = np.load(PACKED / "s2" / f"{site}.npy", mmap_mode="r")
    plab = np.load(PACKED / "label" / f"{site}.npy", mmap_mode="r")
    if not (np.array_equal(ps2, np.clip(s2, 0, None)) and np.array_equal(plab, lab > 0)):
        raise AssertionError(f"{site}: packed arrays differ from the GeoTIFFs")
    n = 0
    for r, c in tile_origins(*lab.shape[1:]):
        win = (lab[:, r:r + TILE, c:c + TILE] > 0).astype(np.uint8)
        back = decode_years(torch.from_numpy(encode_years(win))).numpy()
        if not np.array_equal(back, win):
            raise AssertionError(f"{site}@r{r}c{c}: label bits do not round-trip")
        n += 1
    return n


def verify(workers: int):
    sites = sorted(split_of())
    with ProcessPoolExecutor(workers) as ex:
        n = sum(ex.map(verify_site, sites))
    print(f"verified {len(sites)} sites, {n} tiles: packed == GeoTIFF")


def pack_seasonal_site(site: str) -> dict:
    """data/EuroMineNet/seasonal/<site>/<y>Q<q>.npz (exp.euromine.fetch_seasonal) -> SEASONAL
    s2/<site>.npy (Q, 10, H, W) uint16 and clear/<site>.npy (Q, H, W) uint8 (clear observations
    behind each composite pixel; 0: no clear observation, the pixel is 0)."""
    from exp.euromine.fetch_seasonal import OUT, QUARTERS
    H, W = json.loads((PACKED / "index.json").read_text())["sites"][site]["shape"]
    s2 = np.zeros((len(QUARTERS), len(BANDS), H, W), np.uint16)
    n = np.zeros((len(QUARTERS), H, W), np.uint8)
    scenes = []
    for i, (y, q) in enumerate(QUARTERS):
        z = np.load(OUT / site / f"{y}Q{q}.npz")
        if z["s2"].shape != (len(BANDS), H, W):
            raise ValueError(f"{site} {y}Q{q}: {z['s2'].shape}, release {H} x {W}")
        s2[i], n[i] = z["s2"], z["n_clear"]
        scenes.append(int(z["n_scenes"]))
    np.save(SEASONAL / "s2" / f"{site}.npy", s2)
    np.save(SEASONAL / "clear" / f"{site}.npy", n)
    return {"clear_frac": [round(float(v), 4) for v in (n > 0).mean((1, 2))], "scenes": scenes}


def pack_seasonal(workers: int):
    from exp.euromine.fetch_seasonal import QUARTERS
    for d in ("s2", "clear"):
        (SEASONAL / d).mkdir(parents=True, exist_ok=True)
    sites = sorted(split_of())
    with ProcessPoolExecutor(workers) as ex:
        info = dict(zip(sites, ex.map(pack_seasonal_site, sites)))
    idx = {"quarters": [f"{y}Q{q}" for y, q in QUARTERS], "bands": list(BANDS), "sites": info}
    (SEASONAL / "index.json").write_text(json.dumps(idx, indent=1))
    cf = np.array([info[s]["clear_frac"] for s in sites])                  # (sites, Q)
    print(f"{len(sites)} sites x {len(QUARTERS)} quarters; clear coverage per quarter (mean over "
          f"sites): " + " ".join(f"{k} {v:.2f}" for k, v in zip(idx["quarters"], cf.mean(0))))
    print(f"site-quarters with < 50 % clear pixels: {(cf < 0.5).mean():.3f}; with none: "
          f"{(cf == 0).mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("pack", "verify", "pack_seasonal"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"pack": pack, "verify": verify, "pack_seasonal": pack_seasonal}[a.cmd](a.workers)


if __name__ == "__main__":
    main()
