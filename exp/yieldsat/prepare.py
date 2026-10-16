"""Build the packed YieldSAT store that manyup/data/datasets/yieldsat.py reads.

From data/YieldSAT/{preprocessed-24-ts, original-preprocessed} (Both.zip, extracted) to
data/YieldSAT/packed/:

    index.json                     per field: country, crop, year, farm, provider, quality,
                                   seeding / harvest (ISO), crs, transform, H, W, fold,
                                   tiles [[row, col], ...], raw_days (every acquisition,
                                   days since 1970-01-01), ml24_days (24 slots, -1 = none);
                                   top level: target_mean / target_std (train pixels, t/ha)
    target/<country>/<field>.npy   (H, W) float32 t/ha, NaN = unlabelled (from the NetCDF)
    s2/<country>/<field>.npy       (A, 12, H, W) uint16, every acquisition, GeoTIFF band order
    scl/<country>/<field>.npy      (A, H, W) uint8 scene classification

Steps (each resumable / idempotent):
    index   NetCDF small variables (field, row, col, target, times) + raw metadata and file
            lists -> index.json, target/. Checks: every field has a raw directory on the same
            grid; the NetCDF dates are constant within a field and each is a raw acquisition.
    pack    raw GeoTIFFs -> s2/, scl/ (parallel over fields; a packed field is skipped)
    verify  the ml24 view against the NetCDF `sample` S2 bands at every labelled pixel (or
            --max_px per field): finite exactly where the view is valid, equal values there.

    python -u -m exp.yieldsat.prepare index
    python -u -m exp.yieldsat.prepare pack --workers 32
    python -u -m exp.yieldsat.prepare verify --workers 8
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import rasterio

from manyup.data.datasets.yieldsat import (CLEAR_SCL, COUNTRIES, FOLDS, PACKED, ROOT,
                                           S2_BANDS, fold_splits, tile_origins)
from manyup.data.time import to_days

# the extracted releases; --ml_dir / --raw_dir point elsewhere (e.g. a one-country test)
ML_DIR, RAW_DIR = ROOT / "preprocessed-24-ts", ROOT / "original-preprocessed"
NC_NAME = "merge_s2-soil-dem-weather-coords.nc"
YIELD_MASK = "mean_scaled_yield_masked_regional_statistical_outlier.tif"


def nc_path(country: str) -> Path:
    return ML_DIR / country / NC_NAME


def codes(ds: h5py.Dataset) -> dict[int, object]:
    """An integer-encoded NetCDF variable's code -> value table (kept in its attributes)."""
    return {int(k): v for k, v in ds.attrs.items() if k.isdigit()}


def decode(ds: h5py.Dataset) -> np.ndarray:
    table = codes(ds)
    lut = np.zeros(max(table) + 1, dtype=np.int64)
    for k, v in table.items():
        lut[k] = int(v)
    return lut[ds[:]]


def nc_days(ds: h5py.Dataset) -> np.ndarray:
    """CF 'days since YYYY-MM-DD' float times -> int days since 1970-01-01, -1 = NaT."""
    units = ds.attrs["units"]
    units = units.decode() if isinstance(units, bytes) else units
    base = to_days(dt.date.fromisoformat(units.split("since")[1].strip()[:10]))
    t = ds[:]
    ok = np.isfinite(t)
    if not np.allclose(t[ok], np.round(t[ok])):
        raise ValueError(f"{ds.name}: non-integer day offsets")
    return np.where(ok, np.round(np.nan_to_num(t)).astype(np.int64) + base, -1)


def raw_days(field_dir: Path) -> list[int]:
    names = sorted(p.name for p in (field_dir / "s2_images").glob("S2_L2A_*.tif"))
    return [to_days(dt.datetime.strptime(n[-12:-4], "%Y%m%d").date()) for n in names]


def iso(ddmmyyyy: str) -> str:
    return dt.datetime.strptime(ddmmyyyy, "%d.%m.%Y").date().isoformat()


def field_rows(fs: np.ndarray) -> dict[int, np.ndarray]:
    """field code -> sorted pixel indices."""
    order = np.argsort(fs, kind="stable")
    cuts = np.flatnonzero(np.diff(fs[order])) + 1
    return {int(fs[g[0]]): np.sort(g) for g in np.split(order, cuts)}


# ---- index -----------------------------------------------------------------------------------
def build_index(out: Path, countries: tuple[str, ...]) -> None:
    fields, report = {}, collections.Counter()
    slot_off = collections.Counter()
    for country in countries:
        t0 = time.time()
        with h5py.File(nc_path(country), "r") as f:
            names = {k: (v.decode() if isinstance(v, bytes) else v)
                     for k, v in codes(f["field_shared_name"]).items()}
            fs = f["field_shared_name"][:]
            row, col = decode(f["row"]), decode(f["col"])
            target = f["target"][:]
            days = nc_days(f["times"])                                   # (N, 24)
        raw_dirs = {p.name for p in (RAW_DIR / country).iterdir() if p.is_dir()}
        missing = sorted(set(names.values()) - raw_dirs)
        if missing:
            raise FileNotFoundError(f"{country}: {len(missing)} NetCDF fields without a raw "
                                    f"directory, e.g. {missing[:3]}")
        report[f"{country}_raw_only"] = len(raw_dirs - set(names.values()))
        (out / "target" / country).mkdir(parents=True, exist_ok=True)
        for code, idx in field_rows(fs).items():
            name = names[code]
            fdir = RAW_DIR / country / name
            meta = json.loads((fdir / f"metadata-{name}.json").read_text())
            with rasterio.open(fdir / "yield_masks" / YIELD_MASK) as r:
                H, W = r.shape
                crs, transform = str(r.crs), list(r.transform)[:6]
            if row[idx].max() >= H or col[idx].max() >= W:
                raise ValueError(f"{name}: NetCDF pixels outside the raw {H}x{W} grid")
            t = days[idx]
            if not (t == t[0]).all():
                raise ValueError(f"{name}: NetCDF dates differ between pixels of the field")
            raw = raw_days(fdir)
            ml24 = [int(d) for d in t[0]]
            absent = [d for d in ml24 if d >= 0 and d not in raw]
            if absent:
                raise ValueError(f"{name}: NetCDF dates {absent} are not raw acquisitions")
            dated = [d for d in ml24 if d >= 0]
            if dated != sorted(set(dated)):
                raise ValueError(f"{name}: NetCDF dates not strictly increasing")
            # slot i = month i of the two years ending with the HARVEST year (the metadata
            # "year" can differ: Germany_DUP3_farm5_field272_rapeseed_2018 was harvested 2017)
            y0 = int(iso(meta["harvesting_date"])[:4]) - 1
            for i, d in enumerate(ml24):
                if d >= 0:
                    date = dt.date(1970, 1, 1) + dt.timedelta(days=d)
                    slot_off[(date.year - y0) * 12 + date.month - 1 - i] += 1
            tgt = np.full((H, W), np.nan, np.float32)
            tgt[row[idx], col[idx]] = target[idx]
            np.save(out / "target" / country / f"{name}.npy", tgt)
            lab = np.isfinite(tgt)
            tiles = [[r, c] for r, c in tile_origins(H, W)
                     if lab[max(r, 0):r + 64, max(c, 0):c + 64].any()]
            fields[name] = {
                "country": country, "crop": meta["crop"], "year": int(meta["year"]),
                "farm": meta["farm_identifier"], "provider": meta["data_provider"],
                "quality": meta.get("yieldmap_quality"),
                "seeding": iso(meta["seeding_date"]), "harvest": iso(meta["harvesting_date"]),
                "crs": crs, "transform": transform, "H": H, "W": W,
                "labelled": int(lab.sum()), "tiles": tiles,
                "raw_days": raw, "ml24_days": ml24}
            report["fields"] += 1
            report["tiles"] += len(tiles)
        print(f"{country}: {sum(f['country'] == country for f in fields.values())} fields "
              f"({time.time() - t0:.0f} s)", flush=True)
    assign_folds(fields)
    train = fold_splits(0)["train"]
    ys = np.concatenate([np.load(out / "target" / f["country"] / f"{n}.npy").ravel()
                         for n, f in fields.items() if f["fold"] in train])
    ys = ys[np.isfinite(ys)]
    idx = {"target_mean": float(ys.mean()), "target_std": float(ys.std()),
           "fields": fields}
    (out / "index.json").write_text(json.dumps(idx))
    print(dict(report))
    print("ML-ready slot i vs the calendar month of its acquisition (month - i: count):",
          dict(sorted(slot_off.items())))
    print(f"train target: mean {idx['target_mean']:.4f} std {idx['target_std']:.4f} t/ha "
          f"over {len(ys)} pixels")


def assign_folds(fields: dict) -> None:
    """Within each country x crop, fields in md5-of-name order take folds 0..FOLDS-1 in turn:
    grouped by field (all its tiles in one fold), stratified by country x crop."""
    groups = collections.defaultdict(list)
    for name, f in fields.items():
        groups[(f["country"], f["crop"])].append(name)
    for names in groups.values():
        for i, n in enumerate(sorted(names, key=lambda n: hashlib.md5(n.encode()).hexdigest())):
            fields[n]["fold"] = i % FOLDS


# ---- pack ------------------------------------------------------------------------------------
def pack_field(args) -> str:
    name, f, out, raw_dir = args
    s2_path = out / "s2" / f["country"] / f"{name}.npy"
    scl_path = out / "scl" / f["country"] / f"{name}.npy"
    if s2_path.exists() and scl_path.exists():
        return "skipped"
    fdir = raw_dir / f["country"] / name
    H, W, A = f["H"], f["W"], len(f["raw_days"])
    s2 = np.zeros((A, len(S2_BANDS), H, W), np.uint16)
    scl = np.zeros((A, H, W), np.uint8)
    names = sorted(p.name for p in (fdir / "s2_images").glob("S2_L2A_*.tif"))
    shifted = 0
    for a, n in enumerate(names):
        with rasterio.open(fdir / "s2_images" / n) as r:
            bands = tuple(d.split(" ")[0] for d in r.descriptions)
            # same grid as the yield mask: equal shape; transforms agree to float noise
            # (15% of Germany's files differ by < 1 m, 4 of 27k by up to 5.6 m)
            shift = float(np.abs(np.array(list(r.transform)[:6]) - f["transform"]).max())
            if bands != S2_BANDS or r.shape != (H, W) or shift > 10:
                raise ValueError(f"{name}/{n}: bands {bands}, shape {r.shape} or transform "
                                 f"(off by {shift:.1f} m) differ from the field's")
            shifted += shift > 1
            s2[a] = r.read()
        with rasterio.open(fdir / "scl_masks" / n.replace("S2_L2A_", "S2_L2A_SCL_")) as r:
            if r.shape != (H, W):
                raise ValueError(f"{name}/{n}: SCL shape {r.shape}")
            scl[a] = r.read(1)
    for path, arr in ((s2_path, s2), (scl_path, scl)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, path)
    return "packed, a file > 1 m off the field's transform" if shifted else "packed"


def pack(out: Path, workers: int) -> None:
    fields = json.loads((out / "index.json").read_text())["fields"]
    jobs = [(n, f, out, RAW_DIR) for n, f in sorted(fields.items())]
    t0, done = time.time(), collections.Counter()
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(pack_field, jobs, chunksize=4)):
            done[r] += 1
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} {dict(done)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"pack: {dict(done)} in {time.time() - t0:.0f} s")


# ---- verify ----------------------------------------------------------------------------------
def verify_country(args) -> dict:
    """ml24 values (packed acquisition at the slot's date, kept only at clear labelled pixels)
    vs the NetCDF's S2 bands, for every field of one country."""
    country, out, max_px, ml_dir = args
    fields = json.loads((out / "index.json").read_text())["fields"]
    rng = np.random.default_rng(0)
    stats = collections.Counter()
    with h5py.File(ml_dir / country / NC_NAME, "r") as f:
        names = {k: (v.decode() if isinstance(v, bytes) else v)
                 for k, v in codes(f["field_shared_name"]).items()}
        bands = [b.decode() if isinstance(b, bytes) else b for b in f["band"][:]]
        cols = [bands.index(b) for b in S2_BANDS]
        fs = f["field_shared_name"][:]
        row, col = decode(f["row"]), decode(f["col"])
        sample = f["sample"]
        for code, idx in field_rows(fs).items():
            name = names[code]
            fl = fields[name]
            if max_px and len(idx) > max_px:
                idx = np.sort(rng.choice(idx, max_px, replace=False))
            lo, hi = int(idx[0]), int(idx[-1]) + 1      # a field's pixels are ~contiguous
            nc = sample[lo:hi][idx - lo] if hi - lo <= 4 * len(idx) else sample[idx]
            nc = nc[:, :, cols]                                             # (n, 24, 12)
            s2 = np.load(out / "s2" / country / f"{name}.npy", mmap_mode="r")
            scl = np.load(out / "scl" / country / f"{name}.npy", mmap_mode="r")
            r, c = row[idx], col[idx]
            for t, d in enumerate(fl["ml24_days"]):
                fin = np.isfinite(nc[:, t]).all(1)
                part = np.isfinite(nc[:, t]).any(1) & ~fin
                stats["partial_nan_px"] += int(part.sum())
                if d < 0:
                    stats["undated_slot_finite_px"] += int(fin.sum())
                    continue
                a = fl["raw_days"].index(d)
                ok = np.isin(scl[a][r, c], CLEAR_SCL)
                vals = s2[a][:, r, c].T.astype(np.float32)
                stats["px_checked"] += len(idx)
                stats["valid_mismatch_px"] += int((ok != fin).sum())
                stats["value_mismatch_px"] += int((vals[fin & ok] != nc[:, t][fin & ok]).any(1).sum())
            stats["fields"] += 1
    stats["country"] = country
    return dict(stats)


def verify(out: Path, workers: int, max_px: int, countries: tuple[str, ...]) -> None:
    with ProcessPoolExecutor(min(workers, len(countries))) as ex:
        results = list(ex.map(verify_country, [(c, out, max_px, ML_DIR) for c in countries]))
    bad = 0
    for s in results:
        print(s)
        bad += s.get("valid_mismatch_px", 0) + s.get("value_mismatch_px", 0) \
            + s.get("undated_slot_finite_px", 0) + s.get("partial_nan_px", 0)
    print("VERIFY OK: the ml24 view equals the NetCDF" if bad == 0 else
          f"VERIFY FAILED: {bad} mismatching pixel-steps")


def main():
    global ML_DIR, RAW_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("index", "pack", "verify"))
    ap.add_argument("--out", default=str(PACKED))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_px", type=int, default=0, help="verify: pixels per field (0 = all)")
    ap.add_argument("--countries", default=",".join(COUNTRIES))
    ap.add_argument("--ml_dir", default=str(ML_DIR))
    ap.add_argument("--raw_dir", default=str(RAW_DIR))
    a = ap.parse_args()
    ML_DIR, RAW_DIR = Path(a.ml_dir), Path(a.raw_dir)
    countries = tuple(a.countries.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "index":
        build_index(out, countries)
    elif a.step == "pack":
        pack(out, a.workers)
    else:
        verify(out, a.workers, a.max_px, countries)


if __name__ == "__main__":
    main()
