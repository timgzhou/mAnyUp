"""Cut GEOID-Flood 1024x1024 tiles into fixed-size sub-tiles for the OlmoEarth study.

One .pt per KEPT sub-tile holding the S1 bands plus the label crop, so feature extraction
later applies the OlmoEarth band mapping/normalization.

WHAT WE STORE per sub-tile:
    data/geoidflood_tiles_t<tile>/<split>/<idx>.pt = {
        "s1":    (2, 2, tile, tile) float16,  # (T=2 [pre,post], C=2 [VV,VH]) in dB
        "label": (tile, tile) int16,          # {0,1,2}, and 255 -> IGNORE_LABEL(-1)
        "src":   "<event>-<tileidx>",
        "pos":   (row, col),
        "date_pre": "YYYYMMDD", "date_post": "YYYYMMDD",
    }

WHY dB AT PREP TIME: GEOID ships
s1grd as LINEAR sigma0, but OlmoEarth's S1 encoder was pretrained on dB. Converting here
means the stored tile is already in the encoder's units and every downstream config reads
the identical array. Zeros/negatives in GRD are nodata, so they become NaN under log10 and
are then filled with DB_FLOOR rather than propagating NaN into the encoder.

LABEL HANDLING -- the part that silently corrupts metrics if skipped. GEOID's 255 means
"outside the CEMS-mapped area", i.e. genuinely UNLABELLED, not background. We remap it to
IGNORE_LABEL = -1 here so the loss and the confusion matrix can drop it (the LP passes
ignore_index=-1). Folding 255 into class 0 would score predictions against labels no
analyst ever drew.

FILTERING: keep a sub-tile only if it has >= --min_valid_frac labelled pixels, and (when
--flood_only, the default) at least one flood pixel (class 2). Flood pixels are ~2% of the
corpus, so without the flood filter almost every chip is pure background and the probe
learns the trivial all-background solution. NOTE this makes val/test flood-conditional --
the numbers are "given a flooded chip", not whole-scene rates -- but it applies IDENTICALLY
to all three configs, so the patch-size comparison stays fair. That is what this study is
measuring.

Run:
    python -u -m exp.geoidflood.prep_tiles --splits train,val --tile_size 128
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

TILE_PX = 1024                 # GEOID tiles are 1024x1024 @10 m
IGNORE_LABEL = -1              # what 255 becomes in the stored label
GEOID_IGNORE = 255             # what GEOID writes for "outside the mapped AOI"
DB_FLOOR = -50.0               # dB value substituted for GRD nodata (linear <= 0)
_DATE_RE = re.compile(r"_(\d{8})T\d{6}")


def to_db(x: np.ndarray) -> np.ndarray:
    """Linear sigma0 -> dB, with nodata (<=0) mapped to DB_FLOOR rather than -inf/NaN."""
    x = np.asarray(x, dtype=np.float32)
    out = np.full_like(x, DB_FLOOR)
    valid = np.isfinite(x) & (x > 0)
    out[valid] = 10.0 * np.log10(x[valid])
    return np.clip(out, DB_FLOOR, 50.0)


def _one(paths):
    return sorted(paths)[0] if paths else None


def find_layer(root: Path, event: str, tile: str, layer: str, pas: str | None):
    pat = f"{event}-{tile}_{layer}" + (f"_{pas}_*" if pas else "") + ".tif"
    return _one(glob.glob(str(root / event / layer / pat)))


def load_split_map(csv_path: Path) -> dict[str, str]:
    """tile stem ("EMSR856-1-0") -> split, from the dataset's chip metadata CSV.

    The download unpacks every split into ONE <tree>/<event>/ namespace, so the split is
    NOT recoverable from the path -- it only exists in this CSV. And it is assigned per
    TILE, not per event: 51 of the 210 events have tiles in more than one split (the
    dataset splits by AoI, and one activation can cover several). Mapping at event level
    would therefore leak train tiles into val. Hence a tile-level map.
    """
    m: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label_id = row["label_id"]                   # "<event>-<tile>_label"
            if label_id.endswith("_label"):
                m[label_id[: -len("_label")]] = row["split"]
    return m


def iter_tiles(root: Path, split_map: dict[str, str], split: str):
    """Yield (event, tile_idx) for tiles in `split` that have a label AND both S1 passes."""
    for lab in sorted(glob.glob(str(root / "*" / "label" / "*_label.tif"))):
        stem = Path(lab).name[: -len("_label.tif")]      # "<event>-<tile>"
        if split_map.get(stem) != split:
            continue
        event, tile = stem.rsplit("-", 1)
        if find_layer(root, event, tile, "s1grd", "pre") and \
           find_layer(root, event, tile, "s1grd", "post"):
            yield event, tile


def prep_split(root: Path, split: str, out_root: Path, tile_size: int, flood_only: bool,
               min_valid_frac: float, limit: int, overwrite: bool,
               split_map: dict[str, str]) -> None:
    if TILE_PX % tile_size != 0:
        raise ValueError(f"tile_size {tile_size} must divide {TILE_PX}")
    n_side = TILE_PX // tile_size
    out_dir = out_root / split
    existing = len(list(out_dir.glob("*.pt"))) if out_dir.exists() else 0
    if existing and not overwrite:
        print(f"[{split}] {existing} sub-tiles already in {out_dir}; skipping "
              f"(--overwrite to redo).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = list(iter_tiles(root, split_map, split))
    if limit > 0:
        tiles = tiles[:limit]
    if not tiles:
        print(f"[{split}] no tiles found under {root} for this split "
              f"-- is the download finished?")
        return

    kept = 0
    for event, tidx in tqdm(tiles, desc=f"prep {split} t{tile_size}"):
        p_pre = find_layer(root, event, tidx, "s1grd", "pre")
        p_post = find_layer(root, event, tidx, "s1grd", "post")
        p_lab = find_layer(root, event, tidx, "label", None)
        try:
            with rasterio.open(p_pre) as r:
                pre = to_db(r.read())                       # (2,1024,1024) VV,VH
            with rasterio.open(p_post) as r:
                post = to_db(r.read())
            with rasterio.open(p_lab) as r:
                lab = r.read(1).astype(np.int16)
        except Exception as e:                              # corrupt/truncated COG
            print(f"  !! skip {event}-{tidx}: {e}")
            continue

        lab = np.where(lab == GEOID_IGNORE, IGNORE_LABEL, lab)
        s1 = np.stack([pre, post], axis=0)                  # (T=2, C=2, H, W)

        m_pre = _DATE_RE.search(Path(p_pre).name)
        m_post = _DATE_RE.search(Path(p_post).name)
        d_pre = m_pre.group(1) if m_pre else ""
        d_post = m_post.group(1) if m_post else ""

        for r_ in range(n_side):
            for c_ in range(n_side):
                ys, xs = r_ * tile_size, c_ * tile_size
                lc = lab[ys:ys + tile_size, xs:xs + tile_size]
                valid = lc != IGNORE_LABEL
                if valid.mean() < min_valid_frac:
                    continue
                if flood_only and not (lc == 2).any():
                    continue
                sc = s1[:, :, ys:ys + tile_size, xs:xs + tile_size]
                if not np.isfinite(sc).all():
                    continue
                torch.save({"s1": torch.from_numpy(sc).half(),
                            "label": torch.from_numpy(lc).short(),
                            "src": f"{event}-{tidx}", "pos": (r_, c_),
                            "date_pre": d_pre, "date_post": d_post},
                           out_dir / f"{kept}.pt")
                kept += 1
    print(f"[{split}] kept {kept} sub-tiles of {tile_size}x{tile_size} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/GEOID-Flood-full/geoid-flood")
    ap.add_argument("--out_root", default=None,
                    help="default: data/geoidflood_tiles_t<tile_size>")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--split_csv",
                    default="data/GEOID-Flood-full/geoid-flood/data_tiles_s256_st128.csv",
                    help="chip metadata CSV carrying the per-tile split assignment")
    ap.add_argument("--tile_size", type=int, default=128)
    ap.add_argument("--flood_only", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--min_valid_frac", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="cap source tiles (debug)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root or f"data/geoidflood_tiles_t{args.tile_size}")
    split_map = load_split_map(Path(args.split_csv))
    for split in [s for s in args.splits.split(",") if s]:
        prep_split(Path(args.data_root), split, out_root, args.tile_size,
                   args.flood_only, args.min_valid_frac, args.limit, args.overwrite,
                   split_map)


if __name__ == "__main__":
    main()
