"""Paper material, datasets: the dataset table and one example (input time series, label) per
dataset.

    source env_setup/env_olmo.sh && python -m exp.viz.paper.datasets [--only table|examples]

Writes to figures/paper/datasets/:
    dataset_stats.json            every number in the table, as counted here from the data
    dataset_table.md / .tex       one row per modality of each release; the modalities the
                                  models read are in bold
    examples.{png,pdf}            the five examples in one figure
    example_<dataset>.{png,pdf}   each example on its own
    frames/<dataset>/             every panel as its own PNG (nearest-neighbour upscaled)

Timestep counts are per sample: PASTIS and PLANTEUR from metadata.geojson (all 2,433 / 2,750
patches), BioMassters
from the tortilla index (every --bio_every-th chip), GEOID from the packed series' metadata
(test + val chips' tiles, the fixed 6-month evaluation window), YieldSAT from the packed index
(all 2,173 fields).

Rendering (as dataset_visualization/poster_timeseries.py): S2 true colour B04/B03/B02, S1 as
R = VV, G = VH, B = VV - VH (dB); one stretch per series, so brightness changes are real.
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from manyup.data import All  # noqa: E402
from manyup.data.datasets.biomassters import MONTHS, BiomasstersDataset  # noqa: E402
from manyup.data.datasets.geoidflood_series import GeoidSeriesDataset, _meta  # noqa: E402
from manyup.data.datasets.pastis import CLASS_NAMES, PastisDataset  # noqa: E402
from manyup.data.datasets.planteur import CLASS_NAMES as PLANTEUR_CLASSES  # noqa: E402
from manyup.data.datasets.planteur import DEFAULT_ROOT as PLANTEUR_ROOT  # noqa: E402
from manyup.data.datasets.planteur import FOLDS as PLANTEUR_FOLDS  # noqa: E402
from manyup.data.datasets.planteur import PlanteurDataset  # noqa: E402
from manyup.data.datasets.yieldsat import CLEAR_SCL, PACKED, load_index  # noqa: E402
from manyup.data.time import from_days  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "figures" / "paper" / "datasets"
_spec = importlib.util.spec_from_file_location(
    "poster_timeseries", REPO / "dataset_visualization" / "poster_timeseries.py")
poster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(poster)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
INK, MUTED, RULE = "#111111", "#555555", "#bbbbbb"
GEOID_COLORS = {0: (0.90, 0.90, 0.88), 1: (0.16, 0.47, 0.84), 2: (0.92, 0.41, 0.20),
                255: (1.0, 1.0, 1.0)}
GEOID_NAMES = {0: "background", 1: "permanent water", 2: "flood", 255: "unmapped (ignored)"}
# the release's own class colours (documentation/colormap.txt); void white
PLANTEUR_COLORS = {**{int(r[0]): tuple(int(v) / 255 for v in r[1:]) for r in
                      (ln.split() for ln in open(PLANTEUR_ROOT / "documentation/colormap.txt"))
                      if r},
                   255: (1.0, 1.0, 1.0)}


def rng3(v):
    """min / median / max of a list of counts, as ints."""
    v = np.asarray(v)
    return [int(v.min()), int(np.median(v)), int(v.max())]


def span(r):
    return f"{r[0]}" if r[0] == r[2] else f"{r[0]}–{r[2]} (median {r[1]})"


# ---------------------------------------------------------------- statistics
def stats(bio_every: int = 10) -> dict:
    out = {}
    # PASTIS-R
    feats = json.load(open(REPO / "data/PASTIS-R/metadata.geojson"))["features"]
    n = {m: [len(f["properties"][f"dates-{m}"]) for f in feats] for m in ("S2", "S1A", "S1D")}
    out["pastis"] = {"patches": len(feats), "steps": {m: rng3(v) for m, v in n.items()},
                     "tiles": {s: len(TASKS["pastis"].dataset(s)) for s in ("train", "val", "test")}}
    # PLANTEUR
    feats = json.load(open(PLANTEUR_ROOT / "metadata.geojson"))["features"]
    n = {m: [len(f["properties"][f"dates-{m}"]) for f in feats] for m in ("S2", "S1", "LS")}
    out["planteur"] = {"patches": len(feats), "steps": {m: rng3(v) for m, v in n.items()},
                       "tiles": {s: len(TASKS["planteur"].dataset(s))
                                 for s in ("train", "val", "test")}}
    # BioMassters: months with a frame, per chip
    counts = {"S1": [], "S2": []}
    sizes = {}
    for split in ("train", "val", "test"):
        ds = BiomasstersDataset(split, {"S1": All(), "S2": All()}, tiling=None)
        sizes[split] = len(ds)
        for chip, pos in list(ds._pos.items())[::bio_every]:
            inner = ds._shards.read(pos)
            for m in counts:
                counts[m].append(int((inner["modality"] == m).sum()))
    out["biomassters"] = {"chips": sizes, "steps": {m: rng3(v) for m, v in counts.items()},
                          "chips_counted": len(counts["S1"]),
                          "tiles": {s: len(TASKS["biomassters"].dataset(s))
                                    for s in ("train", "val", "test")}}
    # GEOID-Flood S1 series: acquisitions in the stored year and in the 6-month window
    year, window, chips = [], [], {}
    for split in ("train", "val", "test"):
        ds = GeoidSeriesDataset(split, {"S1": All()})
        chips[split] = len(ds)
        if split == "train":                 # its window is redrawn at random every read
            continue
        seen = set()
        for cid in ds._native_ids(split):
            tile = cid.split("@")[0]
            if tile in seen:
                continue
            seen.add(tile)
            path = ds._tile_path(tile)
            meta = _meta(str(path.with_name(path.stem + "_meta.npz")))
            post = int(meta["post_day"])
            start = ds.window_start(cid, post, ds._event_day.get(tile, post))
            year.append(len(meta["day"]))
            window.append(int(((meta["day"] >= start) & (meta["day"] < start + 183)).sum()))
    out["geoid"] = {"chips": chips, "tiles_counted": len(year),
                    "steps": {"S1_year": rng3(year), "S1_window": rng3(window)}}
    # YieldSAT
    idx = load_index(str(PACKED))
    raw = [len(f["raw_days"]) for f in idx["fields"].values()]
    ml = [sum(d >= 0 for d in f["ml24_days"]) for f in idx["fields"].values()]
    out["yieldsat"] = {"fields": len(raw), "steps": {"S2_raw": rng3(raw), "S2_ml24_filled": rng3(ml)},
                       "tiles": {s: len(TASKS["yieldsat_raw"].dataset(s))
                                 for s in ("train", "val", "test")}}
    return out


# ---------------------------------------------------------------- table
def table_rows(st: dict) -> list[dict]:
    """One row per modality. used: the models read it. `steps`: timesteps per sample in the
    release; `ours`: what the mixed-resolution model is given."""
    p, b, g, y = st["pastis"], st["biomassters"], st["geoid"], st["yieldsat"]
    pl = st["planteur"]
    tiles = lambda t: f"{t['train']:,} / {t['val']:,} / {t['test']:,}"  # noqa: E731
    return [
        dict(ds="PASTIS-R", task="crop type, 19 classes (mIoU)",
             size=f"{p['patches']:,} patches, 128 px; {tiles(p['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A", bands="10 bands", used=True,
             steps=f"{span(p['steps']['S2'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means → single dates"),
        dict(mod="Sentinel-1 ascending", bands="VV, VH, VV−VH", used=True,
             steps=f"{span(p['steps']['S1A'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means → single dates"),
        dict(mod="Sentinel-1 descending", bands="VV, VH, VV−VH", used=False,
             steps=f"{span(p['steps']['S1D'])} dates", cadence="irregular, own dates", ours="—"),
        dict(ds="PLANTEUR", task="tropical crop type, 8 classes (mIoU)",
             size=f"{pl['patches']:,} patches, 128 px; {tiles(pl['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A", bands="10 bands", used=True,
             steps=f"{span(pl['steps']['S2'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means (2019) → single dates"),
        dict(mod="Sentinel-1 RTC, both orbits", bands="VV, VH (linear → dB)", used=True,
             steps=f"{span(pl['steps']['S1'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means (2019) → single dates"),
        dict(mod="Landsat-8 L1 (+ panchromatic)", bands="10 bands at 20 m (+ B8 at 10 m)",
             used=False, steps=f"{span(pl['steps']['LS'])} dates", cadence="irregular, 16-day",
             ours="—"),
        dict(mod="S2 / S1 / Landsat monthly mosaics", bands="as above", used=False,
             steps="12 months", cadence="least-cloudy scene per 30 days, 2019", ours="—"),
        dict(mod="Embeddings (AlphaEarth, Tessera, OlmoEarth)", bands="64–128 dims", used=False,
             steps="static", cadence="annual, 2019", ours="—"),
        dict(ds="BioMassters", task="above-ground biomass, t/ha (RMSE)",
             size=f"{sum(b['chips'].values()):,} chips, 256 px; {tiles(b['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A", bands="10 bands", used=True,
             steps=f"{span(b['steps']['S2'])} months", cadence="monthly composites, Sep–Aug",
             ours="4 seasons → 12 months (no dates in the release)"),
        dict(mod="Sentinel-1 ascending", bands="VV, VH", used=True,
             steps=f"{span(b['steps']['S1'])} months", cadence="monthly composites, Sep–Aug",
             ours="4 seasons → 12 months"),
        dict(mod="Sentinel-1 descending", bands="VV, VH", used=False,
             steps="≤ 12 months", cadence="monthly composites", ours="—"),
        dict(mod="S2 cloud probability", bands="1", used=False,
             steps="as Sentinel-2", cadence="monthly", ours="—"),
        dict(ds="GEOID-Flood", task="flood / permanent water / background (mIoU of background + "
             "flood; flood IoU)",
             size=f"{tiles(g['chips'])} chips of 64 px (≥ 50% mapped, with flood)",
             mod="Sentinel-1 RTC year series (ours, Planetary Computer)", bands="VV, VH",
             used=True,
             steps=f"{span(g['steps']['S1_year'])} dates per year stored; "
                   f"{span(g['steps']['S1_window'])} in the 6-month window read",
             cadence="irregular, every pass of GEOID's orbits",
             ours="calendar months → single dates (consecutive-date pairs)"),
        dict(mod="Sentinel-1 GRD", bands="VV, VH", used=False,
             steps="2 (pre, post)", cadence="event pair",
             ours="— (earlier pre/post baseline only)"),
        dict(mod="Sentinel-1 RTC", bands="VV, VH", used=False, steps="2 (pre, post)",
             cadence="event pair", ours="—"),
        dict(mod="Sentinel-2 L2A", bands="multispectral", used=False, steps="1 (pre-event)",
             cadence="composite", ours="—"),
        dict(mod="Copernicus GLO-30 DEM", bands="1", used=False, steps="static", cadence="—",
             ours="—"),
        dict(ds="YieldSAT", task="crop yield, t/ha (pixel RMSE)",
             size=f"{y['fields']:,} field-seasons; {tiles(y['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A, every acquisition", bands="12 bands (+ SCL)", used=True,
             steps=f"{span(y['steps']['S2_raw'])} dates", cadence="irregular, seeding → harvest",
             ours="8 seasons → 24 monthly clear-pixel means → single dates"),
        dict(mod="Sentinel-2 L2A, ML-ready", bands="12 bands", used=False,
             steps=f"24 month slots, {span(y['steps']['S2_ml24_filled'])} filled",
             cadence="one date per month slot", ours="— (fixed-resolution baselines only)"),
        dict(mod="Soil properties", bands="8 layers", used=False, steps="static", cadence="—",
             ours="—"),
        dict(mod="DEM + terrain derivatives", bands="5 layers", used=False, steps="static",
             cadence="—", ours="—"),
        dict(mod="Weather (per field)", bands="4 variables", used=False, steps="daily",
             cadence="daily", ours="—"),
    ]


HEAD = ("Dataset", "Label (metric)", "Samples (train / val / test)", "Modality", "Bands",
        "Timesteps per sample", "Cadence", "Given to the model as")


def write_table(st: dict):
    rows = table_rows(st)
    md = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    tex = [r"\begin{tabular}{lllllll}", r"\toprule",
           r"Dataset & Modality & Bands & Timesteps per sample & Cadence & Given to the model as \\"]
    for r in rows:
        bold = (lambda s: f"**{s}**") if r["used"] else (lambda s: s)
        md.append("| " + " | ".join([
            f"**{r['ds']}**" if "ds" in r else "", r.get("task", ""), r.get("size", ""),
            bold(r["mod"]), r["bands"], r["steps"], r["cadence"], r["ours"]]) + " |")
        esc = lambda s: (s.replace("%", r"\%").replace("→", r"$\rightarrow$")      # noqa: E731
                         .replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("−", "$-$")
                         .replace("–", "--").replace("—", "---"))
        if "ds" in r:
            tex.append(r"\midrule")
            tex.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{r['ds']}}} --- {esc(r['task'])}; "
                       rf"{esc(r['size'])}}} \\")
        m = rf"\textbf{{{esc(r['mod'])}}}" if r["used"] else esc(r["mod"])
        tex.append(f" & {m} & {esc(r['bands'])} & {esc(r['steps'])} & {esc(r['cadence'])} & "
                   f"{esc(r['ours'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    note = ("\n\nBold: the modalities the models read. Timesteps are per sample, counted from "
            "the data (minimum–maximum, median). \"Given to the model as\": the coarse ps16 "
            "level → the finer levels crops are taken from.\n")
    (OUT / "dataset_table.md").write_text("\n".join(md) + note)
    (OUT / "dataset_table.tex").write_text("\n".join(tex) + "\n")
    print("wrote", OUT / "dataset_table.md")


# ---------------------------------------------------------------- examples
def evenly(n: int, k: int) -> list[int]:
    return sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()))


def fmt(day) -> str:
    return from_days(int(day)).strftime("%Y-%m-%d")


def pastis_example(pid: str, ncols: int) -> dict:
    ds = PastisDataset("test", {"S2": All(), "S1A": All()}, tiling=None)
    s = ds[ds.index_of(pid)]
    s2, s1 = s.modalities["S2"], s.modalities["S1A"]
    f2 = poster.s2_rgb_series(s2.data.numpy(), [2, 1, 0])
    f1 = poster.s1_rgb_series(s1.data[:, 0].numpy(), s1.data[:, 1].numpy())
    i2, i1 = evenly(len(f2), ncols), evenly(len(f1), ncols)
    label = s.target.numpy().astype(int)
    present = [c for c in np.unique(label)]
    return dict(
        name="pastis", title=f"PASTIS-R · patch {pid}",
        sub="crop type · 128 × 128 px at 10 m",
        rows=[(f"Sentinel-2 RGB\n{len(i2)} of {len(f2)} dates",
               [(f2[i], fmt(s2.time.start[i])) for i in i2]),
              (f"Sentinel-1 asc.\n{len(i1)} of {len(f1)} dates",
               [(f1[i], fmt(s1.time.start[i])) for i in i1])],
        label=poster.colorize(label, poster.PASTIS_COLORS), label_title="crop type",
        legend=[Patch(fc=poster.PASTIS_COLORS[c], ec="0.5", lw=0.4, label=CLASS_NAMES[c])
                for c in present])


PLANTEUR_CLEAR_B02 = 1000     # display only: brighter in blue (B02) = cloud / haze
PLANTEUR_LIT_B08 = 1500       # darker in NIR (B08) = cloud shadow (the patches are mostly green)


def planteur_clear(s2: np.ndarray) -> np.ndarray:
    """(T, C, H, W) -> (T,) share of pixels with data, B02 below PLANTEUR_CLEAR_B02 and B08 above
    PLANTEUR_LIT_B08. PLANTEUR ships no cloud mask; this only chooses which dates the figure
    shows (B02 alone let thin cloud and shadowed dates through)."""
    b2, b8 = s2[:, 0], s2[:, 6]
    return ((b2 > 0) & (b2 < PLANTEUR_CLEAR_B02) & (b8 > PLANTEUR_LIT_B08)).reshape(
        len(s2), -1).mean(1)


def planteur_pick(ncols: int) -> str:
    """A test patch with >= 4 classes, >= ncols clear 2019 S2 dates (planteur_clear >= 0.9), and
    the most crop (non-background) pixels."""
    ds = PlanteurDataset("test", {"S2": All()}, tiling=None)
    best, best_id = -1.0, ds.sample_id(0)
    for pid in ds._native_ids("test"):
        lab = np.load(PLANTEUR_ROOT / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0]
        crop = float(((lab > 0) & (lab != 255)).mean())
        if len(set(np.unique(lab).tolist()) - {255}) < 4 or crop <= best:
            continue
        dates = ds._meta[pid]["dates-S2"]
        in_2019 = [int(k) for k, d in dates.items() if d // 10000 == 2019]
        s2 = np.load(PLANTEUR_ROOT / "DATA_S2" / f"S2_{pid}.npy", mmap_mode="r")
        if int((planteur_clear(np.asarray(s2[in_2019])) >= 0.9).sum()) >= ncols:
            best, best_id = crop, pid
    return best_id


def planteur_example(pid: str | None, ncols: int) -> dict:
    pid = pid or planteur_pick(ncols)
    for split in ("test", "val", "train"):
        ds = PlanteurDataset(split, {"S2": All(), "S1": All()}, tiling=None)
        if ds._meta[pid]["Fold"] in PLANTEUR_FOLDS[split]:
            break
    s = ds[ds.index_of(pid)]
    s2, s1 = s.modalities["S2"], s.modalities["S1"]
    year = lambda md: [t for t, d in enumerate(md.time.start.tolist())  # noqa: E731
                       if from_days(d).year == 2019]
    x2 = s2.data.numpy()
    good = [t for t in year(s2) if planteur_clear(x2[t:t + 1])[0] >= 0.9]
    i2 = [good[i] for i in evenly(len(good), ncols)]
    f2 = poster.s2_rgb_series(x2[i2], [2, 1, 0])           # one stretch over the shown dates
    y1 = year(s1)
    i1 = [y1[i] for i in evenly(len(y1), ncols)]
    f1 = poster.s1_rgb_series(s1.data[i1, 0].numpy(), s1.data[i1, 1].numpy(),
                              s1.pixel_valid[i1].numpy())
    label = s.target.numpy().astype(int)
    return dict(
        name="planteur", title=f"PLANTEUR · patch {pid} ({s.meta['island']}, {split})",
        sub=f"crop type · 128 × 128 px at 10 m · {s2.num_frames} S2 and {s1.num_frames} S1 dates "
            f"in 2018-09..2019-12; {len(good)} of {len(year(s2))} S2 dates in 2019 clear (display test: ≥ 90% of pixels "
            f"with B02 < {PLANTEUR_CLEAR_B02} and B08 > {PLANTEUR_LIT_B08})",
        rows=[(f"Sentinel-2 RGB\n{len(i2)} of {len(good)} clear\n2019 dates",
               [(f, fmt(s2.time.start[t])) for f, t in zip(f2, i2)]),
              (f"Sentinel-1 A + D\n{len(i1)} of {len(y1)}\n2019 dates",
               [(f, fmt(s1.time.start[t])) for f, t in zip(f1, i1)])],
        label=poster.colorize(label, PLANTEUR_COLORS), label_title="crop type",
        legend=[Patch(fc=PLANTEUR_COLORS[c], ec="0.5", lw=0.4,
                      label=PLANTEUR_CLASSES[c] if c != 255 else "void (ignored)")
                for c in np.unique(label)])


def biomassters_pick(n_scan: int = 120) -> str:
    """A test chip with Sentinel-1 in every month and the most mostly-cloud-free S2 months."""
    ds = BiomasstersDataset("test", {"S1": All(), "S2": All()}, tiling=None)
    best, best_id = -1.0, ds.sample_id(0)
    for i in range(0, len(ds), max(1, len(ds) // n_scan)):
        s = ds[i]
        s1, s2 = s.modalities["S1"], s.modalities["S2"]
        n1 = int(((s1.data[:, :2] != 0).any(1).flatten(1).float().mean(1) > 0.99).sum())
        cloud = s2.aux["cloud_prob"].flatten(1).mean(1)
        n2 = int((s2.frame_valid & (cloud < 20)).sum())
        score = (n1 == 12) * (n2 + float(s.target.std()) / 1000)
        if score > best:
            best, best_id = score, ds.sample_id(i)
    return best_id


def biomassters_example(chip: str | None, ncols: int) -> dict:
    chip = chip or biomassters_pick()
    for split in ("test", "val", "train"):
        ds = BiomasstersDataset(split, {"S1": All(), "S2": All()}, tiling=None)
        if chip in ds._pos:
            break
    s = ds[ds.index_of(chip)]
    s2, s1 = s.modalities["S2"], s.modalities["S1"]
    f2 = poster.s2_rgb_series(s2.data.numpy(), [2, 1, 0], mask_fill=True)
    ok = (s1.data[:, :2] != 0).any(1).numpy()     # the ascending pass's own nodata
    f1 = poster.s1_rgb_series(s1.data[:, 0].numpy(), s1.data[:, 1].numpy(), ok)
    mon = [m[:3] for m in MONTHS]
    agbm = s.target.numpy()
    vmax = float(np.percentile(agbm, 99.5))
    return dict(
        name="biomassters", title=f"BioMassters · chip {chip} ({split})",
        sub="above-ground biomass · 256 × 256 px at 10 m · year withheld",
        rows=[("Sentinel-2 RGB\n12 monthly slots",
               [(f2[i] if bool(s2.frame_valid[i]) else None, mon[i]) for i in range(12)]),
              ("Sentinel-1 asc.\n12 monthly slots",
               [(f1[i] if bool(s1.frame_valid[i]) else None, mon[i]) for i in range(12)])],
        label=plt.get_cmap("YlGn")(np.clip(agbm / vmax, 0, 1))[..., :3],
        label_title="biomass (t/ha)", cbar=("YlGn", 0, vmax, "t/ha"))


def geoid_pick(ds, n_scan: int = 4000, seed: int = 0) -> int:
    """A chip whose flood is visible in S1: 25-60% flood pixels and the largest VV drop over
    them between the last date before the post scene and the post scene."""
    rng = np.random.default_rng(seed)
    best, best_i = -1.0, 0
    for i in rng.choice(len(ds), min(n_scan, len(ds)), replace=False):
        s = ds[int(i)]
        lab = s.target.numpy()
        fl = (lab == 2).mean()
        if not 0.25 <= fl <= 0.6 or (lab == 255).any():
            continue
        m = s.modalities["S1"]
        days = m.time.start.numpy()
        post = int(np.argmin(np.abs(days - s.meta["post_day"])))
        if post < 6 or post > len(days) - 6 or len(days) < 24:
            continue
        vv = m.data[:, 0].numpy()
        drop = float((vv[post - 1] - vv[post])[lab == 2].mean()
                     - (vv[post - 1] - vv[post])[lab == 0].mean())
        if drop > best:
            best, best_i = drop, int(i)
    return best_i


def geoid_example(chip: str | None, ncols: int) -> dict:
    ds = GeoidSeriesDataset("test", {"S1": All()})
    ids = ds._native_ids("test")
    i = ids.index(chip) if chip else geoid_pick(ds)
    s = ds[i]
    m = s.modalities["S1"]
    days = m.time.start.numpy()
    post = int(np.argmin(np.abs(days - s.meta["post_day"])))
    f = poster.gray_series(m.data[:, 0].numpy(), m.pixel_valid.numpy())
    lo = max(0, min(post - ncols // 2, len(days) - ncols))
    shown = list(range(lo, min(lo + ncols, len(days))))
    label = s.target.numpy()
    frac = {k: (label == k).mean() for k in GEOID_NAMES}
    return dict(
        name="geoid", title=f"GEOID-Flood S1 series · chip {ids[i]}",
        sub=f"flood mapping · 64 × 64 px at 10 m · {len(days)} dates in the 6-month window",
        rows=[(f"Sentinel-1 VV (dB)\n{len(shown)} of {len(days)} dates",
               [(f[t], fmt(days[t])) for t in shown])],
        mark=(0, shown.index(post)), mark_text="post-event scene",
        label=poster.colorize(label, GEOID_COLORS), label_title="flood map",
        legend=[Patch(fc=GEOID_COLORS[k], ec="0.5", lw=0.4, label=GEOID_NAMES[k])
                for k in GEOID_NAMES if frac[k] > 0])


def yieldsat_example(field: str | None, ncols: int) -> dict:
    idx = load_index(str(PACKED))
    fields = idx["fields"]
    if field is None:           # a test field, mid-sized, mostly labelled, many acquisitions
        cand = [(f["labelled"] / (f["H"] * f["W"]), n) for n, f in sorted(fields.items())
                if f["fold"] == 0 and 90 <= f["H"] <= 160 and 90 <= f["W"] <= 160
                and len(f["raw_days"]) >= 40]
        field = max(cand)[1]
    f = fields[field]
    s2 = np.load(PACKED / "s2" / f["country"] / f"{field}.npy", mmap_mode="r")
    scl = np.load(PACKED / "scl" / f["country"] / f"{field}.npy", mmap_mode="r")
    target = np.load(PACKED / "target" / f["country"] / f"{field}.npy")
    clear = np.isin(scl, CLEAR_SCL).reshape(len(scl), -1).mean(1)
    inside = (np.asarray(s2[:, 1:4]) != 0).any(1)               # the field polygon
    # shown: dates whose field pixels are >= 80% clear (SCL vegetation / bare soil), evenly
    # spaced in that list; one stretch over them
    px = max(int(inside.any(0).sum()), 1)
    clear_in = (np.isin(scl, CLEAR_SCL) & inside).reshape(len(scl), -1).sum(1) / px
    good = np.nonzero(clear_in >= 0.8)[0]
    shown = [int(good[i]) for i in evenly(len(good), ncols)]
    rgb = np.asarray(s2[shown][:, [3, 2, 1]]).astype(np.float32)          # (n, 3, H, W)
    lo, hi = np.percentile(rgb.transpose(0, 2, 3, 1)[inside[shown]], [1, 99])   # one for R, G, B
    frames = {t: poster._stretch(x.transpose(1, 2, 0), lo, hi, 0.8) for t, x in zip(shown, rgb)}
    imgs = []
    for t in shown:
        img = frames[t].copy()
        img[~inside[t]] = 1.0                                   # outside the polygon: white
        imgs.append((img, fmt(f["raw_days"][t])))
    vmin, vmax = np.nanpercentile(target, [1, 99])
    lab = plt.get_cmap("viridis")(np.clip((target - vmin) / (vmax - vmin), 0, 1))[..., :3]
    lab[~np.isfinite(target)] = 1.0
    return dict(
        name="yieldsat", title=f"YieldSAT · {f['country']}, {f['crop']} {f['year']}",
        sub=f"yield · field box {f['H']} × {f['W']} px at 10 m · seeded {f['seeding']}, "
            f"harvested {f['harvest']} · {len(clear)} dates, {len(good)} of them clear "
            f"over the field",
        rows=[(f"Sentinel-2 RGB\n{len(shown)} of {len(good)} clear dates", imgs)],
        label=lab, label_title="yield (t/ha)", cbar=("viridis", float(vmin), float(vmax), "t/ha"),
        field=field)


def block_height(ex: dict, cell: float) -> float:
    return 0.53 + len(ex["rows"]) * (cell + 0.2) + 0.25


def draw_block(fig, ex: dict, y_top: float, W: float, H: float, cell: float, ncols: int):
    """One dataset: title line, its modality rows (time runs left to right), the label on the
    right with its legend / colour bar. Coordinates in inches from the top-left."""
    left, gap, date_h = 1.15, 0.04, 0.2
    fig.text(left / W, 1 - (y_top + 0.14) / H, ex["title"], fontsize=10, weight="bold",
             color=INK, va="center")
    fig.text(left / W, 1 - (y_top + 0.33) / H, ex["sub"], fontsize=7.5, color=MUTED,
             va="center")
    y = y_top + 0.5
    for r, (name, frames) in enumerate(ex["rows"]):
        fig.text((left - 0.1) / W, 1 - (y + date_h + cell / 2) / H, name, ha="right",
                 va="center", fontsize=7.5, color=INK)
        for c, (img, date) in enumerate(frames):
            x0 = left + c * (cell + gap)
            fig.text((x0 + cell / 2) / W, 1 - (y + date_h - 0.03) / H, date, ha="center",
                     va="bottom", fontsize=5.6, color=MUTED)
            if img is None:                     # no frame in this slot
                fig.patches.append(plt.Rectangle(
                    (x0 / W, 1 - (y + date_h + cell) / H), cell / W, cell / H, fill=False,
                    ec=RULE, lw=0.4, ls=(0, (2, 2)), transform=fig.transFigure))
                continue
            ax = fig.add_axes([x0 / W, 1 - (y + date_h + cell) / H, cell / W, cell / H])
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            marked = ex.get("mark") == (r, c)
            for sp in ax.spines.values():
                sp.set_linewidth(1.8 if marked else 0.4)
                sp.set_color("#eb6834" if marked else RULE)
            if marked:
                fig.text((x0 + cell / 2) / W, 1 - (y + date_h + cell + 0.04) / H,
                         ex["mark_text"], ha="center", va="top", fontsize=6, color="#eb6834")
        y += cell + date_h
    # the label: as tall as the rows together (at most 2 cells), right of the grid
    rows_h = len(ex["rows"]) * (cell + date_h) - date_h
    lab = min(rows_h, 2 * cell + date_h)
    lx = left + ncols * (cell + gap) + 0.35
    ly = y_top + 0.5 + date_h + (rows_h - lab) / 2
    h, w = ex["label"].shape[:2]
    lw = lab * w / h
    ax = fig.add_axes([lx / W, 1 - (ly + lab) / H, lw / W, lab / H])
    ax.imshow(ex["label"], interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4), sp.set_color(RULE)
    fig.text((lx + lw / 2) / W, 1 - (ly - 0.04) / H, "label: " + ex["label_title"],
             ha="center", va="bottom", fontsize=7.5, color=INK)
    kx = lx + lw + 0.15
    if ex.get("legend"):
        fig.legend(handles=ex["legend"], loc="upper left", frameon=False, fontsize=6.2,
                   ncol=2 if len(ex["legend"]) > 6 else 1, handlelength=1.0, columnspacing=0.8,
                   labelspacing=0.3, bbox_to_anchor=(kx / W, 1 - (ly - 0.05) / H))
    if ex.get("cbar"):
        cmap, vmin, vmax, unit = ex["cbar"]
        cax = fig.add_axes([kx / W, 1 - (ly + lab) / H, 0.09 / W, lab / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin, vmax), cmap), cax=cax)
        cb.set_label(unit, fontsize=6.5)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.4)


def save_examples(examples: list[dict], ncols: int, cell: float = 0.86):
    W = 1.15 + ncols * (cell + 0.04) + 0.35 + 2 * cell + 2.6

    def render(exs, name):
        H = sum(block_height(e, cell) for e in exs) + 0.15
        fig = plt.figure(figsize=(W, H), facecolor="white")
        y = 0.1
        for e in exs:
            draw_block(fig, e, y, W, H, cell, ncols)
            y += block_height(e, cell)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print("wrote", OUT / f"{name}.png")

    render(examples, "examples")
    for e in examples:
        render([e], f"example_{e['name']}")
        d = OUT / "frames" / e["name"]
        d.mkdir(parents=True, exist_ok=True)
        for name, frames in e["rows"]:
            tag = name.split("\n")[0].replace(" ", "_").replace(".", "")
            for i, (img, date) in enumerate(frames):
                if img is not None:
                    poster.save_png(img, d / f"{tag}_{i:02d}_{date}.png", 512)
        poster.save_png(e["label"], d / "label.png", 512)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["table", "examples"])
    ap.add_argument("--ncols", type=int, default=12, help="timesteps shown per row")
    ap.add_argument("--pastis_id", default="40091")
    ap.add_argument("--planteur_id", default=None)
    ap.add_argument("--biomassters_chip", default=None)
    ap.add_argument("--geoid_chip", default=None)
    ap.add_argument("--yieldsat_field", default=None)
    ap.add_argument("--bio_every", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only != "examples":
        path = OUT / "dataset_stats.json"
        st = stats(a.bio_every)
        path.write_text(json.dumps(st, indent=1))
        write_table(st)
    if a.only != "table":
        ex = [pastis_example(a.pastis_id, a.ncols), planteur_example(a.planteur_id, a.ncols),
              biomassters_example(a.biomassters_chip, a.ncols),
              geoid_example(a.geoid_chip, a.ncols), yieldsat_example(a.yieldsat_field, a.ncols)]
        (OUT / "examples.json").write_text(json.dumps(
            {e["name"]: e["title"] for e in ex}, indent=1))
        save_examples(ex, a.ncols)


if __name__ == "__main__":
    main()
