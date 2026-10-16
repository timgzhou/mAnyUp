"""The benchmark tasks, each declared once for every model trained on it.

A Task says which data a model sees (dataset, modalities, time setup, sample size), what it
predicts (target, loss, metrics) and, per model family, how its inputs are packed:

  task        data (OlmoEarth)                         head       target / primary metric
  pastis      S2 + S1A, monthly MEAN on the agri year,  time_mean  19 crop classes (void
              64 px tiles of the 128 px patches                    ignored) / mIoU
  biomassters S1 (ascending) + S2, the 12 month slots,  time_mean  AGBM t/ha, standardized
              64 px tiles of the 256 px chips                      for the loss / RMSE
  geoidflood  S1 pre + post (dB), 64 px chips with       diff       background / permanent
              flood (c64_mapped0.5_class2)                          water / flood / mIoU of
                                                                    background + flood
  yieldsat_*  S2, 24 month slots (ML-ready as-is, or    time_mean  yield t/ha, standardized
              clear-pixel monthly means of every                    for the loss / pixel
              acquisition), 64 px tiles of field boxes              RMSE (+ field level)

Tiling to 64 px changes nothing an OlmoEarth model sees at tile_size <= 64 (windows are
encoded independently either way); it bounds the tokens per sample, so fine-tuning fits.

UTAE sees the same 64 px samples, with every raw frame: PASTIS every acquisition on its own
date, BioMassters the 12 month slots, GEOID the pre/post pair.

Data is read from the task's tile store (TILESTORE/<store_name>, built once by
exp/common/build_tile_store.py) when it exists, else from the raw dataset: the samples are
identical (tests/test_tile_store.py), the store is just much faster to read.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import torch
from torch import nn

from exp.common.metrics import RegMeter, SegMeter, YieldMeter
from exp.common.paths import TILESTORE
from manyup.data import All, MTMMBatch, MTMMDataset, Pool, PoolYears, TemporalPolicy, Tiling
from manyup.data.datasets.biomassters import PLACEHOLDER_YEAR, BiomasstersDataset
from manyup.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset
from manyup.data.datasets.pastis import AGRI_YEAR, CLASS_NAMES, PastisDataset
from manyup.data.datasets.yieldsat import PACKED as YIELDSAT_PACKED
from manyup.data.datasets.yieldsat import S2_BANDS as YIELDSAT_S2_BANDS
from manyup.data.datasets.yieldsat import YieldSat24Dataset, YieldSatDataset
from manyup.data.datasets.yieldsat import load_index as load_yieldsat_index
from manyup.data.guidance import S2_RGB
from manyup.data.tile_store import open_store
from manyup.data.time import to_days

IGNORE = -1                     # ignored target pixels, for every classification task
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class OlmoEarthSetup:
    modalities: dict[str, str]                  # dataset modality -> OlmoEarth modality
    impute: dict[str, dict[str, str]] = field(default_factory=dict)  # modality -> {band: source}


@dataclass(frozen=True)
class UTAESetup:
    modalities: tuple[str, ...]                 # dataset modalities, early-fused in this order
    reference_day: int | None                   # dates = days since this; None = first frame
    norm: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] | None  # per-modality mean/std
    batch_size: int


class Task:
    name: str
    kind: str                     # "classification" | "regression"
    num_outputs: int
    reduction: str                # head's time reduction, see exp/common/heads.py
    olmoearth: OlmoEarthSetup
    utae: UTAESetup
    batch_size: dict[int, int]    # OlmoEarth patch size -> batch size (LP and FFT alike)
    stored_modalities: tuple[str, ...]     # what the tile store keeps
    store_name: str               # tile store of source()
    store_dtypes: dict[str, str] = {}      # compact storage dtypes (build_tile_store)
    store_shard: int = 16                  # native samples per store shard
    # feature upsamplers (exp/upsamplers): the full guidance stack (UPMA, mAnyUp, timAnyUp)
    # and the 3 channels standing in for RGB (AnyUp, UPA), as {modality: bands}
    guidance: dict[str, tuple[str, ...]]
    rgb: dict[str, tuple[str, ...]]
    # seasonal timAnyUp (manyup/seasonal.py): steps per season of the low-res input
    # (GroupSteps after policies()): 3 months, or GEOID's pre/post pair as one step
    season: int

    def policies(self) -> dict[str, TemporalPolicy]:
        """The time setup OlmoEarth sees, per modality."""
        raise NotImplementedError

    def lr_policy(self) -> TemporalPolicy:
        """How the low-res steps (seasons) are formed from policies()'s steps."""
        from manyup.data import GroupSteps
        return GroupSteps(self.season)

    def raw(self, split: str, policies: dict[str, TemporalPolicy]) -> MTMMDataset:
        """The raw dataset, cut into this task's samples, read with these policies."""
        raise NotImplementedError

    def source(self, split: str) -> MTMMDataset:
        """What the tile store holds: every stored modality, every raw frame."""
        return self.raw(split, {m: All() for m in self.stored_modalities})

    def open(self, split: str, policies: dict[str, TemporalPolicy]) -> MTMMDataset:
        ds = open_store(TILESTORE / self.store_name, split, policies)
        return ds if ds is not None else self.raw(split, policies)

    def dataset(self, split: str) -> MTMMDataset:
        """The data OlmoEarth trains on."""
        return self.open(split, self.policies())

    def utae_dataset(self, split: str) -> MTMMDataset:
        return self.open(split, {m: All() for m in self.utae.modalities})

    def target(self, batch: MTMMBatch) -> torch.Tensor:
        raise NotImplementedError

    # events whose val samples score nothing (their target is IGNORE on the val split only);
    # recorded in training state so a resumed run knows which val it was selecting on
    val_excluded_events: tuple[str, ...] = ()

    def loss(self) -> nn.Module:
        raise NotImplementedError

    def meter(self):
        raise NotImplementedError


# ---- PASTIS ----------------------------------------------------------------------------------
# S2 L2A lacks B01/B09, which OlmoEarth's S2 input has; copy neighbours as OlmoEarth's own
# PASTIS eval did.
S2_IMPUTE = {"B01": "B02", "B09": "B8A"}


class FoldView(torch.utils.data.Dataset):
    """Samples of several store splits, re-cut by benchmark fold: the tile store is written as
    train = folds 1-3, val = 4, test = 5, and a rotated protocol fold needs other groupings.

    parts: [(dataset of one store split, indices into it)]; item i is the i-th of the
    concatenated indices, relabelled with the protocol `split` it now belongs to. Everything
    else (SPEC, ...) is read from the first dataset."""

    def __init__(self, parts: list[tuple[MTMMDataset, list[int]]], split: str):
        self.parts, self.split = parts, split
        self.where = [(p, i) for p, (_, idx) in enumerate(parts) for i in idx]

    def __len__(self):
        return len(self.where)

    def __getitem__(self, i):
        p, j = self.where[i]
        from dataclasses import replace
        return replace(self.parts[p][0][j], split=self.split)

    def __getattr__(self, name):            # only reached for attributes not set above
        return getattr(self.__dict__["parts"][0][0], name)


class Pastis(Task):
    name, kind, num_outputs, reduction = "pastis", "classification", 19, "time_mean"
    olmoearth = OlmoEarthSetup({"S2": "sentinel2_l2a", "S1A": "sentinel1"}, {"S2": S2_IMPUTE})
    # dates in days since 2018-09-01, normalization from the split's folds (UTAEInput)
    utae = UTAESetup(("S2", "S1A"), to_days(dt.date(2018, 9, 1)), None, batch_size=16)
    batch_size = {4: 8, 16: 32}
    VOID = PastisDataset.SPEC.target.ignore_index
    stored_modalities = ("S2", "S1A")
    guidance = {"S2": PastisDataset.SPEC.modalities["S2"].bands, "S1A": ("VV", "VH")}
    rgb = {"S2": S2_RGB}
    store_name, store_dtypes = "pastis_tile64", {"target": "int16"}
    season = 3

    def policies(self):
        month_mean = Pool("month", "mean", window=AGRI_YEAR)
        return {"S2": month_mean, "S1A": month_mean}

    # The benchmark's 5-fold protocol (Garnot & Landrieu 2021, train_semantic.py): protocol
    # fold f trains on dataset folds {f, f+1, f+2}, validates on f+3, tests on f+4 (1-based,
    # wrapping at 5). fold = 1 is the split every earlier run used (train 1-3, val 4, test 5).
    fold = 1
    STORE_SPLIT_OF_FOLD = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}

    def fold_groups(self) -> dict[str, tuple[int, ...]]:
        """split -> the dataset folds it holds under protocol fold self.fold."""
        f = [(self.fold - 1 + i) % 5 + 1 for i in range(5)]
        return {"train": tuple(f[:3]), "val": (f[3],), "test": (f[4],)}

    def raw(self, split, policies):
        return PastisDataset(split, policies, Tiling(64, "col-major"))

    def open(self, split, policies):
        if self.fold == 1:
            return super().open(split, policies)
        wanted = self.fold_groups()[split]
        fold_of = None
        parts = []
        for store_split in ("train", "val", "test"):
            if not any(self.STORE_SPLIT_OF_FOLD[g] == store_split for g in wanted):
                continue
            ds = super().open(store_split, policies)
            if fold_of is None:             # patch id -> its dataset fold (metadata.geojson)
                import json
                from manyup.data.datasets.pastis import DEFAULT_ROOT
                feats = json.load(open(DEFAULT_ROOT / "metadata.geojson"))["features"]
                fold_of = {str(f["properties"]["ID_PATCH"]): f["properties"]["Fold"]
                           for f in feats}
            # the sample's patch: its native id (store) or the id before the tile suffix (raw)
            patch = (lambda i: ds.native_of(i)[0]) if hasattr(ds, "native_of") else (
                lambda i: ds._index[i][0])
            idx = [i for i in range(len(ds)) if fold_of[str(patch(i))] in wanted]
            parts.append((ds, idx))
        return FoldView(parts, split)

    def target(self, batch):
        t = batch.target.clone()
        t[t == self.VOID] = IGNORE
        return t

    def loss(self):
        return nn.CrossEntropyLoss(ignore_index=IGNORE)

    def meter(self):
        return SegMeter(self.num_outputs, tuple(n.lower().replace(" ", "_").replace(",", "")
                                                for n in CLASS_NAMES[:19]))


class PastisS2(Pastis):
    """PASTIS with Sentinel-2 only: what the PASTIS semantic-segmentation benchmark numbers
    (U-TAE, TSViT, Exchanger) use; `pastis` adds S1A (the PASTIS-R setting)."""
    name = "pastis_s2"
    olmoearth = OlmoEarthSetup({"S2": "sentinel2_l2a"}, {"S2": S2_IMPUTE})
    utae = UTAESetup(("S2",), to_days(dt.date(2018, 9, 1)), None, batch_size=16)
    guidance = {"S2": PastisDataset.SPEC.modalities["S2"].bands}

    def policies(self):
        return {"S2": Pool("month", "mean", window=AGRI_YEAR)}


# ---- BioMassters -----------------------------------------------------------------------------
# From 600 random train chips (every 17th pixel): AGBM is heavy-tailed (max > 3000 t/ha), so
# the loss sees (y - mean) / std; metrics are reported in t/ha.
AGBM_MEAN, AGBM_STD = 63.98, 72.68
# per-channel mean / std over valid pixels of valid frames, same 600 chips (UTAE input norm)
BIO_NORM = {
    "S1": ((-11.377, -18.003, -11.374, -18.087), (3.138, 4.341, 3.218, 4.464)),
    "S2": ((1653.7, 1630.0, 1621.4, 1941.9, 2502.5, 2611.8, 2755.7, 2702.9, 1019.8, 692.3),
           (2518.8, 2327.0, 2405.5, 2408.9, 2221.2, 2110.0, 2195.6, 2032.1, 928.3, 752.1)),
}


class Biomassters(Task):
    name, kind, num_outputs, reduction = "biomassters", "regression", 1, "time_mean"
    # OlmoEarth takes one S1 (VV, VH): the ascending pass. BioMassters S2 is L2A without
    # B01/B09.
    olmoearth = OlmoEarthSetup({"S1": "sentinel1", "S2": "sentinel2_l2a"},
                               {"S1": {"vv": "VV_asc", "vh": "VH_asc"}, "S2": S2_IMPUTE})
    utae = UTAESetup(("S1", "S2"), to_days(dt.date(PLACEHOLDER_YEAR, 9, 1)), BIO_NORM,
                     batch_size=64)
    batch_size = {4: 8, 16: 32}
    stored_modalities = ("S1", "S2")
    guidance = {"S2": BiomasstersDataset.SPEC.modalities["S2"].bands,
                "S1": BiomasstersDataset.SPEC.modalities["S1"].bands}
    rgb = {"S2": S2_RGB}
    store_name = "biomassters_tile64"
    season = 3
    store_dtypes = {"S2": "uint16", "S2.cloud_prob": "uint8"}

    def policies(self):
        return {"S1": All(), "S2": All()}

    def raw(self, split, policies):
        return BiomasstersDataset(split, policies, Tiling(64, "row-major"))

    def target(self, batch):
        return (batch.target.float() - AGBM_MEAN) / AGBM_STD

    def loss(self):
        return _MSE()

    def meter(self):
        return RegMeter(AGBM_STD)


class _MSE(nn.Module):
    def forward(self, pred, target):          # pred (B, 1, H, W), target (B, H, W)
        return nn.functional.mse_loss(pred[:, 0].float(), target)


# ---- GEOID-Flood -----------------------------------------------------------------------------
GEOID_CLASSES = ("background", "permanent_water", "flood")
# Inverse pixel frequency over 8000 random train chips (63.1% background, 9.6% permanent
# water, 27.3% flood), normalized to mean 1: flood and
# permanent water are minorities even among chips selected for having flood, and an
# unweighted loss collapses toward background (exp/geoidflood/lp.py).
GEOID_CE_WEIGHTS = (0.3032, 1.9972, 0.6996)
GEOID_NORM = {"S1GRD": ((-11.824, -19.731), (5.527, 7.087))}   # dB, same chips (UTAE input)


class GeoidFlood(Task):
    name, kind, num_outputs, reduction = "geoidflood", "classification", 3, "diff"
    olmoearth = OlmoEarthSetup({"S1GRD": "sentinel1"})
    utae = UTAESetup(("S1GRD",), None, GEOID_NORM, batch_size=64)
    batch_size = {4: 64, 16: 128}
    CHIPS = ChipSelection(64)
    stored_modalities = ("S1GRD",)
    guidance = {"S1GRD": ("VV", "VH")}
    rgb = {"S1GRD": ("VV", "VH", "VV")}      # no optical data: a SAR false-colour stand-in
    store_name, store_dtypes, store_shard = ("geoidflood_c64_mapped0.5_class2",
                                             {"target": "uint8"}, 4096)
    season = 2

    def policies(self):
        return {"S1GRD": All()}

    def raw(self, split, policies):
        return GeoidFloodDataset(split, policies, self.CHIPS)

    # EMSR342 (Townsville, Feb 2019) is 28% of val chips, and its labels are not visible in its
    # S1 pair: the post image (2019-02-02) predates the flood peak, so flood-labelled pixels do
    # not darken. It made GEOID val (0.45 for a model that tests 0.67) uninformative for model
    # selection: val-test rank correlation over 34 checkpoints 0.50 -> 0.83 without it
    # (exp/geoidflood/eval_per_event.py). Train and test keep it.
    val_excluded_events = ("EMSR342",)

    def target(self, batch):
        t = batch.target.clone()
        t[t == GeoidFloodDataset.SPEC.target.ignore_index] = IGNORE
        if batch.split == "val":
            for b, sid in enumerate(batch.ids):
                if sid.split("-")[0] in self.val_excluded_events:
                    t[b] = IGNORE
        return t

    def loss(self):
        return nn.CrossEntropyLoss(ignore_index=IGNORE, weight=torch.tensor(GEOID_CE_WEIGHTS))

    def meter(self):
        # headline: background + flood; permanent water is dark in both dates, so it is not
        # a change signal and is reported on its own (iou_permanent_water)
        return SegMeter(3, GEOID_CLASSES, headline=(0, 2))


# ---- GEOID-Flood S1 time series ------------------------------------------------------------
# The same chips and labels, with every Sentinel-1 acquisition (Planetary Computer RTC, GEOID's
# orbits) in a 6-month window containing the flood (manyup/data/datasets/geoidflood_series.py).
# Heads compare CONSECUTIVE dates, whatever their orbit (exp/common/heads.py "pair_*"): one
# shared head per date pair, pooled over pairs, so a location nothing changed at gives a clean
# no-change. (Orbit barely matters on this terrain-corrected product: on dry land, dates <= 12
# days apart differ by 1.95 dB per pixel across orbits vs 1.73 within one -- mostly speckle.)
# The low-res steps are calendar months (Pool), the high-res steps single dates.
# EMSR342 is excluded from val here too: in the series its flood pixels still brighten at the
# post date (+2.3 dB, dry land +2.6; only 8% of flood pixels darken by > 3 dB, vs 60-80% for
# clear events) -- its next acquisition after GEOID's post image is 24 days later.
# Flood-only: flood vs everything else (permanent water is "not flood"); inverse-frequency CE
# weights from GEOID's pixel frequencies (72.7% not flood, 27.3% flood), normalized to mean 1.
GEOID_FLOOD_CE_WEIGHTS = (0.5459, 1.4541)


class GeoidSeries(Task):
    name, kind, num_outputs, reduction = "geoidseries", "classification", 3, "pair_diffmean"
    olmoearth = OlmoEarthSetup({"S1": "sentinel1"})
    utae = UTAESetup(("S1",), None, None, batch_size=32)
    batch_size = {4: 16, 16: 64}
    stored_modalities = ("S1",)
    guidance = {"S1": ("VV", "VH")}
    rgb = {"S1": ("VV", "VH", "VV")}
    store_name = "geoidseries_c64"
    season = 0                            # months, see lr_policy

    def policies(self):
        return {"S1": All()}

    def lr_policy(self):
        return Pool("month")

    def raw(self, split, policies):
        from manyup.data.datasets.geoidflood_series import GeoidSeriesDataset
        return GeoidSeriesDataset(split, policies)

    def open(self, split, policies):
        return self.raw(split, policies)         # read straight from the packed memmaps

    val_excluded_events = ("EMSR342",)

    def target(self, batch):
        return GeoidFlood.target(self, batch)            # IGNORE, and the val exclusion

    def loss(self):
        return nn.CrossEntropyLoss(ignore_index=IGNORE, weight=torch.tensor(GEOID_CE_WEIGHTS))

    def meter(self):
        return SegMeter(3, GEOID_CLASSES, headline=(0, 2))


class GeoidSeriesFlood(GeoidSeries):
    name, num_outputs, reduction = "geoidseries_flood", 2, "pair_diff"

    def target(self, batch):
        t = super().target(batch)
        return torch.where(t == IGNORE, t, (t == 2).long())

    def loss(self):
        return nn.CrossEntropyLoss(ignore_index=IGNORE, weight=torch.tensor(GEOID_FLOOD_CE_WEIGHTS))

    def meter(self):
        return SegMeter(2, ("not_flood", "flood"), headline=(1,))     # primary = flood IoU


# ---- YieldSAT ----------------------------------------------------------------------------------
# Pixel yield (t/ha) of 2,173 field-seasons, 4 countries x 4 crops, ONE model for all of them
# (manyup/data/datasets/yieldsat.py; store built by exp/yieldsat/prepare.py). The target
# travels as (B, 2, H, W): [:, 0] the yield standardized by the train pixels' mean / std
# (NaN = unlabelled: the loss and the meter skip it), [:, 1] the pixel's field index, so
# the meter can score fields (a field's mean prediction vs its mean yield, the paper's
# field level) and every country x crop subset. primary = pooled pixel RMSE.
#   yieldsat_ml24  the ML-ready release's 24 month slots, rasterized as-is (values only at
#                  labelled clear pixels): the fixed psN-tileM OlmoEarth baselines
#   yieldsat_raw   every acquisition (clear = SCL 4/5 as pixel_valid); OlmoEarth sees the 24
#                  month slots of clear-pixel means (PoolYears), the date level reads the
#                  acquisitions themselves: the mixed-resolution encoder
class _YieldSat(Task):
    kind, num_outputs, reduction = "regression", 1, "time_mean"
    olmoearth = OlmoEarthSetup({"S2": "sentinel2_l2a"})       # the same 12 bands, no impute
    utae = UTAESetup(("S2",), None, None, batch_size=32)
    # ps4 tile64: 16 x 16 patches x 24 steps x 3 S2 band groups = 18k tokens (PASTIS: 12k);
    # base FFT at batch 4 peaks at 11 GB on an L40S
    batch_size = {4: 8, 8: 16, 16: 32}
    stored_modalities = ("S2",)
    store_name = ""                       # no tile store: read from the packed memmaps
    season = 3                            # 24 month slots -> 8 seasons
    guidance = {"S2": YIELDSAT_S2_BANDS}
    rgb = {"S2": S2_RGB}
    dataset_cls: type

    def raw(self, split, policies):
        return self.dataset_cls(split, policies)

    def open(self, split, policies):
        return self.raw(split, policies)

    @property
    def index(self) -> dict:
        return load_yieldsat_index(str(YIELDSAT_PACKED))

    @property
    def field_ids(self) -> dict[str, int]:
        if not hasattr(self, "_field_ids"):
            self._field_ids = {n: i for i, n in enumerate(sorted(self.index["fields"]))}
        return self._field_ids

    def target(self, batch):
        idx = self.index
        z = (batch.target.float() - idx["target_mean"]) / idx["target_std"]
        fid = torch.tensor([self.field_ids[s.rsplit("@r", 1)[0]] for s in batch.ids],
                           dtype=torch.float32)
        return torch.stack([z, fid.view(-1, 1, 1).expand_as(z)], 1)

    def loss(self):
        return _MaskedMSE()

    def meter(self):
        idx = self.index
        groups = tuple(f"{f['country'].lower()}_{f['crop']}"
                       for _, f in sorted(idx["fields"].items()))
        return YieldMeter(idx["target_mean"], idx["target_std"], groups)


class _MaskedMSE(nn.Module):
    """MSE over the labelled pixels: pred (B, 1, H, W), target (B, 2, H, W) (see _YieldSat)."""

    def forward(self, pred, target):
        y = target[:, 0]
        keep = torch.isfinite(y)
        if not keep.any():
            return pred.float().sum() * 0          # a batch without labels: no gradient
        return nn.functional.mse_loss(pred[:, 0].float()[keep], y[keep])


class YieldSatML24(_YieldSat):
    name, dataset_cls = "yieldsat_ml24", YieldSat24Dataset

    def policies(self):
        return {"S2": All()}                   # the release's 24 slots as they are


class YieldSatRaw(_YieldSat):
    name, dataset_cls = "yieldsat_raw", YieldSatDataset

    def policies(self):
        return {"S2": PoolYears(2, "mean")}    # the same 24 slots, clear-pixel monthly means


TASKS: dict[str, Task] = {t.name: t for t in (Pastis(), PastisS2(), Biomassters(), GeoidFlood(),
                                              GeoidSeries(), GeoidSeriesFlood(),
                                              YieldSatML24(), YieldSatRaw())}
