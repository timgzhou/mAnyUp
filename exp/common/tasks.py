"""The three benchmark tasks, each declared once for every model trained on it.

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

from exp.common.metrics import RegMeter, SegMeter
from exp.common.paths import TILESTORE
from manyup.data import All, MTMMBatch, MTMMDataset, Pool, TemporalPolicy, Tiling
from manyup.data.datasets.biomassters import PLACEHOLDER_YEAR, BiomasstersDataset
from manyup.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset
from manyup.data.datasets.pastis import AGRI_YEAR, CLASS_NAMES, PastisDataset
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

    def policies(self) -> dict[str, TemporalPolicy]:
        """The time setup OlmoEarth sees, per modality."""
        raise NotImplementedError

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

    def loss(self) -> nn.Module:
        raise NotImplementedError

    def meter(self):
        raise NotImplementedError


# ---- PASTIS ----------------------------------------------------------------------------------
# S2 L2A lacks B01/B09, which OlmoEarth's S2 input has; copy neighbours as OlmoEarth's own
# PASTIS eval did.
S2_IMPUTE = {"B01": "B02", "B09": "B8A"}


class Pastis(Task):
    name, kind, num_outputs, reduction = "pastis", "classification", 19, "time_mean"
    olmoearth = OlmoEarthSetup({"S2": "sentinel2_l2a", "S1A": "sentinel1"}, {"S2": S2_IMPUTE})
    # dates in days since 2018-09-01, normalization from the split's folds (UTAEInput)
    utae = UTAESetup(("S2", "S1A"), to_days(dt.date(2018, 9, 1)), None, batch_size=16)
    batch_size = {4: 8, 16: 32}
    VOID = PastisDataset.SPEC.target.ignore_index
    stored_modalities = ("S2", "S1A")
    store_name, store_dtypes = "pastis_tile64", {"target": "int16"}

    def policies(self):
        month_mean = Pool("month", "mean", window=AGRI_YEAR)
        return {"S2": month_mean, "S1A": month_mean}

    def raw(self, split, policies):
        return PastisDataset(split, policies, Tiling(64, "col-major"))

    def target(self, batch):
        t = batch.target.clone()
        t[t == self.VOID] = IGNORE
        return t

    def loss(self):
        return nn.CrossEntropyLoss(ignore_index=IGNORE)

    def meter(self):
        return SegMeter(self.num_outputs, tuple(n.lower().replace(" ", "_").replace(",", "")
                                                for n in CLASS_NAMES[:19]))


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
    store_name = "biomassters_tile64"
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
    store_name, store_dtypes, store_shard = ("geoidflood_c64_mapped0.5_class2",
                                             {"target": "uint8"}, 4096)

    def policies(self):
        return {"S1GRD": All()}

    def raw(self, split, policies):
        return GeoidFloodDataset(split, policies, self.CHIPS)

    def target(self, batch):
        t = batch.target.clone()
        t[t == GeoidFloodDataset.SPEC.target.ignore_index] = IGNORE
        return t

    def loss(self):
        return nn.CrossEntropyLoss(ignore_index=IGNORE, weight=torch.tensor(GEOID_CE_WEIGHTS))

    def meter(self):
        # headline: background + flood; permanent water is dark in both dates, so it is not
        # a change signal and is reported on its own (iou_permanent_water)
        return SegMeter(3, GEOID_CLASSES, headline=(0, 2))


TASKS: dict[str, Task] = {t.name: t for t in (Pastis(), Biomassters(), GeoidFlood())}
