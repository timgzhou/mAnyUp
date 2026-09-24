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

UTAE sees each dataset as its authors / the benchmark intended: PASTIS every acquisition on
its own date (whole 128 px patches, the published UTAE setup), BioMassters the 12 month slots
on whole 256 px chips, GEOID the same pre/post chips as OlmoEarth.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import torch
from torch import nn

from exp.common.metrics import RegMeter, SegMeter
from manyup.data import All, MTMMBatch, MTMMDataset, Pool, Tiling
from manyup.data.datasets.biomassters import PLACEHOLDER_YEAR, BiomasstersDataset
from manyup.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset
from manyup.data.datasets.pastis import AGRI_YEAR, CLASS_NAMES, PastisDataset
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

    def dataset(self, split: str) -> MTMMDataset:
        """The data OlmoEarth trains on."""
        raise NotImplementedError

    def utae_dataset(self, split: str) -> MTMMDataset:
        raise NotImplementedError

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
    utae = UTAESetup(("S2", "S1A"), to_days(dt.date(2018, 9, 1)), None, batch_size=4)
    batch_size = {4: 8, 16: 32}
    VOID = PastisDataset.SPEC.target.ignore_index

    def dataset(self, split):
        month_mean = Pool("month", "mean", window=AGRI_YEAR)
        return PastisDataset(split, {"S2": month_mean, "S1A": month_mean},
                             Tiling(64, "col-major"))

    def utae_dataset(self, split):
        return PastisDataset(split, {"S2": All(), "S1A": All()}, tiling=None)

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
                     batch_size=4)
    batch_size = {4: 8, 16: 32}

    def dataset(self, split):
        return BiomasstersDataset(split, {"S1": All(), "S2": All()}, Tiling(64, "row-major"))

    def utae_dataset(self, split):
        return BiomasstersDataset(split, {"S1": All(), "S2": All()})

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

    def dataset(self, split):
        return GeoidFloodDataset(split, {"S1GRD": All()}, self.CHIPS)

    def utae_dataset(self, split):
        return self.dataset(split)

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
