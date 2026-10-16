"""Concrete MTMM datasets, one module each."""
from manyup.data.datasets.biomassters import BiomasstersDataset
from manyup.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset
from manyup.data.datasets.geoidflood_series import GeoidSeriesDataset
from manyup.data.datasets.pastis import PastisDataset
from manyup.data.datasets.yieldsat import YieldSat24Dataset, YieldSatDataset

__all__ = ["BiomasstersDataset", "ChipSelection", "GeoidFloodDataset", "GeoidSeriesDataset",
           "PastisDataset", "YieldSat24Dataset", "YieldSatDataset"]
