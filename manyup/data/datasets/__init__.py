"""Concrete MTMM datasets, one module each."""
from manyup.data.datasets.biomassters import BiomasstersDataset
from manyup.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset
from manyup.data.datasets.pastis import PastisDataset

__all__ = ["BiomasstersDataset", "ChipSelection", "GeoidFloodDataset", "PastisDataset"]
