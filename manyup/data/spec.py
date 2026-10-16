"""What a dataset declares about itself. Checked against the data by the conformance tests;
nothing here is inferred at run time."""
from __future__ import annotations

from dataclasses import dataclass, field

TEMPORAL_KINDS = ("irregular", "monthly", "fixed", "static")
TASKS = ("segmentation", "regression")


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    bands: tuple[str, ...]
    units: str                        # e.g. "reflectance*1e4", "dB", "m"
    gsd_m: float | None               # native ground sampling distance; None if unknown
    temporal: str                     # one of TEMPORAL_KINDS
    frames: tuple[int, int]           # (min, max) native frames per sample; (1, 1) for static
    time_precision: str | None        # "day" | "month"; None for static
    pool_reducer: str | None = None   # reducer Pool uses for this modality unless given one
    nodata: float | None = None
    aux_layers: tuple[str, ...] = ()  # optional per-pixel layers, e.g. ("cloud_prob",)

    def __post_init__(self):
        if self.temporal not in TEMPORAL_KINDS:
            raise ValueError(f"{self.name}: temporal {self.temporal!r} not in {TEMPORAL_KINDS}")
        if (self.temporal == "static") != (self.time_precision is None):
            raise ValueError(f"{self.name}: static modalities (only) have no time_precision")
        if len(set(self.bands)) != len(self.bands):
            raise ValueError(f"{self.name}: duplicate band names")

    @property
    def is_static(self) -> bool:
        return self.temporal == "static"


@dataclass(frozen=True)
class TargetSpec:
    task: str                               # one of TASKS
    units: str                              # "class" or e.g. "t/ha" (tonnes per hectare)
    class_names: tuple[str, ...] | None = None
    ignore_index: int | None = None         # label value excluded from loss/metrics

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task {self.task!r} not in {TASKS}")
        if (self.task == "segmentation") != (self.class_names is not None):
            raise ValueError("segmentation targets (only) declare class_names")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modalities: dict[str, ModalitySpec]
    target: TargetSpec
    native_shape: tuple[int, int]           # (H, W) of one native sample
    split_sizes: dict[str, int]             # native samples per split, exact
    georeferenced: bool                     # False -> every Geometry is Geometry.unknown
    notes: tuple[str, ...] = field(default=())   # exclusions, quirks, provenance

    def __post_init__(self):
        for k, m in self.modalities.items():
            if k != m.name:
                raise ValueError(f"modality key {k!r} != spec name {m.name!r}")
