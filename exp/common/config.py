"""Typed run config for exp/pastis/finetune_olmoearth.py.

Two tiers, by design (explicit-over-implicit, so you can't accidentally launch the
wrong experiment):
  - REQUIRED architecture fields (no default): model_size, input_modalities,
    head_mode, freeze_backbone. If any is unset, load_config errors.
  - Tuning knobs with defaults on the Config dataclass below: epochs, lr, batch_size,
    num_workers, seed, data_splits, dataset.

A run = Config defaults  (+ optional --config YAML)  (+ --set key=value CLI).
CLI overrides win. Example:
    python -m exp.pastis.finetune_olmoearth --set \
        model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t1 freeze_backbone=true

Dependency-light (stdlib dataclasses + pyyaml only).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, asdict
from typing import Optional

import yaml

# model_size -> olmoearth_pretrain ModelID member. A bare size is OlmoEarth v1, so every
# existing cache/checkpoint/CSV name ("oe_base_...") still means v1. Later releases are keyed
# <version>_<size> and that key lands in the names verbatim ("oe_v1_2_base_s2_ps4_tile64").
# A new release = new entries here (the ModelID names are in olmoearth_pretrain/model_loader.py).
MODEL_SIZE_TO_ID = {
    "nano": "OLMOEARTH_V1_NANO",
    "tiny": "OLMOEARTH_V1_TINY",
    "base": "OLMOEARTH_V1_BASE",
    "large": "OLMOEARTH_V1_LARGE",
    "v1_1_nano": "OLMOEARTH_V1_1_NANO",
    "v1_1_tiny": "OLMOEARTH_V1_1_TINY",
    "v1_1_base": "OLMOEARTH_V1_1_BASE",
    "v1_2_nano": "OLMOEARTH_V1_2_NANO",
    "v1_2_tiny": "OLMOEARTH_V1_2_TINY",
    "v1_2_small": "OLMOEARTH_V1_2_SMALL",
    "v1_2_base": "OLMOEARTH_V1_2_BASE",
}
ModelSize = str  # a MODEL_SIZE_TO_ID key
ALLOWED_MODALITIES = ("sentinel2_l2a", "sentinel1")
# lp_tcat = lp, but the encoder's time axis is CONCATENATED into the probe input
# (T*D) instead of mean-pooled away. Same backbone cost; a 12x wider linear head.
ALLOWED_HEADS = ("lp", "lp_tcat", "anyup", "anyup_t2", "anyup_t1")

# Architecture fields that MUST be set explicitly (no usable default).
REQUIRED = ("model_size", "input_modalities", "head_mode", "freeze_backbone")


@dataclass
class Config:
    # --- REQUIRED architecture fields (None = unset -> error) ---
    model_size: Optional[ModelSize] = None
    input_modalities: Optional[list[str]] = None
    # head_mode: lp | lp_tcat | anyup | anyup_t2 | anyup_t1
    head_mode: Optional[str] = None
    # freeze_backbone: True -> encoder frozen all epochs (+ frozen AnyUp) -> only head trains.
    freeze_backbone: Optional[bool] = None

    # --- tuning knobs ---
    # patch_size: token grid = 64/patch_size per side. OlmoEarth's LP eval uses 4
    # (16x16 grid); we previously hardcoded 8 (8x8), which lowered mIoU. Default 4.
    patch_size: int = 4
    epochs: int = 64
    lr: float = 1e-3
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 0
    data_splits: str = "data/pastis_olmoearth"
    dataset: str = "pastis"

    def __post_init__(self) -> None:
        missing = [f for f in REQUIRED if getattr(self, f) is None]
        if missing:
            raise ValueError(
                "Required config fields not set (give via --set or YAML): "
                + ", ".join(missing)
            )
        if self.model_size not in MODEL_SIZE_TO_ID:
            raise ValueError(f"model_size={self.model_size!r} not in {list(MODEL_SIZE_TO_ID)}")
        if not self.input_modalities:
            raise ValueError("input_modalities must be non-empty")
        bad = [m for m in self.input_modalities if m not in ALLOWED_MODALITIES]
        if bad:
            raise ValueError(f"input_modalities {bad} not in {list(ALLOWED_MODALITIES)}")
        if self.head_mode not in ALLOWED_HEADS:
            raise ValueError(f"head_mode={self.head_mode!r} not in {list(ALLOWED_HEADS)}")

    # --- derived ---
    @property
    def model_id_name(self) -> str:
        return MODEL_SIZE_TO_ID[self.model_size]

    @property
    def head(self) -> str:
        return self.head_mode

    @property
    def image_size(self) -> int:
        """Sample size of the prepared splits, read off the data_splits path.
        prepare_data.py --image_size 128 writes data/pastis128_olmoearth; the default
        64 prep has no digits in its name."""
        return 128 if "128" in os.path.basename(self.data_splits.rstrip("/")) else 64

    @property
    def run_name(self) -> str:
        mods = "".join(
            {"sentinel2_l2a": "s2", "sentinel1": "s1"}[m] for m in self.input_modalities
        )
        frz = "_frozen" if self.freeze_backbone else ""
        # image_size 64 adds no suffix (every pre-existing run/checkpoint is 64); other
        # sizes append _img<N>. REQUIRED for correctness, not tidiness: patch_size alone
        # does not disambiguate 64 from 128, so p8 at both sizes would otherwise share
        # one ckpt_path and silently overwrite each other. Mirrors extract_features.cfg_name().
        img = "" if self.image_size == 64 else f"_img{self.image_size}"
        return (f"oe_{self.dataset}_{self.model_size}_{mods}_{self.head}{frz}"
                f"_p{self.patch_size}{img}_lr{self.lr:g}_ep{self.epochs}")

    @property
    def ckpt_path(self) -> str:
        return f"checkpoints/{self.run_name}_best.pt"


# Accept "modalities" as a friendly CLI alias for "input_modalities".
_ALIASES = {"modalities": "input_modalities"}
_FIELD_TYPES = {f.name: f.type for f in fields(Config)}


def _coerce(name: str, value):
    """Coerce a string CLI value to the dataclass field's type. YAML values pass through."""
    if not isinstance(value, str):
        return value
    # field types are strings under `from __future__ import annotations`
    t = str(_FIELD_TYPES.get(name, "str"))
    if "bool" in t:
        return value.strip().lower() in ("true", "1", "yes")
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)
    if "list" in t:
        return [v for v in value.split(",") if v]
    return value


def _apply_overrides(data: dict, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        key = _ALIASES.get(key, key)
        data[key] = _coerce(key, val)
    return data


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    """Build a Config from its defaults + optional --config YAML + --set overrides.
    CLI overrides win. Unknown keys and missing required fields raise clear errors."""
    data: dict = {}
    if path:
        with open(path) as f:
            y = yaml.safe_load(f) or {}
        if not isinstance(y, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping")
        data.update({_ALIASES.get(k, k): v for k, v in y.items()})
    _apply_overrides(data, overrides)

    known = set(Config.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}; allowed: {sorted(known)}")
    return Config(**data)


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
