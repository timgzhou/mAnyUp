"""Run configuration: plain dataclasses, composed into one RunConfig, saved as JSON next to the
checkpoint. build_selector checks the composition up front; a wrong one raises before training.

    cfg = RunConfig(task="pastis", select=SelectConfig(scorer="entropy", allocator="halves",
                                                       keep="all", n=8, crop_px=8, loc_stride=8))
    cfg.name()   # mixres_pastis_ps16_entropy-halves-all_n8_c8s8_<hash>
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

from manyup.mixres.model import ModelConfig

SCORERS = ("label", "entropy", "residual", "random", "season_grid", "season_random")
ALLOCATORS = ("top", "pairs", "halves")
# what each scorer produces / each allocator needs (window and / or per-season scores)
SCORER_OUT = {"label": {"window", "season"}, "entropy": {"window", "season"},
              "residual": {"window", "season"}, "random": {"window"},
              "season_grid": {"season"}, "season_random": {"season"}}
ALLOC_IN = {"top": {"window"}, "pairs": {"season"}, "halves": {"window", "season"}}
CLASSIFICATION_ONLY = ("entropy", "residual")


@dataclass
class ResidualConfig:
    arch: str = "conv"          # conv | hr | xattn | xattn_conv (ResidualScorer inputs)
    train: str = "e2e"          # e2e | oracle | both
    tau: float = 0.5            # Gumbel noise on the picks while training (e2e / both)
    decay: float = 0.5          # both: the oracle weight 1 -> 0 over this share of the epochs


@dataclass
class Level2Config:
    patch: int = 4              # finer month crops inside the first layer's (4 or 2 px)
    n: int = 8                  # cells of the patch grid
    pick: str = "oracle"        # oracle | random
    source: str = "months"      # months: composites | dates: every raw acquisition in them


@dataclass
class SelectConfig:
    scorer: str = "label"
    allocator: str = "halves"
    keep: str = "better"        # halves: better | worse | all
    n: int = 8                  # budget: n windows x the sample's present months
    crop_px: int = 8
    loc_stride: int | None = 8  # candidate lattice stride; None: a window centred on every cell
    residual: ResidualConfig | None = None
    level2: Level2Config | None = None


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int | None = None      # None: the task's ps16 batch size
    seed: int = 0
    init_ckpt: str = ""                # checkpoints/<name>.pt to warm start from (legacy or new)
    num_workers: int = 8
    max_hours: float | None = None     # stop before an epoch that would end past this; rerun resumes
    limit_batches: int = 0             # smoke test: a few batches, nothing written to results


@dataclass
class RunConfig:
    task: str = "pastis"
    model: ModelConfig = field(default_factory=ModelConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def name(self) -> str:
        """A readable tag + a hash of everything that defines the run (not num_workers,
        max_hours, limit_batches)."""
        d = self.to_dict()
        for k in ("num_workers", "max_hours", "limit_batches"):
            d["train"].pop(k)
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        s = self.select
        tag = f"{s.scorer}-{s.allocator}" + (f"-{s.keep}" if s.allocator == "halves" else "")
        if s.residual:
            tag += f"-res{s.residual.train}{s.residual.arch}"
        if s.level2:
            tag += (f"-l2{s.level2.source[0]}{s.level2.patch}n{s.level2.n}"
                    f"{'' if s.level2.pick == 'oracle' else 'r'}")
        geo = f"c{s.crop_px}" + (f"s{s.loc_stride}" if s.loc_stride else "")
        warm = "_warm" if self.train.init_ckpt else ""
        return (f"mixres_{self.task}_ps{self.model.lr_patch}_{tag}_n{s.n}_{geo}{warm}"
                f"_seed{self.train.seed}_{h}")

    def check(self, task_kind: str):
        """Raise on a composition that cannot run."""
        s = self.select
        if s.scorer not in SCORERS or s.allocator not in ALLOCATORS:
            raise ValueError(f"scorer {s.scorer!r} / allocator {s.allocator!r}")
        missing = ALLOC_IN[s.allocator] - SCORER_OUT[s.scorer]
        if missing:
            raise ValueError(f"allocator {s.allocator} needs {sorted(missing)} scores; scorer "
                             f"{s.scorer} gives {sorted(SCORER_OUT[s.scorer])}")
        if s.scorer in CLASSIFICATION_ONLY and task_kind != "classification":
            raise ValueError(f"scorer {s.scorer}: classification tasks only")
        if (s.scorer == "residual") != (s.residual is not None):
            raise ValueError("select.residual is set exactly when select.scorer = residual")
        if s.level2 and s.level2.patch >= s.crop_px and s.level2.source == "months":
            raise ValueError(f"level2 patch {s.level2.patch} not finer than crop_px {s.crop_px}")
        if s.level2 and s.level2.patch > s.crop_px:
            raise ValueError(f"level2 patch {s.level2.patch} larger than crop_px {s.crop_px}")
        if (s.level2 is not None and s.level2.source == "dates") != self.model.date_tokens:
            raise ValueError("model.date_tokens is set exactly when level2.source = dates")


def from_dict(cls, d: dict):
    """Nested dataclass from a (partial) dict; unknown keys raise."""
    kw = {}
    names = {f.name: f for f in fields(cls)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"{cls.__name__} has no field {k!r}")
        sub = _dataclass_of(names[k].type)
        kw[k] = from_dict(sub, v) if sub is not None and isinstance(v, dict) else v
    return cls(**kw)


def _dataclass_of(tp):
    """The dataclass in a field annotation like `Level2Config | None`, else None."""
    for c in (ResidualConfig, Level2Config, SelectConfig, TrainConfig, ModelConfig):
        if tp is c or (isinstance(tp, str) and c.__name__ in tp) or c in getattr(tp, "__args__", ()):
            return c
    return None


def apply_overrides(cfg: RunConfig, overrides: list[str]) -> RunConfig:
    """`select.n=16`, `select.residual.arch=xattn`, `train.init_ckpt=...`, `select.level2=none`
    (None). A section that is None is created with its defaults when a field below it is set."""
    d = cfg.to_dict()
    for o in overrides:
        key, _, raw = o.partition("=")
        if not _:
            raise ValueError(f"override {o!r}: want key=value")
        *path, last = key.split(".")
        node, cls = d, RunConfig
        for p in path:
            sub = _dataclass_of({f.name: f for f in fields(cls)}[p].type)
            if node.get(p) is None:
                node[p] = asdict(sub())
            node, cls = node[p], sub
        ftype = {f.name: f for f in fields(cls)}[last].type
        node[last] = _parse(raw, ftype)
    return from_dict(RunConfig, d)


def _parse(raw: str, tp):
    if raw.lower() in ("none", "null"):
        return None
    sub = _dataclass_of(tp)
    if sub is not None:
        return asdict(sub(**json.loads(raw))) if raw.startswith("{") else asdict(sub())
    s = str(tp)
    if "bool" in s:
        return raw.lower() in ("1", "true", "yes")
    if "int" in s and "float" not in s:
        return int(raw)
    if "float" in s:
        return float(raw)
    return raw
