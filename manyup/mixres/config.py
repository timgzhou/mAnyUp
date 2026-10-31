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

SCORERS = ("label", "entropy", "residual", "random", "grid", "season_grid", "season_random",
           "none", "entropy_sign", "season_residual", "lossup", "attention")
ALLOCATORS = ("top", "pairs", "season_top", "halves", "every_location", "month_grid")
# what each scorer produces / each allocator needs (window and / or per-season scores)
SCORER_OUT = {"label": {"window", "season"}, "entropy": {"window", "season"},
              "residual": {"window", "season"}, "random": {"window"}, "grid": {"window"},
              "season_grid": {"season"}, "season_random": {"season"}, "none": set(),
              "entropy_sign": {"window", "season"}, "season_residual": {"season"},
              "lossup": {"window", "season"}, "attention": {"season"}}
ALLOC_IN = {"top": {"window"}, "pairs": {"season"}, "season_top": {"season"},
            "halves": {"window", "season"}, "every_location": {"season"}, "month_grid": set()}
KEEPS = {"halves": ("better", "worse", "all"), "every_location": ("half", "one")}
CLASSIFICATION_ONLY = ("entropy", "residual", "entropy_sign", "season_residual", "lossup")


@dataclass
class ResidualConfig:
    arch: str = "conv"          # conv | hr | xattn | xattn_conv (ResidualScorer inputs)
    train: str = "e2e"          # e2e | oracle | both
    tau: float = 0.5            # Gumbel noise on the picks while training (e2e / both)
    decay: float = 0.5          # both: the oracle weight 1 -> 0 over this share of the epochs
    prior: str = "entropy"      # season_residual: score = z(entropy) + r (entropy) or r alone (none)
    # season_residual capacity (residual.SeasonResidual); the defaults are the arms as first run
    pix_enc: str = "linear"     # linear | conv: the pixel encoder
    month_pool: str = "mean"    # mean | attn: months -> season
    hidden: int = 64            # head width
    depth: int = 1              # head 3x3 conv (MLP hidden) layers
    season_mix: int = 0         # transformer layers across a location's seasons
    pixel: float = 1.0          # lossup: weight of the per-pixel loss regression next to ListNet


@dataclass
class Level2Config:
    patch: int = 4              # finer month crops inside the first layer's (4 or 2 px)
    n: int = 8                  # cells of the patch grid
    pick: str = "oracle"        # oracle | random | entropy (label-free: entropy where +
    #                             attention when) | entropy_r | r_attention (one axis random) |
    #                             entropy0 | entropy0_r (from the crop-free pass, no 2nd pass)
    source: str = "months"      # months: composites | dates: every raw acquisition in them


@dataclass
class QuadrantConfig:
    """selectors.Quadrant: per low-res cell, spatial uncertainty (pooled prediction entropy) x
    temporal uncertainty (entropy of the readout's attention over the cell's seasons)."""
    q_s: float = 0.25           # share of cells "high spatial" (sample: per sample; global: of
    q_t: float = 0.5            # the training distribution); q_t the same for "high temporal"
    mode: str = "sample"        # sample | global (thresholds tracked on the training batches)
    lowspace: str = "skip"      # low spatial + high temporal: skip | lr_all (low-res month tokens
    #                             at every present month of the cell)
    random: bool = False        # training draws q_s ~ U(0, 2 q_s), q_t ~ U(0, 1) per batch (one
    #                             model for every threshold); evaluation uses q_s / q_t
    hs_ht: str = "all"          # high space, high time: every present month (all), or the
    hs_lt: str = "one"          # most-attended season (one) / half the used seasons (half, as
    #                             EveryLocation keep); hs_lt the same for high space, low time
    space: str = "cell"         # u_s per low-res token cell (cell), or per crop window (crop: the
    #                             windows of Entropy + TopWindows on the crop grid)
    rank: str = "weight"        # seasons ranked by the attention weight (score + log months) or
    #                             the readout score alone (score: as the ReadoutAttention scorer);
    #                             oracle: per crop WINDOW, the season(s) whose own low-res map has
    #                             the lowest label loss there (reads the label, also at test)
    # offgrid: each sample's crop count is still its grid windows above the threshold, but the
    # windows are the best non-overlapping crop_px windows at ANY pixel offset (stride-1 mean
    # uncertainty, local-maximum suppression; selectors.Quadrant.offgrid_windows)
    offgrid: bool = False
    unc: str = "entropy"        # spatial uncertainty (classification): pixel entropy, or msr =
    #                             softmax response, 1 - the top class probability


@dataclass
class SelectConfig:
    scorer: str = "label"
    allocator: str = "halves"
    keep: str = "better"        # halves: better | worse | all; every_location: half | one
    month_step: int = 2         # month_grid: every month_step-th month at every location
    season_order: str = "low"   # every_location: the lowest-score seasons (low) or highest (high)
    n: int = 8                  # budget: n windows x the sample's present months
    crop_px: int = 8
    loc_stride: int | None = 8  # candidate lattice stride; None: a window centred on every cell
    residual: ResidualConfig | None = None
    level2: Level2Config | None = None
    # space then time (selectors.SpaceTime): scorer + allocator top pick the n locations, then
    # time_scorer + time_allocator (every_location: keep / season_order; month_grid: month_step)
    # pick each location's months. time_allocator None: one selector (scorer + allocator);
    # time_scorer None with month_grid: no scores ("none" parses as None here).
    quadrant: QuadrantConfig | None = None   # the quadrant policy (replaces scorer / allocator)
    time_scorer: str | None = None
    time_allocator: str | None = None
    route_alpha: float | None = None  # selectors.Route: n crops routed between ps8 windows and
    #                                   ps4 cells by entropy mass x alpha for ps4 (inf: ps4 only)
    entropy_T: bool = False     # season entropy with a per-season temperature (SeasonTemperature)
    entropy_T_fit: str = "train"  # its fit: train (NLL of the training batches) | val (the val
    #                               split's NLL after every epoch, exp.mixres.train)


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
    compute_batches: int = 4           # val batches the reported GMACs/sample are traced on
    init_selector: str = ""            # checkpoints/<name>.pt whose selector networks to load
    init_exit: str = ""                # <file>:<key>: model.exit from checkpoints/<file>.pt entry
    # selector schedule (one run instead of a location-oracle checkpoint + a warm start): the
    # first stage1_epochs of the epochs train with the location oracle (label scorer, top
    # stage1_n windows on the crop grid, every present month) and model.attn stage1_attn; then
    # the configured selector and model.attn, optimizer and LR schedule restarted as a warm
    # start (backbone at lr x UNFREEZE_LR_FACTOR), the best val epoch taken from stage 2 only
    stage1_epochs: int = 0
    stage1_n: int = 0
    stage1_attn: str = "joint"
    #                                    <key> (exp.mixres.early_exit_probe heads, e.g. distill_4)
    season_loss: float = 0.0           # weight of the per-season loss: every used season token's
    #                                    OWN map (selectors.season_maps) against the label, on the
    #                                    training forward's season tokens (reaches the backbone)
    season_loss_detach: bool = False   # the per-season loss stops at model.season_heads (detached
    #                                    season tokens): the backbone learns from the task alone
    freeze_model: bool = False         # train only the selector's networks on the init_ckpt
    #                                    model (frozen, eval mode); epochs=0: evaluate only
    epoch_samples: int = 0             # > 0: each epoch a random subset of that many training
    #                                    samples (legacy --epoch_samples), 0: the whole split
    # oracle mixing (select.quadrant only; one run, no init_ckpt / stage 1): while training, the
    # quadrant's high-space decision comes from the label oracle (each sample's q_s share of
    # crop windows with the highest loss of the crop-free prediction; seasons by attention as
    # always) with probability p, per batch or per crop window, else from the entropy rule;
    # p = oracle_p, annealed linearly to 0 over the first oracle_anneal share of the epochs
    # (0: constant). Evaluation never uses the label.
    oracle_mix: str = ""               # "" | batch | crop
    oracle_p: float = 1.0
    oracle_anneal: float = 0.5


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
        max_hours, limit_batches, compute_batches). Only HASHED fields always enter the hash; a
        field added later enters it only when set away from its default, so adding a field never
        renames existing runs (a chain's later links must find its state file)."""
        d = _hashable(self.to_dict(), _defaults())
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        s = self.select
        def part(sc, al):
            return (f"{sc}-{al}" + (f"-{s.keep}" if al in KEEPS else "")
                    + (f"{s.month_step}" if al == "month_grid" else "")
                    + ("-high" if s.season_order == "high" and al == "every_location" else ""))
        if s.quadrant is not None:
            q = s.quadrant
            tag = (f"quad{q.mode[0]}-{q.lowspace}-qs{q.q_s:g}-qt{q.q_t:g}"
                   + ("-rand" if q.random else "")
                   + (f"-hh{q.hs_ht}" if q.hs_ht != "all" else "")
                   + (f"-hl{q.hs_lt}" if q.hs_lt != "one" else "")
                   + ("-crop" if q.space == "crop" else "")
                   + ("-score" if q.rank == "score" else "-orT" if q.rank == "oracle" else "")
                   + ("-msr" if q.unc == "msr" else "")
                   + ("-og" if q.offgrid else ""))
        elif s.time_allocator is None:
            tag = part(s.scorer, s.allocator)
        else:
            tag = f"{s.scorer}-{s.allocator}+{part(s.time_scorer or 'none', s.time_allocator)}"
        if s.residual:
            tag += f"-res{s.residual.train}{s.residual.arch}"
            if s.residual.prior != "entropy":
                tag += f"-{s.residual.prior}prior"
            r = s.residual
            tag += ((f"-p{r.pix_enc}" if r.pix_enc != "linear" else "")
                    + (f"-m{r.month_pool}" if r.month_pool != "mean" else "")
                    + (f"-h{r.hidden}d{r.depth}" if (r.hidden, r.depth) != (64, 1) else "")
                    + (f"-sm{r.season_mix}" if r.season_mix else ""))
        if s.level2:
            tag += (f"-l2{s.level2.source[0]}{s.level2.patch}n{s.level2.n}"
                    f"{dict(oracle='', random='r', entropy='e', entropy_r='er', r_attention='ra', entropy0='e0',
                             entropy0_r='e0r')[s.level2.pick]}")
        if s.route_alpha is not None:
            tag += f"-route{s.route_alpha:g}"
        geo = (f"c{s.crop_px}" + (f"p{self.model.crop_patch}" if self.model.crop_patch else "")
               + (f"s{s.loc_stride}" if s.loc_stride else ""))
        tag += ("-T" + ("v" if s.entropy_T_fit == "val" else "") if s.entropy_T else "") + (f"-sh{self.model.season_heads}"
                                                 if self.model.season_heads else "") \
            + (f"-sl{self.train.season_loss:g}" if self.train.season_loss else "") \
            + ("d" if self.train.season_loss_detach else "") \
            + {"joint": "", "cross": "-xlr", "cross_hr": "-xhr"}[self.model.attn] \
            + (f"-ex{self.model.exit_layer}" if self.model.exit_layer else "") \
            + ("-ctf" if self.model.crop_time == "frac" else "") \
            + (f"-s1o{self.train.stage1_epochs}"
               + {"joint": "", "cross": "xlr", "cross_hr": "xhr"}[self.train.stage1_attn]
               if self.train.stage1_epochs else "") \
            + (f"-om{self.train.oracle_mix[0]}{self.train.oracle_p:g}a{self.train.oracle_anneal:g}"
               if self.train.oracle_mix else "")
        init = self.train.init_ckpt   # _warmlr: from a fixed-resolution model (no crops)
        warm = (("_warmlr" if "_none_native" in init else "_warm") if init else "") \
            + ("_frozen" if self.train.freeze_model else "")
        return (f"mixres_{self.task}_ps{self.model.lr_patch}_{tag}_n{s.n}_{geo}{warm}"
                f"_seed{self.train.seed}_{h}")

    def check(self, task_kind: str):
        """Raise on a composition that cannot run."""
        s = self.select
        cp = self.model.crop_patch
        if cp and (cp >= s.crop_px or s.crop_px % cp or s.level2):
            raise ValueError("model.crop_patch: a divisor of select.crop_px, finer, no level2")
        if self.model.attn not in ("joint", "cross", "cross_hr"):
            raise ValueError(f"model.attn {self.model.attn!r}: joint | cross | cross_hr")
        t = self.train
        if t.stage1_epochs and not (0 < t.stage1_epochs < t.epochs and t.stage1_n > 0
                                    and t.stage1_attn in ("joint", "cross", "cross_hr")
                                    and not t.init_ckpt and not t.freeze_model):
            raise ValueError("train.stage1_epochs: < epochs, stage1_n > 0, no init_ckpt / freeze")
        if t.oracle_mix and not (t.oracle_mix in ("batch", "crop") and s.quadrant is not None
                                 and not t.stage1_epochs and not t.freeze_model
                                 and 0 < t.oracle_p <= 1 and 0 <= t.oracle_anneal <= 1):
            raise ValueError("train.oracle_mix batch | crop: select.quadrant, no stage1 / freeze, "
                             "oracle_p in (0, 1], oracle_anneal in [0, 1]")
        if s.quadrant is not None:
            q = s.quadrant
            if (q.mode not in ("sample", "global") or q.lowspace not in ("skip", "lr_all")
                    or q.hs_ht not in ("all", "one", "half")
                    or q.hs_lt not in ("all", "one", "half")
                    or q.space not in ("cell", "crop") or q.rank not in ("weight", "score", "oracle")
                    or q.unc not in ("entropy", "msr")
                    or not (0 <= q.q_s <= 1 and 0 <= q.q_t <= 1)):
                raise ValueError(f"select.quadrant {q}")
            if s.time_allocator is not None or s.level2 or s.residual:
                raise ValueError("select.quadrant: no time_allocator / level2 / residual")
            if (self.model.crop_patch or s.crop_px) >= self.model.lr_patch:
                raise ValueError("select.quadrant: crop_px must be finer than model.lr_patch")
        if s.scorer not in SCORERS or s.allocator not in ALLOCATORS:
            raise ValueError(f"scorer {s.scorer!r} / allocator {s.allocator!r}")
        if s.time_scorer is not None and s.time_allocator is None:
            raise ValueError("select.time_scorer without select.time_allocator")
        if s.time_allocator is not None:
            ts = s.time_scorer or "none"
            if s.allocator != "top":
                raise ValueError("space then time: the spatial allocator must be top")
            if ts not in SCORERS or s.time_allocator not in ("every_location",
                                                                         "month_grid"):
                raise ValueError(f"time_scorer {ts!r} / time_allocator "
                                 f"{s.time_allocator!r}: every_location | month_grid")
            if s.time_allocator == "every_location" and s.keep not in KEEPS["every_location"]:
                raise ValueError(f"time_allocator every_location: keep in "
                                 f"{KEEPS['every_location']}, not {s.keep!r}")
            if s.time_allocator == "month_grid" and ts != "none":
                raise ValueError("time_allocator month_grid reads no scores: time_scorer=none")
            miss = ALLOC_IN[s.time_allocator] - SCORER_OUT[ts]
            if miss:
                raise ValueError(f"time_allocator {s.time_allocator} needs {sorted(miss)} scores; "
                                 f"time_scorer {ts} gives "
                                 f"{sorted(SCORER_OUT[ts])}")
            if ts in CLASSIFICATION_ONLY and task_kind != "classification":
                raise ValueError(f"time_scorer {ts}: classification tasks only")
            if ts in ("residual", "entropy_sign") or s.scorer == "season_residual":
                raise ValueError("space then time: a residual scores time (season_residual)")
        if s.allocator in KEEPS and s.keep not in KEEPS[s.allocator]:
            raise ValueError(f"allocator {s.allocator}: keep in {KEEPS[s.allocator]}, not {s.keep!r}")
        if s.season_order not in ("low", "high"):
            raise ValueError(f"select.season_order {s.season_order!r}: low | high")
        if s.season_order == "high" and "every_location" not in (s.allocator, s.time_allocator):
            raise ValueError("select.season_order=high: allocator every_location only")
        if s.allocator == "month_grid" and s.scorer != "none":
            raise ValueError("allocator month_grid reads no scores: select.scorer=none")
        missing = ALLOC_IN[s.allocator] - SCORER_OUT[s.scorer]
        if missing:
            raise ValueError(f"allocator {s.allocator} needs {sorted(missing)} scores; scorer "
                             f"{s.scorer} gives {sorted(SCORER_OUT[s.scorer])}")
        if (s.scorer in CLASSIFICATION_ONLY and task_kind != "classification"
                and s.quadrant is None):    # the quadrant reads no scorer
            raise ValueError(f"scorer {s.scorer}: classification tasks only")
        if s.scorer == "grid" and s.loc_stride is not None:
            raise ValueError("scorer grid: a lattice of centred windows; set select.loc_stride=none")
        learned = ("residual", "season_residual", "lossup")
        if (s.scorer in learned or s.time_scorer in learned) != (s.residual is not None):
            raise ValueError("select.residual is set exactly when select.scorer = residual or "
                             "season_residual")
        if s.residual and s.residual.prior not in ("entropy", "none"):
            raise ValueError(f"select.residual.prior {s.residual.prior!r}: entropy | none")
        rs = "season_residual" in (s.scorer, s.time_scorer)
        if s.residual and s.residual.prior != "entropy" and not (rs or s.scorer == "residual"):
            raise ValueError("select.residual.prior=none: scorer residual / season_residual only")
        if (s.residual and s.scorer == "residual" and s.residual.train != "oracle"
                and s.time_allocator is not None):
            raise ValueError("space then time with a spatial residual: residual.train=oracle")
        r = s.residual
        if r and not rs and (
                (r.pix_enc, r.month_pool, r.hidden, r.depth, r.season_mix)
                != ("linear", "mean", 64, 1, 0)):
            raise ValueError("select.residual pix_enc / month_pool / hidden / depth / season_mix: "
                             "scorer season_residual only")
        if r and r.arch == "conv" and (r.pix_enc, r.month_pool) != ("linear", "mean"):
            raise ValueError("select.residual.arch=conv reads no pixels: pix_enc / month_pool unused")
        if s.route_alpha is not None and (
                (s.scorer, s.allocator, s.time_scorer, s.time_allocator, s.crop_px, s.loc_stride)
                != ("entropy", "top", "attention", "every_location", 8, 8)
                or s.level2 or s.quadrant or s.residual or self.model.lr_patch != 16
                or s.route_alpha < 1):
            raise ValueError("select.route_alpha (>= 1): entropy + attention, ps8 windows at "
                             "stride 8 (c8s8), lr_patch 16, nothing else")
        if s.entropy_T_fit not in ("train", "val") or (s.entropy_T_fit != "train" and not s.entropy_T):
            raise ValueError("select.entropy_T_fit train | val, with select.entropy_T")
        if s.entropy_T and "entropy" not in (s.time_scorer or s.scorer, ):
            raise ValueError("select.entropy_T: the season scorer must be entropy")
        if self.train.season_loss_detach and not (self.model.season_heads and self.train.season_loss):
            raise ValueError("train.season_loss_detach: needs model.season_heads and train.season_loss"
                             " (with the shared head the loss would still move the readout's head)")
        if (self.train.season_loss or self.model.season_heads) and task_kind != "classification":
            raise ValueError("train.season_loss / model.season_heads: classification tasks only")
        if self.train.freeze_model and not self.train.init_ckpt:
            raise ValueError("train.freeze_model needs train.init_ckpt (the model to freeze)")
        if "lossup" in (s.scorer, s.time_scorer):
            if s.residual.arch not in ("stats", "bilinear", "manyup") or s.residual.train != "oracle":
                raise ValueError("scorer lossup: residual.arch stats | bilinear | manyup, "
                                 "residual.train oracle")
        elif s.residual and s.residual.arch not in ("conv", "hr", "xattn", "xattn_conv"):
            raise ValueError(f"select.residual.arch {s.residual.arch!r}")
        if "lossup" in (s.scorer, s.time_scorer) and s.time_allocator is None and s.scorer != "lossup":
            raise ValueError("time_scorer lossup: space then time only")
        if s.level2 and s.level2.patch >= s.crop_px and s.level2.source == "months":
            raise ValueError(f"level2 patch {s.level2.patch} not finer than crop_px {s.crop_px}")
        if s.level2 and s.level2.patch > s.crop_px:
            raise ValueError(f"level2 patch {s.level2.patch} larger than crop_px {s.crop_px}")
        if (s.level2 is not None and s.level2.source == "dates") != self.model.date_tokens:
            raise ValueError("model.date_tokens is set exactly when level2.source = dates")


# the config fields as of 2026-10-23: always hashed (run names of everything launched until then)
HASHED = {
    "task", "model.model_size", "model.lr_patch", "model.inject_layer", "model.date_tokens",
    "select.scorer", "select.allocator", "select.keep", "select.month_step", "select.n",
    "select.crop_px", "select.loc_stride", "select.residual", "select.residual.arch",
    "select.residual.train", "select.residual.tau", "select.residual.decay", "select.level2",
    "select.level2.patch", "select.level2.n", "select.level2.pick", "select.level2.source",
    "train.epochs", "train.lr", "train.batch_size", "train.seed", "train.init_ckpt"}
# fields added to a section hashed whole (quadrant): hashed only when off these defaults
LATE_DEFAULTS = {"select.quadrant.hs_ht": "all", "select.quadrant.hs_lt": "one",
                 "select.quadrant.space": "cell", "select.quadrant.rank": "weight",
                 "select.quadrant.unc": "entropy", "select.quadrant.offgrid": False}
NOT_HASHED = {"train.num_workers", "train.max_hours", "train.limit_batches",
              "train.compute_batches"}


def _defaults() -> dict:
    d = asdict(RunConfig())
    d["select"]["residual"] = asdict(ResidualConfig())
    d["select"]["level2"] = asdict(Level2Config())
    return d


def _hashable(d: dict, dflt: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        p = pre + k
        if p in NOT_HASHED or (p in LATE_DEFAULTS and v == LATE_DEFAULTS[p]):
            continue
        dv = dflt.get(k) if isinstance(dflt, dict) else None
        if isinstance(v, dict):
            out[k] = _hashable(v, dv if isinstance(dv, dict) else {}, p + ".")
        elif p in HASHED or v != dv:
            out[k] = v
    return out


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
    for c in (ResidualConfig, Level2Config, QuadrantConfig, SelectConfig, TrainConfig,
              ModelConfig):
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
    if raw.lower() in ("none", "null") and "None" in str(tp):     # only for optional fields
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
