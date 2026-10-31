"""Selectors for per-step tasks (Task.per_step: a label at every step, e.g. EuroMineNet's annual
mine masks; the model decodes one map per step, MixedResModel.decode).

A step's season token predicts that step (with one step per season, each year is its own
low-res token), so a (season, window) pair has its own label and its own uncertainty: the
per-season scores are the uniform pass's loss / entropy of THAT season's map at its own steps'
labels, which is what makes a time scorer testable here (on PASTIS every season was scored
against the one static label).

The candidates are the non-overlapping crop_px windows that tile the tile (loc_stride =
crop_px), for every scorer, so arms at the same n differ only in where the crops go:

    scorer          season (pair) score                        window score
    label           loss of the season's map vs its steps'     loss of the decoded steps,
                    labels (oracle)                            mean over present steps
    entropy         entropy of the season's map                entropy of the decoded steps,
                                                               mean over present steps
    season_random   uniform                                    --
    season_grid     a lattice of n windows per season, one     --
                    coset per season (random phase in
                    training, fixed at test)
    random          --                                         uniform (selectors.RandomWindows)
    none            -- (allocator month_grid)

Allocators are selectors' own: pairs (the best (season, window) pairs within n windows x the
sample's present steps; highest score first: the most wrong / most uncertain step-windows),
top (n windows, every step), every_location (each window its lowest / highest-score seasons),
month_grid. select.n = 0 with pairs: no crops (the fixed-resolution arms).
"""
import torch
import torch.nn.functional as F

from manyup.mixres.plan import TILE
from manyup.mixres.selectors import (H, NEG, Geometry, NoScores, RandomWindows, Scorer, Scores,
                                     Selector, _allocator, pixel_entropy, season_maps)

STEP_SCORERS = ("label", "entropy", "season_random", "season_grid", "random", "none")
STEP_ALLOCATORS = ("pairs", "top", "every_location", "month_grid")


def _members(ctx) -> torch.Tensor:
    """(B, NS, T) float: step t is a present member of season s."""
    oh = F.one_hot(ctx.season.clamp(min=0), ctx.NS).float() * ctx.valid.float()[..., None]
    return oh.transpose(1, 2)


def _season_mean(px: torch.Tensor, ctx) -> torch.Tensor:
    """(B, T, 64, 64) per-step pixel values -> (B, NS, 64, 64): each season's mean over its
    present steps (0 for an unused season)."""
    m = _members(ctx)
    return torch.einsum("bst,bthw->bshw", m, px) / m.sum(2).clamp(min=1)[..., None, None]


def _step_mean(px: torch.Tensor, ctx) -> torch.Tensor:
    """(B, T, 64, 64) -> (B, 64, 64): the mean over present steps."""
    v = ctx.valid.float()
    return torch.einsum("bt,bthw->bhw", v, px) / v.sum(1).clamp(min=1)[:, None, None]


def step_label_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N, T, C, 64, 64) vs (N, T, 64, 64) -> (N, T, 64, 64) cross-entropy, 0 where
    unlabelled."""
    N, T = target.shape[:2]
    return F.cross_entropy(logits.flatten(0, 1).float(), target.flatten(0, 1).long(),
                           ignore_index=-1, reduction="none").view(N, T, *target.shape[2:])


class StepLabelLoss(Scorer):
    """The label oracle per step: a (season, window) pair scores the loss of the season token's
    own map against its steps' labels."""

    def __call__(self, ctx, geo, need=("window", "season")):
        if ctx.target is None:
            raise ValueError("StepLabelLoss: no target in the context")
        out0 = ctx.out0
        sc = win = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if "season" in need:
                s_log = season_maps(ctx.model, out0.lr_F).float()           # (B NS, C, 64, 64)
                s_log = s_log.view(ctx.B, ctx.NS, *s_log.shape[1:])
                # season s's map against each of its steps' labels: (B, T, 64, 64)
                own = s_log[torch.arange(ctx.B, device=s_log.device)[:, None],
                            ctx.season.clamp(min=0)]
                px = step_label_loss(own, ctx.target)
                sc = geo.window_mean(_season_mean(px, ctx).flatten(0, 1)).view(
                    ctx.B, ctx.NS, H, H)
            if "window" in need:
                px = step_label_loss(out0.logits, ctx.target)
                win = geo.window_mean(_step_mean(px, ctx))
        return Scores(win, sc, pooled=out0.logits, lr_F=out0.lr_F)


class StepEntropy(Scorer):
    """Label-free: the entropy of each season token's own map (pairs), or of every decoded
    step, mean over present steps (windows); the uniform pass of the model being trained."""

    def __call__(self, ctx, geo, need=("window", "season")):
        if ctx.model.task.kind != "classification":
            raise ValueError(f"StepEntropy: {ctx.model.task.name} is not a classification task")
        out0 = ctx.out0
        sc = win = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if "season" in need:
                e = pixel_entropy(season_maps(ctx.model, out0.lr_F).float())   # (B NS, 64, 64)
                sc = geo.window_mean(e).view(ctx.B, ctx.NS, H, H)
            if "window" in need:
                e = pixel_entropy(out0.logits.float().flatten(0, 1)).view(
                    ctx.B, ctx.T, TILE, TILE)
                win = geo.window_mean(_step_mean(e, ctx))
        return Scores(win, sc, pooled=out0.logits, lr_F=out0.lr_F)


def _tiling_mask(geo, device) -> torch.Tensor:
    """(16, 16) bool: the anchor cells of the windows that tile the tile."""
    cell = geo.tiling().to(device)
    m = torch.zeros(H, H, dtype=torch.bool, device=device)
    m[cell[:, None], cell[None, :]] = True
    return m


class StepRandom(Scorer):
    """Content-blind: a uniform score per (season, window) pair."""

    def __call__(self, ctx, geo, need=("season",)):
        dev = ctx.present.device
        s = torch.rand(ctx.B, ctx.NS, H, H, device=dev)
        return Scores(season=s.masked_fill(~_tiling_mask(geo, dev), NEG))


class StepGrid(Scorer):
    """Content-blind: in every season a lattice of n of the S x S tiling windows, point (i, j)
    kept when (i + c j + phase + season) mod m == 0, m = S^2 / n, c = round(sqrt(m)) (exactly
    n per season when m divides S), so consecutive seasons take shifted cosets and every window
    is covered once every m seasons. Random phase per sample while training, 0 at test; ties
    (none within a lattice) broken at random."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, ctx, geo, need=("season",)):
        dev = ctx.present.device
        cell = geo.tiling().to(dev)
        S = len(cell)
        m = max(1, round(S * S / max(self.n, 1)))
        c = max(1, round(m ** 0.5))
        i, j = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev),
                              indexing="ij")
        phase = (torch.randint(0, m, (ctx.B, 1, 1, 1), device=dev) if ctx.model.training
                 else torch.zeros(ctx.B, 1, 1, 1, dtype=torch.long, device=dev))
        s = torch.arange(ctx.NS, device=dev)[None, :, None, None]
        on = ((i + c * j)[None, None] + phase + s) % m == 0                   # (B, NS, S, S)
        sc = torch.full((ctx.B, ctx.NS, H, H), NEG, device=dev)
        sc[:, :, cell[:, None], cell[None, :]] = on.float() + 1e-3 * torch.rand(
            ctx.B, ctx.NS, S, S, device=dev)
        return Scores(season=sc)


class StepThreshold(Selector):
    """select.quadrant on a per-step task: space and time jointly, per (step, window) pair of
    the tiling windows. A pair's uncertainty is the entropy of that step's decoded prediction
    (the crop-free pass) over the window; a crop at each pair above the training distribution's
    q_s quantile (mode global: QuadrantState, refit on val at test by exp.mixres.train), or at
    each sample's top q_s share of its present pairs (mode sample). Each step is its own season
    here, so the pair is the time choice too (the quadrant's attention rules have nothing to
    rank)."""

    def __init__(self, geometry: Geometry, cfg, device=None):
        super().__init__(geometry, NoScores(), None)
        from manyup.mixres.selectors import QuadrantState
        self.cfg = cfg
        self.state = QuadrantState().to(device)
        self.last = {}

    def modules(self) -> list:
        return [self.state]

    def uncertainty(self, ctx):
        """-> u (B, T, 16, 16) on the windows' anchor cells, cand (B, T, 16, 16) bool: a tiling
        window at a present step."""
        with torch.no_grad():
            e = pixel_entropy(ctx.out0.logits.float().flatten(0, 1))           # (B T, 64, 64)
            u = self.geo.window_mean(e).view(ctx.B, ctx.T, H, H)
        cand = _tiling_mask(self.geo, u.device) & ctx.valid[:, :, None, None]
        return u, cand

    def threshold_values(self, ctx) -> torch.Tensor:
        u, cand = self.uncertainty(ctx)
        return u[cand]

    def __call__(self, ctx):
        from manyup.mixres.selectors import Selection
        if ctx.model.task.kind != "classification":
            raise ValueError(f"StepThreshold: {ctx.model.task.name} is not a classification task")
        u, cand = self.uncertainty(ctx)
        if ctx.model.training and self.cfg.mode == "global":
            self.state.update(u[cand], u[cand])
        q = self.cfg.q_s
        if self.cfg.mode == "global":
            mask = cand & (u >= self.state.threshold(self.state.tab_s, q))
        else:
            k = (q * cand.flatten(1).sum(1).float()).round().long()            # (B,)
            r = u.masked_fill(~cand, NEG).flatten(1).argsort(1, descending=True).argsort(1)
            mask = (r < k[:, None]).view_as(cand) & cand
        n = float(cand.sum().clamp(min=1))
        self.last = {"q_s": q, "picked": float(mask.sum()) / n}
        return Selection(self.geo.plan(mask), mask)


class NoCrops(Selector):
    """select.n = 0: the uniform tokens alone (the fixed-resolution arms)."""

    def __init__(self, geometry: Geometry):
        super().__init__(geometry, NoScores(), None)

    def __call__(self, ctx):
        from manyup.mixres.selectors import Selection
        mask = torch.zeros(ctx.B, ctx.T, H, H, dtype=torch.bool, device=ctx.present.device)
        return Selection(self.geo.plan(mask), mask)


def build_step_selector(cfg, model, example=None) -> Selector:
    """manyup.mixres.config.SelectConfig -> Selector for a per-step task. Raises on what is not
    defined per step (residuals, level 2, space then time)."""
    if cfg.residual is not None or cfg.level2 is not None or cfg.time_allocator is not None:
        raise ValueError("per-step tasks: no residual / level2 / space-then-time selectors")
    if cfg.scorer not in STEP_SCORERS or cfg.allocator not in STEP_ALLOCATORS:
        raise ValueError(f"per-step tasks: scorer in {STEP_SCORERS}, allocator in "
                         f"{STEP_ALLOCATORS}; not {cfg.scorer!r} / {cfg.allocator!r}")
    if cfg.loc_stride != cfg.crop_px:
        raise ValueError("per-step tasks: candidates are the tiling windows, "
                         f"select.loc_stride = crop_px ({cfg.crop_px}), not {cfg.loc_stride}")
    geo = Geometry(cfg.crop_px, cfg.loc_stride)
    tile = getattr(model.task, "tile", TILE)
    if tile != TILE and not (cfg.n == 0 and cfg.allocator == "pairs"):
        raise ValueError(f"{model.task.name}: {tile} px tiles; crop selectors work on the "
                         f"{TILE} px tile's cell grid (no crops: select.n = 0, allocator pairs)")
    if cfg.quadrant is not None:
        if cfg.quadrant.mode not in ("sample", "global") or cfg.quadrant.lowspace != "skip":
            raise ValueError("per-step select.quadrant: mode sample | global, lowspace skip")
        return StepThreshold(geo, cfg.quadrant, model.device)
    if cfg.n == 0:
        if cfg.allocator != "pairs":
            raise ValueError("select.n = 0 (no crops): allocator pairs")
        return NoCrops(geo)
    scorer = {"label": StepLabelLoss, "entropy": StepEntropy, "season_random": StepRandom,
              "season_grid": lambda: StepGrid(cfg.n), "random": RandomWindows,
              "none": NoScores}[cfg.scorer]()
    return Selector(geo, scorer, _allocator(cfg.allocator, cfg))
