"""The mixed-resolution model: tokenize -> encode -> decode, with one encoder for every patch size.

    out0 = model.uniform(ms, ss, season, present)          season tokens only (the LR pass)
    plan = selector(..., out0, ...)                        manyup.mixres.plan.Plan
    out  = model(ms, ss, season, present, plan)            season tokens + the plan's crops

Submodule names (mix, head, readout) match the legacy exp.upsamplers.finetune_inject.InjectFT,
so its checkpoints load unchanged (load_legacy).
"""
from dataclasses import dataclass, replace

import copy

import torch
import torch.nn.functional as F
from torch import nn

from exp.common.config import MODEL_SIZE_TO_ID
from exp.common.heads import Pa2PxHead
from exp.common.tasks import Task
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput, _to_device, crop_patches
from manyup.encoders.prithvi import MODELS as PRITHVI, PrithviOE
from manyup.inject import MixedResOE, Packed
from manyup.mixres.decoder import NativeSetReadout
from manyup.mixres.plan import LOOKUP_PS, TILE, Plan

LR_PS = 16          # the head maps a token to a 16 x 16 px block (finer tokens: area-pooled)


@dataclass
class ModelConfig:
    model_size: str = "tiny"       # an OlmoEarth size, or prithvi_tiny / prithvi_100m
    lr_patch: int = 16          # season token patch size (16, 14, 12, 10, 8, 6 or 4)
    # > 0: every picked window is tokenized as its sub-windows of this patch (Plan.split): the
    # selector still picks select.crop_px windows (e.g. ps4 seasons, 4 px windows, ps2 tokens)
    crop_patch: int = 0
    inject_layer: int = 0       # crops join the season tokens before this block
    date_tokens: bool = False   # a learned embedding marking raw-acquisition crops (date_type)
    # per-step tasks: encode every step ALONE (one sequence per (sample, step), no attention
    # across steps), as a single-image model sees a series; uniform tokens only (no crops)
    independent_steps: bool = False
    # > 0: that many per-season heads (one per season slot, copies of the shared head once the
    # weights are loaded) giving each season token's OWN map (selectors.season_maps): what the
    # selectors and the per-season loss (train.season_loss) read; the readout keeps the shared
    season_heads: int = 0
    # who attends to whom once crops join (manyup.inject.MixedResOE attn): "joint" one sequence
    # per sample; "cross" crop tokens read only the season tokens; "cross_hr" + every crop
    # token of the sample. In both cross modes the season tokens never read crops, so inference
    # reuses the crop-free pass and pays for the crop tokens alone
    attn: str = "joint"
    # > 0: the crop-free pass (what the selectors read) exits after this block through an
    # ExitHead instead of running the whole model; the head is distilled online from the full
    # pass while training (MixedResModel.exit_loss), so it tracks the fine-tuned backbone
    exit_layer: int = 0
    # a month crop's time slot: its season's ("slot"), or the season's plus the month's offset
    # from the season's middle in season lengths ("frac": -1/3, 0, 1/3 for 3-month seasons), so
    # the months of one season stay apart in the sin-cos time encoding (Prithvi has no month
    # encoding of its own; OlmoEarth's date crops already use fractional slots)
    crop_time: str = "slot"


@dataclass
class Out:
    """One encode + decode.
    logits  (B, C, 64, 64); a per-step task (Task.per_step): (B, T, C, 64, 64), one map per step
    lr_F    (B, NS, h, w, D) final season tokens
    hr      (N, D) final crop tokens, in plan order (N = 0 for the uniform pass)
    seasons the embedded season tokens it started from (the uniform pass; see forward)
    lr_enc  cross attn modes, the uniform pass: the season tokens entering each block and the
            final ones (MixedResOE.lr_trace), reused by forward at inference"""
    logits: torch.Tensor
    lr_F: torch.Tensor
    hr: torch.Tensor
    seasons: Packed | None = None
    lr_enc: tuple | None = None


class ExitHead(nn.Module):
    """Block L's season tokens -> the crop-free pass: the model's final norm, a residual MLP
    adapter (zero-init output: starts as the model's own head on block L), and its own copy of
    the readout and head (copy_from)."""

    def __init__(self, dim: int, readout: nn.Module, head: nn.Module):
        super().__init__()
        self.adapt = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                   nn.Linear(dim, dim))
        nn.init.zeros_(self.adapt[-1].weight)
        nn.init.zeros_(self.adapt[-1].bias)
        self.readout, self.head = copy.deepcopy(readout), copy.deepcopy(head)

    def copy_from(self, model: "MixedResModel"):
        self.readout.load_state_dict(model.readout.state_dict())
        self.head.load_state_dict(model.head.state_dict())

    def forward(self, norm: nn.Module, lr: Packed, xL, season, present) -> Out:
        x = norm(xL)
        x = x + self.adapt(x)
        lr_F = lr.pooled(x).float()
        z = torch.zeros(0, dtype=torch.long, device=lr_F.device)
        hr_F = lr_F.new_zeros(0, lr_F.shape[-1])
        logits = self.readout(self.head.linear, lr_F, season, present, hr_F, (z,) * 4)
        return Out(logits.float(), lr_F, hr_F, lr)


class MixedResModel(nn.Module):
    def __init__(self, task: Task, cfg: ModelConfig, pretrained: bool = True):
        super().__init__()
        if cfg.independent_steps and not task.per_step:
            raise ValueError("independent_steps: per-step tasks only")
        if cfg.attn != "joint" and (cfg.independent_steps or cfg.inject_layer):
            raise ValueError(f"attn {cfg.attn}: inject_layer 0, no independent_steps")
        if (cfg.lr_patch not in (4, 6, 8, 10, 12, 14, 16)
                and not (cfg.lr_patch == 32 and task.per_step)):
            raise ValueError(f"lr_patch {cfg.lr_patch}: 16, 14, 12, 10, 8, 6 or 4 "
                             "(32: per-step tasks)")
        self.task, self.cfg = task, cfg
        if cfg.model_size in PRITHVI:   # Prithvi-EO-2.0, PI-resized (manyup.encoders.prithvi)
            encoder = PrithviOE(cfg.model_size, pretrained)
        else:
            encoder = OlmoEarthEncoder(MODEL_SIZE_TO_ID[cfg.model_size],
                                       OlmoEarthInput(task.olmoearth.modalities,
                                                      task.olmoearth.impute),
                                       LR_PS, TILE, name=cfg.model_size,
                                       pretrained=pretrained).encoder
        self.mix = MixedResOE(encoder, cfg.inject_layer, "center", cfg.lr_patch, attn=cfg.attn)
        D = encoder.embedding_size
        # the head writes a head_px x head_px map per token: 16, or a ps32 token's own 32
        self.head_px = max(LR_PS, cfg.lr_patch)
        self.head = Pa2PxHead(task.reduction, D, task.num_outputs, self.head_px)
        # ps6 seasons cover the tile padded to 66 px (MixedResOE.embed): the readout works on
        # 2 px cells of that frame, crops placed lo px in, the map cropped back to the tile
        self.lo = (-(-TILE // cfg.lr_patch) * cfg.lr_patch - TILE) // 2
        self.readout = NativeSetReadout(D, lr_px=self.head_px, base_px=cfg.lr_patch,
                                        **({"crop_px": 2, "out_px": TILE} if self.lo else {}))
        self.date_type = nn.Parameter(torch.zeros(D)) if cfg.date_tokens else None
        self.season_head = (nn.ModuleList(copy.deepcopy(self.head.linear)
                                          for _ in range(cfg.season_heads))
                             if cfg.season_heads else None)
        if cfg.exit_layer and not (0 < cfg.exit_layer < len(encoder.blocks)
                                   and cfg.attn == "joint" and not cfg.independent_steps):
            raise ValueError(f"exit_layer {cfg.exit_layer}: in (0, blocks), attn joint")
        self.exit = ExitHead(D, self.readout, self.head) if cfg.exit_layer else None

    @property
    def readout0(self) -> nn.Module:
        """The readout of the crop-free pass (the exit head's under exit_layer)."""
        return self.readout if self.exit is None else self.exit.readout

    @property
    def device(self):
        return self.head.linear.weight.device

    # -- tokenize ---------------------------------------------------------------------------
    def tokenize_seasons(self, ss) -> Packed:
        """The season composites at the season patch size; blocks [0, L) applied.
        independent_steps: B x NS one-step sequences (encode(steps=NS) joins them back)."""
        if self.cfg.independent_steps:
            ss = split_steps(ss)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.mix.head_blocks(self.mix.embed(ss, self.cfg.lr_patch))

    def tokenize_crops(self, ms, plan: Plan, season, lr: Packed, dates=None,
                       season_ts=None) -> Packed:
        """The plan's windows (of the monthly images, or of one modality's raw acquisitions),
        one single-patch sequence each, placed at their pixel position and time slot; blocks
        [0, L) applied."""
        if plan.src == "months":
            src = ms
        else:
            if dates is None or self.date_type is None:
                raise ValueError(f"date crops ({plan.src}): need the batch's dates and a model "
                                 "with date_tokens")
            src = _to_device(dates[plan.src][0], self.device, torch.bfloat16)
        st = plan.t if plan.src_t is None else plan.src_t
        crops = crop_patches(src, plan.b, st, plan.py, plan.px, plan.patch, pixels=True)
        slot = season[plan.b, plan.t] if plan.slot is None else plan.slot
        if self.cfg.crop_time == "frac" and plan.slot is None and plan.src == "months":
            # month middle vs season middle (season_ts: the season steps' mid timestamps)
            mid = _days(crops.timestamps[:, 0].to(slot.device)) + 15.2
            centre = _days(season_ts.to(slot.device)[plan.b, slot])
            slot = slot.float() + (mid - centre) / (30.44 * self.task.season)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crops, plan.patch, slot=slot, py=plan.py + self.lo,
                                px=plan.px + self.lo)
            if plan.src != "months":
                hr.x = (hr.x + self.date_type).to(hr.x.dtype)
            return self.mix.head_blocks(hr, lr, plan.b)

    # -- encode / decode --------------------------------------------------------------------
    def encode(self, lr: Packed, crops: list[tuple[Packed, Plan]], steps: int | None = None,
               lr_enc: tuple | None = None):
        """Blocks [L, n) with every sample's season tokens, then its crops group by group, in
        one sequence. -> (season tokens (B, NS, h, w, D), crop tokens (N, D) in group order).
        steps: NS of independent_steps sequences (B x NS of them, one step each). lr_enc: the
        uniform pass's MixedResOE.lr_trace (cross attn modes, inference)."""
        if self.cfg.independent_steps and crops:
            raise ValueError("independent_steps: no crops (the uniform tokens only)")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr_x, outs = self.mix.tail_multi(lr, [(hr, p.b) for hr, p in crops], lr_enc)
        lr_F = lr.pooled(lr_x).float()
        if self.cfg.independent_steps:
            lr_F = lr_F.view(-1, steps, *lr_F.shape[2:])
        hr_F = [hr.pooled(x).float()[:, 0, 0, 0] for (hr, _), x in zip(crops, outs)]
        return lr_F, (torch.cat(hr_F) if hr_F else lr_F.new_zeros(0, lr_F.shape[-1]))

    def decode(self, lr_F, hr_F, plans: list[Plan], season, present, logw=None):
        """Every token through the head at its own patch size, pooled per pixel over the
        tokens covering it (NativeSetReadout). plans: the non-empty crop groups, hr_F's order.

        One group and several follow the legacy conventions of a plain plan and of --level2
        respectively (footprints as ps4 cell counts, pixel positions only when a group is off
        the ps4 grid); they are kept apart to stay bit-exact with the legacy checkpoints.

        A per-step task decodes every step t on its own: (sample, step) is one readout sample
        whose candidates are the token of t's season and the crops of step t
        -> (B, T, C, 64, 64)."""
        if self.task.per_step:
            return self._decode_steps(lr_F, hr_F, plans, season, present, logw)
        return self._decode(lr_F, hr_F, plans, season, present, logw)

    def _decode_steps(self, lr_F, hr_F, plans, season, present, logw):
        B, T = season.shape
        bt = torch.arange(B * T, device=lr_F.device)
        s = season.clamp(min=0).flatten()
        lr_bt = lr_F[bt // T, s][:, None]                                  # (B T, 1, h, w, D)
        season_bt = torch.zeros(B * T, 1, dtype=torch.long, device=lr_F.device)
        present_bt = (present & (season >= 0)).reshape(B * T, 1)
        plans = [replace(p, b=p.b * T + p.t, t=torch.zeros_like(p.t)) for p in plans]
        out = self._decode(lr_bt, hr_F, plans, season_bt, present_bt, logw)
        return out.view(B, T, *out.shape[1:])

    def _decode(self, lr_F, hr_F, plans: list[Plan], season, present, logw=None):
        if not plans:
            z = torch.zeros(0, dtype=torch.long, device=lr_F.device)
            return self.readout(self.head.linear, lr_F, season, present, hr_F, (z,) * 4)
        if len(plans) == 1:
            p = plans[0]
            off = not p.on_grid or self.lo
            return self.readout(self.head.linear, lr_F, season, present, hr_F,
                                (p.b, p.t, p.y, p.x), logw, torch.ones_like(p.b),
                                hr_px=(p.py + self.lo, p.px + self.lo) if off else None,
                                hr_fp=p.patch)
        if logw is not None:
            raise ValueError("crop gates with several crop groups: not ported")
        cat = lambda f: torch.cat([f(p) for p in plans])                         # noqa: E731
        idx = tuple(cat(lambda p, i=i: (p.b, p.t, p.y, p.x)[i]) for i in range(4))
        sizes = cat(lambda p: torch.full_like(p.b, p.cells))
        off = any(p.patch < LOOKUP_PS or p.offgrid for p in plans) or self.lo
        return self.readout(self.head.linear, lr_F, season, present, hr_F, idx, None, sizes,
                            hr_px=(cat(lambda p: p.py + self.lo), cat(lambda p: p.px + self.lo))
                            if off else None,
                            hr_fp=cat(lambda p: torch.full_like(p.b, p.patch)) if off
                            else plans[0].patch)

    # -- passes -----------------------------------------------------------------------------
    def _inputs(self, ms, ss, season, present):
        dev = self.device
        return (_to_device(ms, dev, torch.bfloat16), _to_device(ss, dev, torch.bfloat16),
                season.to(dev), present.to(dev))

    def uniform(self, ms, ss, season, present) -> Out:
        """The season tokens alone (the plan with no crops): what a selector reads. Eval mode and
        bfloat16 throughout, as the legacy crop-free pass (InjectFT._crop_free). Call it OUTSIDE
        any autocast region: autocast caches its bfloat16 weight copies until the outermost
        region exits, and copies made under no_grad would leave the next pass's weights without
        gradients."""
        if torch.is_autocast_enabled():
            raise RuntimeError("MixedResModel.uniform inside an autocast region (see docstring)")
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        was = self.training
        if was:             # (eval() walks every module: skipped when already in eval mode)
            self.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lr = self.tokenize_seasons(ss)
                if self.exit is not None:
                    xL = self.mix.blocks(lr.x, lr.seqlens, 0, self.cfg.exit_layer)
                    return self.exit(self.mix.encoder.norm, lr, xL, season, present)
                lr_enc = None
                if self.cfg.attn.startswith("cross"):   # (= encode(lr, []), its trace kept)
                    lr_enc = self.mix.lr_trace(lr)
                    lr_F = lr.pooled(lr_enc[1]).float()
                    hr_F = lr_F.new_zeros(0, lr_F.shape[-1])
                else:
                    lr_F, hr_F = self.encode(lr, [], ss.timestamps.shape[1])
                logits = self.decode(lr_F, hr_F, [], season, present).float()
        finally:
            if was:
                self.train()
        return Out(logits, lr_F, hr_F, lr, lr_enc)

    def exit_loss(self, ms, ss, season, present) -> torch.Tensor:
        """Online distillation of the exit head (training): KL of its per-pixel class
        distribution and of its per-cell softmax of the readout score over the used seasons
        to the full crop-free pass's, on the frozen (no-grad, eval) backbone's tokens."""
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        was = self.training
        self.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lr = self.tokenize_seasons(ss)
                trace, x = self.mix._lr_trace(lr.x, lr.seqlens, 0, len(self.mix.encoder.blocks))
                lr_F = lr.pooled(self.mix.encoder.norm(x)).float()
                full = self.decode(lr_F, lr_F.new_zeros(0, lr_F.shape[-1]), [], season,
                                   present).float()
                s_full = self.readout._score(lr_F).float()
        finally:
            self.train(was)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.exit(self.mix.encoder.norm, lr, trace[self.cfg.exit_layer].detach(),
                          season, present)
            s_exit = self.exit.readout._score(o.lr_F).float()
        NS = lr_F.shape[1]
        valid = present & (season >= 0)
        months = (F.one_hot(season.clamp(min=0), NS).float() * valid.float()[..., None]).sum(1)
        unused = (months == 0)[:, :, None, None]
        lp = lambda s: s.masked_fill(unused, -1e4).log_softmax(1)                # noqa: E731
        kl = lambda a, b: F.kl_div(a, b, log_target=True, reduction="none").sum(1).mean()  # noqa: E731
        return kl(o.logits.log_softmax(1), full.log_softmax(1)) + kl(lp(s_exit), lp(s_full))

    def split(self, plan: Plan | list[Plan], logw=None):
        """-> the non-empty crop groups, each window split into model.crop_patch sub-windows
        (the crop gate log-weights repeated with them)."""
        plans = [p for p in ([plan] if isinstance(plan, Plan) else plan) if len(p)]
        cp = self.cfg.crop_patch
        if not cp or not any(p.patch > cp for p in plans):
            return plans, logw
        if logw is not None:
            logw = logw.repeat_interleave((plans[0].patch // cp) ** 2)
        return [p.split(cp) if p.patch > cp else p for p in plans], logw

    def forward(self, ms, ss, season, present, plan: Plan | list[Plan], logw=None,
                dates=None, seasons: Packed | None = None, lr_enc: tuple | None = None) -> Out:
        """plan: one crop group, or several (e.g. Level2: layer-1 crops, then finer ones).
        dates: the batch's raw acquisitions, for date crop groups. seasons: the uniform pass's
        embedded season tokens, reused at inference (they carry no gradient: never in
        training)."""
        if (seasons is not None or lr_enc is not None) and (self.training
                                                             or torch.is_grad_enabled()):
            raise ValueError("forward(seasons=/lr_enc=...) reuses no-grad tokens: inference only")
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        plans, logw = self.split(plan, logw)
        lr = self.tokenize_seasons(ss) if seasons is None else seasons
        crops = [(self.tokenize_crops(ms, p, season, lr, dates, ss.timestamps), p)
                 for p in plans]
        lr_F, hr_F = self.encode(lr, crops, ss.timestamps.shape[1], lr_enc)
        return Out(self.decode(lr_F, hr_F, plans, season, present, logw), lr_F, hr_F)


_MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def _days(ts: torch.Tensor) -> torch.Tensor:
    """(..., 3) [day, month (0-11), year] -> (...,) float days (365.25-day years)."""
    start = torch.tensor(_MONTH_START, device=ts.device, dtype=torch.float32)
    return ts[..., 2].float() * 365.25 + start[ts[..., 1]] + ts[..., 0].float()


def split_steps(s):
    """MaskedOlmoEarthSample of B samples x T steps -> B x T samples of one step each (sample
    b's step t at b * T + t). Spatial multitemporal modalities only."""
    B, T = s.timestamps.shape[:2]
    out = {"timestamps": s.timestamps.reshape(B * T, 1, -1)}
    for k, v in s._asdict().items():
        if k == "timestamps" or v is None:
            continue
        if v.ndim != 5 or v.shape[3] != T:
            raise ValueError(f"split_steps: {k} {tuple(v.shape)} is not (B, H, W, T, C)")
        out[k] = v.permute(0, 3, 1, 2, 4).reshape(B * T, v.shape[1], v.shape[2], 1, v.shape[4])
    return type(s)(**out)


def load_legacy(model: MixedResModel, ckpt: str) -> list[str]:
    """Load checkpoints/<ckpt>.pt of a legacy InjectFT run; -> the legacy keys not used here
    (selector heads, residual scorers, ...)."""
    sd = torch.load(f"checkpoints/{ckpt}.pt", map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    missing = [k for k in own if k not in sd]
    exit_new = any(k.startswith("exit.") for k in missing)
    missing = [k for k in missing if not k.startswith("exit.")]
    new_heads = all(k.startswith("season_head.") for k in missing) and missing
    if missing and not new_heads:
        raise KeyError(f"{ckpt}: missing {missing[:5]}")
    model.load_state_dict({k: sd[k] for k in own if k in sd},
                          strict=not (new_heads or exit_new))
    if new_heads:                       # per-season heads start as copies of the shared head
        for lin in model.season_head:
            lin.load_state_dict(model.head.linear.state_dict())
    if exit_new:                        # an exit head starts as the loaded readout + head
        model.exit.copy_from(model)
    return sorted(k for k in sd if k not in own)
