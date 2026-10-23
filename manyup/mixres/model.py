"""The mixed-resolution model: tokenize -> encode -> decode, with one encoder for every patch size.

    out0 = model.uniform(ms, ss, season, present)          season tokens only (the LR pass)
    plan = selector(..., out0, ...)                        manyup.mixres.plan.Plan
    out  = model(ms, ss, season, present, plan)            season tokens + the plan's crops

Submodule names (mix, head, readout) match the legacy exp.upsamplers.finetune_inject.InjectFT,
so its checkpoints load unchanged (load_legacy).
"""
from dataclasses import dataclass

import torch
from torch import nn

from exp.common.config import MODEL_SIZE_TO_ID
from exp.common.heads import Pa2PxHead
from exp.common.tasks import Task
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput, _to_device, crop_patches
from manyup.inject import MixedResOE, Packed
from manyup.mixres.plan import LOOKUP_PS, TILE, Plan

LR_PS = 16          # the head maps a token to a 16 x 16 px block (finer tokens: area-pooled)


@dataclass
class ModelConfig:
    model_size: str = "tiny"
    lr_patch: int = 16          # season token patch size (16 or 8)
    inject_layer: int = 0       # crops join the season tokens before this block


@dataclass
class Out:
    """One encode + decode.
    logits  (B, C, 64, 64)
    lr_F    (B, NS, h, w, D) final season tokens
    hr      (N, D) final crop tokens, in plan order (N = 0 for the uniform pass)"""
    logits: torch.Tensor
    lr_F: torch.Tensor
    hr: torch.Tensor


class MixedResModel(nn.Module):
    def __init__(self, task: Task, cfg: ModelConfig, pretrained: bool = True):
        super().__init__()
        from exp.upsamplers.finetune_inject import NativeSetReadout   # TODO: move here
        if cfg.lr_patch not in (8, 16):
            raise ValueError(f"lr_patch {cfg.lr_patch}: 16 or 8")
        self.task, self.cfg = task, cfg
        oe = OlmoEarthEncoder(MODEL_SIZE_TO_ID[cfg.model_size],
                              OlmoEarthInput(task.olmoearth.modalities, task.olmoearth.impute),
                              LR_PS, TILE, name=cfg.model_size, pretrained=pretrained)
        self.mix = MixedResOE(oe.encoder, cfg.inject_layer, "center", cfg.lr_patch)
        D = oe.encoder.embedding_size
        self.head = Pa2PxHead(task.reduction, D, task.num_outputs, LR_PS)
        self.readout = NativeSetReadout(D, base_px=cfg.lr_patch)

    @property
    def device(self):
        return self.head.linear.weight.device

    # -- tokenize ---------------------------------------------------------------------------
    def tokenize_seasons(self, ss) -> Packed:
        """The season composites at the season patch size; blocks [0, L) applied."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.mix.head_blocks(self.mix.embed(ss, self.cfg.lr_patch))

    def tokenize_crops(self, ms, plan: Plan, season, lr: Packed) -> Packed:
        """The plan's windows of the monthly images, one single-patch sequence each, placed at
        their pixel position and their month's season slot; blocks [0, L) applied."""
        crops = crop_patches(ms, plan.b, plan.t, plan.py, plan.px, plan.patch, pixels=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crops, plan.patch, slot=season[plan.b, plan.t],
                                py=plan.py, px=plan.px)
            return self.mix.head_blocks(hr, lr, plan.b)

    # -- encode / decode --------------------------------------------------------------------
    def encode(self, lr: Packed, crops: list[tuple[Packed, Plan]]):
        """Blocks [L, n) with every sample's season tokens, then its crops group by group, in
        one sequence. -> (season tokens (B, NS, h, w, D), crop tokens (N, D) in group order)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr_x, outs = self.mix.tail_multi(lr, [(hr, p.b) for hr, p in crops])
        lr_F = lr.pooled(lr_x).float()
        hr_F = [hr.pooled(x).float()[:, 0, 0, 0] for (hr, _), x in zip(crops, outs)]
        return lr_F, (torch.cat(hr_F) if hr_F else lr_F.new_zeros(0, lr_F.shape[-1]))

    def decode(self, lr_F, hr_F, plans: list[Plan], season, present, logw=None):
        """Every token through the head at its own patch size, pooled per pixel over the
        tokens covering it (NativeSetReadout). plans: the non-empty crop groups, hr_F's order.

        One group and several follow the legacy conventions of a plain plan and of --level2
        respectively (footprints as ps4 cell counts, pixel positions only when a group is off
        the ps4 grid); they are kept apart to stay bit-exact with the legacy checkpoints."""
        if not plans:
            z = torch.zeros(0, dtype=torch.long, device=lr_F.device)
            return self.readout(self.head.linear, lr_F, season, present, hr_F, (z,) * 4)
        if len(plans) == 1:
            p = plans[0]
            return self.readout(self.head.linear, lr_F, season, present, hr_F,
                                (p.b, p.t, p.y, p.x), logw, torch.ones_like(p.b),
                                hr_px=None if p.on_grid else (p.py, p.px), hr_fp=p.patch)
        if logw is not None:
            raise ValueError("crop gates with several crop groups: not ported")
        cat = lambda f: torch.cat([f(p) for p in plans])                         # noqa: E731
        idx = tuple(cat(lambda p, i=i: (p.b, p.t, p.y, p.x)[i]) for i in range(4))
        sizes = cat(lambda p: torch.full_like(p.b, p.cells))
        off = any(p.patch < LOOKUP_PS for p in plans)
        return self.readout(self.head.linear, lr_F, season, present, hr_F, idx, None, sizes,
                            hr_px=(cat(lambda p: p.py), cat(lambda p: p.px)) if off else None,
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
        self.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lr_F, hr_F = self.encode(self.tokenize_seasons(ss), [])
                logits = self.decode(lr_F, hr_F, [], season, present).float()
        finally:
            self.train(was)
        return Out(logits, lr_F, hr_F)

    def forward(self, ms, ss, season, present, plan: Plan | list[Plan], logw=None) -> Out:
        """plan: one crop group, or several (e.g. --level2: layer-1 crops, then finer ones)."""
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        plans = [p for p in ([plan] if isinstance(plan, Plan) else plan) if len(p)]
        lr = self.tokenize_seasons(ss)
        crops = [(self.tokenize_crops(ms, p, season, lr), p) for p in plans]
        lr_F, hr_F = self.encode(lr, crops)
        return Out(self.decode(lr_F, hr_F, plans, season, present, logw), lr_F, hr_F)


def load_legacy(model: MixedResModel, ckpt: str) -> list[str]:
    """Load checkpoints/<ckpt>.pt of a legacy InjectFT run; -> the legacy keys not used here
    (selector heads, residual scorers, ...)."""
    sd = torch.load(f"checkpoints/{ckpt}.pt", map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    missing = [k for k in own if k not in sd]
    if missing:
        raise KeyError(f"{ckpt}: missing {missing[:5]}")
    model.load_state_dict({k: sd[k] for k in own})
    return sorted(k for k in sd if k not in own)
