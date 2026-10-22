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
    def encode(self, lr: Packed, hr: Packed | None, plan: Plan | None):
        """Blocks [L, n) with every sample's season tokens and its crops in one sequence."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr_x, hr_x = self.mix.tail(lr, hr, plan.b if hr is not None else None)
        lr_F = lr.pooled(lr_x).float()
        hr_F = (hr.pooled(hr_x).float()[:, 0, 0, 0] if hr is not None
                else lr_F.new_zeros(0, lr_F.shape[-1]))
        return lr_F, hr_F

    def decode(self, lr_F, hr_F, plan: Plan | None, season, present, logw=None):
        """Every token through the head at its own patch size, pooled per pixel over the
        tokens covering it (NativeSetReadout)."""
        if plan is None or not len(plan):
            z = torch.zeros(0, dtype=torch.long, device=lr_F.device)
            return self.readout(self.head.linear, lr_F, season, present, hr_F, (z,) * 4)
        return self.readout(self.head.linear, lr_F, season, present, hr_F,
                            (plan.b, plan.t, plan.y, plan.x), logw, torch.ones_like(plan.b),
                            hr_px=None if plan.on_grid else (plan.py, plan.px),
                            hr_fp=plan.patch)

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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lr_F, hr_F = self.encode(self.tokenize_seasons(ss), None, None)
                logits = self.decode(lr_F, hr_F, None, season, present).float()
        finally:
            self.train(was)
        return Out(logits, lr_F, hr_F)

    def forward(self, ms, ss, season, present, plan: Plan, logw=None) -> Out:
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        lr = self.tokenize_seasons(ss)
        hr = self.tokenize_crops(ms, plan, season, lr) if len(plan) else None
        lr_F, hr_F = self.encode(lr, hr, plan if hr is not None else None)
        return Out(self.decode(lr_F, hr_F, plan, season, present, logw), lr_F, hr_F)


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
