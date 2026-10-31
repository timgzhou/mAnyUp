"""What the baselines share: months as the token sequence, no crops, RGB frames of a sample."""
import torch
from olmoearth_pretrain.data.constants import Modality

from manyup.encoders.olmoearth import _pad_space, _spatial_size
from manyup.mixres.model import MixedResModel
from manyup.mixres.plan import Plan
from manyup.mixres.selectors import Selection


class BaselineModel(MixedResModel):
    """MixedResModel on seasons, or (baseline.tokens months) with the monthly images in the
    season sample's place and every month its own slot, the fixed monthly models' input."""

    def __init__(self, task, cfg, pretrained: bool = True):
        super().__init__(task, cfg.model, pretrained)
        self.run_cfg = cfg
        self.months = cfg.baseline.tokens == "months"

    def _inputs(self, ms, ss, season, present):
        ms, ss, season, present = super()._inputs(ms, ss, season, present)
        if self.months:
            B, T = present.shape
            ss, season = ms, torch.arange(T, device=present.device).expand(B, T).contiguous()
        return ms, ss, season, present


class NoCrops:
    """The selector of a fixed-token baseline: nothing to pick."""
    scorer = None

    def modules(self) -> list:
        return []

    def __call__(self, ctx) -> Selection:
        empty = torch.zeros(ctx.B, ctx.T, 16, 16, dtype=torch.bool, device=ctx.season.device)
        return Selection(Plan.empty(4, ctx.season.device), empty)


def frame_side(sample, patch: int) -> int:
    """The padded square side a patch tiles (MixedResOE.embed): the tile if it divides it."""
    h = _spatial_size(sample)
    return -(-h // patch) * patch


def rgb_bands(task) -> tuple[str, list[int]]:
    """The OlmoEarth attribute and channels of the task's natural-colour bands, in R, G, B order
    (OlmoEarth keeps S2 in its own band order)."""
    if len(task.rgb) != 1:
        raise ValueError(f"{task.name}: one RGB modality expected, got {task.rgb}")
    mod, bands = next(iter(task.rgb.items()))
    name = task.olmoearth.modalities[mod]
    order = [b.upper() for b in Modality.get(name).band_order]
    return name, [order.index(b.upper()) for b in bands]


def rgb_frames(task, sample, patch: int) -> torch.Tensor:
    """(B, T, 3, H, W) float, the raw values of a MaskedOlmoEarthSample, on the frame `patch`
    tiles (replicate-padded like the tokens). The sample's linear normalization is kept: callers
    stretch them."""
    side = frame_side(sample, patch)
    if side != _spatial_size(sample):
        sample = _pad_space(sample, side)
    name, ch = rgb_bands(task)
    return getattr(sample, name)[..., ch].float().permute(0, 3, 4, 1, 2)
