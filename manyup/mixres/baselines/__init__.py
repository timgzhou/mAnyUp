"""Published methods compared with MiSTR, each behind RunConfig.baseline and trained, evaluated
and traced by exp.mixres.train like any mixed-resolution run (docs/baselines.md):

    upsample   a fixed model's final tokens through AnyUp / Upsample Anything / bilinear
    tldr       vid-TLDR token merging inside a trained fixed model, training-free
    apt        APT adaptive patches
"""
from manyup.mixres.baselines.apt import AptModel, AptSelector
from manyup.mixres.baselines.base import NoCrops
from manyup.mixres.baselines.upsample import UpsampleModel
from manyup.mixres.baselines.vid_tldr import TldrModel
from manyup.mixres.model import MixedResModel
from manyup.mixres.selectors import build_selector
from manyup.mixres.steps import build_step_selector

MODELS = {"upsample": UpsampleModel, "tldr": TldrModel, "apt": AptModel}


def build_model(task, cfg, pretrained: bool = True) -> MixedResModel:
    """cfg: RunConfig."""
    if cfg.baseline is None:
        return MixedResModel(task, cfg.model, pretrained)
    if task.per_step:
        raise ValueError("baselines: one map per time series (not per-step tasks)")
    return MODELS[cfg.baseline.method](task, cfg, pretrained)


def build_selector_for(task, cfg, model, example):
    if cfg.baseline is not None:
        return AptSelector(task, cfg, model) if cfg.baseline.method == "apt" else NoCrops()
    return (build_step_selector if task.per_step else build_selector)(cfg.select, model, example)
