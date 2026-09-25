"""The feature upsamplers compared on each task, built the same way for LP and training.

Time: a task whose head averages over time (PASTIS, BioMassters) gets time-pooled features
and guidance, the setting the image upsamplers were designed for; a change-detection task
(GEOID, head post - pre) upsamples every timestep with its OWN date's image, so the
difference compares like with like. timAnyUp always works per timestep, and so does
manyup_t: mAnyUp trained per timestep on every task (timAnyUp without the high-res lookup).
"""
from __future__ import annotations

from pathlib import Path

from exp.common.paths import FEATURES
from exp.common.tasks import Task
from manyup.upsamplers import JBU, AnyUp, Bicubic, Bilinear, FeatureUpsampler, MAnyUp, TimAnyUp
from manyup.upsamplers.jbu import KernelCache

TRAINING_FREE = ("bilinear", "bicubic", "anyup", "upa", "upma")
TRAINED = ("manyup", "manyup_t", "timanyup")
METHODS = TRAINING_FREE + TRAINED
SHARED_PROBE = ("upa", "upma")      # per-image optimization: fitted kernels are cached
                                    # (exp/upsamplers/fit_jbu.py); without them, test only


def kernel_dir(task: Task, method: str) -> Path:
    return FEATURES / "jbu_kernels" / f"{task.name}_{method}"


def pooled(task: Task) -> bool:
    return task.reduction == "time_mean"


def build(name: str, task: Task, ckpt: str | None = None, fit_steps: int = 50,
          k: int | None = None) -> FeatureUpsampler:
    """k: timAnyUp's lookup budget (default: the trained one; 0 = no lookups)."""
    tp = "mean" if pooled(task) else None
    if name in TRAINED and not ckpt:
        raise ValueError(f"{name} needs a checkpoint (exp.upsamplers.train)")
    if name == "bilinear":
        return Bilinear(pooled=pooled(task))
    if name == "bicubic":
        return Bicubic(pooled=pooled(task))
    if name == "anyup":
        return AnyUp(tp, rgb=task.rgb)
    if name in ("upa", "upma"):
        d = kernel_dir(task, name)
        kernels = KernelCache(d) if KernelCache.exists(d) else None
        if name == "upa":
            return JBU.upa(tp, rgb=task.rgb, fit_steps=fit_steps, kernels=kernels)
        return JBU.upma(tp, bands=task.guidance, fit_steps=fit_steps, kernels=kernels)
    if name in ("manyup", "manyup_t"):         # the checkpoint fixes pooled / per timestep
        return MAnyUp.from_checkpoint(ckpt)
    if name == "timanyup":
        return TimAnyUp.from_checkpoint(ckpt, k=k)
    raise ValueError(f"unknown upsampler {name!r}; one of {METHODS}")


def roles(name: str) -> tuple[str, ...]:
    """Feature caches an upsampler reads (exp/upsamplers/task_data.py)."""
    return ("lr", "hr") if name == "timanyup" else ("lr",)
