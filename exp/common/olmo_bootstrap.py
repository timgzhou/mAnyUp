"""Make olmoearth_pretrain.evals importable on this cluster. Call apply() FIRST:

    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()

We use a few pieces of OlmoEarth's eval package (PASTIS dataset/processor, collate,
BackboneWithHead, get_eval_wrapper), but its package __init__s eagerly import every
competitor model wrapper (Clay, Satlas, Galileo, ...) and every eval dataset. Two of those
cannot be satisfied here:

  - evals.models: the competitor models' own packages (claymodel, satlaspretrain_models,
    ...). get_eval_wrapper only isinstance()-checks against these classes, and our encoder
    matches the FlexiVitBase branch first, so dummy classes are enough.
  - evals.datasets.geobench_dataset: needs `geobench`, which has no installable release
    compatible with this stack.

Both are replaced with permissive modules that hand out a fresh dummy class for ANY name,
so a newer olmoearth-pretrain that adds competitors or renames them still imports.

Also restores torch's "file_system" sharing strategy, which pastis_dataset set at import
in olmoearth-pretrain 0.1.0 and our DataLoader workers rely on (the default strategy exhausts
/dev/shm on the cluster).
"""
import sys
import types

import torch.multiprocessing

_STUBBED = ("olmoearth_pretrain.evals.models",
            "olmoearth_pretrain.evals.datasets.geobench_dataset")


class _Permissive(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls


def apply() -> None:
    """Install the shims. Idempotent; call before importing olmoearth_pretrain.evals."""
    for name in _STUBBED:
        sys.modules.setdefault(name, _Permissive(name))
    torch.multiprocessing.set_sharing_strategy("file_system")
