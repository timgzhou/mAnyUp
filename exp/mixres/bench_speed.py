"""Inference peak memory and throughput on this GPU (exp.bench.speed: real test batches, data
loading excluded), every arm through the SAME code (manyup.mixres), so the numbers compare the
models and not two implementations (the legacy finetune_inject forward is ~5x slower on the same
network: its own overhead, not the model's):

    MiSTR runs      the trained checkpoint, everything exp.mixres.train.step runs at test time
                    (the crop-free pass, the selector, the crops, the readout); one seed per
                    configuration (the cost depends on the seed only through the adaptive crop
                    count) -> results/<task>/mixres_speed.csv
    --fixed         the fixed-resolution models as MixedResModel's crop-free pass: seasonal
                    tokens (the season composites) or monthly tokens (the monthly images in
                    their place, every month its own time slot), every patch size of the
                    figure. Untrained heads: the cost does not depend on the weights. Their
                    traced MACs equal the legacy finetune_inject traces (inject_flops.csv) at
                    every patch size checked (seasonal 16 / 8 / 4 / 2, monthly 16 / 8 / 4)
                    -> results/<task>/fixed_speed.csv

    python -u -m exp.mixres.bench_speed --task pastis [--runs <checkpoint name> ...]
    python -u -m exp.mixres.bench_speed --task pastis --fixed
        [--mem_batch_size 64] [--batch_size 512]

Peak memory at --mem_batch_size (13 batches, 3 of them warm-up), the arm's model inputs as given
(seasonal models: seasons; monthly: months; MiSTR: both); throughput at --batch_size (3 batches
cycled, exp.bench.speed.TIMED of them timed), halved until it fits (speed_batch_size), from the raw
frames on the GPU: each arm's compositing is timed with it (exp.bench.composite, disk read not).

Without --runs: every cross-attention arm the datasets_tradeoff figure draws (entropy space +
attention time, one season per location, half-patch crops, no ps4 seasons), one-run and two-step.
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import re  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.bench.composite import Composer, check  # noqa: E402
from exp.bench.speed import append_row, peak_memory, take, throughput  # noqa: E402
from exp.common.tasks import TASKS  # noqa: E402
from exp.mixres.train import step  # noqa: E402
from manyup.inject import MixedResOE  # noqa: E402
from manyup.encoders.olmoearth import _to_device  # noqa: E402
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.decoder import NativeSetReadout  # noqa: E402
from manyup.mixres.model import ModelConfig  # noqa: E402
from manyup.mixres.selectors import build_selector  # noqa: E402
from manyup.mixres.steps import build_step_selector  # noqa: E402

ARM = re.compile(r"^mixres_(?P<task>\w+?)_ps(?P<ps>\d+)_quadg-skip-qs[\d.]+-qt0\.5-hhone-crop-score"
                 r"-xhr(?:-s1o\d+)?_n0_c(?P<c>\d+)(?:p(?P<p>\d+))?(?:_warm)?_seed\d+_")
FIXED = {"sea": (16, 14, 12, 10, 8, 6, 4, 2), "mon": (16, 14, 12, 10, 8, 6, 4)}


def plotted_runs(task: str) -> list[str]:
    """One finished checkpoint per configuration of the figure's MiSTR arms (lowest seed)."""
    d = pd.read_csv(f"results/{task}/mixres.csv", usecols=["run"]).drop_duplicates()
    keep = {}
    for run in sorted(d.run):
        m = ARM.match(run)
        if not m or m["task"] != task or int(m["ps"]) <= 4:
            continue
        if 2 * int(m["p"] or m["c"]) != int(m["ps"]):     # half-patch crops only
            continue
        cfg = re.sub(r"_seed\d+_\w+$", "", run)
        if cfg not in keep and Path(f"checkpoints/{run}.pt").exists():
            keep[cfg] = run
    return list(keep.values())


def load(run: str, nw: int):
    """The run's model and selector as tested (exp.mixres.compute_breakdown.load)."""
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    task = TASKS[cfg.task]
    model = MixedResModel(task, cfg.model).cuda().eval()
    load_legacy(model, run)
    b0 = next(iter(loader(task, "val", 2, nw, False)))
    build = build_step_selector if task.per_step else build_selector
    sel = build(cfg.select, model, (b0.ms, b0.ss))
    nets = nn.ModuleList(sel.modules())
    nets.load_state_dict(ck["selector"])
    nets.eval()
    return task, model, sel


def fixed_model(task, ps: int) -> MixedResModel:
    """The fixed-resolution model at patch ps. ps2 is outside MixedResModel's season patches:
    built at ps4, then the tokenizer and readout swapped for ps2's (as the legacy InjectFT
    builds them: 2 px readout cells)."""
    model = MixedResModel(task, ModelConfig(lr_patch=max(ps, 4))).cuda().eval()
    if ps == 2:
        model.cfg = replace(model.cfg, lr_patch=2)
        model.mix = MixedResOE(model.mix.encoder, 0, "center", 2, attn="joint").cuda()
        model.readout = NativeSetReadout(model.mix.encoder.embedding_size, base_px=2,
                                         crop_px=2).cuda()
    return model


def fixed_forward(model, fam: str):
    """One batch through the crop-free pass: season tokens, or every month its own slot. Only
    that arm's input reaches the GPU, once (uniform moves both of its samples; moved already,
    the same tensors go through)."""
    def forward(b):
        x = _to_device(b.ss if fam == "sea" else b.ms, model.device, torch.bfloat16)
        if fam == "sea":
            return model.uniform(x, x, b.season, b.present)
        B, T = b.present.shape
        slots = torch.arange(T, device=b.present.device).expand(B, T).contiguous()
        return model.uniform(x, x, slots, b.present)
    return forward


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--fam", nargs="*", default=list(FIXED), help="--fixed: families")
    ap.add_argument("--patches", nargs="*", type=int, default=None, help="--fixed: only these")
    ap.add_argument("--mem_batch_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=512, help="throughput (halved on OOM)")
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    task = TASKS[a.task]
    mem_batches = take(loader(task, "test", a.mem_batch_size, a.num_workers, False), 13)
    speed_batches, comp = {}, Composer(task)
    n_of = lambda item: item[0].target.shape[0]     # noqa: E731

    def measure(forward, composed) -> dict:
        """forward(batch) at peak memory (the composites as given); composed(batch, raw) at
        throughput (the arm composites what it needs from the raw frames, timed)."""
        r = {"mem_batch_size": a.mem_batch_size, **peak_memory(forward, mem_batches)}
        bs = a.batch_size
        while True:
            if bs not in speed_batches:
                speed_batches.clear()       # (one batch size's batches held at a time)
                # (no workers, no pinning: their prefetch and the CUDA host allocator's kept
                # blocks went over a 64 GB job limit when the batch size changed)
                dl = loader(task, "test", bs, 0, False)
                dl.pin_memory = False
                speed_batches[bs] = list(zip(take(dl, 3), comp.raw_batches("test", bs, 3)))
            try:
                return {**r, "speed_batch_size": bs,
                        **throughput(lambda it: composed(*it), speed_batches[bs], n_of)}
            except torch.OutOfMemoryError:
                if bs == 1:
                    raise
                torch.cuda.empty_cache()
                bs //= 2
                print(f"  out of memory: throughput at batch {bs}", flush=True)

    if a.fixed:
        for fam in a.fam:
            for ps in FIXED[fam] if a.patches is None else a.patches:
                model = fixed_model(task, ps)
                forward = fixed_forward(model, fam)

                def composed(b, raw, fam=fam, forward=forward):
                    months = comp.months(raw)
                    if fam == "sea":
                        return forward(replace(b, ss=comp.pack(b.ss, comp.seasons(months))))
                    return forward(replace(b, ms=comp.pack(b.ms, months)))

                r = {"family": fam, "patch": ps, **measure(forward, composed)}
                append_row(Path(f"results/{a.task}/fixed_speed.csv"), r)
                print(f"fixed {fam} ps{ps}: {r['peak_mem_gb']:.3f} GB peak | "
                      f"{r['samples_per_s']:.1f} samples/s (batch {r['speed_batch_size']})",
                      flush=True)
                del model
                torch.cuda.empty_cache()
        return
    runs = a.runs if a.runs is not None else plotted_runs(a.task)
    print(f"{len(runs)} runs", flush=True)
    for run in runs:
        _, model, sel = load(run, a.num_workers)
        crops = [0, 0]

        def forward(b):
            crops[0] += step(model, sel, b)[2]
            crops[1] += b.target.shape[0]

        def composed(b, raw):
            months = comp.months(raw)
            forward(replace(b, ms=comp.pack(b.ms, months),
                            ss=comp.pack(b.ss, comp.seasons(months))))

        r = {"run": run, **measure(forward, composed), "crops_per_sample": round(crops[0] / crops[1], 1)}
        append_row(Path(f"results/{a.task}/mixres_speed.csv"), r)
        print(f"{run}: {r['peak_mem_gb']:.3f} GB peak ({r['act_mem_gb']:.3f} activations) | "
              f"{r['samples_per_s']:.1f} samples/s (batch {r['speed_batch_size']}) | "
              f"{r['crops_per_sample']} crops/sample", flush=True)
        del model, sel
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
