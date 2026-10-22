"""The ported selectors vs the legacy InjectFT, IN ONE PROCESS: for each key the legacy
finetune_inject records its reference picks / logits (--regress_out, as
scripts/slurm/regress_record.sh does), then manyup.mixres replays them from the same checkpoint,
seed and test batches. Per batch the crop mask must be identical and the logits within --tol;
with --full also the whole test split's metrics.

Why one process: the legacy model is not reproducible ACROSS processes. Run to run (same node,
same data, same seed) its bfloat16 numerics land in one of two states (e.g. logits off by up to
0.08 with the same picks; the label oracle's near-tied windows flipping, 2743 vs 2740 crops);
within a process they are stable. cache/regress/ keeps the cross-process record.

    python -m exp.mixres.check_selectors                       # every key
    python -m exp.mixres.check_selectors --keys random_s8m4_n32 --full
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from manyup.mixres import MixedResModel, ModelConfig, load_legacy  # noqa: E402
from manyup.mixres.selectors import (Context, Entropy, Geometry, LabelLoss,  # noqa: E402
                                     RandomWindows, Residual, SeasonGrid, SeasonHalves,
                                     SeasonPairs, Selector, TopWindows)

SCORER16 = "inject_fft_pastis_tiny_none_native_set_lr0.001_ep50"
LOC16 = "inject_fft_pastis_tiny_oracle_L0_k0.0312_loc_native_set_cpx8_st8_poscenter_lr0.001_ep50"
S16 = ["--select_unit", "location", "--inject_layer", "0", "--readout", "native_set",
       "--crop_px", "8", "--pos", "center", "--k_frac", "0.03125"]
S8 = ["--select_unit", "location", "--inject_layer", "0", "--readout", "native_set",
      "--lr_patch", "8", "--crop_px", "4", "--pos", "center", "--k_frac", "0.125"]
P = ["--task", "pastis", "--epochs", "50"]
ENT = ["--score_src", "lr_entropy", "--scorer", SCORER16, "--season_rule", "all",
       "--init_ckpt", LOC16]
# key -> the legacy finetune_inject arguments (as scripts/slurm/regress_record.sh)
LEGACY = {
    "loc_oracle_s16m8_n8": P + ["--selector", "oracle"] + S16 + ["--loc_stride", "8"],
    "bothalf_s16m8_n8": P + ["--selector", "oracle_bothalf"] + S16 + ["--loc_stride", "8"],
    "bothalf_s8m4_n32": P + ["--selector", "oracle_bothalf"] + S8,
    "season_grid_s16m8_n8": P + ["--selector", "season_grid"] + S16,
    "random_s8m4_n32": P + ["--selector", "random"] + S8,
    "lrent_all_s16m8_n8": P + ["--selector", "oracle_bothalf"] + S16 + ["--loc_stride", "8"] + ENT,
    "lrent_rese2e_s16m8_n8": (P + ["--selector", "oracle_bothalf"] + S16 + ["--loc_stride", "8"]
                              + ENT + ["--res_mode", "e2e"]),
    "yieldsat_bothalf_s16m8_n8": (["--task", "yieldsat_raw", "--epochs", "50", "--selector",
                                   "oracle_bothalf"] + S16 + ["--loc_stride", "8"]),
}
# key -> (task, season patch, crop px, loc stride, scorer, allocator); scorer / allocator are
# built once the model exists (frozen sources and learned residuals load weights)
SPECS = {
    "loc_oracle_s16m8_n8":   ("pastis", 16, 8, 8, "label", ("top", 8)),
    "bothalf_s16m8_n8":      ("pastis", 16, 8, 8, "label", ("halves", 8, "better")),
    "bothalf_s8m4_n32":      ("pastis", 8, 4, None, "label", ("halves", 32, "better")),
    "season_grid_s16m8_n8":  ("pastis", 16, 8, None, ("season_grid", 8), ("pairs", 8)),
    "random_s8m4_n32":       ("pastis", 8, 4, None, "random", ("top", 32)),
    "lrent_all_s16m8_n8":    ("pastis", 16, 8, 8, ("frozen_entropy", SCORER16),
                              ("halves", 8, "all")),
    "lrent_rese2e_s16m8_n8": ("pastis", 16, 8, 8, ("residual_e2e", SCORER16),
                              ("halves", 8, "all")),
    "yieldsat_bothalf_s16m8_n8": ("yieldsat_raw", 16, 8, 8, "label", ("halves", 8, "better")),
}


def frozen(task, lp, ckpt):
    m = MixedResModel(task, ModelConfig(lr_patch=lp)).cuda().eval()
    load_legacy(m, ckpt)
    m.requires_grad_(False)
    return m


def build(key, run):
    task_name, lp, cp, stride, sc, al = SPECS[key]
    task = TASKS[task_name]
    model = MixedResModel(task, ModelConfig(lr_patch=lp)).cuda().eval()
    unused = load_legacy(model, run)
    if sc == "label":
        scorer = LabelLoss()
    elif sc == "random":
        scorer = RandomWindows()
    elif sc[0] == "season_grid":
        scorer = SeasonGrid(sc[1])
    elif sc[0] == "frozen_entropy":
        scorer = Entropy(frozen(task, lp, sc[1]))
    elif sc[0] == "residual_e2e":
        src = frozen(task, lp, sc[1])
        net = fi.ResidualScorer(src.mix.encoder.embedding_size,
                                src.head.linear.out_features // 16 ** 2).cuda().eval()
        sd = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)["model"]
        net.load_state_dict({k[len("res_sel."):]: v for k, v in sd.items()
                             if k.startswith("res_sel.")})
        unused = [k for k in unused if not k.startswith("res_sel.")]
        scorer = Residual(Entropy(src), net, train="e2e")
    allocator = {"top": lambda n: TopWindows(n), "pairs": lambda n: SeasonPairs(n),
                 "halves": lambda n, keep: SeasonHalves(n, keep)}[al[0]](*al[1:])
    return task, model, Selector(Geometry(cp, stride), scorer, allocator), unused


def legacy_reference(key, a):
    """Run the legacy finetune_inject --regress_out for key in this process -> its record."""
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    argv = sys.argv
    sys.argv = (["finetune_inject"] + LEGACY[key] + ["--num_workers", "8", "--regress_out", path,
                "--regress_batches", str(a.batches)]
                + ([] if a.full else ["--limit_batches", str(a.batches)]))
    try:
        fi.main()
    finally:
        sys.argv = argv
    ref = torch.load(path, weights_only=False)
    os.remove(path)
    return ref


@torch.no_grad()
def check(key, a):
    ref = legacy_reference(key, a)
    task, model, sel, unused = build(key, ref["run"])
    print(f"== {key}  ({ref['run']})\n   legacy keys unused: {unused}", flush=True)
    dl = fi.loader(task, "test", task.batch_size[16], 8, False)
    ok = True

    def run(i, x):
        ms, ss, season, present, _, extra = x
        s = sel(Context(model, ms, ss, season, present, extra["target"]))
        return s, model(ms, ss, season, present, s.plan, s.logw).logits

    torch.manual_seed(1234)
    for i, (*x, _) in enumerate(dl):
        if i >= len(ref["batches"]):
            break
        r = ref["batches"][i]
        s, logits = run(i, x)
        same = bool((s.mask.cpu() == r["pick"]).all())
        d = (logits.float().cpu() - r["logits"]).abs().max().item()
        ok &= same and d <= a.tol and len(s.plan) == r["n_crops"]
        print(f"   batch {i}: picks {'same' if same else 'DIFFERENT'} | logits max diff "
              f"{d:.2e} | crops {len(s.plan)} vs {r['n_crops']}", flush=True)
    if a.full:
        torch.manual_seed(1234)
        meter, crops, n = task.meter(), 0, 0
        for i, (*x, target) in enumerate(dl):
            s, logits = run(i, x)
            meter.update(logits, target.cuda())
            crops, n = crops + len(s.plan), n + target.shape[0]
        res = meter.compute()
        diff = max(abs(res[k] - ref["test"][k]) for k in ref["test"])
        ok &= diff <= a.metric_tol
        print(f"   full test: primary {res['primary']:.5f} vs {ref['test']['primary']:.5f} | "
              f"max metric diff {diff:.2e} | crops/sample {crops / n:.1f} vs "
              f"{ref['crops_per_sample']:.1f}", flush=True)
    print(f"   {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=list(SPECS))
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--metric_tol", type=float, default=0.0)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    results = {k: check(k, a) for k in a.keys}
    print("\n".join(f"{'PASS' if v else 'FAIL'}  {k}" for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
