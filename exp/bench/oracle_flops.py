"""Inference GMACs per sample of the location-constrained arms (rounds 49 - 56), measured the
way `finetune_inject --flops` does (exp.bench.flop_trace over a few val batches, untrained
weights: cost does not depend on them) but with ONE model build per (task, setup): the selector,
budget and candidate stride are switched on the same model. One row per arm ->
results/<task>/oracle_flops.csv (setup, selector, stride, n, gmacs, crops_per_sample).

    python -m exp.bench.oracle_flops [--tasks pastis,biomassters,yieldsat_raw] [--batches 4]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.bench.flop_trace import append_row, count_macs  # noqa: E402
from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402

SETUPS = {   # setup -> (lr_patch, crop_px, budgets n, strides, selectors)
    "s16m8": (16, 8, (8, 16, 32), (4, 6, 8),
              ("oracle", "oracle_season", "oracle_tophalf", "oracle_bothalf", "grid", "random",
               "season_random", "season_grid")),
    "s8m4": (8, 4, (32, 64, 128), (4, 2),
             ("oracle", "oracle_season", "oracle_tophalf", "oracle_bothalf", "grid", "random",
              "season_random", "season_grid")),
}
STRIDED = ("oracle", "oracle_season", "oracle_tophalf", "oracle_bothalf")
# --level2 (round 59): better-half layer 1 at stride 8 + a second layer; selector column =
# "l2<arm>[r]" (r: random picks), stride 8, n = n1, n2 in its own column
L2_ARMS = {   # setup -> task -> arms (the first d* arm, or BioMassters' m*, gets a random control)
    "s16m8": {"pastis": ("m4", "d8", "d4"), "yieldsat_raw": ("m4", "d8", "d4"),
              "biomassters": ("m4",)},
    "s8m4": {"pastis": ("m2", "d2"), "yieldsat_raw": ("m2", "d2"), "biomassters": ("m2",)},
}
L2_SETUP = {"s16m8": (16, 8, 8, (8, 16, 32)), "s8m4": (8, 4, 4, (32, 64, 128))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", default="pastis,biomassters,yieldsat_raw")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--setups", default=",".join(SETUPS))
    ap.add_argument("--strides", default="", help="only these candidate strides (e.g. 2)")
    ap.add_argument("--level2", action="store_true", help="only the round-59 level-2 arms")
    a = ap.parse_args()
    for name in a.tasks.split(","):
        task = TASKS[name]
        batches = []
        for i, (*x, target) in enumerate(fi.loader(task, "val", 32, 4, False)):
            if i >= a.batches:
                break
            batches.append((x, target.shape[0]))
        out = Path(f"results/{name}/oracle_flops.csv")
        if a.level2:
            for setup in a.setups.split(","):
                level2(name, task, a.batches, out, setup)
            continue
        only = {int(v) for v in a.strides.split(",") if v}
        for setup, (lp, cp, ns, strides, sels) in SETUPS.items():
            if setup not in a.setups.split(","):
                continue
            m = fi.InjectFT(task, "tiny", 0, "oracle", 1 / 256, "location", readout="native_set",
                            lr_patch=lp, crop_px=cp, pos="center").cuda().eval()
            n_blocks = len(m.backbone().blocks)
            for sel in sels:
                for stride in (strides if sel in STRIDED else (4,)):
                    if only and stride not in only:
                        continue
                    m.loc_px, m.loc_fine = None, False
                    if stride != 4:
                        m.set_loc_stride(stride)
                    for n in ns:
                        m.sel_name, m.k_frac = sel, n / 256
                        torch.manual_seed(0)
                        crops = samples = 0
                        with count_macs() as count:
                            for x, bs in batches:
                                crops += m(*x)[2]
                                samples += bs
                        row = {"setup": setup, "selector": sel, "stride": stride, "n": n,
                               "gmacs": round(count.total / samples / 1e9, 3),
                               "crops_per_sample": round(crops / samples, 1),
                               "samples_traced": samples}
                        append_row(out, row)
                        print(name, row, flush=True)


def level2(name, task, n_batches, out, setup):
    """GMACs of the rounds-59 / 60 arms: one model, layer 1 = the better-half oracle."""
    lp, cp, st, ns = L2_SETUP[setup]
    batches = []
    for i, (*x, target) in enumerate(fi.loader(task, "val", 32, 4, False, dates=True)):
        if i >= n_batches:
            break
        batches.append((x, target.shape[0]))
    m = fi.InjectFT(task, "tiny", 0, "oracle_bothalf", 1 / 256, "location",
                    readout="native_set", lr_patch=lp, crop_px=cp, pos="center").cuda().eval()
    if st != 4:
        m.set_loc_stride(st)
    arms = L2_ARMS[setup][name]
    ctrl = next((x for x in arms if x[0] == "d"), arms[0])
    rows = [("l2none", None, "oracle", (0,))]
    for arm in arms:
        rows.append((f"l2{arm}", arm, "oracle", None))
        if arm == ctrl:
            rows.append((f"l2{arm}r", arm, "random", None))
    for sel, arm, pick, n2s in rows:
        for n in ns:
            for n2 in n2s or (n // 2, n, 2 * n):
                m.k_frac = n / 256
                if arm is None:
                    m.l2 = None
                else:
                    m.set_level2(arm, n2, pick)
                torch.manual_seed(0)
                crops = samples = 0
                with count_macs() as count:
                    for x, bs in batches:
                        crops += m(*x)[2]
                        samples += bs
                        torch.cuda.empty_cache()
                row = {"setup": setup, "selector": sel, "stride": st, "n": n, "n2": n2,
                       "gmacs": round(count.total / samples / 1e9, 3),
                       "crops_per_sample": round(crops / samples, 1), "samples_traced": samples}
                append_row(out.with_name("level2_flops.csv"), row)
                print(name, row, flush=True)


if __name__ == "__main__":
    main()
