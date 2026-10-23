"""exp.mixres.train vs the legacy finetune_inject training, IN ONE PROCESS (see check_selectors
for why): the same arm trained for --epochs epochs of --batches batches each by both, then the
val / test metrics of their smoke runs compared (exact unless --tol).

    python -m exp.mixres.check_train --keys bothalf_cold_s16m8 sent_all_s16m8
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from exp.mixres import train as new  # noqa: E402
from exp.mixres.check_selectors import LOC16, S16  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402

P = ["--task", "pastis"]
CASES = {   # key -> (legacy finetune_inject args, exp.mixres.train overrides)
    "bothalf_cold_s16m8": (
        P + ["--selector", "oracle_bothalf"] + S16 + ["--loc_stride", "8"],
        ["select.scorer=label", "select.allocator=halves", "select.keep=better", "select.n=8"]),
    "sent_all_s16m8": (
        P + ["--selector", "oracle_bothalf"] + S16 + ["--loc_stride", "8", "--score_src", "entropy",
                                                       "--season_rule", "all", "--init_ckpt", LOC16],
        ["select.scorer=entropy", "select.allocator=halves", "select.keep=all", "select.n=8",
         f"train.init_ckpt={LOC16}"]),
    "sent_rese2e_s16m8": (
        P + ["--selector", "oracle_bothalf"] + S16 + ["--loc_stride", "8", "--score_src", "entropy",
                                                       "--season_rule", "all", "--init_ckpt", LOC16,
                                                       "--res_mode", "e2e"],
        ["select.scorer=residual", "select.residual.train=e2e", "select.allocator=halves",
         "select.keep=all", "select.n=8", f"train.init_ckpt={LOC16}"]),
}


def last_row(path):
    with open(path) as f:
        return list(csv.DictReader(f))[-1]


def check(key, a):
    old_args, new_over = CASES[key]
    common = ["--epochs", str(a.epochs), "--limit_batches", str(a.batches), "--num_workers", "8"]
    argv = sys.argv
    sys.argv = ["finetune_inject"] + old_args + common
    try:
        fi.main()
    finally:
        sys.argv = argv
    old = last_row("results/pastis/inject_fft_smoke.csv")
    res = new.main(new_over + [f"train.epochs={a.epochs}", f"train.limit_batches={a.batches}"])
    rows = {"val": (float(old["val_primary"]), res["val"]["primary"]),
            "test": (float(old["test_primary"]), res["test"]["primary"]),
            "crops": (float(old["test_crops_per_sample"]), res["crops"])}
    ok = all(abs(o - n) <= a.tol for o, n in rows.values())
    print(f"== {key}: " + " | ".join(f"{k} legacy {o:.5f} new {n:.5f}" for k, (o, n) in rows.items())
          + f"  {'PASS' if ok else 'FAIL'}", flush=True)
    for f in glob.glob("checkpoints/*_smoke_state.pt"):     # this check's resumable states
        if os.path.getmtime(f) > a.t0:
            os.remove(f)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=list(CASES))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    import time
    a.t0 = time.time()
    res = {k: check(k, a) for k in a.keys}
    print("\n".join(f"{'PASS' if v else 'FAIL'}  {k}" for k, v in res.items()))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
