"""GMACs/sample of a ported arm (exp.mixres.train.measure_compute) vs the legacy --flops row of the
same checkpoint in results/<task>/inject_flops.csv.

    python -m exp.mixres.check_compute --key loc_oracle_s16m8_n8
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402

from exp.mixres.check_selectors import build  # noqa: E402
from exp.mixres.train import measure_compute  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402

RUNS = {"loc_oracle_s16m8_n8":
        "inject_fft_pastis_tiny_oracle_L0_k0.0312_loc_native_set_cpx8_st8_poscenter_lr0.001_ep50"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="loc_oracle_s16m8_n8", choices=list(RUNS))
    a = ap.parse_args()
    run = RUNS[a.key]
    task, model, sel, _ = build(a.key, run)
    with open(f"results/{task.name}/inject_flops.csv") as f:
        ref = [r for r in csv.DictReader(f) if r["run"] == run][-1]
    dl = loader(task, "val", task.batch_size[16], 8, False)
    new = measure_compute(model, sel, dl, round(int(ref["samples_traced"]) / task.batch_size[16]))
    for k in ("gmacs", "gmacs_attention", "tokens_per_block", "longest_sequence"):
        print(f"{k:18s} legacy {float(ref[k]):9.3f}  new {float(new[k]):9.3f}  "
              f"({100 * (float(new[k]) / float(ref[k]) - 1):+.2f}%)")


if __name__ == "__main__":
    main()
