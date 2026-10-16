"""Test-time pick ceiling for a model trained with whole-cell (location-unit) crops: evaluate the
SAME weights with different test-time cell picks -- oracle (the cells with the highest crop-free
loss against the true label), entropy (most uncertain cells; classification only), random, and
no crops. Appends one row per pick mode to results/<task>/inject_picks_ceiling.csv.

    python -m exp.upsamplers.eval_inject_picks --task pastis --run <run name> --readout native_attn \
        --k_frac 0.1667 [--split test]
"""
import os

from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims)

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402


def main():
    olmo_bootstrap.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", required=True, help="run name; weights at checkpoints/<run>.pt")
    ap.add_argument("--readout", required=True, choices=fi.READOUTS)
    ap.add_argument("--k_frac", type=float, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV row")
    a = ap.parse_args()
    task = TASKS[a.task]
    st = torch.load(f"checkpoints/{a.run}.pt", map_location="cpu", weights_only=False)
    # built as an oracle location-unit model: the weights are the same for every selector
    m = fi.InjectFT(task, "tiny", 0, "oracle", a.k_frac, "location", readout=a.readout).cuda()
    m.load_state_dict(st["model"])
    m.eval()
    dl = fi.loader(task, a.split, a.batch_size, 8, False)
    modes = (("oracle", "entropy", "random", "none") if task.kind == "classification"
             else ("oracle", "random", "none"))
    for mode in modes:
        torch.manual_seed(0)
        m.sel_name = "oracle" if mode == "none" else mode
        m.k_frac = 0.0 if mode == "none" else a.k_frac
        meter, crops, n = task.meter(), 0.0, 0
        with torch.no_grad():
            for i, (*x, tgt) in enumerate(dl):
                if a.limit_batches and i >= a.limit_batches:
                    break
                logits, _, nc = m(*x)
                meter.update(logits, tgt.cuda())
                crops += float(nc)
                n += tgt.shape[0]
        res = meter.compute()
        print(mode, round(res["primary"], 4), "crops/sample", round(crops / n, 1),
              flush=True)
        if a.limit_batches:
            continue
        _append(Path(f"results/{a.task}/inject_picks_ceiling.csv"), {
            "job": os.environ.get("SLURM_JOB_ID", ""), "run": a.run, "readout": a.readout,
            "k_frac": a.k_frac, "split": a.split, "test_pick": mode,
            "crops_per_sample": round(crops / n, 1),
            **{f"{a.split}_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
