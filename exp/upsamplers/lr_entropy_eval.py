"""Label-free picks from the LR-only model's prediction entropy (--score_src lr_entropy), tested
on the FROZEN better-half oracle checkpoint (no training): which windows get crops = the highest
entropy of the LR-only model's pooled prediction; which seasons = each window's lowest-entropy
half of seasons (the LR-only model's own per-season maps).
    picks   label        the oracle itself
            lr_entropy   both rankings from the LR-only model's entropy
            lrent_osea   LR-entropy windows + oracle seasons   (what the window choice costs)
            lrent_oloc   oracle windows + LR-entropy seasons   (what the season choice costs)
Also the overlap of the lr_entropy picks with the oracle's on --recall_batches test batches:
location recall (oracle windows also picked) and season agreement (on shared windows).
-> results/<task>/lr_entropy_eval.csv (rows already there are skipped).

    python -m exp.upsamplers.lr_entropy_eval --task pastis --setup s16m8 --n 8
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.learned_select import EPOCHS, SETUPS, build, run_name  # noqa: E402

SCORER = {("pastis", "s16m8"): "inject_fft_pastis_tiny_none_native_set_lr0.001_ep50",
          ("pastis", "s8m4"): "inject_fft_pastis_tiny_none_native_set_lrps8_lr0.001_ep50"}
PICKS = ("label", "lr_entropy", "lrent_osea", "lrent_oloc")


@torch.no_grad()
def overlap(m, dl, batches):
    """(location recall, season agreement) of the lr_entropy picks vs the oracle's."""
    lh = lt = sh = st = 0.0
    for i, (*x, _) in enumerate(dl):
        if i >= batches:
            break
        masks = {}
        for src in ("label", "lr_entropy"):
            m.score_src, m.last_pick = src, None
            m(*x)
            masks[src] = m.last_pick[0]                                     # (B, T, H, W)
        ref, got = masks["label"], masks["lr_entropy"]
        rw, gw = ref.any(1), got.any(1)
        both = (rw & gw)[:, None]
        lh += float((rw & gw).sum()); lt += float(rw.sum())
        sh += float((ref & got & both).sum()); st += float((ref & both).sum())
    return lh / max(lt, 1), sh / max(st, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPOCHS))
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--recall_batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV")
    a = ap.parse_args()
    task = TASKS[a.task]
    name = run_name(a.task, a.setup, a.n)
    m = build(a.task, a.setup, a.n)
    m.rank_head = None
    m.attach_scorer(SCORER[(a.task, a.setup)])
    dl = fi.loader(task, "test", a.batch_size, a.num_workers, False)
    out = Path(f"results/{a.task}/lr_entropy_eval.csv")
    done = set()
    if out.exists() and not a.limit_batches:
        with open(out) as f:
            done = {(r["setup"], r["n"], r["picks"]) for r in csv.DictReader(f)}
    loc_r, sea_a = overlap(m, dl, a.limit_batches or a.recall_batches)
    print(f"lr_entropy vs oracle picks: location recall {loc_r:.3f} | season agreement {sea_a:.3f}",
          flush=True)
    for src in PICKS:
        if (a.setup, str(a.n), src) in done:
            continue
        m.score_src = src
        torch.manual_seed(0)
        res, crops = fi.evaluate(m, task, dl, a.limit_batches)
        print(f"TEST {src}: primary {res['primary']:.4f} | {crops:.0f} crops/sample", flush=True)
        if a.limit_batches:
            continue
        _append(out, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "setup": a.setup,
                      "n": a.n, "picks": src, "scorer": SCORER[(a.task, a.setup)],
                      "location_recall": round(loc_r, 4), "season_agreement": round(sea_a, 4),
                      "crops_per_sample": round(crops, 1),
                      **{f"test_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
