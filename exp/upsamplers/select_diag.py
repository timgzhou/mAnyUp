"""Where does the learned (season, window) selector fall short of the better-half oracle: the
choice of WINDOWS (spatial ranking) or of SEASONS within them (temporal ranking)?

For one (task, setup, n) and two sets of weights -- the frozen oracle checkpoint with its linear
rank head, and arm A (the same, fine-tuned with the head's picks) -- on the test split:
  recall split   location recall: share of the oracle's windows the learned picks also crop;
                 season agreement: on windows both crop, share of the oracle's (season, window)
                 picks the learned picks match (the rest of a full pick recall)
  test metric    with picks from: label (the oracle), learned, mix_oloc (oracle windows +
                 learned seasons), mix_osea (learned windows + oracle seasons)
-> results/<task>/select_diag.csv, one row per (weights, picks).

    python -m exp.upsamplers.select_diag --task pastis --setup s16m8 --n 8
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.learned_select import EPOCHS, SETUPS, build, collect, picks, run_name  # noqa: E402


def arm_a_name(task, setup, n):
    pre, post = ((SETUPS[setup][3][:-3], "") if setup == "s16m8" else ("_poscenter", "_lrps8"))
    return (f"inject_fft_{task}_tiny_oracle_bothalf_L0_k{n / 256:.3g}_loc_native_set{pre}"
            f"_rklinear_warm{post}_lr0.001_ep{EPOCHS[task]}")


@torch.no_grad()
def recall_split(m, dl, limit):
    """-> (location recall, season agreement on shared windows, full pick recall)."""
    loc_hit = loc_tot = sea_hit = sea_tot = all_hit = all_tot = 0.0
    for i, (*x, _) in enumerate(dl):
        if limit and i >= limit:
            break
        rb = collect(m, x)
        ref = picks(m, rb, rb["sc"], rb["loc"])                              # (B, T, H, W)
        sc, loc = m.rank_scores(rb["lr_F"], rb["season"], rb["present"], rb.get("e"))
        got = picks(m, rb, sc, loc)
        rw, gw = ref.any(1), got.any(1)                                      # windows (B, H, W)
        loc_hit += float((rw & gw).sum())
        loc_tot += float(rw.sum())
        both = (rw & gw)[:, None]
        sea_hit += float((ref & got & both).sum())
        sea_tot += float((ref & both).sum())
        all_hit += float((ref & got).sum())
        all_tot += float(ref.sum())
    return loc_hit / max(loc_tot, 1), sea_hit / max(sea_tot, 1), all_hit / max(all_tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPOCHS))
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--recall_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test: no CSV")
    a = ap.parse_args()
    task = TASKS[a.task]
    name = run_name(a.task, a.setup, a.n)
    dl = fi.loader(task, "test", a.batch_size, a.num_workers, False)
    out = Path(f"results/{a.task}/select_diag.csv")
    for weights in ("frozen", "armA"):
        m = build(a.task, a.setup, a.n)
        if weights == "frozen":
            m.rank_head.load_state_dict(torch.load(f"checkpoints/{name}_rankhead.pt",
                                                   map_location="cpu"))
        else:
            ck = arm_a_name(a.task, a.setup, a.n)
            sd = torch.load(f"checkpoints/{ck}.pt", map_location="cpu", weights_only=False)["model"]
            m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")})
        m.requires_grad_(False)
        m.eval()
        lr_, sa, fr = recall_split(m, dl, a.limit_batches or a.recall_batches)
        print(f"{weights}: location recall {lr_:.3f} | season agreement {sa:.3f} | "
              f"pick recall {fr:.3f}", flush=True)
        for src in ("label", "learned", "mix_oloc", "mix_osea"):
            m.score_src = src
            torch.manual_seed(0)
            res, crops = fi.evaluate(m, task, dl, a.limit_batches)
            print(f"TEST {weights} {src}: primary {res['primary']:.4f} | {crops:.0f} crops/sample",
                  flush=True)
            if a.limit_batches:
                continue
            _append(out, {"job": os.environ.get("SLURM_JOB_ID", ""), "run": name,
                          "setup": a.setup, "n": a.n, "weights": weights, "picks": src,
                          "location_recall": round(lr_, 4), "season_agreement": round(sa, 4),
                          "pick_recall": round(fr, 4), "crops_per_sample": round(crops, 1),
                          **{f"test_{k}": round(v, 5) for k, v in res.items()}})


if __name__ == "__main__":
    main()
