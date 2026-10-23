"""Training mechanics of exp.mixres.train vs the legacy finetune_inject, on ONE optimizer step
from the same warm start. Training is not bit-reproducible in either (the same run twice in one
process ends at val .396 vs .422 after 6 steps: the backward pass is nondeterministic), so the
check is on the update each parameter receives: Δθ_new vs Δθ_legacy per tensor. A wrong
learning-rate group, freeze or loss term shows up as a norm ratio far from 1 or a low cosine;
backward noise as small deviations.

    python -m exp.mixres.check_step
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import glob  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from exp.mixres import train as new  # noqa: E402
from exp.mixres.check_train import CASES  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402


_save = torch.save
CAPTURED = []


def _capture(obj, *a, **k):
    """The per-epoch training state (it carries the optimizer): keep a CPU copy of its weights;
    the file itself is overwritten by the run's "done" marker at the end."""
    if isinstance(obj, dict) and "opt" in obj:
        cp = lambda d: {n: (t.detach().cpu().clone() if torch.is_tensor(t) else cp(t))  # noqa: E731
                        for n, t in d.items()}
        CAPTURED.append(cp(obj["model"]))
    return _save(obj, *a, **k)


torch.save = _capture


def cleanup(t0):
    for f in glob.glob("checkpoints/*_smoke_state.pt"):
        if os.path.getmtime(f) > t0:
            os.remove(f)


def run_legacy(old_args):
    argv = sys.argv
    sys.argv = ["fi"] + old_args + ["--epochs", "1", "--limit_batches", "1", "--num_workers", "8"]
    try:
        fi.main()
    finally:
        sys.argv = argv
    return CAPTURED.pop()


def stats(th0, a, b):
    """Per moved tensor: cosine(Δa, Δb) and |Δa| / |Δb| -> (cosines, ratios)."""
    cos, ratio = [], []
    for k, v in a.items():
        if k not in th0 or k not in b or not v.is_floating_point():
            continue
        da, db = (v - th0[k]).float().flatten(), (b[k] - th0[k]).float().flatten()
        if da.norm() == 0 and db.norm() == 0:
            continue
        cos.append(float(torch.nn.functional.cosine_similarity(da, db, dim=0)))
        ratio.append(float(da.norm() / db.norm().clamp(min=1e-30)))
    return torch.tensor(cos), torch.tensor(ratio)


def summary(c, r):
    return (f"{len(c)} tensors | cosine median {c.median():.4f} p5 {c.quantile(0.05):.4f} "
            f"min {c.min():.4f} | norm ratio median {r.median():.4f} "
            f"range [{r.min():.3f}, {r.max():.3f}]")


def check(key):
    old_args, new_over = CASES[key]
    if "--init_ckpt" not in old_args:
        raise SystemExit(f"{key}: needs a warm start (a cold start's random head init differs)")
    init = old_args[old_args.index("--init_ckpt") + 1]
    th0 = torch.load(f"checkpoints/{init}.pt", map_location="cpu", weights_only=False)["model"]
    t0 = time.time()
    leg1 = run_legacy(old_args)
    cleanup(t0)
    leg2 = run_legacy(old_args)
    cleanup(t0)
    new.main(new_over + ["train.epochs=1", "train.limit_batches=1"])
    cleanup(t0)
    th_new = CAPTURED.pop()["model"]
    c0, r0 = stats(th0, leg2, leg1)
    c1, r1 = stats(th0, th_new, leg1)
    print(f"== {key}\n   legacy vs legacy: {summary(c0, r0)}\n   new vs legacy:    "
          f"{summary(c1, r1)}", flush=True)
    # same mechanics: the new run's agreement with legacy is within the legacy run-to-run noise
    ok = (len(c1) == len(c0) and c1.median() >= c0.median() - 0.01
          and c1.quantile(0.05) >= c0.quantile(0.05) - 0.05
          and abs(float(r1.median()) - 1) <= abs(float(r0.median()) - 1) + 0.01)
    print(f"   {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    keys = sys.argv[1:] or ["sent_all_s16m8", "sent_rese2e_s16m8"]
    res = {k: check(k) for k in keys}
    print("\n".join(f"{'PASS' if v else 'FAIL'}  {k}" for k, v in res.items()))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
