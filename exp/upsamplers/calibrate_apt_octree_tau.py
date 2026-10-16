"""Calibrate --apt_tau so --apt_octree's average token count lands near the SAME budget the
random/grid/apt baselines use at a given --k_frac (1/6, 1/3) -- the octree has no budget
concept of its own (InjectFT._apt_octree_mask is pure entropy-threshold, token count is
whatever the recursion produces), so comparing it against k_frac-budgeted arms needs a tau
that happens to produce a comparable average count (user-confirmed 2026-10-09/12, see
[[apt-spacetime-octree]]).

Binary-searches tau per (task, k_frac) by measuring the REAL InjectFT._apt_octree_mask's
average leaf count (= token count, one token per leaf) over --n_samples real training-split
chips -- not a synthetic estimate, since real content drives the recursion. Writes results to
results/apt_octree_tau.json (task -> k_frac -> {tau, mean_tokens, target_tokens}), which
scripts/slurm/inject.sh's --apt_octree launcher reads to fill in --apt_tau per run.

    python -u -m exp.upsamplers.calibrate_apt_octree_tau --tasks pastis,biomassters,geoidflood,yieldsat_raw
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers.finetune_inject import OUT_GRID, loader  # noqa: E402
from exp.upsamplers.viz_apt_picks import apt_stub  # noqa: E402

OUT_PATH = Path("results/apt_octree_tau.json")


def mean_tokens(obj, batches, tau: float, space_only: bool = False) -> float:
    """Average total leaf count (= token count) of --apt_octree at this tau, over the given
    pre-loaded (ms, present) batches -- ps_min=4, matching the training wiring (see
    [[apt-spacetime-octree]]: ps2 has no footprint representation in NativeSetReadout).
    space_only: calibrate the 2026-10-13 isolating variant (--apt_octree_space_only) instead --
    needs its OWN tau since it structurally uses MORE tokens than the time-capable recursion
    at the same tau (confirmed on PASTIS: 720 vs 1012 average at tau=10.8693), as every month
    now builds its own independent quadtree instead of sharing entropy savings across time."""
    total, n = 0, 0
    for ms, present in batches:
        leaves = obj._apt_octree_mask(ms, present, present.shape[0], tau, ps_min=4,
                                      space_only=space_only)
        total += sum(len(ls) for ls in leaves)
        n += present.shape[0]
    return total / max(n, 1)


def calibrate(task, obj, batches, target: float, lo: float = -5.0, hi: float = 30.0,
             iters: int = 16, space_only: bool = False) -> tuple[float, float]:
    """Binary search: LOWER tau -> MORE splitting -> MORE tokens (tau gates "keep coarse if
    entropy < tau", so a smaller tau keeps fewer nodes coarse). Returns (tau, mean_tokens at
    that tau) after `iters` halvings -- not exact, but converges well within a few percent of
    target in practice (see printed mean_tokens per iteration if more precision is ever
    needed).

    lo=-5.0 (not 0 or near-0): entropy is always >= 0, so tau<=0 already means "never keep
    coarse on the threshold test alone" -- the ACTUAL ceiling on token count is the separate
    gain-per-token test in _apt_octree_recurse (a split is only taken if it has POSITIVE gain,
    independent of tau), which can and does cap total tokens well below a budget on some
    (dataset, k_frac) pairs -- confirmed directly on PASTIS at k_frac=1/3 (target 1024): tau=0,
    tau=-1, and tau=0.05 ALL gave the identical 726 tokens, since the gain test -- not the
    threshold -- was already the binding constraint. Starting lo well past 0 confirms this is
    a REAL structural ceiling (the recursion has nothing left it considers worth splitting),
    not an under-aggressive search bracket."""
    # bracket check: hi must give FEWER tokens than target, lo must give MORE
    m_lo = mean_tokens(obj, batches, lo, space_only)
    m_hi = mean_tokens(obj, batches, hi, space_only)
    if m_lo < target:
        m_floor = mean_tokens(obj, batches, -1e6, space_only)   # confirm this is the TRUE
        if abs(m_floor - m_lo) > 0.5:                 # ceiling, not an insufficiently negative
            print(f"  NOTE: tau={lo} gives {m_lo:.1f} but tau=-1e6 gives {m_floor:.1f} -- "   # bracket
                 f"bracket was not aggressive enough, using the true floor instead")
            m_lo, lo = m_floor, -1e6
        print(f"  UNREACHABLE: this task's apt_octree tops out at {m_lo:.1f} tokens (the "
             f"gain-per-token test itself stops splitting further, independent of tau -- see "
             f"docstring) -- target {target:.1f} cannot be reached by ANY tau. User-confirmed "
             f"2026-10-12: train at this MAX token count anyway, report actual vs target "
             f"honestly rather than skip this (dataset, k_frac) pair.")
        return lo, m_lo
    if m_hi > target:
        print(f"  WARNING: even tau={hi} (near-infinite) gives {m_hi:.1f} tokens > target "
             f"{target:.1f} -- every cell already costs >= 1 token (16 ps16 cells is the "
             f"floor); returning tau={hi} as the closest achievable (LESS splitting is not "
             f"possible above this)")
        return hi, m_hi
    for i in range(iters):
        mid = (lo + hi) / 2
        m = mean_tokens(obj, batches, mid, space_only)
        print(f"  iter {i}: tau={mid:.4f} -> {m:.1f} tokens (target {target:.1f})")
        if m > target:               # too many tokens -> need to split LESS -> raise tau
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    return tau, mean_tokens(obj, batches, tau, space_only)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", default="pastis,biomassters,geoidflood,yieldsat_raw")
    ap.add_argument("--k_fracs", default="0.16667,0.33333")
    ap.add_argument("--n_samples", type=int, default=64,
                    help="train-split chips to measure mean_tokens over, per calibration step")
    ap.add_argument("--split", default="train")
    ap.add_argument("--space_only", action="store_true",
                    help="calibrate --apt_octree_space_only instead (2026-10-13 isolating "
                         "experiment -- see InjectFT._apt_octree_mask's space_only docstring). "
                         "Needs its own tau (structurally more tokens at the same tau than the "
                         "time-capable recursion); written under a '<k_frac>_sonly' key so it "
                         "never overwrites the time-capable calibration already in this file.")
    a = ap.parse_args()
    k_fracs = [float(f) for f in a.k_fracs.split(",")]

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    for name in a.tasks.split(","):
        task = TASKS[name]
        obj = apt_stub(task)
        dl = loader(task, a.split, batch_size=8, num_workers=2, shuffle=True)
        it = iter(dl)
        batches = []
        n = 0
        while n < a.n_samples:
            ms, ss, season, present, pairs, extra, target = next(it)
            batches.append((ms, present))
            n += present.shape[0]
        print(f"=== {name} ({n} chips sampled) ===")
        results.setdefault(name, {})
        for k_frac in k_fracs:
            T = present.shape[1]
            target_tokens = round(k_frac * T * OUT_GRID * OUT_GRID)
            print(f" k_frac={k_frac:.5g} (target {target_tokens} tokens, T={T}"
                 f"{', space_only' if a.space_only else ''}):")
            tau, got = calibrate(task, obj, batches, target_tokens, space_only=a.space_only)
            reachable = got >= target_tokens * 0.95   # within 5% counts as "reached"
            print(f" -> tau={tau:.4f} (got {got:.1f} tokens, target {target_tokens}, "
                 f"reachable={reachable})")
            key = f"{k_frac:.5g}" + ("_sonly" if a.space_only else "")
            results[name][key] = {
                "tau": tau, "mean_tokens": got, "target_tokens": target_tokens, "T": T,
                "reachable": reachable}
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
