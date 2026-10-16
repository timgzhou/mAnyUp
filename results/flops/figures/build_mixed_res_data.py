"""Extract mixed-resolution (injection) points at a FIXED architecture -- inject_layer=3,
and for the learned selector, joint_target=drop (the "drop-path value predictor") -- rather
than whichever architecture variant scores best at each budget k. Joined with GMACs.

GEOID-Flood is excluded: its injection results (geoidseries_flood) use a different task
(flood-only IoU over a full acquisition series) than its fixed-resolution panel (3-class
mIoU from a single pre/post pair) -- not a fair same-axis comparison, and inject_layer=3
was never run there anyway.
"""
import csv
import json
import re
from pathlib import Path

RESULTS = Path("/scratch/timz/mAnyUp/results")

# injection task dir -> the fixed-res panel key it overlays onto (GEOID-Flood excluded)
PANEL_MAP = {
    "pastis": "pastis",
    "biomassters": "biomassters",
    "yieldsat_raw": "yieldsat_ml24",
}
METRIC = {
    "pastis": ("test_miou", True, "mIoU"),
    "biomassters": ("test_rmse", False, "RMSE (t/ha)"),
    "yieldsat_raw": ("test_rmse", False, "pixel RMSE (t/ha)"),
}

FIXED_LAYER = "3"

STRIP_SUFFIX_PATTERNS = [
    r"_lr[\d.eE+-]+", r"_ep\d+", r"_seed\d+", r"_warm\d*", r"_teach\w*", r"_gfix",
    r"_haoneway", r"_poscenter", r"_dt\w*", r"_jitter\w*",
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def round_k(k_frac):
    if k_frac in (None, ""):
        return None
    return round(float(k_frac), 3)


def is_plain_two_level(row):
    """Same guard as before: exclude a 3rd date-level crop tier, non-default hr crop size,
    acquisition pooling, or a multi-resolution mix -- architecture changes this chart can't
    label, not fair comparisons at "the same" budget k."""
    if row.get("date_level", "none") not in ("", "none"):
        return False
    if row.get("res_mix", ""):
        return False
    if row.get("acq_pool", ""):
        return False
    if row.get("crop_px", "") not in ("", "4"):
        return False
    return True


def bucket_of(row):
    """'random' (plain baseline, inject_layer=3, joint off) or 'value_predictor'
    (inject_layer=3, joint=True, joint_target=drop -- the drop-path value predictor) or None.
    Both buckets are pinned to the SAME inject_layer so they're an apples-to-apples pair."""
    if row.get("selector") != "random" or not is_plain_two_level(row):
        return None
    if row.get("inject_layer") != FIXED_LAYER:
        return None
    joint = row.get("joint", "")
    if joint not in ("True", "true", "1"):
        return "random"
    if row.get("joint_target") == "drop":
        return "value_predictor"
    return None   # a joint run at L3 but not the drop-path target -- not what was asked for


def strip_arch_key(run):
    s = run
    for pat in STRIP_SUFFIX_PATTERNS:
        s = re.sub(pat, "", s)
    return s


def best_rows(task, exact, arch):
    """Best (by val_primary) row per (bucket, k) WITHIN the fixed architecture -- this now
    only varies over seeds/minor training hyperparameters, not over readout/layer search."""
    metric_col, higher, label = METRIC[task]
    rows = read_csv(RESULTS / task / "inject_fft.csv")
    groups = {}
    for r in rows:
        bucket = bucket_of(r)
        if bucket is None:
            continue
        if r.get(metric_col) in (None, ""):
            continue
        k = round_k(r.get("k_frac"))
        if k is None or k == 0:
            continue
        fl = exact.get(r["run"]) or arch.get(strip_arch_key(r["run"]))
        if fl is None:
            continue
        groups.setdefault((bucket, k), []).append((r, fl))

    out = []
    for (bucket, k), rs in groups.items():
        key = (lambda pair: float(pair[0]["val_primary"]))
        best, fl = max(rs, key=key) if higher else min(rs, key=key)
        out.append({
            "bucket": bucket, "k": k,
            "test_metric": float(best[metric_col]),
            "lr_patch": best.get("lr_patch") or "16",
            "lr_tile": best.get("lr_tile") or "64",
            "inject_layer": best.get("inject_layer", ""),
            "select_unit": best.get("select_unit", ""),
            "run": best["run"],
            "hours": float(best["hours"]) if best.get("hours") else None,
            "gmacs": float(fl["gmacs"]),
            "flops_run": fl["run"],
            "n_candidates": len(rs),
        })
    return out, label, higher


def flops_lookup(task):
    path = RESULTS / task / "inject_flops.csv"
    if not path.exists():
        return {}, {}
    rows = read_csv(path)
    exact, arch = {}, {}
    for r in rows:
        exact[r["run"]] = r
        arch.setdefault(strip_arch_key(r["run"]), r)
    return exact, arch


def main():
    all_points = {}
    for task in PANEL_MAP:
        exact, arch = flops_lookup(task)
        pts, label, higher = best_rows(task, exact, arch)
        panel = PANEL_MAP[task]
        all_points.setdefault(panel, [])
        for p in pts:
            all_points[panel].append({
                "bucket": p["bucket"], "k": p["k"], "test_metric": p["test_metric"],
                "gmacs": p["gmacs"], "lr_ps": int(p["lr_patch"]), "lr_ts": int(p["lr_tile"]),
                "hr_ps": 4, "inject_layer": p["inject_layer"], "select_unit": p["select_unit"],
                "run": p["run"], "flops_run": p["flops_run"], "hours": p["hours"],
            })
    print(json.dumps(all_points, indent=2))
    out_path = Path(__file__).parent / "mixed_res_data.json"
    with open(out_path, "w") as f:
        json.dump(all_points, f, indent=2)


if __name__ == "__main__":
    main()
