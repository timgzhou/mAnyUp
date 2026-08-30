"""Merge the per-(arm, image_size) speed CSVs into one olmoearth_ps-tile_speed.csv.

exp/bench/olmo_throughput.py writes one CSV per sbatch job (one arm x one image_size) so
concurrent jobs never contend for a file. This collapses them into the single combined CSV,
de-duplicating on (arm, patch, tile, image_size) so re-running one arm replaces its rows
rather than appending duplicates.

Run after the bench jobs finish:
    python -m exp.bench.merge_speed_csvs
"""
import csv, glob, os

import subprocess

# Refuse to merge while a bench job is running: each job REWRITES its CSV from scratch on
# every flush, so mid-run a file holds fewer rows than a completed earlier run left there.
# Merging then silently drops good rows (seen live: an 11-config figure fell back to 7).
try:
    running = subprocess.run(["squeue", "-h", "-n", "oe_bench", "-o", "%i"],
                             capture_output=True, text=True, timeout=30).stdout.split()
except Exception:
    running = []
if running and os.environ.get("MERGE_ANYWAY") != "1":
    raise SystemExit(
        f"{len(running)} oe_bench job(s) still running ({', '.join(running)}); their CSVs are "
        f"mid-write and merging now would drop rows.\n"
        f"Wait for them, or set MERGE_ANYWAY=1 to override.")

parts = sorted(glob.glob("results/bench/olmoearth_ps-tile_speed_*_img*.csv"))
rows = []
for p in parts:
    rows.extend(csv.DictReader(open(p)))
if not rows:
    raise SystemExit("no per-arm speed CSVs found")

# dedupe on the config identity; later files win
by = {(r["arm"], int(r["patch_size"]), int(r["tile_size"]), int(r.get("image_size", 64) or 64)): r
      for r in rows}
out = [by[k] for k in sorted(by, key=lambda k: (k[0], k[3], -k[1], k[2]))]
dest = "results/bench/olmoearth_ps-tile_speed.csv"
with open(dest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(f"merged {len(parts)} files -> {dest}  ({len(out)} rows)")
for arm in ("s2", "s1", "s2s1"):
    n = sum(1 for k in by if k[0] == arm)
    print(f"  {arm}: {n} configs")
