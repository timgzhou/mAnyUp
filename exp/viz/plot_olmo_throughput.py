"""Visualize OlmoEarth feature-extraction throughput vs resolution and modality.

Reads results/pastis/bench/olmo_throughput.csv (written by exp/bench/olmo_throughput.py) and
draws the four things that decide whether a config is affordable:

  1. throughput vs tokens-per-tile, log-log, one line per modality arm. The slope IS the
     scaling exponent -- a slope of -1 would be linear in tokens, steeper means attention's
     quadratic term dominates. This is the panel that says what a finer grid actually costs.
  2. ms per PASTIS sample as grouped bars per (patch, tile) config. Same data, absolute
     units, for reading off wall-clock budgets.
  3. max batch size that fits + peak GPU memory -- the capacity side of the same tradeoff.
  4. relative cost of each arm vs the S2-only baseline, showing whether S1 rides along
     nearly free or effectively doubles the sequence.

Run:
    python -u -m exp.viz.plot_olmo_throughput
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The sbatch runner writes one CSV per modality arm (results/pastis/bench/olmo_throughput_<arm>.csv)
# so the three jobs never contend for a single file; a plain glob merges them back.
CSV_GLOB = "results/pastis/bench/olmo_throughput*.csv"
OUT = "results/pastis/bench/olmo_throughput.png"

# PASTIS sample side in px; a config with tile_size == IMAGE_SIZE is untiled (one encoder
# call per sample). Must match exp/bench/olmo_throughput.IMAGE_SIZE.
IMAGE_SIZE = 64

ARM_STYLE = {
    "s2":   ("#1f77ff", "o", "-",  "S2 only (13 bands)"),
    "s1":   ("#00a05a", "s", "--", "S1 only (2 bands)"),
    "s2s1": ("#e4002b", "^", "-.", "S2 + S1 (multi-modal)"),
}

NUMERIC = ("patch_size", "tile_size", "grid", "tiles_per_sample", "tokens_per_tile",
           "batch_size", "samples_per_s", "ms_per_sample", "step_s_median",
           "step_s_mean", "step_s_std", "peak_mem_gb")


def load(pattern: str) -> list[dict]:
    """Read every per-arm CSV matching the glob and concatenate them.

    Later files win on a duplicate (arm, patch, tile) key, so re-running one arm and
    leaving the old combined CSV in place does not double-count it."""
    paths = sorted(Path().glob(pattern)) if not Path(pattern).is_file() else [Path(pattern)]
    if not paths:
        raise SystemExit(f"no CSVs matched {pattern}")
    rows: list[dict] = []
    for path in paths:
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
                if k in ("patch_size", "tile_size", "grid", "tiles_per_sample",
                         "tokens_per_tile", "batch_size"):
                    r[k] = int(r[k])
    dedup = {(r["arm"], r["patch_size"], r["tile_size"]): r for r in rows}
    return list(dedup.values())


def cfg_label(r: dict) -> str:
    return f"ps{r['patch_size']}\ntile{r['tile_size']}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=CSV_GLOB)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    rows = load(args.csv)
    if not rows:
        raise SystemExit(f"{args.csv} is empty")
    arms = [a for a in ARM_STYLE if any(r["arm"] == a for r in rows)]
    gpu = rows[0].get("gpu", "?")
    model_size = rows[0].get("model_size", "?")

    # Config order: by full-sample token count (grid^2), i.e. cheapest grid first.
    configs = sorted({(r["patch_size"], r["tile_size"]) for r in rows},
                     key=lambda c: (64 // c[0], c[1]))
    by = {(r["arm"], r["patch_size"], r["tile_size"]): r for r in rows}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle(
        f"OlmoEarth-{model_size} feature extraction throughput on {gpu}\n"
        f"PASTIS geometry: 64x64 px @ 10 m, T=12 - dummy input, max batch per config",
        fontsize=13, fontweight="bold")

    # --- 1. throughput vs sequence length, UNTILED configs only (log-log) ---------
    # Only tile_size == IMAGE_SIZE (one encoder call per sample) goes on this fit. Mixing in
    # tiled configs would put two points at the SAME x (tokens per call) with different
    # throughput -- e.g. ps4:tile32 and ps2:tile64 both hit 3072 tokens/call but do 4x vs 1x
    # calls per sample -- producing vertical jumps and a slope fit that averages across two
    # different tiling regimes. Tiled configs are shown separately in panel 2.
    ax = axes[0, 0]
    for arm in arms:
        color, marker, ls, label = ARM_STYLE[arm]
        pts = sorted(((by[(arm, ps, ts)]["tokens_per_tile"],
                       by[(arm, ps, ts)]["samples_per_s"])
                      for ps, ts in configs
                      if (arm, ps, ts) in by and ts == IMAGE_SIZE))
        if not pts:
            continue
        x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        ax.plot(x, y, color=color, marker=marker, ls=ls, label=label, lw=2, ms=7)
        # slope of the log-log fit = empirical scaling exponent in sequence length.
        # -1 would be linear (compute bound by token count); steeper means the O(n^2)
        # attention term is taking over.
        if len(x) > 1:
            slope = np.polyfit(np.log(x), np.log(y), 1)[0]
            ax.annotate(f"slope {slope:.2f}", xy=(x[-1], y[-1]),
                        xytext=(4, -12), textcoords="offset points",
                        color=color, fontsize=9, fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tokens per encoder call  (64/patch)$^2$ x T")
    ax.set_ylabel("throughput (PASTIS samples / s)")
    ax.set_title("Throughput vs sequence length (untiled only)\n"
                 "slope -1 = linear; steeper => attention dominates", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # --- 2. ms per sample, grouped bars -------------------------------------------
    ax = axes[0, 1]
    xs = np.arange(len(configs))
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        color, _, _, label = ARM_STYLE[arm]
        vals = [by[(arm, ps, ts)]["ms_per_sample"] if (arm, ps, ts) in by else np.nan
                for ps, ts in configs]
        bars = ax.bar(xs + i * w - 0.4 + w / 2, vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([cfg_label(by[(arms[0], ps, ts)]) for ps, ts in configs], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("ms per PASTIS sample (all tiles)")
    ax.set_title("Cost per sample by config\n(log scale)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # --- 3. max batch size + peak memory ------------------------------------------
    ax = axes[1, 0]
    for i, arm in enumerate(arms):
        color, _, _, label = ARM_STYLE[arm]
        vals = [by[(arm, ps, ts)]["batch_size"] if (arm, ps, ts) in by else np.nan
                for ps, ts in configs]
        bars = ax.bar(xs + i * w - 0.4 + w / 2, vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v)}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([cfg_label(by[(arms[0], ps, ts)]) for ps, ts in configs], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("max batch size that fits")
    ax.set_title("Usable batch size per config\n(auto-tuned, 10% safety margin)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # --- 4. the tiling effect + multi-modal overhead ------------------------------
    # Pairs of configs with the SAME token grid (same total tokens per sample) that differ
    # only in how those tokens are split across encoder calls. Attention is O(n^2) in the
    # per-call sequence length, so splitting one big call into k smaller ones cuts the
    # attention term by ~k even though the token count is identical -- the single most
    # actionable result here, and invisible in a per-config bar chart.
    ax = axes[1, 1]
    pairs = []                                   # (grid, tiled_cfg, untiled_cfg)
    for ps_t, ts_t in configs:
        if ts_t == IMAGE_SIZE:
            continue
        grid_t = IMAGE_SIZE // ps_t
        for ps_u, ts_u in configs:
            if ts_u == IMAGE_SIZE and IMAGE_SIZE // ps_u == grid_t:
                pairs.append((grid_t, (ps_t, ts_t), (ps_u, ts_u)))
    pairs.sort()

    if pairs:
        px = np.arange(len(pairs))
        pw = 0.8 / max(len(arms), 1)
        for i, arm in enumerate(arms):
            color, _, _, label = ARM_STYLE[arm]
            vals = []
            for _g, tiled, untiled in pairs:
                a, b = by.get((arm,) + tiled), by.get((arm,) + untiled)
                # >1 means the tiled variant is faster at identical total token count
                vals.append(b["ms_per_sample"] / a["ms_per_sample"] if a and b else np.nan)
            bars = ax.bar(px + i * pw - 0.4 + pw / 2, vals, pw, color=color, label=label)
            for bb, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.2f}x",
                            ha="center", va="bottom", fontsize=8)
        ax.axhline(1.0, color="#666", lw=1.2, ls=":")
        ax.set_xticks(px)
        ax.set_xticklabels(
            [f"grid {g}\nps{t[0]}:tile{t[1]}\nvs ps{u[0]}:tile{u[1]}" for g, t, u in pairs],
            fontsize=8)
        ax.set_ylabel("speedup of tiled vs untiled (same grid)")
        ax.set_title("Tiling speedup at identical token count\n"
                     "(>1 = splitting the sequence wins)", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
