"""Plot downstream accuracy against extraction cost for the ps x tile grid.

The question this answers: for a fixed compute budget, which (patch_size, tile_size) gives
the best PASTIS segmentation? Throughput alone says ps16 wins; mIoU alone says ps1 wins.
The Pareto front is the only honest answer, and it is what this draws.

Hue = tile size, shape = patch size, every point labelled with its config and its cost
(both axes are log, so reading a value off one by eye is imprecise).

The combined figure is a single panel on --x (default measured samples/s). --per_tile
additionally writes one figure per tile size, and those are SIDE BY SIDE: measured
throughput left, GMACs/sample right. Within a fixed tile size the question worth asking is
whether the hardware ranking matches the arithmetic ranking -- where they disagree, the
config is bound by memory traffic or kernel efficiency rather than by raw arithmetic.

Inputs (produced by separate jobs, joined on the extraction config):
  results/pastis/bench/olmoearth_ps-tile_speed_*.csv   exp/bench/olmo_throughput.py  (dummy input)
  results/pastis/lp_olmoearth_pastis.csv        exp/pastis/lp_cached_features.py (lp_pa2px)

Speed is measured on DUMMY tensors and accuracy on real PASTIS, so the join key is the
config (patch_size, tile_size, image_size), not any per-sample identity. Only the s2 arm is
joined -- every cached feature set is S2-only -- while the s1/s2s1 speed rows stay in the
speed CSV for reference.

Run:
    python -u -m exp.viz.plot_miou_vs_speed
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPEED_GLOB = "results/pastis/bench/olmoearth_ps-tile_speed.csv"
LP_CSV = "results/pastis/lp_olmoearth_pastis.csv"
FFT_CSV = "results/pastis/fft_olmoearth_pastis.csv"
OUT = "results/pastis/bench/miou_vs_throughput.png"
IMAGE_SIZE = 128        # only this prep is plotted (--image_size to change)

# Hue = tile size, shape = patch size.
TS_COLOR = {16: "#1f77ff", 64: "#00a05a", 128: "#e4002b"}
PS_MARKER = {1: "o", 2: "s", 4: "^", 8: "D", 16: "v"}

# Full-finetune overlay. FFT never tiles (it always encodes the whole image), so its points
# are tile==image_size by construction and get ONE hue rather than the tile-size scale.
# Star marker separates them from the frozen-LP dots at a glance.
FFT_COLOR = "#7d3cc6"
FFT_MARKER = "*"
# The FFT sweep was run S2+S1 while every cached-feature LP point is S2-only. The backbone
# forward is identical for LP and FFT -- finetuning changes only whether gradients flow, not
# inference GMACs -- so an S2 FFT point would land exactly on its LP twin's x. The S1 tokens
# are what move it right (~1.5-1.7x). We therefore look its cost up under the s2s1 arm, and
# the caption says so, rather than pretending it sits at the s2 x.
FFT_ARM = "s2s1"

# features dir name -> (patch, tile, image_size), e.g. oe_base_s2_ps4_tile64[_img128]
FEAT_RE = re.compile(r"_ps(\d+)_tile(\d+)(?:_img(\d+))?$")


def parse_features(name: str):
    m = FEAT_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 64)


FLOP_CACHE = "results/pastis/bench/flops_cache.json"
# base|s2|ps4|tile16|img128
FLOP_KEY_RE = re.compile(r"^[^|]+\|([^|]+)\|ps(\d+)\|tile(\d+)\|img(\d+)$")


def load_speed(pattern: str, arm: str = "s2", flop_cache: str = FLOP_CACHE) -> dict:
    """(patch, tile, image_size) -> a dict of cost metrics for one arm.

    Timing comes from the bench CSVs, but GMACs are ALSO merged in from flops_cache.json.
    That cache is written by a separate, much cheaper pass (one batch-1 trace per config, no
    GPU in its key), so it is complete long before the timing sweep is. Reading it directly
    means a mIoU-vs-GMACs figure never waits on timing it does not use."""
    out: dict = {}
    for path in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(path)):
            if r.get("arm") != arm:
                continue
            key = (int(r["patch_size"]), int(r["tile_size"]),
                   int(r.get("image_size", 64) or 64))
            out[key] = dict(r)

    if Path(flop_cache).exists():
        try:
            cached = json.loads(Path(flop_cache).read_text())
        except json.JSONDecodeError:
            cached = {}
        for k, flops in cached.items():
            m = FLOP_KEY_RE.match(k)
            if not m or m.group(1) != arm:
                continue
            key = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            row = out.setdefault(key, {"patch_size": key[0], "tile_size": key[1],
                                       "image_size": key[2], "arm": arm})
            # the CSV's own value wins if present; otherwise fill from cache
            if not row.get("gmacs_per_sample"):
                row["gmacs_per_sample"] = float(flops) / 2e9
                row["gflops_per_sample"] = float(flops) / 1e9
    return out


def load_miou(csv_path: str, head: str, epochs: int | None = None) -> dict:
    """(patch, tile, image_size) -> best test mIoU for the given head.

    epochs, when given, keeps ONLY runs at that budget. This matters: at a fixed batch_size
    the img128 prep takes 4x fewer optimizer steps per epoch than img64, and 32 epochs left
    every config short of convergence (ps8:tile64 went 0.444 -> 0.463 from 32 to 128 epochs,
    closing almost all of its gap to the img64 twin). Mixing budgets across points would put
    under-trained and converged configs on the same axes, so pin one budget per figure.

    Within a budget we take the MAX over repeats (they differ only by seed/lr), since the
    best-achieved probe is the fair representative of what the features support."""
    out = {}
    if not Path(csv_path).exists():
        return out
    for r in csv.DictReader(open(csv_path)):
        if r.get("head_mode") != head:
            continue
        key = parse_features(r.get("features", ""))
        if key is None:
            continue
        if epochs is not None:
            try:
                if int(r.get("epochs", -1)) != epochs:
                    continue
            except (ValueError, TypeError):
                continue
        try:
            miou = float(r["test_miou"])
        except (KeyError, ValueError, TypeError):
            continue
        if key not in out or miou > out[key]:
            out[key] = miou
    return out


def load_fft(csv_path: str, image_size: int) -> dict:
    """(patch, tile, prep_image_size) -> (best FFT test mIoU, ground_scale).

    FFT has no tiling knob (the whole image is encoded in one pass), so the tile slot is
    filled with the prep's image_size -- which is what tile_size means for an untiled
    forward, and what joins these rows to the tile<N> speed/GMACs entries.

    BOTH preps are plotted, not just `image_size`. They cover the SAME ground: the 64px
    prep is the quadrant split of the 128px one (128-sample i == 64-samples 4i..4i+3), so
    a 64px sample is a quarter of a 128px sample. ground_scale is the multiplier that puts
    a point's cost on a per-equal-ground footing with the reference prep -- 4x for the 64px
    rows when the figure is img128. mIoU needs no such correction: it is computed over the
    same test pixels either way, just grouped into different numbers of samples."""
    out: dict = {}
    if not Path(csv_path).exists():
        return out
    for r in csv.DictReader(open(csv_path)):
        try:
            img = int(r["image_size"])
            ps = int(r["patch_size"])
            miou = float(r["test_miou"])
        except (KeyError, ValueError, TypeError):
            continue
        # cost scale to the figure's reference prep: (ref_px / this_px)^2 samples of this
        # prep tile the same ground as one reference sample.
        scale = (image_size / img) ** 2
        key = (ps, img, img)
        if key not in out or miou > out[key][0]:
            out[key] = (miou, scale)
    return out


def pareto(points):
    """Indices on the upper-left Pareto front: no other point is both faster and better."""
    keep = []
    for i, (x_i, y_i) in enumerate(points):
        if not any(x_j >= x_i and y_j >= y_i and (x_j > x_i or y_j > y_i)
                   for j, (x_j, y_j) in enumerate(points) if j != i):
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", default=SPEED_GLOB)
    p.add_argument("--lp", default=LP_CSV)
    p.add_argument("--head", default="lp_pa2px")
    p.add_argument("--x", default="samples_per_s",
                   choices=("samples_per_s", "gmacs_per_sample"),
                   help="cost axis: measured throughput (hardware-specific) or GMACs per "
                        "sample (hardware-independent arithmetic)")
    p.add_argument("--epochs", type=int, default=None,
                   help="keep only LP runs at this epoch budget (default: any, taking the "
                        "best per config). Use 128 for the converged grid.")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                   help="plot only configs from this prep (default 128)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--per_tile", action="store_true",
                   help="also write one figure per tile size, named <out stem>_tile<N>.png")
    p.add_argument("--fft", default=FFT_CSV,
                   help="full-finetune results CSV, overlaid as star markers")
    p.add_argument("--no_fft", action="store_true", help="omit the full-finetune overlay")
    args = p.parse_args()

    speed = load_speed(args.speed)
    miou = load_miou(args.lp, args.head, args.epochs)
    # FFT points carry their own cost lookup: the runs are S2+S1, so their GMACs come from
    # the s2s1 arm, not the s2 arm the LP grid uses.
    fft = {} if args.no_fft else load_fft(args.fft, args.image_size)
    fft_speed = load_speed(args.speed, arm=FFT_ARM) if fft else {}
    dropped = {k for k in fft if fft_speed.get(k, {}).get(args.x) in ("", None)}
    if dropped:
        print(f"note: {len(dropped)} FFT run(s) have no {FFT_ARM} {args.x} and are "
              f"omitted: {sorted((k[0], k[1]) for k in dropped)}")
    fft = {k: v for k, v in fft.items() if k not in dropped}
    # Single image_size, so samples/s is a common unit across every point and needs no
    # pixel normalisation -- an img128 sample is an img128 sample.
    def has(k, col):
        return speed.get(k, {}).get(col) not in ("", None)

    keys = sorted(k for k in (set(speed) & set(miou))
                  if k[2] == args.image_size and has(k, args.x))
    if not keys:
        raise SystemExit(
            f"no img{args.image_size} configs have BOTH speed and {args.head} mIoU.\n"
            f"  speed: {sorted(k for k in speed if k[2] == args.image_size)}\n"
            f"  miou : {sorted(k for k in miou if k[2] == args.image_size)}")
    only_speed = sorted(k for k in set(speed) - set(miou) if k[2] == args.image_size)
    if only_speed:
        print(f"note: {len(only_speed)} img{args.image_size} configs have speed but no "
              f"{args.head} mIoU yet: {[(k[0], k[1]) for k in only_speed]}")

    if fft:
        print(f"fft overlay: {len(fft)} point(s) at {FFT_ARM} cost "
              f"{[(k[0]) for k in sorted(fft)]}")
    draw(keys, speed, miou, args, Path(args.out), fft=fft, fft_speed=fft_speed)
    if args.per_tile:
        stem = Path(args.out)
        for ts in sorted({k[1] for k in keys}):
            sub = [k for k in keys if k[1] == ts]
            subf = {k: v for k, v in fft.items() if k[1] == ts}
            draw(sub, speed, miou, args,
                 stem.with_name(f"{stem.stem}_tile{ts}{stem.suffix}"), tile_only=ts,
                 fft=subf, fft_speed=fft_speed)


def _panel(ax, keys, speed, miou, args, xcol, tile_only, fft=None, fft_speed=None) -> None:
    """Draw one accuracy-vs-cost panel on `ax`, with `xcol` as the cost metric.

    xcol is either samples_per_s (higher is better, so the good corner is upper-right) or
    gmacs_per_sample (a cost, so the good corner is upper-left). The Pareto helper always
    keeps upper-right, so for a cost axis we negate x before calling it."""
    speed_axis = xcol == "samples_per_s"
    pts = []
    for k in keys:
        raw = speed[k].get(xcol, "")
        if raw in ("", None):
            raise SystemExit(f"{xcol} missing for {k}; re-run the bench to populate it")
        pts.append((float(raw), miou[k]))

    placed: list[tuple[float, float]] = []
    for k, (x, y) in zip(keys, pts):
        ps, ts, _img = k
        ax.scatter(x, y, s=150, color=TS_COLOR.get(ts, "#666"),
                   marker=PS_MARKER.get(ps, "o"), edgecolor="black", linewidth=0.7, zorder=3)
        # Near-coincident points (the ps16 cluster is three configs within ~10% throughput
        # and 0.01 mIoU) would overprint. Stack each collision one label-height lower, and
        # flip to the left of the marker once stacked so the block does not run off-axis.
        n_hit = sum(1 for px_, py_ in placed
                    if abs(np.log10(x) - np.log10(px_)) < 0.12 and abs(y - py_) < 0.012)
        dx, ha = (7, "left") if n_hit == 0 else (-9, "right")
        dy = 5 - 20 * n_hit
        # The x axis is log, so reading a value off it by eye is imprecise -- print it.
        if speed_axis:
            rate = f"{x:.1f} samp/s" if x < 10 else f"{x:.0f} samp/s"
        else:
            rate = f"{x:,.0f} GMACs" if x >= 10 else f"{x:.1f} GMACs"
        ax.annotate(f"ps{ps}/t{ts}\n{rate}", xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, linespacing=1.25, ha=ha)
        placed.append((x, y))

    # --- full-finetune overlay -------------------------------------------------------
    # Same cost axis (finetuning does not change the inference forward), different marker,
    # and its own hue since tile size is not a variable for these runs.
    for k in sorted(fft or {}):
        row = (fft_speed or {}).get(k, {})
        raw = row.get(xcol, "")
        if raw in ("", None):
            continue
        y, scale = fft[k]
        img = k[2]
        # Put every prep on a per-equal-ground cost. GMACs/sample scales UP by the number
        # of small samples needed to cover one reference sample; samples/s scales DOWN by
        # the same factor (you must push 4x as many 64px samples through per unit ground).
        x = float(raw) * (1 / scale if speed_axis else scale)
        ax.scatter(x, y, s=340, color=FFT_COLOR, marker=FFT_MARKER,
                   edgecolor="black", linewidth=0.8, zorder=4,
                   facecolor=FFT_COLOR if img == args.image_size else "none")
        if img != args.image_size:   # hollow star = rescaled from the other prep
            ax.scatter(x, y, s=340, facecolor="white", edgecolor=FFT_COLOR,
                       marker=FFT_MARKER, linewidth=1.6, zorder=5)
        n_hit = sum(1 for px_, py_ in placed
                    if abs(np.log10(x) - np.log10(px_)) < 0.12 and abs(y - py_) < 0.012)
        dx, ha = (9, "left") if n_hit == 0 else (-11, "right")
        rate = (f"{x:.1f} samp/s" if x < 10 else f"{x:.0f} samp/s") if speed_axis else (
            f"{x:,.0f} GMACs" if x >= 10 else f"{x:.1f} GMACs")
        ax.annotate(f"FFT ps{k[0]}/img{img}\n{rate}", xy=(x, y), xytext=(dx, 5 - 20 * n_hit),
                    textcoords="offset points", fontsize=7.5, linespacing=1.25, ha=ha,
                    color=FFT_COLOR, fontweight="bold")
        placed.append((x, y))

    # The Pareto front is only meaningful when several tile sizes compete. In a single-tile
    # view every point is on it by construction (mIoU falls monotonically with cost), so the
    # dashed line would just retrace the series and say nothing.
    # NOTE: the front is computed over the frozen-LP points only. FFT is a different training
    # regime (and a different modality set here), so folding it in would produce a front that
    # no single sweep actually offers.
    front = ([] if tile_only is not None else
             pareto(pts if speed_axis else [(-a, b) for a, b in pts]))
    if len(front) > 1:
        ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front],
                ls="--", color="#333", lw=1.4, zorder=2, label="Pareto front")

    ax.set_xscale("log")
    ax.margins(x=0.16)
    ax.set_xlabel("extraction throughput (samples / s, log)" if speed_axis
                  else "arithmetic cost (GMACs / sample, log)")
    ax.set_ylabel(f"test mIoU ({args.head})")
    ax.set_title("Accuracy vs measured speed\nupper-right is better" if speed_axis
                 else "Accuracy vs arithmetic cost\nupper-LEFT is better (x is cost)",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    # Two legends: hue and shape carry independent variables, so they need separate keys.
    ts_seen = sorted({k[1] for k in keys})
    ps_seen = sorted({k[0] for k in keys})
    h_ts = [plt.Line2D([], [], ls="", marker="s", ms=9, color=TS_COLOR.get(t, "#666"),
                       label=f"tile {t}") for t in ts_seen]
    h_ps = [plt.Line2D([], [], ls="", marker=PS_MARKER.get(q, "o"), ms=8, color="#444",
                       label=f"patch {q}") for q in ps_seen]
    if len(front) > 1:
        h_ps.append(plt.Line2D([], [], ls="--", color="#333", label="Pareto front"))
    if fft:
        h_ts = h_ts + [plt.Line2D([], [], ls="", marker=FFT_MARKER, ms=13, color=FFT_COLOR,
                                  label=f"full finetune ({FFT_ARM}, untiled)")]
        if any(k[2] != args.image_size for k in fft):
            h_ts.append(plt.Line2D([], [], ls="", marker=FFT_MARKER, ms=13, mfc="white",
                                   mec=FFT_COLOR, mew=1.6,
                                   label=f"  (hollow = img64 prep,\n   cost x4 to equal ground)"))
    if tile_only is None:
        # lower-left overlaps the cheap/low-mIoU corner (the ps16 cluster), so park the
        # hue legend outside the axes rather than on top of data.
        leg1 = ax.legend(handles=h_ts, fontsize=9, loc="upper left",
                         bbox_to_anchor=(1.01, 0.62), borderaxespad=0,
                         title="tile size (hue) / regime")
        ax.add_artist(leg1)
    ax.legend(handles=h_ps, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, title="patch size (shape)")


def has_gmacs(keys, speed) -> bool:
    """True when every key carries a GMACs figure (an older bench CSV has none)."""
    return bool(keys) and all(
        speed[k].get("gmacs_per_sample") not in ("", None) for k in keys)


def draw(keys, speed, miou, args, out_path, tile_only=None, fft=None, fft_speed=None) -> None:
    """Render the figure for `keys`.

    Per-tile views get BOTH cost axes side by side -- measured throughput on the left,
    GMACs/sample on the right -- because within one tile size the interesting question is
    whether the hardware ranking matches the arithmetic ranking. The combined view stays
    single-panel on --x, where the tile-size comparison is already carrying the figure.
    """
    # Side-by-side needs BOTH metrics on every point. GMACs land well before timing does
    # (separate, cheaper pass), so during a partial sweep fall back to a single panel on the
    # metric that is actually complete rather than failing outright.
    have_speed = all(speed[k].get("samples_per_s") not in ("", None) for k in keys)
    both = tile_only is not None and has_gmacs(keys, speed) and have_speed
    ncols = 2 if both else 1
    fig, axes = plt.subplots(1, ncols, figsize=(15.5, 7.0) if both else (9.5, 7.0),
                             squeeze=False)
    scope = (f"tile {tile_only}" if tile_only is not None else "all tile sizes")
    ep = f", {args.epochs} ep" if args.epochs else ""
    reg = (f"\ncircles/diamonds = frozen LP (S2)   stars = full finetune ({FFT_ARM}, 64 ep) "
           f"-- FFT costs more here because of the added S1 tokens, not the finetuning"
           if fft else "")
    fig.suptitle(f"PASTIS {args.head} mIoU vs OlmoEarth extraction cost "
                 f"(OlmoEarth-base, image_size={args.image_size}, {scope}{ep}){reg}",
                 fontsize=13, fontweight="bold")

    if both:
        _panel(axes[0][0], keys, speed, miou, args, "samples_per_s", tile_only, fft, fft_speed)
        _panel(axes[0][1], keys, speed, miou, args, "gmacs_per_sample", tile_only, fft, fft_speed)
    else:
        _panel(axes[0][0], keys, speed, miou, args, args.x, tile_only, fft, fft_speed)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  ({len(keys)} configs{', 2 panels' if both else ''})")


if __name__ == "__main__":
    main()
