"""Render a PASTIS Sentinel-2 time series as a video, for the "RGB video vs. remote sensing
time series" slide.

The point of the slide is that an RS time series is NOT like an RGB video:
  - it spans a YEAR, not seconds, so change is seasonal (crop growth, harvest, bare soil);
  - acquisitions are IRREGULAR -- gaps of 5 to 30+ days, not a fixed frame rate;
  - many frames are partly or fully CLOUD, i.e. the observation is simply missing;
  - each frame is 10+ spectral bands, of which RGB is a small and unrepresentative slice.

So we read the RAW PASTIS-R arrays (T up to 61 real acquisition dates) rather than the
processed 12-month averages the models consume -- averaging into monthly composites is exactly
the step that hides the irregularity and the clouds.

Each frame is captioned with its true date and the day gap since the previous acquisition, so
the irregular sampling is visible rather than implied. --pace real gives each frame a duration
proportional to that gap (long gap = long hold), which makes the irregularity felt; the default
--pace fixed gives every frame the same --sec_per_step.

Writes MP4 (ffmpeg, for slides) and optionally GIF (--gif, for embedding anywhere).

    source env_setup/env_olmo.sh
    python -u -m exp.viz.make_rs_video --list                 # candidate patches
    python -u -m exp.viz.make_rs_video --patches 20013,20021  # render these
    python -u -m exp.viz.make_rs_video --patches 20013 --pace real --gif
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless cluster: never try an interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

DATA_DIR = "data/PASTIS-R"
RGB_IDX = [2, 1, 0]            # B04,B03,B02 (R,G,B) within PASTIS's 10-band S2 stack


def load_meta(data_dir: Path) -> list[dict]:
    with open(data_dir / "metadata.geojson") as f:
        return json.load(f)["features"]


def patch_dates(props: dict) -> list[str]:
    """Acquisition dates (YYYYMMDD strings) in chronological order."""
    return [str(d) for d in sorted(props["dates-S2"].values())]


def to_rgb(frames: np.ndarray, lo_p: float, hi_p: float) -> np.ndarray:
    """(T,3,H,W) int16 reflectance -> (T,H,W,3) float in [0,1].

    ONE percentile stretch across the whole series (not per frame): a per-frame stretch would
    renormalize every frame to full contrast and so erase the very thing the video is meant to
    show -- that some frames are washed out by cloud and others are dark. Shared limits keep
    frames comparable to each other."""
    x = frames.astype(np.float32).transpose(0, 2, 3, 1)      # (T,H,W,3)
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def cloud_score(rgb: np.ndarray) -> np.ndarray:
    """Per-frame cloudiness proxy in [0,1]: fraction of pixels that are bright and grey.

    Cloud is high-reflectance and near-neutral across R/G/B, whereas bright soil or senescent
    crop keeps a colour cast -- so 'bright AND low saturation' separates them better than
    brightness alone."""
    bright = rgb.mean(-1) > 0.55
    sat = rgb.max(-1) - rgb.min(-1)
    return (bright & (sat < 0.16)).mean(axis=(1, 2))


def day_gaps(dates: list[str]) -> list[int]:
    """Days since the previous acquisition; 0 for the first frame."""
    from datetime import datetime
    ds = [datetime.strptime(d, "%Y%m%d") for d in dates]
    return [0] + [(b - a).days for a, b in zip(ds, ds[1:])]


def render(pid: int, data_dir: Path, out_dir: Path, args) -> None:
    props = next(f["properties"] for f in load_meta(data_dir)
                 if f["properties"]["ID_PATCH"] == pid)
    dates = patch_dates(props)
    arr = np.load(data_dir / "DATA_S2" / f"S2_{pid}.npy", mmap_mode="r")
    T = min(len(dates), arr.shape[0])
    dates, arr = dates[:T], np.asarray(arr[:T, RGB_IDX])
    rgb = to_rgb(arr, args.lo_pct, args.hi_pct)
    clouds = cloud_score(rgb)
    gaps = day_gaps(dates)

    fig, ax = plt.subplots(figsize=(args.size, args.size))
    fig.subplots_adjust(0, 0, 1, 1)                # image fills the frame; caption drawn on it
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    im = ax.imshow(rgb[0], interpolation="nearest")
    # Caption boxes drawn IN the image so the video needs no external legend on the slide.
    txt = ax.text(0.03, 0.955, "", transform=ax.transAxes, fontsize=args.size * 2.6,
                  color="white", va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.32", fc="black", ec="none", alpha=0.55))
    note = ax.text(0.97, 0.955, "", transform=ax.transAxes, fontsize=args.size * 2.6,
                   color="white", va="top", ha="right",
                   bbox=dict(boxstyle="round,pad=0.32", fc="#c53030", ec="none", alpha=0.0))

    def draw(i):
        im.set_data(rgb[i])
        d = dates[i]
        gap = f"  (+{gaps[i]}d)" if i else ""
        txt.set_text(f"{d[:4]}-{d[4:6]}-{d[6:]}{gap}\nframe {i + 1}/{T}")
        cloudy = clouds[i] > args.cloud_thresh
        note.set_text("CLOUD" if cloudy else "")
        note.get_bbox_patch().set_alpha(0.75 if cloudy else 0.0)
        return im, txt, note

    if args.pace == "real":
        # Frame duration proportional to the real gap, so a 30-day gap holds ~6x a 5-day one.
        base = np.array([max(g, 1) for g in gaps], dtype=float)
        base[0] = np.median(base[1:]) if T > 1 else 1.0
        durations = base / np.median(base) * args.sec_per_step * 1000.0
        durations = np.clip(durations, 120, 2500)
    else:
        durations = np.full(T, args.sec_per_step * 1000.0)

    anim = animation.FuncAnimation(fig, draw, frames=T, blit=False,
                                   interval=float(durations.mean()))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rs_timeseries_{pid}_{args.pace}"

    mp4 = out_dir / f"{stem}.mp4"
    # Constant fps for MP4; variable pacing is approximated by repeating frames.
    fps = 1.0 / args.sec_per_step
    if args.pace == "real":
        reps = np.maximum(1, np.round(durations / (args.sec_per_step * 1000.0))).astype(int)
        seq = np.repeat(np.arange(T), reps)
    else:
        seq = np.arange(T)
    writer = animation.FFMpegWriter(fps=fps, bitrate=args.bitrate,
                                    extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(mp4), dpi=args.dpi):
        for i in seq:
            draw(int(i))
            writer.grab_frame()
    print(f"  wrote {mp4}  ({len(seq)} rendered frames, {len(seq) / fps:.1f}s)")

    if args.gif:
        gif = out_dir / f"{stem}.gif"
        anim.save(str(gif), writer=animation.PillowWriter(fps=fps), dpi=args.dpi // 2)
        print(f"  wrote {gif}")
    plt.close(fig)

    span = f"{dates[0]} -> {dates[-1]}"
    print(f"  patch {pid}: T={T}  {span}  gaps {min(gaps[1:])}-{max(gaps[1:])}d  "
          f"cloudy frames={int((clouds > args.cloud_thresh).sum())}  parcels={props['N_Parcel']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--patches", default=None,
                    help="comma-separated ID_PATCH values to render (see --list)")
    ap.add_argument("--list", action="store_true",
                    help="print the longest/most-varied candidate patches and exit")
    ap.add_argument("--n_list", type=int, default=12)
    ap.add_argument("--sec_per_step", type=float, default=0.5,
                    help="seconds per timestep (default 0.5)")
    ap.add_argument("--pace", default="fixed", choices=("fixed", "real"),
                    help="fixed: every frame held equally. real: hold proportional to the true "
                         "day gap, so irregular sampling is visible in the playback itself.")
    ap.add_argument("--cloud_thresh", type=float, default=0.25,
                    help="fraction of bright+grey pixels above which a frame is flagged CLOUD")
    ap.add_argument("--lo_pct", type=float, default=2.0)
    ap.add_argument("--hi_pct", type=float, default=98.0)
    ap.add_argument("--size", type=float, default=5.0, help="figure side in inches")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--bitrate", type=int, default=2400)
    ap.add_argument("--gif", action="store_true", help="also write a GIF")
    ap.add_argument("--out_dir", default="results/pastis/feature_viz/rs_video")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "metadata.geojson").exists():
        raise SystemExit(f"ERROR: raw PASTIS-R not found at {data_dir}")

    if args.list or not args.patches:
        feats = sorted(load_meta(data_dir),
                       key=lambda f: -len(f["properties"]["dates-S2"]))[:args.n_list]
        print(f"{'patch':<9}{'T':<5}{'parcels':<9}{'span':<22}{'gaps(d)':<12}{'cloudy'}")
        for f in feats:
            p = f["properties"]
            d = patch_dates(p)
            g = day_gaps(d)[1:]
            arr = np.load(data_dir / "DATA_S2" / f"S2_{p['ID_PATCH']}.npy", mmap_mode="r")
            rgb = to_rgb(np.asarray(arr[:, RGB_IDX]), args.lo_pct, args.hi_pct)
            nc = int((cloud_score(rgb) > args.cloud_thresh).sum())
            print(f"{p['ID_PATCH']:<9}{len(d):<5}{p['N_Parcel']:<9}"
                  f"{d[0]}-{d[-1]:<12}{min(g)}-{max(g):<9}{nc}")
        if not args.patches:
            print("\nPick with --patches <id,id,...>")
        return

    out_dir = Path(args.out_dir)
    for pid in [int(x) for x in args.patches.split(",") if x]:
        print(f"rendering patch {pid} ...")
        render(pid, data_dir, out_dir, args)


if __name__ == "__main__":
    main()
