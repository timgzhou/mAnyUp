"""Compare the anyup and anyup_t1 heads' upsampled FEATURE maps, per sample.

The two heads differ only in WHEN time is collapsed (see lp_cached_features.py's head table):

  anyup     mean over T FIRST, then one AnyUp call:  AnyUp(mean_t feats, mean_t RGB)
  anyup_t1  one AnyUp call PER timestep, then mean:  mean_t AnyUp(feats[t], RGB[t])

So anyup upsamples a time-averaged feature map using a time-averaged (hazy, cloud-smeared)
guidance image, while anyup_t1 upsamples each frame against its OWN guidance and averages
afterwards. If per-frame guidance is sharper than the composite -- the usual case, since
averaging blurs edges across the series -- t1's mean should retain more spatial detail.

Per sample this writes a figure with:
  row 0   raw RGB (mean T) | anyup output | anyup_t1 output | HR reference (native ps1)
  row 1+  the per-timestep AnyUp maps t1 averages over, with their guidance frames

Panels are FEATURES (the frozen AnyUp upsampler's output), not logits: both heads put a
trainable probe on top, and an untrained probe would show nothing. All feature panels share one
PCA basis (fit on the anyup output) so colors are comparable across the figure.

    source env_setup/env_olmo.sh
    python -u -m exp.viz.viz_anyup_t1 --n_images 2
    python -u -m exp.viz.viz_anyup_t1 --lr_dir oe_base_s2_ps4_tile64 --n_timesteps 6
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")        # headless cluster: never try an interactive backend
import matplotlib.pyplot as plt
import torch

from exp.viz.visualize_features import (raw_rgb, pca_rgb, pca_rgb_shared, nearest_resize,
                                        s2_path, DISPLAY_PX)

FEATURES_ROOT = "~/projects/aip-gpleiss/timz/features"
DATA_SPLITS = "data/pastis_olmoearth"
OUT_SIZE = (64, 64)          # AnyUp output grid = PASTIS label resolution


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_root", default=FEATURES_ROOT)
    ap.add_argument("--data_splits", default=DATA_SPLITS)
    ap.add_argument("--lr_dir", default="oe_base_s2_ps8_tile64",
                    help="low-res feature dir both heads consume")
    ap.add_argument("--hr_dir", default="oe_base_s2_ps1_tile64",
                    help="native high-res dir shown as a reference ('' to omit)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_images", type=int, default=2)
    ap.add_argument("--n_timesteps", type=int, default=6,
                    help="how many per-timestep panels to show (<= T; the mean still uses ALL T)")
    ap.add_argument("--time_pool", default="mean", choices=("mean", "median"),
                    help="how anyup collapses the series into its single guidance frame")
    ap.add_argument("--pca", default="shared", choices=("shared", "independent"),
                    help="shared: one PCA basis (fit on the anyup output) for every feature "
                         "panel. independent: per-panel basis (max contrast, no shared meaning).")
    ap.add_argument("--out", default="feature_viz/anyup_t1_compare.png",
                    help="output TEMPLATE: one file per sample is written as "
                         "<stem>_<split><idx>.png next to this path")
    args = ap.parse_args()

    # Imported here, not at module top: pulls olmo bootstrap + torch.hub AnyUp, both heavy.
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()
    from exp.pastis import finetune_olmoearth as fmod
    from exp.pastis.finetune_olmoearth import AnyUpUpsampleProbe, _load_rgb_guidance

    root = Path(args.features_root).expanduser().resolve()
    data_splits = Path(args.data_splits).expanduser().resolve()
    if not data_splits.exists():
        raise SystemExit(f"ERROR: --data_splits not found: {data_splits}")
    fmod.DATA_SPLITS = str(data_splits)          # _load_rgb_guidance reads this global
    lr_d = root / args.lr_dir
    if not (lr_d / f"pastis_r_{args.split}").exists():
        raise SystemExit(f"ERROR: {lr_d} has no pastis_r_{args.split}/ (not extracted?)")
    hr_d = root / args.hr_dir if args.hr_dir else None
    if hr_d is not None and not (hr_d / f"pastis_r_{args.split}").exists():
        print(f"WARNING: {hr_d} not found; omitting the HR reference panel")
        hr_d = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  LR={args.lr_dir}  pca={args.pca}")
    # Only the frozen upsampler is used -- both heads share it, and the probe on top is
    # untrained here (we visualize features, not logits).
    anyup = AnyUpUpsampleProbe(num_classes=1).anyup.to(device).eval()

    out_tmpl = Path(args.out)
    out_dir, stem = out_tmpl.parent, out_tmpl.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in range(args.n_images):
        s2 = torch.load(s2_path(data_splits, args.split, r))            # (T,13,64,64)
        lr = torch.load(lr_d / f"pastis_r_{args.split}" / f"{r}.pt").float()   # (T,gH,gW,D)
        T = lr.shape[0]

        # Guidance: one pooled frame for anyup, the full per-timestep stack for anyup_t1.
        g_mean = _load_rgb_guidance(args.split, r, temporal=False,
                                    time_pool=args.time_pool).unsqueeze(0).to(device)  # (1,3,64,64)
        g_per_t = _load_rgb_guidance(args.split, r, temporal=True).to(device)           # (T,3,64,64)

        with torch.no_grad():
            # anyup: collapse time FIRST, then a single AnyUp call.
            f_mean = lr.mean(0).permute(2, 0, 1).unsqueeze(0).to(device)   # (1,D,gH,gW)
            a_mean = anyup(g_mean, f_mean, output_size=OUT_SIZE)           # (1,D,64,64)

            # anyup_t1: one AnyUp call per timestep, averaged AFTER upsampling.
            per_t = []
            for t in range(T):
                f_t = lr[t].permute(2, 0, 1).unsqueeze(0).to(device)       # (1,D,gH,gW)
                per_t.append(anyup(g_per_t[t:t + 1], f_t, output_size=OUT_SIZE))
            a_t1 = torch.stack(per_t, 0).mean(0)                           # (1,D,64,64)

        def to_hwd(x):                              # (1,D,H,W) -> (H,W,D) on cpu
            return x.float().squeeze(0).permute(1, 2, 0).cpu()

        n_t = min(args.n_timesteps, T)
        maps = [to_hwd(a_mean), to_hwd(a_t1)]
        if hr_d is not None:
            hr = torch.load(hr_d / f"pastis_r_{args.split}" / f"{r}.pt").float()
            maps.append(hr.mean(0))                                        # (64,64,D)
        maps += [to_hwd(p) for p in per_t[:n_t]]

        if args.pca == "shared":
            rgbs = pca_rgb_shared(maps[0], maps)     # basis fit on the anyup output
        else:
            rgbs = [pca_rgb(m) for m in maps]
        rgbs = [nearest_resize(x, DISPLAY_PX) for x in rgbs]

        head = [(raw_rgb(s2.float().mean(0)), f"raw RGB ({args.time_pool} T)"),
                (rgbs[0], "anyup\nmean-T feats + mean-T RGB"),
                (rgbs[1], "anyup_t1\nper-t AnyUp, then mean")]
        if hr_d is not None:
            head.append((rgbs[2], f"HR (native)\n{args.hr_dir.split('_', 3)[-1]}"))
        tail_rgbs = rgbs[3:] if hr_d is not None else rgbs[2:]

        # Per-timestep rows: each panel is one AnyUp(feats[t], RGB[t]) that t1 averages, with
        # its guidance frame directly above so haze/cloud in a frame can be tied to its output.
        ncols = max(len(head), n_t)
        nrows = 3                                    # head row + guidance row + per-t output row
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.75 * nrows),
                                 squeeze=False)
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        for c, (img, title) in enumerate(head):
            ax = axes[0][c]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(img); ax.set_title(title, fontsize=8.5)
        axes[0][0].set_ylabel("time-pooled", fontsize=9.5)

        for c in range(n_t):
            g = g_per_t[c].cpu()
            # Guidance is ImageNet-standardized; min-max back to [0,1] just for display.
            g = (g - g.min()) / (g.max() - g.min() + 1e-6)
            ax = axes[1][c]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(g.permute(1, 2, 0).numpy())
            ax.set_title(f"guidance t={c}", fontsize=8)

            ax = axes[2][c]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(tail_rgbs[c])
            ax.set_title(f"AnyUp(feats[{c}], RGB[{c}])", fontsize=8)
        axes[1][0].set_ylabel("per-timestep guidance", fontsize=9.5)
        axes[2][0].set_ylabel("per-timestep output", fontsize=9.5)

        basis = ("one PCA basis fit on the anyup output, shared by every feature panel"
                 if args.pca == "shared" else "independent per-panel PCA")
        fig.suptitle(f"{args.split} #{r}  --  anyup vs anyup_t1 on {args.lr_dir}  "
                     f"(showing {n_t} of {T} timesteps; the mean uses all {T})\n{basis}",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_path = out_dir / f"{stem}_{args.split}{r}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        delta = (maps[0] - maps[1]).abs().mean().item()
        print(f"  wrote {out_path}   mean|anyup - anyup_t1| = {delta:.4f}")


if __name__ == "__main__":
    main()
