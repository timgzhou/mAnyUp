"""Compare LR features, guided upsamplings of them, and the native HR features.

ONE FIGURE PER SAMPLE (--out is a template: upsample_compare.png -> upsample_compare_test0.png,
_test1.png, ...), each a 3x3 grid:

    raw RGB       | LR (8x8)                  | HR (64x64, native)
                  | AnyUp(LR)  RGB guidance   | mAnyUp(LR) trained on this pair
                  | UPA(LR)    RGB guidance   | UPMA(LR)   multispectral guidance

Row 2 is the LEARNED upsamplers (feed-forward, no per-image fitting); row 3 is the TEST-TIME
OPTIMIZED ones (--fit_steps of optimization per image). Pairing them this way puts the
comparison that matters inside a row -- pretrained vs trained-on-this-pair, and RGB vs
multispectral guidance -- while HR stays in view as a fixed reference.

Each upsampler takes the LR feature map plus a guidance image and produces a 64x64 feature map,
so the interesting question is how close any of them gets to the independently extracted HR map
on the right. They differ in guidance: AnyUp and UPA use 3-band RGB, UPMA uses the multispectral
stack (--guide_bands, default the 10 surface bands), mAnyUp uses all 13 bands ("mean13").

mAnyUp (--manyup_ckpt, or --manyup to auto-discover) is the one upsampler TRAINED on this
LR->HR pair, so unlike the others it has seen HR-like targets -- expect it to track the HR panel
more closely, and read that as "it learned the mapping", not as a like-for-like win over the
zero-shot methods. Its checkpoint fixes the HR grid it upsamples to, so pair it with the --hr_dir
it trained against.

Upsampling reuses exp.upsamplers.eval_pa2pa._upsample_features so the guidance normalization is
bit-identical to the eval path (UPA gets percentile-stretched uint8 RGB; UPMA gets
percentile-stretched multispectral; AnyUp gets its own min-max + ImageNet standardization). No
duplicated preprocessing.

COLOR: every feature panel in a row shares ONE PCA basis, fit on the LR map and applied to the
upsampled and HR maps (pca_rgb_shared). Identical feature vectors therefore get identical
colors, so differences you see between panels are real rather than a per-panel recoloring. The
HR features come from a separate extraction, so sharing the LR basis with them is a deliberate
choice -- use --pca independent to give every panel its own basis instead.

    source env_setup/env_olmo.sh
    python -u -m exp.viz.viz_upsample_compare --n_images 4
    python -u -m exp.viz.viz_upsample_compare --lr_dir oe_base_s2_ps16_tile64 --n_images 2
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_root", default=FEATURES_ROOT)
    ap.add_argument("--data_splits", default=DATA_SPLITS)
    ap.add_argument("--lr_dir", default="oe_base_s2_ps8_tile64",
                    help="low-res INPUT feature dir the upsamplers consume")
    ap.add_argument("--hr_dir", default="oe_base_s2_ps1_tile64",
                    help="native high-res feature dir shown as the rightmost reference")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_images", type=int, default=4)
    ap.add_argument("--fit_steps", type=int, default=50,
                    help="UPA/UPMA test-time optimization steps (~seconds per image each)")
    ap.add_argument("--guide_bands", default="surface",
                    help="UPMA guidance bands: surface|all|comma-separated indices "
                         "(same semantics as eval_pa2pa)")
    ap.add_argument("--manyup", action="store_true",
                    help="add a mAnyUp panel, auto-discovering the latest checkpoint trained to "
                         "upsample --lr_dir (under --manyup_root)")
    ap.add_argument("--manyup_ckpt", default=None,
                    help="explicit mAnyUp .pth to visualize (implies --manyup)")
    ap.add_argument("--manyup_root", default="checkpoints/manyup",
                    help="root scanned for <lr_dir>__to__*/*.pth when --manyup is set")
    ap.add_argument("--manyup_use_proj", action=argparse.BooleanOptionalAction, default=True,
                    help="include the trained projector in the frozen mAnyUp pipeline")
    ap.add_argument("--time_pool", default="mean", choices=("mean", "median"),
                    help="how the S2 series is collapsed into the guidance image")
    ap.add_argument("--pca", default="shared", choices=("shared", "independent"),
                    help="shared: one PCA basis (fit on LR) for every feature panel in a row. "
                         "independent: each panel gets its own basis (max contrast, no "
                         "cross-panel color meaning).")
    ap.add_argument("--out", default="feature_viz/upsample_compare.png",
                    help="output TEMPLATE: one file per sample is written as "
                         "<stem>_<split><idx>.png next to this path")
    args = ap.parse_args()

    # Imported here, not at module top: pulls olmo bootstrap + torch.hub AnyUp, both heavy.
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()
    from exp.upsamplers.eval_pa2pa import _upsample_features
    from exp.pastis.finetune_olmoearth import AnyUpUpsampleProbe
    from exp.upsamplers.upa_anyup import SURFACE_BANDS

    # Same --guide_bands semantics as eval_pa2pa: surface | all | explicit indices.
    if args.guide_bands == "surface":
        guide_band_idx = SURFACE_BANDS
    elif args.guide_bands == "all":
        guide_band_idx = list(range(13))
    else:
        guide_band_idx = [int(b) for b in args.guide_bands.split(",")]

    root = Path(args.features_root).expanduser().resolve()
    data_splits = Path(args.data_splits).expanduser().resolve()
    lr_d, hr_d = root / args.lr_dir, root / args.hr_dir
    for d in (lr_d, hr_d):
        if not (d / f"pastis_r_{args.split}").exists():
            raise SystemExit(f"ERROR: {d} has no pastis_r_{args.split}/ (not extracted?)")
    if not data_splits.exists():
        raise SystemExit(f"ERROR: --data_splits not found: {data_splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  LR={args.lr_dir}  HR={args.hr_dir}  pca={args.pca}")
    anyup_model = AnyUpUpsampleProbe(num_classes=1).anyup.to(device).eval()

    # --- optional mAnyUp: the one upsampler TRAINED on this LR->HR pair ---
    manyup = None
    if args.manyup or args.manyup_ckpt:
        from exp.pastis.lp_cached_features import (CachedManyUp, _discover_manyup_ckpts,
                                                   _load_s2_guidance)
        ckpt = args.manyup_ckpt
        if ckpt is None:
            found = _discover_manyup_ckpts(Path(args.manyup_root), args.lr_dir)
            if not found:
                raise SystemExit(
                    f"ERROR: no mAnyUp checkpoint for LR={args.lr_dir} under {args.manyup_root} "
                    f"(expected {args.lr_dir}__to__*/*.pth). Train one with train_manyup.sh.")
            ckpt = str(found[-1])       # latest by epoch/mtime
        print(f"mAnyUp checkpoint: {ckpt}")
        # embed_dim from the LR cache itself; num_classes is irrelevant (we only call features()).
        probe_dim = torch.load(lr_d / f"pastis_r_{args.split}" / "0.pt").shape[-1]
        manyup = CachedManyUp(probe_dim, 1, 1, ckpt_path=ckpt,
                              use_proj=args.manyup_use_proj, label_size=64,
                              time_pool=args.time_pool).to(device).eval()
        # mAnyUp's guidance is the 13-band S2 composite ("mean13"), NOT the RGB the other
        # upsamplers use, and _load_s2_guidance reads it from finetune_olmoearth.DATA_SPLITS.
        from exp.pastis import finetune_olmoearth as fmod
        fmod.DATA_SPLITS = str(data_splits)

    # One figure per sample, 3 columns x 3 rows:
    #   row 0  context     : raw RGB | LR (what every upsampler starts from) | HR (the target)
    #   row 1  learned     : AnyUp | mAnyUp        -- feed-forward nets, no per-image fitting
    #   row 2  optimized   : UPA   | UPMA          -- test-time optimization, --fit_steps each
    # Grouping by row puts the two methods of a kind side by side, so the comparison that
    # matters (RGB vs multispectral guidance; pretrained vs trained-on-this-pair) is within a
    # row, while HR stays visible in the same figure as a fixed reference.
    out_tmpl = Path(args.out)
    out_dir = out_tmpl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_tmpl.stem

    for r in range(args.n_images):
        s2 = torch.load(s2_path(data_splits, args.split, r))          # (T,13,64,64)
        lr = torch.load(lr_d / f"pastis_r_{args.split}" / f"{r}.pt")  # (T,gH,gW,D)
        hr = torch.load(hr_d / f"pastis_r_{args.split}" / f"{r}.pt")  # (T,64,64,D)

        # _upsample_features wants (1,T,gH,gW,D) and returns (1,D,64,64).
        feats = lr.float().unsqueeze(0)
        ups = {}
        for method in ("anyup", "upa", "upma"):
            out = _upsample_features(method, feats, s2, device, args.fit_steps,
                                     guide_band_idx, anyup_model, tpool=args.time_pool)
            ups[method] = out.detach().float().squeeze(0).permute(1, 2, 0).cpu()  # (64,64,D)
        if manyup is not None:
            # 13-band guidance, batched to (1,13,64,64); features() returns (1,D,64,64).
            g13 = _load_s2_guidance(args.split, r, tpool=args.time_pool).unsqueeze(0).to(device)
            with torch.no_grad():
                m_hr = manyup.features(feats.to(device), g13)
            ups["manyup"] = m_hr.float().squeeze(0).permute(1, 2, 0).cpu()

        lr_map = lr.float().mean(0)                                   # (gH,gW,D)
        hr_map = hr.float().mean(0)                                   # (64,64,D)

        # PCA over EVERY feature panel at once, so colors stay comparable across the figure.
        keys = ["lr", "hr", "anyup", "upa", "upma"] + (["manyup"] if manyup is not None else [])
        pool = {"lr": lr_map, "hr": hr_map, **ups}
        if args.pca == "shared":
            fitted = pca_rgb_shared(lr_map, [pool[k] for k in keys])
        else:
            fitted = [pca_rgb(pool[k]) for k in keys]
        rgb_of = {k: nearest_resize(v, DISPLAY_PX) for k, v in zip(keys, fitted)}

        lr_lbl = args.lr_dir.split("_", 3)[-1]
        hr_lbl = args.hr_dir.split("_", 3)[-1]
        # (image, title) per cell; None leaves the cell blank (e.g. mAnyUp without --manyup).
        grid = [
            [(raw_rgb(s2.float().mean(0)), "raw RGB (mean T)"),
             (rgb_of["lr"], f"LR  {lr_lbl}"),
             (rgb_of["hr"], f"HR (native)  {hr_lbl}")],
            [(rgb_of["anyup"], "AnyUp(LR)  RGB guidance"),
             (rgb_of["manyup"], "mAnyUp(LR)  trained on this pair")
             if manyup is not None else None,
             None],
            [(rgb_of["upa"], f"UPA(LR)  RGB, {args.fit_steps} steps"),
             (rgb_of["upma"], f"UPMA(LR)  {args.guide_bands} bands, {args.fit_steps} steps"),
             None],
        ]
        row_labels = ["input / target", "learned", "test-time optimized"]

        # 3x3: row 0 fills all three columns (raw | LR | HR); the upsampler rows fill the first
        # two and leave column 2 blank, so every row's panels start at the same left edge.
        fig, axes = plt.subplots(3, 3, figsize=(3 * 2.9, 3 * 3.0), squeeze=False)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                ax = axes[i][j]
                ax.set_xticks([]); ax.set_yticks([])
                if cell is None:
                    ax.axis("off")
                    continue
                img, title = cell
                ax.imshow(img)
                ax.set_title(title, fontsize=8.5)
            axes[i][0].set_ylabel(row_labels[i], fontsize=9.5, labelpad=6)

        basis = ("one PCA basis fit on LR, shared by every feature panel"
                 if args.pca == "shared" else "independent per-panel PCA")
        fig.suptitle(f"{args.split} #{r}  --  {args.lr_dir} upsampled to {args.hr_dir}\n"
                     f"{basis}; guidance = {args.time_pool}-pooled S2", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out_path = out_dir / f"{stem}_{args.split}{r}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}  (LR {tuple(lr_map.shape[:2])} -> "
              f"{tuple(hr_map.shape[:2])})")


if __name__ == "__main__":
    main()
