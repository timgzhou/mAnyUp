"""One figure per row of the mAnyUp comparison table: five methods, features + predictions.

Each column is a route from the SAME low-res features to a 64x64 segmentation, so the figure
answers "what does each method's feature map look like, and what does its head predict from it":

    bilinear+px2px  LR feats bilinear-upsampled to label res, 1x1 per-pixel probe
                    (the AnyUp paper's protocol)
    pa2px           LR feats, Conv2d(D -> C*p^2) unfolded into sub-pixels; no upsampling
    td0+own         frozen mAnyUp (transform_depth=0) + its own pa2px head
    td2+own         frozen mAnyUp (transform_depth=2) + its own pa2px head
    oracle          the REAL target-resolution features + pa2px (upper bound)

Row 1 shows each route's feature map under ONE shared PCA basis (fit on the oracle features)
so colours are comparable across columns; row 2 shows the predictions under the PASTIS tab20
palette with per-sample mIoU in the title. Every head is LOADED from a --save_head file, so
the panels correspond to real logged runs rather than probes refitted here.

    source env_setup/env_olmo.sh
    python -u -m exp.pastis.viz_table_row --lr_ps 16 --hr_ps 4 --sample 0 1 2 3
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp.pastis.lp_cached_features import (
    CachedFeatureDataset, build_cached_head, IGNORE_LABEL, NUM_CLASSES,
)
from exp.pastis.viz_manyup_lp import _pca_rgb_shared, _raw_rgb, _miou, CMAP

HEADS = Path("checkpoints/lp_heads")


def _probe_from(path: Path):
    """Rebuild a Conv2d probe from a --save_head file (works for both px2px and pa2px: the
    output-channel count carries the sub-pixel factor)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["head_state"]
    w, b = sd["probe.weight"], sd["probe.bias"]
    conv = torch.nn.Conv2d(w.shape[1], w.shape[0], 1)
    conv.load_state_dict({"weight": w, "bias": b})
    return conv.eval()


def _unfold(logits, num_classes, label_size):
    """(B, C*q*q, g, g) -> (B, C, g*q, g*q), then resize if that is not label res."""
    q = int(round((logits.shape[1] / num_classes) ** 0.5))
    if q > 1:
        logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                           c=num_classes, i=q, j=q)
    if logits.shape[-2:] != (label_size, label_size):
        logits = F.interpolate(logits, size=(label_size, label_size),
                               mode="bilinear", align_corners=True)
    return logits



def _grid(ax, n_cells: int, extent_px: int) -> None:
    """Draw very light lines on the NATIVE token boundaries of a panel.

    extent_px is the panel's displayed pixel size and n_cells the real token count, so a
    display-resampled panel (pa2px shown at the oracle size) still rules its true 4x4 grid.
    Skipped when the cells would be denser than a few pixels -- at 64x64 the lines would
    swamp the image rather than inform it.
    """
    if n_cells < 2 or extent_px / n_cells < 3:
        return
    step = extent_px / n_cells
    for k in range(1, n_cells):
        ax.axhline(k * step - 0.5, color="white", lw=0.4, alpha=0.35)
        ax.axvline(k * step - 0.5, color="white", lw=0.4, alpha=0.35)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr_ps", type=int, required=True, help="source patch size (16 or 8)")
    p.add_argument("--hr_ps", type=int, required=True, help="target patch size (4 or 2)")
    p.add_argument("--mods", default="s2")
    p.add_argument("--sample", type=int, nargs="+", default=[0])
    p.add_argument("--out", default="results/pastis/viz_table")
    p.add_argument("--out_root", default=None)
    p.add_argument("--data_splits", default="data/pastis_olmoearth")
    args = p.parse_args()

    root = Path(args.out_root or (Path.home() / "projects/aip-gpleiss/timz/features"))
    splits = Path(args.data_splits)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_cfg = f"oe_base_{args.mods}_ps{args.lr_ps}_tile64"
    hr_cfg = f"oe_base_{args.mods}_ps{args.hr_ps}_tile64"
    embed = json.loads((root / lr_cfg / "meta.json").read_text())["embed_dim"]

    # max_ram_gb=0 -> per-sample disk reads; we only index a handful of samples.
    mk = lambda cfg, g, mod: CachedFeatureDataset(       # noqa: E731
        root / cfg, splits, "test", guidance=g, max_ram_gb=0, reduce_time=True,
        time_pool="mean", guidance_mod=mod)
    lr_ds = mk(lr_cfg, "mean13", args.mods)              # guidance needed by the mAnyUp heads
    hr_ds = mk(hr_cfg, "none", args.mods)
    label_size = lr_ds.label_size

    # Frozen mAnyUp heads, each with the pa2px probe its own logged run trained.
    mu = {}
    for td in (0, 2):
        ck = (f"checkpoints/manyup/manyup_{td}transform_{lr_cfg}_to_{hr_cfg}"
              f"_manyup_w1_dr0_ep31.pth")
        hd = HEADS / f"lphead_manyup_{td}transform_{lr_cfg}_to_{hr_cfg}_manyup_w1_dr0_ep31.pth"
        if not (Path(ck).exists() and hd.exists()):
            print(f"skip td{td}: missing {'ckpt' if not Path(ck).exists() else 'head'}")
            continue
        h = build_cached_head("manyup", embed, NUM_CLASSES, args.lr_ps, manyup_ckpt=ck,
                              manyup_native_out=True, label_size=label_size).to(dev)
        h.load_state_dict(torch.load(hd, map_location="cpu",
                                     weights_only=False)["head_state"], strict=False)
        mu[td] = h.eval()

    bu = _probe_from(HEADS / f"lphead_{lr_cfg}_lp_bu_px2px.pth").to(dev)
    pa = _probe_from(HEADS / f"lphead_{lr_cfg}_lp_pa2px.pth").to(dev)
    orc = _probe_from(HEADS / f"lphead_{hr_cfg}_lp_pa2px.pth").to(dev)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in args.sample:
        feats, label, guid = lr_ds[idx]
        hr_feats = hr_ds[idx][0]
        f_b = feats.unsqueeze(0).to(dev)
        g_b = guid.unsqueeze(0).to(dev)
        lr2d = feats.float().mean(0).permute(2, 0, 1)                 # (D,gh,gw)
        hr2d = hr_feats.float().mean(0).permute(2, 0, 1)              # (D,GH,GW)

        # --- features per column -------------------------------------------------------
        bu_feat = F.interpolate(lr2d.unsqueeze(0), size=(label_size, label_size),
                                mode="bilinear", align_corners=True)[0]
        # pa2px probes the raw token grid; show it at the oracle's size (nearest, no
        # interpolation) so the feature row is visually comparable column to column.
        lr_disp = F.interpolate(lr2d.unsqueeze(0), size=hr2d.shape[-2:], mode="nearest")[0]
        gh = lr2d.shape[-1]
        cols = [("bilinear+px2px", bu_feat, label_size),
                ("pa2px (LR tokens)", lr_disp, gh)]
        for td in (0, 2):
            if td in mu:
                fm = mu[td].features(f_b, g_b)[0].cpu()
                cols.append((f"td{td}+own", fm, fm.shape[-1]))
        cols.append(("oracle (real HR)", hr2d, hr2d.shape[-1]))

        # --- predictions per column ----------------------------------------------------
        preds = {}
        preds["bilinear+px2px"] = _unfold(bu(bu_feat.unsqueeze(0).to(dev)),
                                          NUM_CLASSES, label_size).argmax(1)[0].cpu()
        preds["pa2px (LR tokens)"] = _unfold(pa(lr2d.unsqueeze(0).to(dev)),
                                             NUM_CLASSES, label_size).argmax(1)[0].cpu()
        for td in (0, 2):
            if td in mu:
                preds[f"td{td}+own"] = mu[td](f_b, g_b).argmax(1)[0].cpu()
        preds["oracle (real HR)"] = _unfold(orc(hr2d.unsqueeze(0).to(dev)),
                                            NUM_CLASSES, label_size).argmax(1)[0].cpu()

        # Shared PCA basis fit on the ORACLE features: every feature panel is then in the
        # same colour space as the target the upsamplers are chasing.
        rgbs = _pca_rgb_shared(hr2d, [c[1] for c in cols])
        lab = label.long()

        n = len(cols) + 1                       # +1 for the raw RGB / ground-truth column
        fig, axes = plt.subplots(2, n, figsize=(n * 2.7, 7.0))
        fig.subplots_adjust(hspace=0.34)
        axes[0][0].imshow(_raw_rgb(guid)); axes[0][0].set_title("raw RGB", fontsize=9)
        axes[1][0].imshow(np.where(lab.numpy() < 0, 19, lab.numpy()), cmap=CMAP,
                          vmin=0, vmax=19, interpolation="nearest")
        axes[1][0].set_title("ground truth", fontsize=9)
        for j, ((name, fmap, ncell), rgb) in enumerate(zip(cols, rgbs), start=1):
            axes[0][j].imshow(rgb)
            _grid(axes[0][j], ncell, rgb.shape[0])
            axes[0][j].set_title(f"{name}\n{ncell}x{ncell}", fontsize=9)
            pr = preds[name]
            axes[1][j].imshow(np.where(pr.numpy() < 0, 19, pr.numpy()), cmap=CMAP,
                              vmin=0, vmax=19, interpolation="nearest")
            _grid(axes[1][j], ncell, label_size)
            axes[1][j].set_title(f"{name}\nmIoU {_miou(pr, lab, NUM_CLASSES, IGNORE_LABEL):.3f}",
                                 fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{args.mods} ps{args.lr_ps} -> ps{args.hr_ps} "
                     f"({(64 // args.hr_ps) // (64 // args.lr_ps)}x)  --  test sample {idx}  --  "
                     f"features (shared PCA on oracle) / predictions (PASTIS tab20)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.0)
        out = out_dir / f"row_{args.mods}_ps{args.lr_ps}_to_ps{args.hr_ps}_s{idx}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
