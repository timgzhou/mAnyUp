"""Last-layer feature maps + predictions for every ep64 FFT checkpoint of the
ps x image_size sweep (ps8/ps16 at img64 and img128).

One figure per test sample, laid out as (n_models + 1) rows:
  row 0            input RGB | ground truth
  row 1..n_models  that model's last-layer feature map | its prediction
Rows are ordered by patch_size, then image_size.

The feature map is the encoder output the segmentation head consumes -- (gH, gW, D) tokens,
projected to RGB by a PCA fit ONCE on the finest-grid model of each figure and reused for
every row, so colors are comparable down the column instead of each panel having its own
arbitrary basis.

The two preps cover the SAME ground: prepare_data.py --image_size 64 quadrant-splits each
128x128 PASTIS patch, so 128-sample i == 64-samples 4i..4i+3. Verified empirically that the
quadrant order is column-major (TL, BL, TR, BR). We therefore pick 128-sample indices, render
them whole, and crop each 64px model's four quadrant predictions back into one 128x128 mosaic
-- so every panel shows identical terrain and the comparison is apples-to-apples.

Run in the OlmoEarth venv:
    source env_setup/env_olmo.sh
    python -u -m exp.pastis.viz_ps_sweep --samples 4
"""
import argparse
import glob
import os
import re
from pathlib import Path
from typing import cast

from exp.common import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()  # before any olmoearth_pretrain.evals import

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402
from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG, TaskType  # noqa: E402
from olmoearth_pretrain.evals.datasets.pastis_dataset import PASTISRDataset  # noqa: E402
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn  # noqa: E402

from exp.pastis import finetune_olmoearth as FT  # noqa: E402
from exp.pastis.visualize import CLASSES, CMAP, compute_metrics  # noqa: E402
from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402

SPLITS = {64: "data/pastis_olmoearth", 128: "data/pastis128_olmoearth"}
# 64px quadrant order within a 128px patch, verified against targets.pt (column-major).
QUAD_ORDER = ["TL", "BL", "TR", "BR"]
QUAD_SLICE = {"TL": (slice(0, 64), slice(0, 64)),   "TR": (slice(0, 64), slice(64, 128)),
              "BL": (slice(64, 128), slice(0, 64)), "BR": (slice(64, 128), slice(64, 128))}


def infer_run(ckpt: str) -> dict:
    """Parse patch_size / image_size / model_size out of a run_name checkpoint."""
    name = os.path.basename(ckpt)
    # longest key first, so "_v1_2_base_" is not read as v1 "_base_"
    size = next((s for s in sorted(MODEL_SIZE_TO_ID, key=len, reverse=True) if f"_{s}_" in name),
                "base")
    m = re.search(r"_p(\d+)[_.]", name)
    ps = int(m.group(1)) if m else 4
    # run_name appends _img<N> for non-64 preps (see exp/common/config.py).
    mi = re.search(r"_img(\d+)_", name)
    img = int(mi.group(1)) if mi else 64
    return {"ckpt": ckpt, "patch_size": ps, "image_size": img, "model_size": size,
            "label": f"ps{ps} img{img}", "grid": img // ps}


def load_rgb128(idx: int) -> np.ndarray:
    """Natural-color RGB for a 128px test sample, time-averaged. 13-band L1C: B04/B03/B02."""
    s2 = torch.load(Path(SPLITS[128]) / "pastis_r_test" / "s2_images" / f"{idx}.pt")
    rgb = s2.float().mean(0)[[3, 2, 1]].permute(1, 2, 0).numpy()
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)   # stretch for visibility
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def build_model(run: dict, task_config, device, encoders: dict) -> nn.Module:
    """Rebuild the FFT head for one run and load its finetuned weights."""
    size = run["model_size"]
    if size not in encoders:
        m = load_model_from_id(getattr(ModelID, MODEL_SIZE_TO_ID[size]), load_weights=True)
        encoders[size] = cast(nn.Module, m.encoder if hasattr(m, "encoder") else m)
    FT.HEAD = "lp"
    FT.DATA_SPLITS = SPLITS[run["image_size"]]
    FT.INPUT_MODALITIES = ["sentinel2_l2a", "sentinel1"]
    ft = FT.build_head("lp", encoders[size], run["patch_size"], task_config).to(device)
    # Lazy-init the probe (in_dim depends on embed dim) before load_state_dict.
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        FT._forward_logits(ft, sample_batch(run["image_size"], 0), device,
                           task_config, run["patch_size"])
    ft.load_state_dict(torch.load(run["ckpt"], map_location=device))
    ft.eval()
    return ft


_DS_CACHE: dict[int, PASTISRDataset] = {}


def sample_batch(image_size: int, idx: int):
    """One test sample from the prep matching image_size, collated as the head expects."""
    if image_size not in _DS_CACHE:
        _DS_CACHE[image_size] = PASTISRDataset(
            path_to_splits=Path(SPLITS[image_size]), split="test",
            norm_stats_from_pretrained=True,
            input_modalities=["sentinel2_l2a", "sentinel1"])
    return eval_collate_fn([_DS_CACHE[image_size][idx]])


@torch.no_grad()
def _encode(ft, batch, device) -> torch.Tensor:
    """Last-layer encoder features for one batch -> (gH, gW, D).

    This is exactly the tensor BackboneWithHead.forward feeds to its linear head
    (`emb, labels = self.wrapper(...)`), i.e. the final representation the probe sees."""
    masked, label = batch
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        emb, _ = ft.wrapper(FT.to_device(masked, device), label.to(device), is_train=False)
    return cast(torch.Tensor, emb)[0].float().cpu()


@torch.no_grad()
def predict_128(ft, run: dict, idx128: int, task_config, device) -> tuple:
    """Prediction, ground truth, and last-layer features for 128-sample idx128.

    pred/gt are 128x128; features are (G, G, D) on that model's token grid over the
    full 128px patch. img128 models see the patch whole. img64 models are run on the
    four constituent 64px samples and their outputs re-assembled -- which is exactly
    how those models would be applied to this ground."""
    ps = run["patch_size"]
    if run["image_size"] == 128:
        batch = sample_batch(128, idx128)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, label = FT._forward_logits(ft, batch, device, task_config, ps)
        return (logits.argmax(1)[0].cpu().numpy(), label[0].cpu().numpy(),
                _encode(ft, batch, device).numpy())

    pred = np.zeros((128, 128), dtype=np.int64)
    gt = np.zeros((128, 128), dtype=np.int64)
    g = 64 // ps                      # tokens per 64px quadrant side
    feat = None
    for k, quad in enumerate(QUAD_ORDER):
        batch = sample_batch(64, 4 * idx128 + k)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, label = FT._forward_logits(ft, batch, device, task_config, ps)
        r, c = QUAD_SLICE[quad]
        pred[r, c] = logits.argmax(1)[0].cpu().numpy()
        gt[r, c] = label[0].cpu().numpy()
        fq = _encode(ft, batch, device).numpy()          # (g, g, D)
        if feat is None:
            feat = np.zeros((2 * g, 2 * g, fq.shape[-1]), dtype=fq.dtype)
        # same quadrant placement as the pixels, on the token grid
        gr = slice(0, g) if r.start == 0 else slice(g, 2 * g)
        gc = slice(0, g) if c.start == 0 else slice(g, 2 * g)
        feat[gr, gc] = fq
    return pred, gt, feat


def fit_pca(feat: np.ndarray):
    """Fit a 3-component PCA on (H, W, D) tokens; returns (mean, components) to reuse."""
    x = feat.reshape(-1, feat.shape[-1])
    mu = x.mean(0)
    # economy SVD on the centered tokens: rows of Vt are the principal directions.
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:3]


def feat_rgb(feat: np.ndarray, basis) -> np.ndarray:
    """Project (H, W, D) features onto a shared 3-component basis -> (H, W, 3) in [0,1].

    Percentile-stretched per figure-column basis (not per panel) so that two rows sharing
    the basis are directly comparable."""
    mu, comps = basis
    x = (feat.reshape(-1, feat.shape[-1]) - mu) @ comps.T
    lo, hi = np.percentile(x, 2, axis=0), np.percentile(x, 98, axis=0)
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return x.reshape(*feat.shape[:2], 3)


def draw_grid(ax, cells: int, quads: bool = False) -> None:
    """Overlay the encoder token grid (cells per 128px side)."""
    for k in range(1, cells):
        x = k * 128 / cells - 0.5
        ax.axvline(x, color="white", lw=0.4, alpha=0.5)
        ax.axhline(x, color="white", lw=0.4, alpha=0.5)
    if quads:  # mark the 64px tile seams the img64 models cannot see across
        ax.axvline(63.5, color="red", lw=1.4, alpha=0.9)
        ax.axhline(63.5, color="red", lw=1.4, alpha=0.9)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--indices", default="", help="comma-separated 128px test indices")
    p.add_argument("--ckpt_glob", default="checkpoints/*_ep64_best.pt")
    p.add_argument("--out_dir", default="results/pastis/predictions/ps_sweep")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_config = DATASET_TO_CONFIG["pastis"]
    assert task_config.task_type == TaskType.SEGMENTATION

    runs = [infer_run(c) for c in sorted(glob.glob(args.ckpt_glob))]
    if not runs:
        raise SystemExit(f"no checkpoints matching {args.ckpt_glob}")
    # rows ordered by patch size, then image size
    runs.sort(key=lambda r: (r["patch_size"], r["image_size"]))
    print("checkpoints:")
    for r in runs:
        print(f"  {r['label']:<12} grid {r['grid']}x{r['grid']}  {os.path.basename(r['ckpt'])}")

    if args.indices:
        idxs = [int(x) for x in args.indices.split(",")]
    else:
        n = len(torch.load(Path(SPLITS[128]) / "pastis_r_test" / "targets.pt"))
        idxs = np.linspace(0, n - 1, args.samples).astype(int).tolist()
    print("test indices (128px):", idxs)

    os.makedirs(args.out_dir, exist_ok=True)
    encoders: dict[str, nn.Module] = {}
    # Build every model once, then sweep samples (model construction dominates runtime).
    models = []
    for r in runs:
        print(f"loading {r['label']} ...", flush=True)
        models.append((r, build_model(r, task_config, device, encoders)))

    for idx in idxs:
        rgb = load_rgb128(idx)
        panels, gt_ref = [], None
        for r, ft in models:
            pred, gt, feat = predict_128(ft, r, idx, task_config, device)
            gt_ref = gt if gt_ref is None else gt_ref
            panels.append((r, pred, feat, compute_metrics(pred, gt)))

        # One PCA basis per figure, fit on the finest token grid (most tokens = best
        # estimate of the principal directions) and reused for every row, so feature
        # colors mean the same thing down the column.
        finest = max(panels, key=lambda p: p[2].shape[0])
        basis = fit_pca(finest[2])

        nrow = 1 + len(panels)
        fig, axes = plt.subplots(nrow, 2, figsize=(8.6, 4.3 * nrow),
                                 squeeze=False)

        axes[0][0].imshow(rgb)
        axes[0][0].set_title("Input (RGB, time-mean)", fontsize=11)
        axes[0][1].imshow(gt_ref, cmap=CMAP, vmin=0, vmax=19, interpolation="nearest")
        axes[0][1].set_title("Ground truth", fontsize=11)

        for row, (r, pred, feat, met) in enumerate(panels, start=1):
            axf, axp = axes[row]
            axf.imshow(feat_rgb(feat, basis), interpolation="nearest")
            axf.set_title(f"{r['label']}  last-layer features "
                          f"({feat.shape[0]}x{feat.shape[1]} tokens, D={feat.shape[-1]})",
                          fontsize=10)
            axp.imshow(pred, cmap=CMAP, vmin=0, vmax=19, interpolation="nearest")
            draw_grid(axp, r["grid"], quads=(r["image_size"] == 64))
            axp.set_title(f"{r['label']}  prediction   "
                          f"mIoU {met[3]:.3f}  acc {met[0]:.3f}", fontsize=10)
            # tile seams on the feature panel too, in token units
            if r["image_size"] == 64:
                h = feat.shape[0] / 2 - 0.5
                axf.axvline(h, color="red", lw=1.4, alpha=0.9)
                axf.axhline(h, color="red", lw=1.4, alpha=0.9)

        for axrow in axes:
            for ax in axrow:
                ax.axis("off")

        present = np.unique(np.concatenate(
            [gt_ref.flatten()] + [p.flatten() for _, p, _, _ in panels])).astype(int)
        handles = [mpatches.Patch(color=CMAP(i), label=f"{i}: {CLASSES[i]}")
                   for i in present if 0 <= i <= 19]
        fig.legend(handles=handles, bbox_to_anchor=(1.005, 0.995), loc="upper left",
                   fontsize=8)
        fig.suptitle(f"PASTIS test sample {idx} (128x128)  -  rows by patch size, then "
                     f"image size\nred seams = 64px tile boundaries the img64 models "
                     f"cannot attend across", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out = os.path.join(args.out_dir, f"test{idx:04d}_ps_sweep.png")
        fig.savefig(out, bbox_inches="tight", dpi=115)
        plt.close(fig)
        print(f"saved {out}   " + "  ".join(
            f"{r['label']}={m[3]:.3f}" for r, _, _, m in panels))


if __name__ == "__main__":
    main()
