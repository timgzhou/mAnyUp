"""Linear-probe flood segmentation on FROZEN OlmoEarth features (GEOID-Flood).

Change-detection framing: each cached
sub-tile holds per-timestep features (T=2, gH, gW, D) for the pre-event (t0) and post-event
(t1) S1 acquisition. The head combines them -- concat [pre,post] or diff (post-pre) -- and
a 1x1 conv maps each token to patch_size^2 sub-pixel logits, which unfold to label
resolution. The backbone stays frozen; only the head trains.

WHY A LINEAR PROBE FOR THIS QUESTION. The study asks what the frozen representation
contains at each (patch_size, tile_size), so the head must stay too weak to compensate for
a bad representation. A linear per-pixel probe is the standard instrument for that.

CLASSES (see exp/geoidflood/prep_tiles.py for the label remap):
    0 background, 1 permanent water, 2 flood; -1 = ignore (outside the CEMS-mapped area).
HEADLINE METRIC is mIoU over {background, flood} -- classes 0 and 2. Permanent water
(class 1) is reported separately rather than folded in, because it is NOT a change signal:
it is dark in BOTH timesteps, so a pre/post model can score it from a single date. Mixing
it into the headline would let a config that is merely good at "is this water" outrank one
that is actually good at "did this become water", which is the question the dataset poses.

CRITICAL: the probe passes ignore_index=-1, so unmapped pixels contribute to neither the
loss nor the confusion matrix. Dropping that would score predictions against labels the
CEMS analysts never drew.

Run (OlmoEarth venv):
    python -u -m exp.geoidflood.lp --features geoid_base_s1_ps8_res10_t128 --weighted_ce
"""
import os
import sys

from exp.common import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.metrics import segmentation_metrics, _build_confusion_matrix
from exp.common.paths import FEATURES

SCHEDULER_MIN_LR = 1e-6
NUM_CLASSES = 3                 # 0 background, 1 permanent water, 2 flood
IGNORE_LABEL = -1
CLASS_NAMES = ["BG", "PW", "FL"]
CLASS_COLORS = ["#f2f2f2", "#1f77ff", "#e4002b"]
HEADLINE_CLASSES = [0, 2]       # background + flood; permanent water reported separately


class TiledFeatureDataset(torch.utils.data.Dataset):
    """Loads cached {"feat": (T,gH,gW,D) fp16, "label": (tile,tile) int16} sub-tiles."""

    def __init__(self, feat_dir: Path, split: str):
        self.dir = feat_dir / split
        self.n = len(list(self.dir.glob("*.pt")))
        if self.n == 0:
            raise FileNotFoundError(f"no feature tiles in {self.dir}")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rec = torch.load(self.dir / f"{i}.pt")
        return rec["feat"].float(), rec["label"].long()


class _SiamesePerPixelHead(nn.Module):
    """Base Siamese per-pixel (pa2px) head: combine pre/post, 1x1 conv to C*p^2 sub-pixel
    logits, unfold to full resolution, interpolate to the label size if needed."""

    in_mult = 1

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int, label_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.label_size = label_size
        self.probe = nn.Conv2d(self.in_mult * embed_dim,
                               num_classes * patch_size * patch_size, kernel_size=1)

    def _combine(self, pre, post):
        raise NotImplementedError

    def forward(self, feats: torch.Tensor) -> torch.Tensor:   # (B,T=2,gH,gW,D)
        pre, post = feats[:, 0], feats[:, 1]
        x = self._combine(pre, post).permute(0, 3, 1, 2).contiguous()
        logits = self.probe(x)
        p = self.patch_size
        logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                           c=self.num_classes, i=p, j=p)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


class ConcatPrePostPerPixelHead(_SiamesePerPixelHead):
    """concat: [pre, post] -> 2D. Learns from both dates jointly."""
    in_mult = 2

    def _combine(self, pre, post):
        return torch.cat([pre, post], dim=-1)


class DiffPrePostPerPixelHead(_SiamesePerPixelHead):
    """diff: post - pre -> D. Signed per-channel change; the pure change-detection framing."""
    in_mult = 1

    def _combine(self, pre, post):
        return post - pre


HEADS = {"concat": ConcatPrePostPerPixelHead, "diff": DiffPrePostPerPixelHead}


def build_head(name, embed_dim, num_classes, patch_size, label_size):
    if name not in HEADS:
        raise ValueError(f"head={name!r} not in {list(HEADS)}")
    return HEADS[name](embed_dim, num_classes, patch_size, label_size)


def per_class_stats(preds, labels):
    """Per-class IoU / precision / recall / F1. segmentation_metrics only returns the
    aggregate mIoU, so the per-class values are recomputed from the confusion matrix."""
    conf = _build_confusion_matrix(preds, labels, NUM_CLASSES, IGNORE_LABEL)
    tp = conf.diagonal().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    union = tp + fp + fn
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {"iou": tp / (union + 1e-8), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall + 1e-8),
            "union": union, "support": tp + fn}


def _fmt_per_class(stats) -> str:
    out = []
    for c in range(NUM_CLASSES):
        if stats["support"][c] == 0:
            out.append(f"    {CLASS_NAMES[c]}: (absent)")
        else:
            out.append(f"    {CLASS_NAMES[c]}: IoU={stats['iou'][c]:.4f} "
                       f"P={stats['precision'][c]:.4f} R={stats['recall'][c]:.4f} "
                       f"F1={stats['f1'][c]:.4f}")
    return "\n".join(out)


@torch.no_grad()
def evaluate(head, loader, device):
    head.eval()
    preds, labels = [], []
    for feats, label in loader:
        preds.append(head(feats.to(device)).argmax(1).cpu())
        labels.append(label)
    preds, labels = torch.cat(preds), torch.cat(labels)
    res = segmentation_metrics(preds, labels, num_classes=NUM_CLASSES,
                               ignore_label=IGNORE_LABEL)
    stats = per_class_stats(preds, labels)
    present = [c for c in HEADLINE_CLASSES if stats["union"][c] > 0]
    headline = (float(sum(stats["iou"][c] for c in present) / len(present))
                if present else float("nan"))
    pw_iou = float(stats["iou"][1]) if stats["union"][1] > 0 else None
    return res, headline, pw_iou, stats


def append_results_csv(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def build_result_row(args, meta, headline, pw_iou, res_metrics, stats) -> dict:
    row = {
        "features": args.features, "head": args.head, "weighted_ce": args.weighted_ce,
        "tile_size": meta.get("tile_size"), "patch_size": meta.get("patch_size"),
        "input_res": meta.get("input_res"),
        "metres_per_token": meta.get("metres_per_token"),
        "grid": meta.get("grid"),
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "miou_BG_FL": round(headline, 4),
        "iou_PW": (round(pw_iou, 4) if pw_iou is not None else ""),
        "overall_acc": round(res_metrics.get("overall_acc", float("nan")), 4),
        "macro_f1": round(res_metrics.get("macro_f1", float("nan")), 4),
    }
    for c, name in enumerate(CLASS_NAMES):
        present = stats["support"][c] > 0
        for k in ("iou", "precision", "recall", "f1"):
            row[f"{name}_{k}"] = (round(float(stats[k][c]), 4) if present else "")
    return row


@torch.no_grad()
def visualize_first_val(head, feat_dir: Path, tiles_root: Path, device, out_path: str,
                        n: int = 3):
    """Prediction vs GT for the first few val sub-tiles, next to the pre/post S1 VV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.4 * n), squeeze=False)
    for r in range(n):
        f = feat_dir / "val" / f"{r}.pt"
        t = tiles_root / "val" / f"{r}.pt"
        if not f.exists() or not t.exists():
            # Say WHICH input is missing rather than emitting a silently blank panel --
            # a blank figure otherwise reads as "the model predicted nothing".
            for c in range(4):
                axes[r][c].axis("off")
            axes[r][0].text(0.5, 0.5, f"missing: {'features' if not f.exists() else ''}"
                                      f"{' and ' if not f.exists() and not t.exists() else ''}"
                                      f"{'raw tiles' if not t.exists() else ''}\n"
                                      f"(idx {r})", ha="center", va="center",
                            fontsize=11, color="#b00")
            print(f"  !! viz: missing input for val idx {r} "
                  f"(feat={f.exists()}, tile={t.exists()})")
            continue
        rec = torch.load(f)
        pred = head(rec["feat"].float().unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        gt = rec["label"].long().numpy()
        s1 = torch.load(t)["s1"].float().numpy()          # (T=2,C=2,H,W) dB

        def g(x):
            lo, hi = np.percentile(x, [2, 98])
            return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

        axes[r][0].imshow(g(s1[0, 0]), cmap="gray"); axes[r][0].set_title("pre VV (dB)")
        axes[r][1].imshow(g(s1[1, 0]), cmap="gray"); axes[r][1].set_title("post VV (dB)")
        # ignore (-1) is rendered as background grey but is excluded from every metric
        axes[r][2].imshow(np.where(gt < 0, 0, gt), cmap=cmap, vmin=0, vmax=2,
                          interpolation="nearest"); axes[r][2].set_title("ground truth")
        axes[r][3].imshow(pred, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        axes[r][3].set_title("prediction")
        for c in range(4):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    handles = [Patch(color=CLASS_COLORS[i], label=f"{i}: {CLASS_NAMES[i]}") for i in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(Path(out_path).stem, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"Saved prediction visualization to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="LP flood seg on frozen OlmoEarth features (GEOID).")
    p.add_argument("--features", required=True,
                   help="folder under --out_root, e.g. geoid_base_s1_ps8_res10_t128")
    p.add_argument("--out_root", default=str(FEATURES))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head", default="concat", choices=list(HEADS))
    p.add_argument("--weighted_ce", action="store_true",
                   help="inverse-frequency class weights in CE. Flood is a small minority "
                        "of pixels even after flood-only chip filtering, so without this "
                        "the probe collapses toward all-background.")
    p.add_argument("--tiles_root", default="data/geoidflood_tiles")
    p.add_argument("--viz_out", default=None)
    p.add_argument("--results_csv", default="results/geoidflood/lp.csv")
    args = p.parse_args()

    if args.viz_out is None:
        wc = "_wce" if args.weighted_ce else ""
        args.viz_out = f"results/geoidflood/{args.features}_{args.head}{wc}.png"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = Path(args.out_root) / args.features
    meta = json.loads((feat_dir / "meta.json").read_text())
    embed_dim, patch_size = meta["embed_dim"], meta["patch_size"]
    label_size = meta.get("label_size", meta["tile_size"])
    tile_size = meta["tile_size"]
    assert meta["timesteps"] == 2, f"expected T=2 (pre/post), got {meta['timesteps']}"
    print(f"Features: {args.features} | shape {meta['feature_shape']} | "
          f"{meta.get('metres_per_token')} m/token | label_size {label_size} | head={args.head}")

    def loader(split, shuffle):
        return DataLoader(TiledFeatureDataset(feat_dir, split), batch_size=args.batch_size,
                          num_workers=args.num_workers, shuffle=shuffle,
                          pin_memory=device.type == "cuda")

    train_loader, val_loader = loader("train", True), loader("val", False)

    head = build_head(args.head, embed_dim, NUM_CLASSES, patch_size, label_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=SCHEDULER_MIN_LR)

    weight = None
    if args.weighted_ce:
        counts = torch.zeros(NUM_CLASSES)
        for _, label in train_loader:
            valid = label[label != IGNORE_LABEL]
            counts += torch.bincount(valid.flatten(), minlength=NUM_CLASSES).float()
        freq = counts / counts.sum().clamp(min=1)
        inv = 1.0 / freq.clamp(min=1e-8)
        weight = (inv / inv.mean()).to(device)
        print(f"weighted CE: pixel counts {counts.tolist()} -> "
              f"weights {[round(w, 3) for w in weight.tolist()]}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, weight=weight)

    # GEOID has a real held-out test split, but this probe reports VAL. We train a fixed
    # number of epochs with a schedule-based lr (no metric coupling) and report the FINAL
    # epoch, so no epoch selection happens on the reported numbers either way.
    final = None
    checked_finite = False
    for epoch in range(args.epochs):
        head.train()
        losses = []
        for feats, label in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}",
                                 leave=False):
            feats, label = feats.to(device), label.to(device)
            if not checked_finite:
                assert torch.isfinite(feats).all(), "non-finite features -- re-run extract"
                checked_finite = True
            loss = loss_fn(head(feats), label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        res, headline, pw, stats = evaluate(head, val_loader, device)
        final = (headline, pw, res.metrics, stats)
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {sum(losses)/max(len(losses),1):.4f} "
              f"| val mIoU(BG,FL) {headline:.4f} | PW IoU "
              f"{pw if pw is None else round(pw, 4)}")
        print(_fmt_per_class(stats))

    headline, pw, metrics, stats = final
    print(f"\nFINAL val mIoU(BG,FL) {headline:.4f} | PW IoU "
          f"{pw if pw is None else round(pw, 4)} | {metrics}")
    print("Per-class (BG/PW/FL):")
    print(_fmt_per_class(stats))

    append_results_csv(args.results_csv,
                       build_result_row(args, meta, headline, pw, metrics, stats))
    print(f"Appended results to {args.results_csv}")

    visualize_first_val(head, feat_dir, Path(f"{args.tiles_root}_t{tile_size}"), device,
                        args.viz_out)


if __name__ == "__main__":
    main()
