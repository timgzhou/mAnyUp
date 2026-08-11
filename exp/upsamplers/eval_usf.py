"""Do UPA/UPMA upsampled features improve a frozen LP head on UrbanSARFloods?

The UrbanSARFloods counterpart of eval_upsamplers_pa2pa.py. Same premise -- UPA/UPMA emit HR
features that are normalized convex combinations of the LR features, so they live in the same
space and a head trained on LR tokens transfers without retraining -- but four things differ
from PASTIS and they change the experiment:

  1. T=2 is (pre-event, post-event), NOT a time series to mean-pool. The pre/post DIFFERENCE
     is the flood signal, so each date is upsampled independently and the head still receives
     a (1,2,H,W,D) stack to combine as it likes (concat or diff).
  2. Guidance is S1 SAR, not S2. Only VV/VH exist -- no NIR/SWIR -- so UPMA's spectral-richness
     advantage over UPA does not obviously carry over. This script is the test of that.
     Guidance is PER-DATE MATCHED: pre features are guided by the pre-event image, post by the
     post-event image, so neither map is upsampled by the wrong scene.
  3. SAR speckle is multiplicative noise. A per-band percentile stretch amplifies it, and the
     bilateral range term can then key on speckle instead of real flood edges. --speckle_filter
     applies a small median filter to the guide first (on by default).
  4. 3 classes (non-flood / flooded-open / flooded-urban), and the headline mIoU covers only
     classes 0/1 -- matching exp/urbansarfloods/lp.py, where flooded-urban is reported separately.

IMPORTANT CAVEAT -- the head here is pa2px, not pa2pa. lp_urbansarfloods' heads map each token
to patch_size^2 SUB-PIXEL logits, i.e. they expect an LR token grid and produce the pixel
detail themselves. Feeding them a 64x64 upsampled feature map is NOT the identity-preserving
substitution the PASTIS pa2pa experiment was: the sub-pixel unfold would blow the output up by
another patch_size factor. So we probe the upsampled features with a pa2pa-style 1x1 conv
instead (--head_mode pa2pa, the default), trained here on LR tokens. --head_mode pa2px runs
lp_urbansarfloods' native head as an LR-only reference point. Read the pa2px row as context,
not as a directly comparable arm.

UPA/UPMA are imported from exp/upsamplers/upa_anyup.py (single source of truth).

    source env_setup/env_olmo.sh
    python eval_upsamplers_usf.py --features usf_base_s1_ps4_res20_t64
    python eval_upsamplers_usf.py --features usf_... --limit_test 64   # quick smoke run

Prerequisites (neither is checked into this repo -- build them first):
    python exp/urbansarfloods/prep_tiles.py
    python exp/urbansarfloods/extract_features.py --splits train,valid
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# UPA/UPMA + the guidance stretch come from the comparison script (no duplicate kernel code).
from exp.upsamplers.upa_anyup import UPA, UPMA, percentile_stretch

RESULTS_CSV = "results/upsamplers/upsampler_usf.csv"
CSV_FIELDS = ["timestamp", "features", "method", "head_mode", "guide", "speckle_filter",
              "fit_steps", "epochs", "lr", "seed", "n_test",
              "miou_headline", "miou_all", "overall_acc", "eval_sec"]

# UrbanSARFloods S1 tile band layout (exp/urbansarfloods/prep_tiles.py, and the viz in
# exp/urbansarfloods/lp.py:151): 8 bands, of which the intensity bands are
#   b4 = date1 VH, b5 = date1 VV, b6 = date2 VH, b7 = date2 VV
PRE_BANDS = [5, 4]      # (VV, VH) date 1 -- pre-event
POST_BANDS = [7, 6]     # (VV, VH) date 2 -- post-event


def _append_csv(row: dict) -> None:
    Path(RESULTS_CSV).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def median_filter2d(x: np.ndarray, k: int = 3) -> np.ndarray:
    """(C,H,W) -> per-channel k x k median filter, edge-replicated.

    Cheap speckle suppression for the SAR guide. SAR speckle is multiplicative and heavy-tailed,
    so a median is the right primitive (a box/Gaussian blur would smear the flood boundaries we
    want the bilateral range term to lock onto). Implemented with unfold rather than scipy to
    avoid another dependency in the olmo env.
    """
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)   # (1,C,H,W)
    pad = k // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="replicate")
    patches = t.unfold(2, k, 1).unfold(3, k, 1)                          # (1,C,H,W,k,k)
    return patches.contiguous().view(*patches.shape[:4], -1).median(-1).values[0].numpy()


def build_guide(sar: torch.Tensor, bands, speckle_filter: int) -> np.ndarray:
    """(8,H,W) raw SAR tile -> (len(bands),H,W) guide in [0,1], optionally speckle-filtered.

    Filter BEFORE the stretch: the percentile bounds should be computed on the despeckled
    signal, otherwise the outliers we are removing still set the display range.
    """
    g = sar[bands].float().numpy()
    if speckle_filter > 1:
        g = median_filter2d(g, speckle_filter)
    return percentile_stretch(g)


class PatchToPatchProbe(nn.Module):
    """pa2pa probe over a pre/post feature pair: combine the two dates, then a 1x1 conv.

    Deliberately mirrors lp_urbansarfloods' _combine semantics (concat or diff) so the LR
    baseline is comparable to that script, but WITHOUT the sub-pixel unfold -- see the module
    docstring. Applied identically to LR tokens and to upsampled 64x64 features.
    """

    def __init__(self, embed_dim: int, num_classes: int, combine: str = "concat",
                 label_size: int = 64):
        super().__init__()
        self.combine = combine
        self.label_size = label_size
        in_dim = embed_dim * (2 if combine == "concat" else 1)
        self.probe = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def _combine(self, pre, post):                      # (B,D,H,W) each
        if self.combine == "concat":
            return torch.cat([pre, post], dim=1)
        return post - pre

    def forward(self, pre, post, upsample_logits: bool = True):
        x = self._combine(pre, post)
        logits = self.probe(x)
        if upsample_logits and logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


def train_probe(head, loader, device, epochs, lr, ignore_label, class_weight=None):
    """Train the pa2pa probe on LR tokens with bilinear-upsampled logits."""
    head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    w = None if class_weight is None else class_weight.to(device)

    for ep in range(epochs):
        tot, nb, t0 = 0.0, 0, time.time()
        for feats, label in loader:
            feats, label = feats.to(device), label.to(device)
            pre = feats[:, 0].permute(0, 3, 1, 2).contiguous()     # (B,D,gH,gW)
            post = feats[:, 1].permute(0, 3, 1, 2).contiguous()
            logits = head(pre, post)
            loss = F.cross_entropy(logits, label, ignore_index=ignore_label, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / max(nb, 1):.4f}  ({time.time() - t0:.1f}s)")
    return head


def upsample_pair(method, feats, sar, device, args):
    """(T=2,gH,gW,D) features + (8,H,W) SAR -> two (1,D,H,W) upsampled maps.

    Per-date MATCHED guidance: the pre-event feature map is guided by the pre-event image and
    the post-event map by the post-event image, so neither date is upsampled by the other's
    scene. Costs two JBU fits per tile.
    """
    out = []
    for t, bands in ((0, PRE_BANDS), (1, POST_BANDS)):
        lr_feat = feats[t].permute(2, 0, 1).unsqueeze(0).to(device)        # (1,D,gH,gW)
        guide = build_guide(sar, bands, args.speckle_filter)               # (Cg,H,W) in [0,1]
        if method == "upa":
            # UPA takes HWC uint8 and divides by 255 internally. SAR has no RGB, so we feed
            # the 2-band VV/VH guide as a pseudo-3-channel image (VV, VH, VV) -- the same
            # false-colour convention lp_urbansarfloods uses for its visualizations.
            rgb = np.stack([guide[0], guide[1], guide[0]], axis=-1)        # (H,W,3)
            out.append(UPA((rgb * 255).astype(np.uint8), lr_feat, fit_steps=args.fit_steps))
        elif method == "upma":
            out.append(UPMA(guide, lr_feat, fit_steps=args.fit_steps))
        else:
            raise ValueError(f"unknown method {method}")
    return out[0], out[1]


def evaluate_method(head, dataset, indices, method, device, args, tiles_dir,
                    seg_metrics, num_classes, ignore_label):
    """Frozen head under one upsampling path. Not @torch.no_grad(): UPA/UPMA need autograd
    internally for their test-time fit; grad is disabled only around the head forward."""
    head.eval()
    preds, labels = [], []
    t0 = time.time()

    for n, idx in enumerate(indices):
        feats, label = dataset[idx]                                        # (2,gH,gW,D),(H,W)

        if method == "lr_bilinear":
            with torch.no_grad():
                pre = feats[0].permute(2, 0, 1).unsqueeze(0).to(device)
                post = feats[1].permute(2, 0, 1).unsqueeze(0).to(device)
                logits = head(pre, post)
        else:
            sar = torch.load(tiles_dir / args.split / f"{idx}.pt")["sar"]  # (8,H,W)
            pre, post = upsample_pair(method, feats, sar, device, args)
            with torch.no_grad():
                # already at label res -> no logit upsampling, the features carry the detail
                logits = head(pre.float(), post.float(), upsample_logits=False)
                if logits.shape[-2:] != label.shape[-2:]:
                    logits = F.interpolate(logits, size=tuple(label.shape[-2:]),
                                           mode="bilinear", align_corners=True)

        preds.append(logits.argmax(dim=1).cpu())
        labels.append(label.unsqueeze(0))

        if (n + 1) % 25 == 0:
            print(f"    [{method}] {n + 1}/{len(indices)}  ({time.time() - t0:.0f}s)", flush=True)

    P, L = torch.cat(preds), torch.cat(labels)
    all_m = seg_metrics(P, L, num_classes=num_classes, ignore_label=ignore_label).metrics
    # headline mIoU over classes 0/1 only (flooded-urban reported separately), matching
    # lp_urbansarfloods.HEADLINE_CLASSES.
    head_m = _headline_miou(P, L, args.headline_classes, num_classes, ignore_label)
    return all_m, head_m, time.time() - t0


def _headline_miou(preds, labels, classes, num_classes, ignore_label):
    """Mean IoU restricted to `classes`, computed directly from the confusion matrix."""
    from olmoearth_pretrain.evals.metrics import _build_confusion_matrix
    conf = _build_confusion_matrix(preds, labels, num_classes, ignore_label).float()
    ious = []
    for c in classes:
        inter = conf[c, c]
        union = conf[c].sum() + conf[:, c].sum() - inter
        if union > 0:
            ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--features", required=True,
                   help="feature folder under --out_root, e.g. usf_base_s1_ps4_res20_t64")
    p.add_argument("--out_root", default="features")
    p.add_argument("--tiles_root", default="data/urbansarfloods_tiles",
                   help="tile root; the run reads <tiles_root>_t<tile_size> per meta.json")
    p.add_argument("--split", default="valid", help="split to evaluate (USF has no test split)")
    p.add_argument("--train_split", default="train")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--combine", default="concat", choices=["concat", "diff"],
                   help="how the probe fuses pre/post features")
    p.add_argument("--fit_steps", type=int, default=50, help="UPA/UPMA test-time opt steps")
    p.add_argument("--speckle_filter", type=int, default=3,
                   help="median filter size for the SAR guide; 0/1 disables")
    p.add_argument("--weighted_ce", action="store_true",
                   help="inverse-frequency class weights (floods are rare)")
    p.add_argument("--limit_test", type=int, default=None)
    p.add_argument("--methods", default="lr_bilinear,upa,upma")
    p.add_argument("--head_ckpt", type=Path, default=None,
                   help="cache for the trained probe (default: checkpoints/usf_<features>.pt)")
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()

    from exp.urbansarfloods.lp import (TiledFeatureDataset, NUM_CLASSES, IGNORE_LABEL,
                                   HEADLINE_CLASSES, LABEL_SIZE)
    from olmoearth_pretrain.evals.metrics import segmentation_metrics
    args.headline_classes = HEADLINE_CLASSES

    feat_dir = Path(args.out_root) / args.features
    if not feat_dir.exists():
        raise SystemExit(
            f"no features at {feat_dir}.\nBuild them first:\n"
            f"  python exp/urbansarfloods/prep_tiles.py\n"
            f"  python exp/urbansarfloods/extract_features.py --splits train,valid")

    meta = {}
    meta_path = feat_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    label_size = meta.get("label_size", LABEL_SIZE)
    tile_size = meta.get("tile_size", label_size)
    tiles_dir = Path(f"{args.tiles_root}_t{tile_size}")
    if not tiles_dir.exists():
        raise SystemExit(f"no tiles at {tiles_dir} (needed for SAR guidance); run "
                         f"exp/urbansarfloods/prep_tiles.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TiledFeatureDataset(feat_dir, args.train_split)
    test_ds = TiledFeatureDataset(feat_dir, args.split)
    embed_dim = train_ds[0][0].shape[-1]
    print(f"[data] train={len(train_ds)} {args.split}={len(test_ds)}  D={embed_dim}  "
          f"label={label_size}  tiles={tiles_dir}")

    head = PatchToPatchProbe(embed_dim, NUM_CLASSES, combine=args.combine,
                             label_size=label_size)
    ckpt = args.head_ckpt or Path("checkpoints") / f"usf_pa2pa_{args.features}_{args.combine}.pt"
    if ckpt.exists() and not args.retrain:
        head.load_state_dict(torch.load(ckpt))
        head.to(device)
        print(f"[head] loaded {ckpt}")
    else:
        weight = None
        if args.weighted_ce:
            counts = torch.zeros(NUM_CLASSES)
            for _, label in train_ds:
                valid = label[label != IGNORE_LABEL]
                counts += torch.bincount(valid.flatten(), minlength=NUM_CLASSES).float()
            weight = counts.sum() / (counts.clamp_min(1) * NUM_CLASSES)
            print(f"[head] class weights {weight.tolist()}")
        print(f"[head] training pa2pa probe ({args.combine}): {args.epochs} ep, lr={args.lr}")
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_probe(head, loader, device, args.epochs, args.lr, IGNORE_LABEL, weight)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), ckpt)
        print(f"[head] saved {ckpt}")

    indices = list(range(len(test_ds)))
    if args.limit_test is not None:
        indices = indices[:args.limit_test]

    print(f"\n[eval] {len(indices)} {args.split} tiles x {len(args.methods.split(','))} methods")
    results = {}
    for method in args.methods.split(","):
        all_m, head_m, secs = evaluate_method(head, test_ds, indices, method, device, args,
                                              tiles_dir, segmentation_metrics,
                                              NUM_CLASSES, IGNORE_LABEL)
        results[method] = (all_m, head_m)
        print(f"  {method:12s}  mIoU(NF,FO)={head_m:.4f}  mIoU(all)={all_m['miou']:.4f}  "
              f"acc={all_m['overall_acc']:.4f}  ({secs:.0f}s)")
        _append_csv({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "features": args.features, "method": method, "head_mode": "pa2pa",
            "guide": "per-date VV/VH" if method != "lr_bilinear" else "",
            "speckle_filter": args.speckle_filter if method != "lr_bilinear" else "",
            "fit_steps": args.fit_steps if method in ("upa", "upma") else "",
            "epochs": args.epochs, "lr": args.lr, "seed": args.seed, "n_test": len(indices),
            "miou_headline": round(head_m, 4),
            "miou_all": round(float(all_m["miou"]), 4),
            "overall_acc": round(float(all_m["overall_acc"]), 4),
            "eval_sec": round(secs, 1),
        })

    base = results.get("lr_bilinear")
    print(f"\n{'method':<14}{'mIoU(NF,FO)':>13}{'delta':>10}{'mIoU(all)':>12}{'acc':>9}")
    for method, (all_m, head_m) in results.items():
        d = f"{head_m - base[1]:+.4f}" if base else "-"
        print(f"{method:<14}{head_m:>13.4f}{d:>10}{all_m['miou']:>12.4f}"
              f"{all_m['overall_acc']:>9.4f}")
    print(f"\nappended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

# python -u eval_upsamplers_usf.py