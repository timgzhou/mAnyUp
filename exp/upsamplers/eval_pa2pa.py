"""Does UPA/UPMA/AnyUp upsampling improve a pa2pa head that was trained on LR features?

The premise: UPA and UPMA output HR features that are normalized convex combinations of LR
features (weights come from a softmax-like normalization, num_s/den_s, so they sum to 1). The
upsampled features therefore live in the SAME space as the LR features they were built from --
no linear probe retraining is needed. A 1x1-conv head trained on the LR token grid can be
applied directly to the 64x64 upsampled map. AnyUp is included on the same assumption (it is
also an attention-weighted combination of LR features).

So we train ONE head on the raw LR features, then evaluate that same frozen head four ways:

    lr_bilinear  -- baseline: probe the LR tokens, bilinear-upsample the LOGITS (= lp_pa2pa_bu)
    upa          -- upsample FEATURES with UPA (RGB guidance), then probe per-pixel
    upma         -- upsample FEATURES with UPMA (multispectral guidance), then probe per-pixel
    anyup        -- upsample FEATURES with pretrained AnyUp, then probe per-pixel

Every path shares the same head weights, so any mIoU difference is attributable to the
upsampler alone.

NOTE ON THE BASELINE: for a purely LINEAR probe, "upsample features then probe" and "probe then
upsample logits" are the same operation when the upsampler is bilinear (1x1 conv and bilinear
interpolation commute -- they act on disjoint axes). That is exactly why lr_bilinear is the
right control: it isolates what the GUIDED, edge-aware upsamplers add over the unguided
smoothness prior, rather than measuring the trivial gain of any upsampling at all.

There is no saved pa2pa checkpoint in checkpoints/ -- exp/pastis/lp_cached_features.py trains heads and
appends metrics to CSV without ever calling torch.save. So this script trains the head itself
(same architecture, optimizer and schedule as lp_pa2pa_bu) and caches it to --head_ckpt for
reuse on later runs.

UPA/UPMA are imported from exp/upsamplers/upa_anyup.py (single source of truth -- no duplicated
kernel code).

    source env_setup/env_olmo.sh
    python eval_upsamplers_pa2pa.py                       # train head, eval all 4 paths
    python eval_upsamplers_pa2pa.py --limit_test 64       # quick smoke run
    python eval_upsamplers_pa2pa.py --time_pool median    # median guidance composite

GUIDANCE TIME POOLING: the upsamplers need a single guidance image, so the S2 series is collapsed
over T -- by mean (default, historical) or median (--time_pool median), which rejects the cloud
and shadow frames a mean smears into the composite. The FEATURE side is always mean-pooled
regardless, because the frozen head was trained that way; only guidance changes.
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

# UPA/UPMA + guidance normalization come from the comparison script (no duplicate kernel code).
from exp.upsamplers.upa_anyup import (UPA, UPMA, percentile_stretch, time_pool, TIME_POOLS,
                                     RGB_BANDS, SURFACE_BANDS)

RESULTS_CSV = "results/upsamplers/upsampler_pa2pa.csv"
CSV_FIELDS = ["timestamp", "features", "method", "guide_bands", "time_pool", "fit_steps",
              "epochs", "lr", "seed", "n_test", "test_miou", "test_overall_acc", "eval_sec"]


def _append_csv(row: dict) -> None:
    new = not Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def build_head(embed_dim: int, num_classes: int) -> nn.Module:
    """The pa2pa probe: a single 1x1 conv D->C. Identical to LPPatchToPatchBU.probe -- kept as
    a bare conv here because we apply it at BOTH the LR token grid and the 64x64 upsampled
    grid, and the label-res interpolation differs per eval path."""
    return nn.Conv2d(embed_dim, num_classes, kernel_size=1)


def train_head(head, train_loader, device, epochs: int, lr: float, label_size: int,
               ignore_label: int):
    """Train the 1x1 probe on LR tokens with bilinear-upsampled logits (the lp_pa2pa_bu
    objective). Mirrors lp_on_cached_features: AdamW + cosine annealing to 1e-6."""
    head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for ep in range(epochs):
        tot, nb = 0.0, 0
        t0 = time.time()
        for feats, label, _ in train_loader:
            feats, label = feats.to(device), label.to(device)
            x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()   # (B,D,gH,gW)
            logits = head(x)                                          # (B,C,gH,gW)
            logits = F.interpolate(logits, size=(label_size, label_size),
                                   mode="bilinear", align_corners=True)
            loss = F.cross_entropy(logits, label, ignore_index=ignore_label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / max(nb, 1):.4f}  ({time.time() - t0:.1f}s)")
    return head


def _upsample_features(method, feats_btgwd, s2_raw, device, fit_steps, guide_bands, anyup_model,
                       tpool="mean"):
    """(1,T,gH,gW,D) cached features + (T,13,64,64) raw S2 -> (1,D,64,64) upsampled features.

    Features are always MEAN-pooled over time -- that is how the head was trained, so changing it
    would invalidate the frozen probe. Only the GUIDANCE composite honours `tpool`, so a
    mean-vs-median difference here is attributable to guidance quality alone.

    Returns None for the bilinear baseline (handled in the logits domain instead).
    """
    lr_feat = feats_btgwd.mean(dim=1).permute(0, 3, 1, 2).contiguous().to(device)  # (1,D,gH,gW)
    s2 = time_pool(s2_raw.float(), tpool)                              # (13,64,64)

    if method == "upa":
        # percentile-stretched display RGB, round-tripped through uint8 (UPA divides by 255)
        rgb = s2[RGB_BANDS].numpy()                                    # (3,64,64)
        rgb = percentile_stretch(rgb).transpose(1, 2, 0)               # (64,64,3) in [0,1]
        return UPA((rgb * 255).astype(np.uint8), lr_feat, fit_steps=fit_steps)

    if method == "upma":
        guide = percentile_stretch(s2[guide_bands].numpy())            # (Cg,64,64)
        return UPMA(guide, lr_feat, fit_steps=fit_steps)

    if method == "anyup":
        # AnyUp expects its own normalization: per-channel min-max then ImageNet standardize.
        from exp.upsamplers.upa_anyup import _norm_rgb
        guide = _norm_rgb(s2[RGB_BANDS])                               # (3,64,64)
        with torch.no_grad():
            return anyup_model(guide.unsqueeze(0).to(device), lr_feat, output_size=(64, 64))

    raise ValueError(f"unknown method {method}")


def evaluate_method(head, dataset, indices, method, device, args, seg_metrics,
                    num_classes, ignore_label, anyup_model=None):
    """Apply the frozen head under one upsampling path; return segmentation metrics.

    UPA/UPMA run a per-image test-time optimization, so this is a strict per-sample loop
    (batching would fit one JBU to a batch-mean guide, which is not the method).

    Deliberately NOT decorated with @torch.no_grad(): UPA/UPMA need autograd internally to fit
    their JBU parameters. Grad is disabled only around the head forward, where it is safe.
    """
    head.eval()
    preds, labels = [], []
    t0 = time.time()

    for n, idx in enumerate(indices):
        feats, label, _ = dataset[idx]
        feats = feats.unsqueeze(0)                                     # (1,T,gH,gW,D)

        if method == "lr_bilinear":
            with torch.no_grad():
                x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous().to(device)
                logits = head(x)                                       # (1,C,gH,gW)
                logits = F.interpolate(logits, size=(args.label_size, args.label_size),
                                       mode="bilinear", align_corners=True)
        else:
            s2_raw = torch.load(args.data_splits / f"pastis_r_{args.split}" /
                                "s2_images" / f"{idx}.pt")
            hr = _upsample_features(method, feats, s2_raw, device, args.fit_steps,
                                    args.guide_band_idx, anyup_model,
                                    tpool=args.time_pool)              # (1,D,64,64)
            with torch.no_grad():
                logits = head(hr.float())                              # (1,C,64,64)

        preds.append(logits.argmax(dim=1).cpu())
        labels.append(label.unsqueeze(0))

        if (n + 1) % 25 == 0:
            print(f"    [{method}] {n + 1}/{len(indices)}  ({time.time() - t0:.0f}s)", flush=True)

    r = seg_metrics(torch.cat(preds), torch.cat(labels),
                    num_classes=num_classes, ignore_label=ignore_label)
    return r.metrics, time.time() - t0        # {'miou','overall_acc','macro_acc','macro_f1'}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--features_root", type=Path,
                   default=Path("~/projects/aip-gpleiss/timz/features").expanduser())
    p.add_argument("--features", default="oe_base_s2_ps4_tile64",
                   help="LR feature config the head is trained on and the upsamplers consume")
    p.add_argument("--split", default="test")
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit_steps", type=int, default=50, help="UPA/UPMA test-time opt steps")
    p.add_argument("--guide_bands", default="surface", help="UPMA bands: surface|all|csv list")
    p.add_argument("--time_pool", default="mean", choices=list(TIME_POOLS),
                   help="how the S2 time series is collapsed into the guidance image for "
                        "upa/upma/anyup. Features are always mean-pooled (the head was trained "
                        "that way); this changes guidance only. median rejects cloud/shadow "
                        "frames that a mean smears into the composite.")
    p.add_argument("--limit_test", type=int, default=None,
                   help="evaluate only the first N test images (UPA/UPMA are ~seconds each)")
    p.add_argument("--methods", default="lr_bilinear,upa,upma,anyup")
    p.add_argument("--head_ckpt", type=Path, default=Path("checkpoints/pa2pa_head.pt"),
                   help="cache for the trained probe; reused if present")
    p.add_argument("--retrain", action="store_true", help="ignore any cached head")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    repo = args.repo
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()

    # lp_on_cached_features owns the dataset + metric; import after the shim like it does.
    from exp.pastis import finetune_olmoearth as fmod
    args.data_splits = repo / "data" / "pastis_olmoearth"
    fmod.DATA_SPLITS = str(args.data_splits)          # dataset guidance loaders read this global
    from exp.pastis.lp_cached_features import (CachedFeatureDataset, NUM_CLASSES,
                                       IGNORE_LABEL, LABEL_SIZE)
    from olmoearth_pretrain.evals.metrics import segmentation_metrics
    args.label_size = LABEL_SIZE

    if args.guide_bands == "surface":
        args.guide_band_idx = SURFACE_BANDS
    elif args.guide_bands == "all":
        args.guide_band_idx = list(range(13))
    else:
        args.guide_band_idx = [int(b) for b in args.guide_bands.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = args.features_root / args.features

    # guidance="none": we load raw S2 ourselves per method, so the dataset needn't build any.
    train_ds = CachedFeatureDataset(feat_dir, args.data_splits, "train",
                                    guidance="none", reduce_time=True)
    test_ds = CachedFeatureDataset(feat_dir, args.data_splits, args.split,
                                   guidance="none", reduce_time=True)
    embed_dim = train_ds[0][0].shape[-1]
    print(f"[data] train={len(train_ds)} {args.split}={len(test_ds)}  D={embed_dim}")

    # ---- head: load the cached probe, else train it (nothing in checkpoints/ saves one) ----
    head = build_head(embed_dim, NUM_CLASSES)
    if args.head_ckpt.exists() and not args.retrain:
        head.load_state_dict(torch.load(args.head_ckpt))
        head.to(device)
        print(f"[head] loaded {args.head_ckpt}")
    else:
        print(f"[head] training pa2pa probe: {args.epochs} epochs, lr={args.lr}")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, drop_last=False)
        train_head(head, train_loader, device, args.epochs, args.lr,
                   LABEL_SIZE, IGNORE_LABEL)
        args.head_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), args.head_ckpt)
        print(f"[head] saved {args.head_ckpt}")

    indices = list(range(len(test_ds)))
    if args.limit_test is not None:
        indices = indices[:args.limit_test]

    methods = args.methods.split(",")
    anyup_model = None
    if "anyup" in methods:
        anyup_model = torch.hub.load("wimmerth/anyup", "anyup_multi_backbone",
                                     use_natten=False, pretrained=True).to(device).eval()

    print(f"\n[eval] {len(indices)} {args.split} images x {len(methods)} methods")
    results = {}
    for method in methods:
        m, secs = evaluate_method(head, test_ds, indices, method, device, args,
                                  segmentation_metrics, NUM_CLASSES, IGNORE_LABEL,
                                  anyup_model=anyup_model)
        results[method] = m
        print(f"  {method:12s}  mIoU={m['miou']:.4f}  acc={m['overall_acc']:.4f}  "
              f"({secs:.0f}s)")
        _append_csv({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "features": args.features,
            "method": method,
            "guide_bands": args.guide_bands if method == "upma" else "",
            # lr_bilinear never touches the guidance image, so time_pool is not a variable for it.
            "time_pool": args.time_pool if method != "lr_bilinear" else "",
            "fit_steps": args.fit_steps if method in ("upa", "upma") else "",
            "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
            "n_test": len(indices),
            "test_miou": round(float(m["miou"]), 4),
            "test_overall_acc": round(float(m["overall_acc"]), 4),
            "eval_sec": round(secs, 1),
        })

    # ---- summary, deltas relative to the bilinear control ----
    print(f"\n{'method':<14}{'mIoU':>9}{'d mIoU':>10}{'acc':>9}{'d acc':>10}")
    base = results.get("lr_bilinear")
    for method, m in results.items():
        if base is not None:
            d_iou = f"{m['miou'] - base['miou']:+.4f}"
            d_acc = f"{m['overall_acc'] - base['overall_acc']:+.4f}"
        else:
            d_iou = d_acc = "-"
        print(f"{method:<14}{m['miou']:>9.4f}{d_iou:>10}"
              f"{m['overall_acc']:>9.4f}{d_acc:>10}")
    print(f"\nappended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
