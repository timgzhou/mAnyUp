"""Side-by-side viz of what a mAnyUp LP run actually predicts, vs the LR and HR routes.

The training viz (train_manyup.save_epoch_viz) shows the FEATURE maps -- raw RGB, LR feats,
mAnyUp output, HR target -- which answers "did the upsampler reconstruct the HR features".
It cannot answer the question the LP runs are actually asking: does that reconstruction turn
into better SEGMENTATION. This module puts both rows in one figure:

    row 1 (features, shared PCA basis fit on HR so the panels are colour-comparable)
        raw RGB | LR feats (gh x gw) | mAnyUp output | HR feats
    row 2 (predictions, shared PASTIS tab20 palette)
        ground truth | LR + pa2px | mAnyUp + probe | HR + pa2px

Reading it: column 2 is the route the mAnyUp arm is trying to beat (probe the coarse features
directly), column 4 is the oracle it is trying to match (probe the real fine features), and
column 3 is what mAnyUp actually delivers. A mAnyUp panel that looks like column 4 in row 1
but like column 2 in row 2 is the signature of a reconstruction that does not survive the
probe -- exactly the outcome the KNN/LP numbers have been hinting at.

The LR/HR reference probes are trained here (a few epochs of 1x1 conv on cached features --
seconds, since the features are already in RAM), because the LP run itself only ever trains
the mAnyUp probe. Per-panel mIoU is printed in each title so the picture and the number agree.
"""
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 20-class PASTIS scheme, same colormap as exp/pastis/visualize.py so figures are consistent.
CMAP = plt.get_cmap("tab20", 20)


def _pca_rgb_shared(fit_chw, maps_chw):
    """Fit PCA (top-3 dirs + min-max) on `fit_chw` (C,H,W); apply the SAME basis to every map.

    Mirrors train_manyup._pca_rgb_shared so this figure's feature row is directly comparable
    to the upsampler's own training viz.
    """
    C = fit_chw.shape[0]
    xf = fit_chw.reshape(C, -1).T.float().cpu().numpy()
    mean = xf.mean(0, keepdims=True)
    cov = ((xf - mean).T @ (xf - mean)) / max(xf.shape[0] - 1, 1)
    _, evecs = np.linalg.eigh(cov)
    dirs = evecs[:, -3:][:, ::-1]
    proj_fit = (xf - mean) @ dirs
    lo, hi = proj_fit.min(0), proj_fit.max(0)
    out = []
    for m in maps_chw:
        H, W = m.shape[-2:]
        x = m.reshape(C, -1).T.float().cpu().numpy()
        proj = ((x - mean) @ dirs).reshape(H, W, 3)
        out.append(np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1))
    return out


def _raw_rgb(guid_chw):
    """(C,64,64) guidance -> display RGB. 13/15-band stacks index B04/B03/B02; 2-band S1
    (VV/VH) has no colour bands, so show VV/VH/VV."""
    c = guid_chw.shape[0]
    bands = [3, 2, 1] if c >= 13 else [0, 1, 0]
    rgb = guid_chw[bands].float().cpu().numpy().transpose(1, 2, 0)
    lo = np.percentile(rgb, 2, (0, 1))
    hi = np.percentile(rgb, 98, (0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def _miou(pred, label, num_classes, ignore_label):
    """Mean IoU over classes present in either pred or label, ignoring void pixels."""
    keep = label != ignore_label
    p, t = pred[keep], label[keep]
    ious = []
    for c in range(num_classes):
        pi, ti = p == c, t == c
        union = (pi | ti).sum().item()
        if union:
            ious.append((pi & ti).sum().item() / union)
    return float(np.mean(ious)) if ious else float("nan")


@torch.enable_grad()
def _train_ref_probe(feats_grid, loader_feats, loader_labels, num_classes, patch_size,
                     label_size, ignore_label, device, epochs=8, lr=0.01):
    """Train a pa2px reference probe (1x1 conv D -> C*p^2, unfolded to sub-pixels) on cached
    features already resident in RAM. This is the LR / HR baseline route the mAnyUp arm is
    being compared against, trained the same way lp_pa2px trains.

    @enable_grad because save_lp_viz is @no_grad (everything else it does is inference) --
    without it the very first loss.backward() here raises "does not require grad".
    """
    from einops import rearrange
    D = loader_feats.shape[-1]
    p = patch_size
    probe = torch.nn.Conv2d(D, num_classes * p * p, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
    n = loader_feats.shape[0]
    bs = 32
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = loader_feats[idx].to(device).float().mean(1).permute(0, 3, 1, 2)
            logits = rearrange(probe(x), "b (c i j) gh gw -> b c (gh i) (gw j)",
                               c=num_classes, i=p, j=p)
            if logits.shape[-2:] != (label_size, label_size):
                logits = F.interpolate(logits, size=(label_size, label_size),
                                       mode="bilinear", align_corners=True)
            loss = lossf(logits, loader_labels[idx].to(device).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


@torch.no_grad()
def _probe_predict(probe, feats_1, num_classes, patch_size, label_size, device):
    from einops import rearrange
    x = feats_1.to(device).float().mean(0, keepdim=True).permute(0, 3, 1, 2)
    logits = rearrange(probe(x), "b (c i j) gh gw -> b c (gh i) (gw j)",
                       c=num_classes, i=patch_size, j=patch_size)
    if logits.shape[-2:] != (label_size, label_size):
        logits = F.interpolate(logits, size=(label_size, label_size),
                               mode="bilinear", align_corners=True)
    return logits.argmax(1)[0].cpu()


@torch.no_grad()
def save_lp_viz(head, test_ds, hr_ds, device, out_path, *, sample_idx=0, run_tag="",
                num_classes=20, ignore_label=-1, label_size=64, hr_patch=4,
                lr_patch=16, ref_epochs=8, train_ds=None, hr_train_ds=None,
                lr_probe=None, hr_probe=None):
    """Write the 2x4 figure for one TEST sample. `head` is the trained mAnyUp LP head;
    `hr_ds` supplies the real fine-patch features for the oracle column (None -> that
    column is skipped and the figure is 2x3)."""
    feats, label, guid = test_ds[sample_idx]
    lr_grid = feats.shape[1]

    # --- mAnyUp route: the frozen upsampled feature map + the trained probe's prediction ---
    f_b = feats.unsqueeze(0).to(device)
    g_b = None if guid.numel() == 0 else guid.unsqueeze(0).to(device)
    mu_feat = head.features(f_b, g_b)[0].cpu()               # (D,S,S)
    mu_pred = head(f_b, g_b).argmax(1)[0].cpu()

    lr_feat = feats.float().mean(0).permute(2, 0, 1)         # (D,gh,gw)
    panels_feat = [_raw_rgb(guid) if guid.numel() else np.zeros((64, 64, 3))]
    titles_feat = ["raw RGB" if guid.numel() else "(no guidance)"]

    hr_feat = hr_pred = None
    if hr_ds is not None:
        hr_feats_1, _hl, _hg = hr_ds[sample_idx]
        hr_feat = hr_feats_1.float().mean(0).permute(2, 0, 1)

    # Shared PCA basis: fit on HR when available (matches the training viz), else on mAnyUp.
    fit = hr_feat if hr_feat is not None else mu_feat
    disp = F.interpolate(lr_feat.unsqueeze(0), size=fit.shape[-2:], mode="nearest")[0]
    maps = [disp, mu_feat] + ([hr_feat] if hr_feat is not None else [])
    rgbs = _pca_rgb_shared(fit, maps)

    panels_feat += [rgbs[0], rgbs[1]]
    titles_feat += [f"LR feats ({lr_grid}x{lr_grid})",
                    f"mAnyUp ({mu_feat.shape[-1]}x{mu_feat.shape[-1]})"]
    if hr_feat is not None:
        panels_feat.append(rgbs[2])
        titles_feat.append(f"HR feats ({hr_feat.shape[-1]}x{hr_feat.shape[-1]})")

    # --- reference probes: LR + pa2px and HR + pa2px, the two routes being compared against ---
    # PREFER a probe saved by a real lp_pa2px run (--save_head): training one here would report a
    # different number than the run it is meant to illustrate. Training is the fallback for when
    # no saved head exists, and is deliberately short.
    lr_pred = None
    if lr_probe is not None:
        lr_pred = _probe_predict(lr_probe.to(device), feats, num_classes, lr_patch,
                                 label_size, device)
    elif train_ds is not None and getattr(train_ds, "_feats", None) is not None:
        probe = _train_ref_probe(lr_grid, train_ds._feats, train_ds.labels, num_classes,
                                 lr_patch, label_size, ignore_label, device, epochs=ref_epochs)
        lr_pred = _probe_predict(probe, feats, num_classes, lr_patch, label_size, device)
    if hr_ds is not None:
        if hr_probe is not None:
            hr_pred = _probe_predict(hr_probe.to(device), hr_ds[sample_idx][0], num_classes,
                                     hr_patch, label_size, device)
        elif hr_train_ds is not None and getattr(hr_train_ds, "_feats", None) is not None:
            probe_hr = _train_ref_probe(None, hr_train_ds._feats, hr_train_ds.labels,
                                        num_classes, hr_patch, label_size, ignore_label,
                                        device, epochs=ref_epochs)
            hr_pred = _probe_predict(probe_hr, hr_ds[sample_idx][0], num_classes, hr_patch,
                                     label_size, device)

    lab = label.long()
    panels_pred = [lab, lr_pred, mu_pred] + ([hr_pred] if hr_ds is not None else [])
    titles_pred = ["ground truth",
                   "LR + pa2px" + (f"  mIoU {_miou(lr_pred, lab, num_classes, ignore_label):.3f}"
                                   if lr_pred is not None else " (n/a)"),
                   f"mAnyUp + probe  mIoU {_miou(mu_pred, lab, num_classes, ignore_label):.3f}"]
    if hr_ds is not None:
        titles_pred.append(
            "HR + pa2px" + (f"  mIoU {_miou(hr_pred, lab, num_classes, ignore_label):.3f}"
                            if hr_pred is not None else " (n/a)"))

    ncol = len(panels_feat)
    # Panels are square, so the default row spacing leaves row 2's titles under row 1's
    # images; hspace buys them their own band.
    fig, axes = plt.subplots(2, ncol, figsize=(ncol * 2.7, 6.4))
    fig.subplots_adjust(hspace=0.28)
    for ax, p, t in zip(axes[0], panels_feat, titles_feat):
        ax.imshow(p)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax, p, t in zip(axes[1], panels_pred, titles_pred):
        if p is None:
            ax.axis("off")
            ax.set_title(t, fontsize=9)
            continue
        # Void pixels are -1; show them as class 19 (void_label) so the palette stays fixed.
        ax.imshow(np.where(p.numpy() < 0, 19, p.numpy()), cmap=CMAP, vmin=0, vmax=19,
                  interpolation="nearest")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    head_txt = f"{run_tag}  --  " if run_tag else ""
    fig.suptitle(f"{head_txt}test sample {sample_idx} -- "
                 f"features (shared PCA on {'HR' if hr_feat is not None else 'mAnyUp'}) "
                 f"/ predictions (PASTIS tab20)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved LP viz {out_path}")


# --------------------------------------------------------------------------------------------
# Standalone entry point. Renders the figure WITHOUT a full LP run: it builds the datasets,
# trains the mAnyUp probe for a few epochs (the upsampler itself is frozen, so only a 1x1 conv
# is fitted -- seconds once the features are resident), then draws. Use --epochs to match a real
# run more closely; the default is deliberately short because this is a picture, not a metric.
#
#   source env_setup/env_olmo.sh
#   python -u -m exp.pastis.viz_manyup_lp \
#       --features oe_base_s2s1_ps16_tile64 --manyup_ckpt "$CK" --manyup_native_out
def main() -> None:
    import argparse
    from pathlib import Path

    from exp.pastis.lp_cached_features import (
        CachedFeatureDataset, build_cached_head, HEAD_GUIDANCE, HEAD_REDUCES_TIME,
        IGNORE_LABEL, NUM_CLASSES, _run_head,
    )
    import json
    import re

    p = argparse.ArgumentParser(description="Render the LR / mAnyUp / HR comparison figure.")
    p.add_argument("--features", required=True, help="LR cached-feature config (the ckpt's lr_cfg)")
    p.add_argument("--manyup_ckpt", required=True)
    p.add_argument("--manyup_native_out", action="store_true",
                   help="run the upsampler at its trained target grid (match your LP run)")
    p.add_argument("--no-manyup_use_proj", dest="manyup_use_proj", action="store_false")
    p.add_argument("--out_root", default=None)
    p.add_argument("--data_splits", default="data/pastis_olmoearth")
    p.add_argument("--out", default="results/pastis/viz", help="output directory")
    p.add_argument("--sample", type=int, nargs="+", default=[0,1,2,3],
                   help="test sample index/indices to render")
    p.add_argument("--head", default=None,
                   help="LP head saved by lp_cached_features --save_head. Loads the probe that "
                        "run actually scored instead of training a fresh one here (preferred: "
                        "the picture then matches the CSV row).")
    p.add_argument("--lr_head", default=None,
                   help="saved lp_pa2px head on the LR features, for the 'LR + pa2px' column")
    p.add_argument("--hr_head", default=None,
                   help="saved lp_pa2px head on the HR features, for the 'HR + pa2px' column")
    p.add_argument("--epochs", type=int, default=8,
                   help="epochs for the mAnyUp probe when --head is not given")
    p.add_argument("--ref_epochs", type=int, default=8, help="epochs for the LR/HR reference probes")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max_ram_gb", type=float, default=32.0)
    p.add_argument("--time_pool", default="mean")
    args = p.parse_args()

    out_root = Path(args.out_root or (Path.home() / "projects/aip-gpleiss/timz/features"))
    splits = Path(args.data_splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # An unset shell variable becomes an EMPTY string, not a missing flag: `--head "$LPHEAD"`
    # with LPHEAD unset silently falls through to training a probe (and the big preload that
    # implies). Fail loudly instead, and catch typo'd paths before minutes of feature loading.
    for _flag in ("head", "lr_head", "hr_head", "manyup_ckpt"):
        _v = getattr(args, _flag)
        if _v is not None and (_v == "" or not Path(_v).is_file()):
            raise SystemExit(f"--{_flag}: {'empty (unset shell variable?)' if not _v else _v}"
                             f" is not a file")

    ck = torch.load(args.manyup_ckpt, map_location="cpu", weights_only=False)
    guidance_mod = ck.get("args", {}).get("guidance_mod") or "s2"
    meta = json.loads((out_root / args.features / "meta.json").read_text())
    patch_size, embed_dim = meta["patch_size"], meta["embed_dim"]

    guidance = HEAD_GUIDANCE["manyup"]
    reduce_time = HEAD_REDUCES_TIME["manyup"]
    # max_ram_gb=0 forces CachedFeatureDataset's per-sample disk path. We only ever INDEX the
    # test set (a handful of samples), so preloading the whole split -- ~1 GB and minutes of
    # Lustre reads -- buys nothing. Training splits still preload, but only when we actually
    # train a probe; with --head / --lr_head / --hr_head they are never built at all.
    mk = lambda sp, ram: CachedFeatureDataset(                # noqa: E731
        out_root / args.features, splits, sp, guidance=guidance, max_ram_gb=ram,
        reduce_time=reduce_time, time_pool=args.time_pool, guidance_mod=guidance_mod)
    test_ds = mk("test", 0)
    train_ds = None                    # built lazily, only if some probe must be trained
    label_size = test_ds.label_size

    head = build_cached_head("manyup", embed_dim, NUM_CLASSES, patch_size,
                             manyup_ckpt=args.manyup_ckpt,
                             manyup_use_proj=getattr(args, "manyup_use_proj", True),
                             manyup_native_out=args.manyup_native_out,
                             time_pool=args.time_pool, label_size=label_size).to(device)

    if args.head:
        saved = torch.load(args.head, map_location="cpu", weights_only=False)
        missing, unexpected = head.load_state_dict(saved["head_state"], strict=False)
        if unexpected:
            raise SystemExit(f"--head {args.head} has unexpected keys: {unexpected[:4]}")
        print(f"loaded LP head from {args.head} "
              f"(test mIoU {saved.get('test_miou', float('nan')):.4f}, "
              f"{saved.get('epochs')} epochs)")
        head.eval()
        _skip_train = True
    else:
        _skip_train = False

    # Train the mAnyUp probe ONLY when no saved head was given (the upsampler is frozen
    # inside CachedManyUp either way). Building train_ds is what costs the big preload, so it
    # stays inside this branch.
    if not _skip_train:
        train_ds = mk("train", args.max_ram_gb)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        opt = torch.optim.AdamW([q for q in head.parameters() if q.requires_grad], lr=args.lr)
        lossf = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        head.train()
        for ep in range(args.epochs):
            last = float("nan")
            for feats, label, rgb in loader:
                loss = lossf(_run_head(head, feats, rgb, device), label.to(device).long())
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = loss.item()
            print(f"probe epoch {ep + 1}/{args.epochs} | loss {last:.4f}", flush=True)
        head.eval()

    # HR ("oracle") features: a different cached config, named by the ckpt's own hr_cfg.
    hr_ds = hr_train_ds = None
    hr_patch = 4
    hr_cfg = ck.get("args", {}).get("hr_cfg", "")
    m = re.search(r"_ps(\d+)_", hr_cfg)
    hr_dir = out_root / hr_cfg if hr_cfg else None
    if m and hr_dir and hr_dir.exists():
        hr_patch = int(m.group(1))
        mkhr = lambda sp, ram: CachedFeatureDataset(          # noqa: E731
            hr_dir, splits, sp, guidance="none", max_ram_gb=ram,
            reduce_time=True, time_pool=args.time_pool)
        hr_ds = mkhr("test", 0)                    # indexed only -> lazy, no preload
        # The HR train split exists solely to fit the oracle probe; with --hr_head we load a
        # trained one instead and never touch it (it is the biggest preload in this script).
        if not args.hr_head:
            hr_train_ds = mkhr("train", args.max_ram_gb)
    else:
        print(f"no HR features at {hr_dir} -- drawing LR vs mAnyUp only")

    def _load_ref(path, patch):
        """Rebuild a pa2px probe (1x1 conv D -> C*p^2) from a --save_head file."""
        if not path:
            return None
        sd = torch.load(path, map_location="cpu", weights_only=False)["head_state"]
        w = sd["probe.weight"]
        conv = torch.nn.Conv2d(w.shape[1], w.shape[0], 1)
        conv.load_state_dict({"weight": w, "bias": sd["probe.bias"]})
        return conv.eval()

    lr_probe = _load_ref(args.lr_head, patch_size)
    hr_probe = _load_ref(args.hr_head, hr_patch)

    # The LR reference column falls back to training a probe when --lr_head is absent, which
    # needs the LR train split. It is already built if we trained the mAnyUp probe above.
    if lr_probe is None and train_ds is None:
        train_ds = mk("train", args.max_ram_gb)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.manyup_ckpt).stem
    for idx in args.sample:
        save_lp_viz(head, test_ds, hr_ds, device, out_dir / f"lpviz_{tag}_s{idx}.png",
                    sample_idx=idx, run_tag=tag, num_classes=NUM_CLASSES,
                    ignore_label=IGNORE_LABEL, label_size=label_size, hr_patch=hr_patch,
                    lr_patch=patch_size, ref_epochs=args.ref_epochs,
                    train_ds=train_ds, hr_train_ds=hr_train_ds,
                    lr_probe=lr_probe, hr_probe=hr_probe)


if __name__ == "__main__":
    main()
