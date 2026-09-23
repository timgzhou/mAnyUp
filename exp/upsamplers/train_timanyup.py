"""Train timAnyUp: temporal, adaptive upsampling with a queryable high-res side channel.

Two CHEAP feature maps in, one EXPENSIVE map approximated out:

  F_lrhc  oe_base_s2_ps16_tile64       (T, 4, 4,C)  low-res, high-context   -- few tokens
  F_hrlc  oe_base_s2_ps4_tile4_single  (T,16,16,C)  high-res, low-context   -- no cross-t attn
  F_hrhc  oe_base_s2_ps4_tile64        (T,16,16,C)  high-res, high-context  <- TARGET

Pipeline (see anyup/timAnyUp.py for the module):

  F_lrhc --mAnyUp per-timestep, image-guided--> F_lrhc_up ------------.
  F_lrhc --bilinear--> query head --> scores --> top-k (k=512) --.    |
  F_hrlc --transform head--> F_hrlc_t ---------------------------+--> blend --> F_fused
                                                                       |
                                                        L_recon = CosMSE(F_fused, F_hrhc)

Three losses:
  L_recon      CosMSE(F_fused, F_hrhc)                              -- the objective
  L_query      BCE(query scores, rank(cosmse_map(F_lrhc_up, F_hrhc)))  -- learn WHERE to look
  L_transform  CosMSE(F_hrlc_t, F_lrhc_up) on the LOW-error locations  -- align the two spaces

Both L_query and L_transform are supervised by DETACHED quantities: the mask must not be able
to help itself by making the upsampler worse, and the transform must not drag F_lrhc_up toward
being easy to regress onto.

Controls (all reported every epoch, this is the point of the script):
  bilinear     no learning at all
  upsample     mAnyUp alone, no lookups        -- does the F_hrlc side channel earn its cost?
  random-k     same budget, locations at random -- did the MASK learn anything?

    source env_setup/env_olmo.sh
    python -m exp.upsamplers.train_timanyup --sanity
    python -m exp.upsamplers.train_timanyup --epochs 20 --batch_size 4 --stage_to_tmpdir
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")        # headless cluster
import matplotlib.pyplot as plt
import matplotlib.colors            # ListedColormap for the binary query mask
import matplotlib.patches           # Patch handles for the mask legend

from exp.upsamplers.common import (
    DATA_ROOT, FEATURES_ROOT, GUIDANCE_BANDS, guidance_mod_for, cfg_bits,
    stage_to_tmpdir, warmup_cosine, pca_rgb_shared, raw_rgb,
)
from exp.upsamplers.data_timanyup import TriFeatureDataset, check_arms

DEFAULT_ANYUP_REPO = "/scratch/timz/mAnyUp/third_party/anyup"


def _import_timanyup(repo: str):
    sys.path.insert(0, repo)
    from anyup.loss import Cosine_MSE, cosmse_map                      # noqa: E402
    from anyup.timAnyUp import (TimAnyUp, topk_mask, random_mask,      # noqa: E402
                                blend, rank_normalize)
    return dict(TimAnyUp=TimAnyUp, Cosine_MSE=Cosine_MSE, cosmse_map=cosmse_map,
                topk_mask=topk_mask, random_mask=random_mask, blend=blend,
                rank_normalize=rank_normalize)


def _flat_bt(x):
    """(B,T,C,H,W) -> (B*T,C,H,W): the losses are per-frame, so fold time into the batch."""
    B, T = x.shape[:2]
    return x.reshape(B * T, *x.shape[2:])


@torch.no_grad()
def evaluate(model, loader, A, args, GH, GW, device, max_batches=None):
    """Mean losses + the three controls over (part of) a loader. Returns a dict of scalars."""
    model.eval()
    agg, n = {}, 0
    for bi, (lrhc, hrlc, hrhc, guide) in enumerate(loader):
        lrhc, hrlc = lrhc.to(device), hrlc.to(device)
        hrhc, guide = hrhc.to(device), guide.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = _forward_losses(model, A, args, lrhc, hrlc, hrhc, guide, (GH, GW), device)
        for k, v in out.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
        if max_batches and bi + 1 >= max_batches:
            break
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def _forward_losses(model, A, args, lrhc, hrlc, hrhc, guide, out_size, device):
    """One forward pass -> every loss and control. Shared by train and eval so they cannot
    drift apart (a classic source of train/eval metric mismatch)."""
    GH, GW = out_size
    lrhc_up = model.upsample(lrhc, guide, out_size)          # (B,T,C,H,W)
    hrlc_t = model.transform(hrlc)
    scores = model.query(lrhc, out_size, lrhc_up=lrhc_up)    # (B,T,H,W) logits

    up_f, hr_f = _flat_bt(lrhc_up).float(), _flat_bt(hrhc).float()

    # --- L_recon: the objective. Learned top-k selection, channel-mean blend, then a LINEAR
    # projection into the target's feature space. We are reweighting low-res (ps16) tokens, so
    # requiring the fused map to EQUAL a ps4 target is ill-posed -- we require only that it can
    # linearly PREDICT it, which is also what a downstream linear probe asks of it.
    mask = A["topk_mask"](scores.detach().float(), args.k)
    fused = A["blend"](lrhc_up, hrlc_t, mask)
    l_recon = A["Cosine_MSE"]()(_flat_bt(model.project(fused)).float(), hr_f)["total"]

    # --- L_query: predict WHERE the cheap upsample fails. Target is detached and
    # rank-normalized per sample -- top-k needs ordering, not error magnitude.
    # Measured on the PROJECTED upsample, so the mask predicts failure of the same quantity
    # L_recon minimizes (an unprojected error would rank locations by a residual the objective
    # never sees -- e.g. a global basis mismatch the projector removes for free).
    up_proj_f = _flat_bt(model.project(lrhc_up)).float()
    err = A["cosmse_map"](up_proj_f, hr_f).reshape(scores.shape).detach()
    target = A["rank_normalize"](err)
    l_query = F.binary_cross_entropy_with_logits(scores.float(), target)

    # --- L_transform: align F_hrlc into F_lrhc_up's space, fit ONLY where the upsampler is
    # already trustworthy (bottom quantile of err) -- those are the locations where
    # F_lrhc_up is a valid regression target. Target detached: the transform must not pull
    # the upsampler toward being easy to regress onto.
    B = err.shape[0]
    flat_err = err.reshape(B, -1)
    n_fit = max(1, int(flat_err.shape[1] * args.transform_frac))
    fit_idx = flat_err.argsort(dim=1)[:, :n_fit]             # lowest-error locations
    fit = torch.zeros_like(flat_err, dtype=torch.bool).scatter_(1, fit_idx, True)
    fit = fit.reshape(err.shape).unsqueeze(2).expand_as(hrlc_t)
    C = hrlc_t.shape[2]
    l_transform = A["Cosine_MSE"]()(
        hrlc_t[fit].reshape(-1, C, 1, 1),
        lrhc_up.detach()[fit].reshape(-1, C, 1, 1))["total"]

    out = {"recon": l_recon, "query": l_query, "transform": l_transform}

    # --- controls (no grad needed, but cheap and computed in the same pass so they always
    # describe the SAME model state as the losses above).
    with torch.no_grad():
        # Every learned control passes through the SAME projector as L_recon: comparing a
        # projected fused map against an unprojected control would credit the projector, not
        # the lookups. Bilinear is the one exception -- it is the no-learning-at-all floor.
        out["c_upsample"] = A["Cosine_MSE"]()(up_proj_f, hr_f)["total"]   # mAnyUp alone
        bil = F.interpolate(_flat_bt(lrhc).float(), size=out_size,
                            mode="bilinear", align_corners=False)
        out["c_bilinear"] = A["Cosine_MSE"]()(bil, hr_f)["total"]        # no learning
        rmask = A["random_mask"](scores.float(), args.k)
        rfused = A["blend"](lrhc_up, hrlc_t, rmask)
        out["c_random"] = A["Cosine_MSE"]()(
            _flat_bt(model.project(rfused)).float(), hr_f)["total"]
        # How much of the learned mask agrees with the true worst-k: the direct read on
        # whether the query head is doing better than chance.
        true_worst = A["topk_mask"](err, args.k)
        inter = (mask & true_worst).reshape(B, -1).sum(1).float()
        out["mask_prec"] = (inter / args.k).mean()
        out["mask_chance"] = torch.tensor(args.k / err[0].numel(), device=device)
    return out


@torch.no_grad()
def save_epoch_viz(model, A, sample, args, GH, GW, device, out_path, epoch, run_tag=""):
    """Per-TIMESTEP viz: one ROW per timestep, columns
        guidance | F_lrhc | F_lrhc_up (proj) | F_fused (proj) | F_hrhc target
        | query mask | mask TARGET

    The last two columns are the pair to read together: the mask the head PREDICTED, and the
    error map it was supervised on (rank-normalized cosmse between the projected upsample and
    the target). Where they disagree is exactly where the query head is still wrong.

    F_lrhc_up is shown PROJECTED, because that is what the loss and the mask target both see
    -- an unprojected panel would be a different quantity from the one being optimized.

    Nothing here is time-pooled -- timAnyUp is a temporal method, and a mean over T would hide
    exactly what it is supposed to exploit (a cloudy frame the low-context arm fails on, and
    the mask spending its budget there). Feature panels share ONE PCA basis fit on the target
    across ALL shown timesteps, so colours are comparable down a column AND across rows.
    The F_fused panel is the PROJECTED map -- what the reconstruction loss actually sees.
    """
    lrhc, hrlc, hrhc, guide = [x.unsqueeze(0).to(device) for x in sample]
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        up = model.upsample(lrhc, guide, (GH, GW))
        hrlc_t = model.transform(hrlc)
        scores = model.query(lrhc, (GH, GW), lrhc_up=up)
        mask = A["topk_mask"](scores.float(), args.k)
        fused = model.project(A["blend"](up, hrlc_t, mask))
        up_proj = model.project(up)
        # The mask's supervision target, built exactly as _forward_losses does it.
        err = A["cosmse_map"](_flat_bt(up_proj).float(),
                              _flat_bt(hrhc).float()).reshape(scores.shape)
        err_rank = A["rank_normalize"](err)
    model.train()

    T = lrhc.shape[1]
    ts = list(range(T)) if args.viz_t < 0 else [min(args.viz_t, T - 1)]
    if args.viz_max_t and len(ts) > args.viz_max_t:      # evenly spaced subset, always incl. 0
        step = max(1, len(ts) // args.viz_max_t)
        ts = ts[::step][:args.viz_max_t]

    # One PCA basis for the whole figure, fit on the target frames being shown.
    fit = torch.cat([hrhc[0, t].float().cpu().reshape(hrhc.shape[2], -1) for t in ts], dim=1)
    fit = fit.reshape(hrhc.shape[2], -1, 1)

    MASK_COL, TGT_COL = 5, 6
    # Two colours only: the query mask is a SELECTION, not a heatmap.
    MASK_CMAP = matplotlib.colors.ListedColormap(["#1b1b2f", "#f2c14e"])
    n_row, n_col = len(ts), 7
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2.2, n_row * 2.35),
                             squeeze=False)
    per_t_mask = []
    for r, t in enumerate(ts):
        lr_c = lrhc[0, t].float().cpu()
        up_c, fu_c = up_proj[0, t].float().cpu(), fused[0, t].float().cpu()
        hr_c = hrhc[0, t].float().cpu()
        lr_disp = F.interpolate(lr_c.unsqueeze(0), size=(GH, GW), mode="nearest").squeeze(0)
        lr_rgb, up_rgb, fu_rgb, hr_rgb = pca_rgb_shared(fit, [lr_disp, up_c, fu_c, hr_c])
        mk = mask[0, t].float().cpu().numpy()
        tgt = err_rank[0, t].float().cpu().numpy()      # what the mask SHOULD have picked
        per_t_mask.append(mk.sum())
        panels = [raw_rgb(guide[0, t].float().cpu()), lr_rgb, up_rgb, fu_rgb, hr_rgb, mk, tgt]
        for c, pan in enumerate(panels):
            ax = axes[r][c]
            if c == MASK_COL:
                # Binary selected/not -- its own two-colour map, NOT the target's continuous
                # one: sharing a colormap invited reading the mask as if it had a magnitude.
                im = ax.imshow(pan, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
            elif c == TGT_COL:
                im_tgt = ax.imshow(pan, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(pan)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                # Per-row: which timestep, and how much of the k budget landed on it.
                ax.set_ylabel(f"t={t}\n{int(mk.sum())}/{args.k}", fontsize=7)
        if r == 0:
            for c, ti in enumerate(["guidance", f"F_lrhc ({lr_c.shape[-2]}x{lr_c.shape[-1]})",
                                    "F_lrhc_up (proj)", "F_fused (proj)", "F_hrhc target",
                                    f"query mask (k={args.k})", "mask target (rank err)"]):
                axes[r][c].set_title(ti, fontsize=8)
    # A binary mask and a [0,1] error map are different KINDS of quantity, so they get
    # different keys: a two-entry legend for the selection, a colourbar for the ranked error.
    axes[0][MASK_COL].legend(
        handles=[matplotlib.patches.Patch(facecolor="#f2c14e", edgecolor="none",
                                          label=f"queried (top-{args.k})"),
                 matplotlib.patches.Patch(facecolor="#1b1b2f", edgecolor="none",
                                          label="not queried")],
        loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=1, fontsize=6,
        frameon=False, handlelength=1.0, handleheight=1.0, borderpad=0.1,
        labelspacing=0.25)
    # Colourbar spanning the target column, so the rank scale is readable: 0 = the upsampler
    # is already accurate here, 1 = its worst location in this sample.
    cax = fig.add_axes([0.915, 0.12, 0.010, 0.70])
    cb = fig.colorbar(im_tgt, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0\nbest", "0.5", "1\nworst"], fontsize=6)
    cb.ax.tick_params(length=2, pad=1)
    cb.set_label("per-sample rank of cosmse err", fontsize=6, labelpad=2)
    cb.outline.set_linewidth(0.3)

    head = f"{run_tag}  --  " if run_tag else ""
    fig.suptitle(f"{head}epoch {epoch} -- test sample, per timestep "
                 f"(shared PCA on target; budget spread {int(min(per_t_mask))}-"
                 f"{int(max(per_t_mask))} per frame)", fontsize=10)
    # Leave room on the right for the colourbar; tight_layout would otherwise overlap it.
    fig.tight_layout(rect=(0, 0, 0.905, 0.955))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved viz {out_path}  ({len(ts)} timesteps)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lrhc_cfg", default="oe_base_s2_ps16_tile64",
                   help="low-res HIGH-context input (cheap: few tokens)")
    p.add_argument("--hrlc_cfg", default="oe_base_s2_ps4_tile4_single",
                   help="high-res LOW-context input (cheap: small tile, per-timestep)")
    p.add_argument("--hrhc_cfg", default="oe_base_s2_ps4_tile64",
                   help="high-res HIGH-context TARGET (expensive)")
    p.add_argument("--split", default="train")
    p.add_argument("--id_half", default="all", choices=("all", "first", "second"),
                   help="train on one half of the split's ids so the upsampler and a downstream "
                        "LP probe can use disjoint samples")
    p.add_argument("--features_root", default=str(FEATURES_ROOT))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--anyup_repo", default=DEFAULT_ANYUP_REPO)
    p.add_argument("--guidance_mod", default=None, choices=list(GUIDANCE_BANDS))
    # --- model
    p.add_argument("--k", type=int, default=512,
                   help="per-sample lookup BUDGET: how many (t,y,x) locations are fetched from "
                        "F_hrlc. Absolute, not a fraction, so the compute cost stays fixed and "
                        "comparable when the target grid changes.")
    p.add_argument("--query_input", default="bilinear", choices=("bilinear", "upsampled"),
                   help="bilinear: mask depends only on cheap inputs, so it can be computed in "
                        "PARALLEL with the upsample. upsampled: reads mAnyUp's output (more "
                        "information, but couples the two) -- the first thing to ablate.")
    p.add_argument("--transform_frac", type=float, default=0.25,
                   help="fraction of lowest-error locations the transform head is fit on")
    p.add_argument("--qk_dim", type=int, default=128)
    p.add_argument("--feat_dim", type=int, default=768)
    p.add_argument("--transform_depth", type=int, default=0,
                   help="resblocks inside mAnyUp after the cross-attention")
    p.add_argument("--window_ratio", type=float, default=0.5)
    p.add_argument("--lambda_query", type=float, default=1.0)
    p.add_argument("--lambda_transform", type=float, default=1.0)
    # --- optim
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16,
                   help="effective batch is batch_size*T frames (T=12 -> 48 at the default)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--stage_to_tmpdir", action="store_true")
    p.add_argument("--out_dir", default="checkpoints/timanyup")
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--eval_batches", type=int, default=25,
                   help="test batches for the per-epoch held-out eval")
    p.add_argument("--viz_t", type=int, default=-1,
                   help="timestep to visualize; -1 (default) shows ALL timesteps as rows -- "
                        "this is a temporal method, so a single frame hides the point")
    p.add_argument("--viz_max_t", type=int, default=6,
                   help="cap on rows when --viz_t -1 (evenly spaced subset)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sanity", action="store_true", help="one batch then exit")
    args = p.parse_args()

    A = _import_timanyup(args.anyup_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Shapes cannot catch an arm mix-up (F_hrlc and F_hrhc are shape-identical), so validate
    # what the cfg NAMES claim before touching any data.
    check_arms(args.lrhc_cfg, args.hrlc_cfg, args.hrhc_cfg)

    froot = Path(args.features_root)
    lrhc_dir, hrlc_dir = froot / args.lrhc_cfg, froot / args.hrlc_cfg
    hrhc_dir = froot / args.hrhc_cfg
    data_root = Path(args.data_root)
    if args.stage_to_tmpdir:
        lrhc_dir, hrlc_dir, hrhc_dir = stage_to_tmpdir([lrhc_dir, hrlc_dir, hrhc_dir])
        (staged,) = stage_to_tmpdir([data_root / f"pastis_r_{args.split}"])
        data_root = staged.parent

    if args.guidance_mod is None:
        args.guidance_mod = guidance_mod_for(args.lrhc_cfg)
    guide_bands = GUIDANCE_BANDS[args.guidance_mod]
    print(f"guidance: {args.guidance_mod} ({guide_bands}-band) from {data_root}")

    ds = TriFeatureDataset(lrhc_dir, hrlc_dir, hrhc_dir, data_root, args.split,
                           guidance_mod=args.guidance_mod, half=args.id_half)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Held-out TEST data: the per-epoch numbers that matter. Feature dirs may be staged, but
    # test imagery only ever lives in the original data_root (only the train split is staged).
    test_loader, viz_sample = None, None
    try:
        test_ds = TriFeatureDataset(lrhc_dir, hrlc_dir, hrhc_dir, Path(args.data_root), "test",
                                    guidance_mod=args.guidance_mod)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=max(2, args.num_workers // 2), pin_memory=True)
        viz_sample = test_ds[0]
    except RuntimeError as e:
        print(f"test eval/viz disabled: {e}")

    lrhc0, hrlc0, hrhc0, guide0 = ds[0]
    T, C = lrhc0.shape[0], lrhc0.shape[1]
    GH, GW = hrhc0.shape[-2:]
    print(f"F_lrhc {tuple(lrhc0.shape)}  +  F_hrlc {tuple(hrlc0.shape)}  ->  "
          f"F_hrhc {tuple(hrhc0.shape)}   (T={T}, C={C}, target {GH}x{GW})")
    n_loc = T * GH * GW
    print(f"lookup budget k={args.k} of {n_loc} locations ({args.k/n_loc:.0%}); "
          f"random-k is the control that matters")

    _, lr_ps = cfg_bits(args.lrhc_cfg)
    _, hr_ps = cfg_bits(args.hrhc_cfg)
    qi_tag = "" if args.query_input == "bilinear" else f"_q{args.query_input}"
    # Batch size is part of the identity, not just the schedule: effective batch is
    # batch_size*T frames, so it changes the LR schedule and thus the trained model. Without
    # it two batch sizes write the same checkpoint and viz filenames.
    run_tag = (f"{args.guidance_mod}_ps{lr_ps}_to_ps{hr_ps}_T{T}_k{args.k}"
               f"_tf{args.transform_frac:g}_bs{args.batch_size}{qi_tag}")
    print(f"run tag: {run_tag}")

    model = A["TimAnyUp"](input_dim=guide_bands, qk_dim=args.qk_dim, feat_dim=C, num_frames=T,
                          transform_depth=args.transform_depth,
                          window_ratio=args.window_ratio,
                          query_input=args.query_input).to(device).train()
    np_ = lambda m: sum(x.numel() for x in m.parameters())
    print(f"params: upsampler {np_(model.upsampler):,}  query {np_(model.query_head):,}  "
          f"transform {np_(model.transform_head):,}  projector {np_(model.projector):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine(opt, args.epochs * len(loader), args.warmup_frac, args.lr, args.lr_min)

    for epoch in range(args.epochs):
        run = {}
        for bi, (lrhc, hrlc, hrhc, guide) in enumerate(loader):
            lrhc, hrlc = lrhc.to(device, non_blocking=True), hrlc.to(device, non_blocking=True)
            hrhc = hrhc.to(device, non_blocking=True)
            guide = guide.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                o = _forward_losses(model, A, args, lrhc, hrlc, hrhc, guide, (GH, GW), device)
                loss = (o["recon"] + args.lambda_query * o["query"]
                        + args.lambda_transform * o["transform"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            for k_, v in o.items():
                run[k_] = run.get(k_, 0.0) + float(v)
            if bi % args.log_every == 0:
                print(f"ep {epoch} b {bi}/{len(loader)} lr={sched.get_last_lr()[0]:.2e} "
                      f"recon={float(o['recon']):.4f} query={float(o['query']):.4f} "
                      f"transform={float(o['transform']):.4f} | "
                      f"up={float(o['c_upsample']):.4f} rand={float(o['c_random']):.4f} "
                      f"prec={float(o['mask_prec']):.3f}")
            if args.sanity:
                print("sanity: one batch done, exiting.")
                return

        n = len(loader)
        tr = {k_: v / n for k_, v in run.items()}
        print(f"== epoch {epoch} TRAIN recon={tr['recon']:.4f} query={tr['query']:.4f} "
              f"transform={tr['transform']:.4f}")
        _report(tr, "train", args)

        if test_loader is not None:
            te = evaluate(model, test_loader, A, args, GH, GW, device, args.eval_batches)
            print(f"== epoch {epoch} TEST  recon={te['recon']:.4f}")
            _report(te, "test ", args)
            viz_dir = out_dir / "viz" / run_tag
            viz_dir.mkdir(parents=True, exist_ok=True)
            save_epoch_viz(model, A, viz_sample, args, GH, GW, device,
                           viz_dir / f"{run_tag}_ep{epoch:03d}.png", epoch, run_tag)

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            ckpt = out_dir / f"timanyup_{run_tag}_ep{epoch}.pth"
            # query_input is promoted to a TOP-LEVEL key, not left buried in args: it
            # changes what the selector reads (and whether the mask can be computed in
            # parallel with the upsample), so every consumer must be able to see it without
            # digging. The arms are promoted for the same reason -- F_hrlc and F_hrhc are
            # shape-identical, so a checkpoint that does not name them is ambiguous.
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "input_dim": guide_bands, "feat_dim": C, "num_frames": T,
                        "run_tag": run_tag, "query_input": args.query_input,
                        "k": args.k, "lrhc_cfg": args.lrhc_cfg,
                        "hrlc_cfg": args.hrlc_cfg, "hrhc_cfg": args.hrhc_cfg}, ckpt)
            print(f"saved {ckpt}")


def _report(d, tag, args):
    """Print the comparisons that decide whether the method works. Lower loss is better."""
    print(f"   [{tag}] fused={d['recon']:.4f}  vs  upsample-only={d['c_upsample']:.4f} "
          f"({d['c_upsample'] - d['recon']:+.4f})  vs  random-k={d['c_random']:.4f} "
          f"({d['c_random'] - d['recon']:+.4f})  vs  bilinear={d['c_bilinear']:.4f}")
    print(f"   [{tag}] mask precision={d['mask_prec']:.3f} (chance={d['mask_chance']:.3f}) "
          f"-- fraction of the k picks that are truly in the worst-k")


if __name__ == "__main__":
    main()
