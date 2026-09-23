"""Visualize a TRAINED timAnyUp checkpoint on several held-out TEST samples.

train_timanyup.py only ever renders test_ds[0] (one sample, once per epoch) -- enough to watch
training progress, useless for judging whether the model behaves sensibly across scenes. This
runs a finished checkpoint over many test samples and writes one figure each, plus a per-sample
metrics table so the pictures can be read against numbers rather than impressions.

Every setting comes from the CHECKPOINT's own recorded args (arms, k, guidance, query_input),
so the visualization cannot silently differ from how the model was trained. --k overrides the
budget on purpose, for seeing what the lookups do at a different budget.

    source env_setup/env_olmo.sh
    python -m exp.upsamplers.viz_timanyup --ckpt checkpoints/timanyup/.../..._ep19.pth
    python -m exp.upsamplers.viz_timanyup --ckpt <ckpt> --n 12 --sort worst
"""
import argparse
import re
import sys
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")

from exp.upsamplers.common import DATA_ROOT, FEATURES_ROOT, GUIDANCE_BANDS, cfg_bits
from exp.upsamplers.data_timanyup import TriFeatureDataset, check_arms
from exp.upsamplers.train_timanyup import (_import_timanyup, _flat_bt, save_epoch_viz,
                                           _forward_losses)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="trained timAnyUp checkpoint")
    p.add_argument("--n", type=int, default=8, help="how many test samples to render")
    p.add_argument("--sort", default="index", choices=("index", "worst", "best"),
                   help="index: the first --n test samples. worst/best: rank ALL scanned "
                        "samples by fused reconstruction loss first -- 'worst' is where to "
                        "look for failure modes, 'best' for what the model does when it works.")
    p.add_argument("--scan", type=int, default=200,
                   help="samples to score before ranking (only used by --sort worst/best)")
    p.add_argument("--k", type=int, default=None, help="override the lookup budget")
    p.add_argument("--split", default="test")
    p.add_argument("--viz_t", type=int, default=-1, help="-1 = all timesteps as rows")
    p.add_argument("--viz_max_t", type=int, default=6)
    p.add_argument("--out_dir", default=None,
                   help="default: <ckpt dir>/viz_samples/<ckpt stem>")
    p.add_argument("--features_root", default=str(FEATURES_ROOT))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--anyup_repo", default="/scratch/timz/mAnyUp/third_party/anyup")
    args = p.parse_args()

    A = _import_timanyup(args.anyup_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ck.get("args", {})

    # Arms and every knob come from the checkpoint, never from fresh defaults: a viz built with
    # different arms or a different guidance modality would not be showing this model at all.
    lrhc_cfg, hrlc_cfg = ca["lrhc_cfg"], ca["hrlc_cfg"]
    hrhc_cfg, gmod = ca["hrhc_cfg"], ca.get("guidance_mod", "s2")
    check_arms(lrhc_cfg, hrlc_cfg, hrhc_cfg)

    froot = Path(args.features_root)
    ds = TriFeatureDataset(froot / lrhc_cfg, froot / hrlc_cfg, froot / hrhc_cfg,
                           Path(args.data_root), args.split, guidance_mod=gmod)

    lrhc0, _, hrhc0, _ = ds[0]
    T, C = lrhc0.shape[0], lrhc0.shape[1]
    GH, GW = hrhc0.shape[-2:]

    model = A["TimAnyUp"](
        input_dim=ck.get("input_dim", GUIDANCE_BANDS[gmod]), qk_dim=ca.get("qk_dim", 128),
        feat_dim=ck.get("feat_dim", C), num_frames=ck.get("num_frames", T),
        transform_depth=ca.get("transform_depth", 0),
        window_ratio=ca.get("window_ratio", 1.0),
        query_input=ca.get("query_input", "bilinear")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # save_epoch_viz and _forward_losses read their settings off an args object; build one that
    # carries the CHECKPOINT's values so both paths agree with training.
    class VA:
        pass
    va = VA()
    va.k = args.k if args.k is not None else ca.get("k", 512)
    va.transform_frac = ca.get("transform_frac", 0.25)
    va.viz_t, va.viz_max_t = args.viz_t, args.viz_max_t
    if args.k is not None and args.k != ca.get("k"):
        print(f"NOTE: rendering at k={va.k} but checkpoint trained with k={ca.get('k')}")

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(args.ckpt).parent / "viz_samples" / Path(args.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Which samples to render. Ranking needs a scoring pass first; plain "index" does not, so
    # we skip the scan entirely in that case rather than paying for it.
    order = list(range(min(args.n, len(ds))))
    scores = {}
    if args.sort != "index":
        n_scan = min(args.scan, len(ds))
        print(f"scoring {n_scan} test samples to pick the {args.sort} {args.n}...")
        for i in range(n_scan):
            lrhc, hrlc, hrhc, guide = [x.unsqueeze(0).to(device) for x in ds[i]]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                o = _forward_losses(model, A, va, lrhc, hrlc, hrhc, guide, (GH, GW), device)
            scores[i] = float(o["recon"])
        order = sorted(scores, key=lambda i: scores[i], reverse=(args.sort == "worst"))[:args.n]

    _, lr_ps = cfg_bits(lrhc_cfg)
    _, hr_ps = cfg_bits(hrhc_cfg)
    tag = f"{gmod}_ps{lr_ps}_to_ps{hr_ps}_T{T}_k{va.k}"
    print(f"{tag}  ->  {out_dir}")

    rows = []
    for rank, i in enumerate(order):
        sample = ds[i]
        lrhc, hrlc, hrhc, guide = [x.unsqueeze(0).to(device) for x in sample]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            o = _forward_losses(model, A, va, lrhc, hrlc, hrhc, guide, (GH, GW), device)
        sid = ds.ids[i]
        save_epoch_viz(model, A, sample, va, GH, GW, device,
                       out_dir / f"{tag}_sample{sid:05d}.png", f"sample {sid}", tag)
        rows.append((sid, float(o["recon"]), float(o["c_upsample"]), float(o["c_random"]),
                     float(o["mask_prec"])))

    # The table is the point of rendering many samples: it says whether a picture is typical or
    # cherry-picked, and whether the lookups help on THIS scene or only on average.
    print(f"\n{'sample':>7} {'fused':>7} {'up-only':>8} {'random-k':>9} {'vs up':>7} "
          f"{'vs rnd':>7} {'maskP':>6}")
    for sid, rec, up, rnd, mp in rows:
        print(f"{sid:>7} {rec:>7.4f} {up:>8.4f} {rnd:>9.4f} {up-rec:>+7.4f} "
              f"{rnd-rec:>+7.4f} {mp:>6.3f}")
    n = len(rows)
    if n:
        mean = lambda j: sum(r[j] for r in rows) / n
        print(f"{'mean':>7} {mean(1):>7.4f} {mean(2):>8.4f} {mean(3):>9.4f} "
              f"{mean(2)-mean(1):>+7.4f} {mean(3)-mean(1):>+7.4f} {mean(4):>6.3f}")
        wins = sum(1 for r in rows if r[2] > r[1])
        print(f"\nfused beats upsample-only on {wins}/{n} samples; "
              f"beats random-k on {sum(1 for r in rows if r[3] > r[1])}/{n}")
    print(f"wrote {n} figures to {out_dir}")


if __name__ == "__main__":
    main()
