"""What SHOULD the query head predict: reconstruction error, or counterfactual lookup gain?

timAnyUp currently supervises the mask on cosmse(F_lrhc_up, F_hrhc) -- "where is the upsampler
wrong". But the decision the mask actually makes is "where would an F_hrlc lookup HELP", which
is an intervention effect, not an error level. A patch can be badly reconstructed AND unhelped
by a lookup (cloud, say, where F_hrlc is equally uninformative). That gap would explain the
k-sweep: the head ranks error well (spearman ~0.61) yet the reconstruction gain grows linearly
in k, as if the selected locations were no better than random ones.

Because blend() is per-location and independent across locations, the counterfactual for EVERY
location is one extra forward: blend everywhere, then compare per-location errors.

    gain[t,y,x] = cosmse(project(up))[t,y,x] - cosmse(project(blend_all))[t,y,x]

positive = a lookup at that location improves the objective.

This measures, on a TRAINED checkpoint and with no training:
  1. is gain predictable at all, and does it differ from error? (correlation between them)
  2. would an ORACLE ranking by gain beat the current one? (upper bound on the whole idea)
  3. does the existing head already rank gain, or only error?

    source env_setup/env_olmo.sh
    python -m exp.upsamplers.check_query_target --ckpt <ckpt> --n 64
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from exp.upsamplers.common import DATA_ROOT, FEATURES_ROOT, GUIDANCE_BANDS
from exp.upsamplers.data_timanyup import TriFeatureDataset, check_arms
from exp.upsamplers.train_timanyup import _import_timanyup, _flat_bt


def _spearman(a, b):
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--split", default="test")
    p.add_argument("--ks", default="64,256,512,1024,2048")
    p.add_argument("--features_root", default=str(FEATURES_ROOT))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--anyup_repo", default="/scratch/timz/mAnyUp/third_party/anyup")
    args = p.parse_args()

    A = _import_timanyup(args.anyup_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ck.get("args", {})
    check_arms(ca["lrhc_cfg"], ca["hrlc_cfg"], ca["hrhc_cfg"])
    gmod = ca.get("guidance_mod", "s2")

    froot = Path(args.features_root)
    ds = TriFeatureDataset(froot / ca["lrhc_cfg"], froot / ca["hrlc_cfg"],
                           froot / ca["hrhc_cfg"], Path(args.data_root), args.split,
                           guidance_mod=gmod)
    lrhc0, _, hrhc0, _ = ds[0]
    T, C = lrhc0.shape[0], lrhc0.shape[1]
    GH, GW = hrhc0.shape[-2:]

    model = A["TimAnyUp"](
        input_dim=ck.get("input_dim", GUIDANCE_BANDS[gmod]), qk_dim=ca.get("qk_dim", 128),
        feat_dim=ck.get("feat_dim", C), num_frames=ck.get("num_frames", T),
        transform_depth=ca.get("transform_depth", 0),
        window_ratio=ca.get("window_ratio", 1.0),
        query_input=ca.get("query_input", "bilinear")).to(device).eval()
    model.load_state_dict(ck["model"])

    err_all, gain_all, q_all = [], [], []
    n = min(args.n, len(ds))
    print(f"scoring {n} {args.split} samples ({T}x{GH}x{GW} locations each)...")
    with torch.no_grad():
        for i in range(n):
            lrhc, hrlc, hrhc, guide = [x.unsqueeze(0).to(device) for x in ds[i]]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                up = model.upsample(lrhc, guide, (GH, GW))
                hrlc_t = model.transform(hrlc)
                scores = model.query(lrhc, (GH, GW), lrhc_up=up)
                up_proj = model.project(up)
                # Blend at EVERY location: since blend is per-location, the resulting error
                # map holds the counterfactual "if this one location were queried" for all
                # locations simultaneously.
                all_mask = torch.ones(1, T, GH, GW, dtype=torch.bool, device=device)
                fused_all = model.project(A["blend"](up, hrlc_t, all_mask))
            hr_f = _flat_bt(hrhc).float()
            err = A["cosmse_map"](_flat_bt(up_proj).float(), hr_f).reshape(1, T, GH, GW)
            err_q = A["cosmse_map"](_flat_bt(fused_all).float(), hr_f).reshape(1, T, GH, GW)
            err_all.append(err.cpu().numpy().ravel())
            gain_all.append((err - err_q).cpu().numpy().ravel())   # >0 => lookup helps
            q_all.append(scores.float().cpu().numpy().ravel())

    err = np.concatenate(err_all); gain = np.concatenate(gain_all); q = np.concatenate(q_all)
    npl = T * GH * GW
    print(f"\npooled {err.size:,} locations")
    print(f"err   mean {err.mean():+.4f}  std {err.std():.4f}")
    print(f"gain  mean {gain.mean():+.4f}  std {gain.std():.4f}  "
          f"positive at {100*(gain>0).mean():.1f}% of locations")

    print(f"\n{'':<26}{'pearson':>9}{'spearman':>10}")
    print(f"{'gain vs err':<26}{np.corrcoef(gain, err)[0,1]:>9.4f}{_spearman(gain, err):>10.4f}"
          f"   <- if low, error is the WRONG target")
    print(f"{'learned query vs err':<26}{np.corrcoef(q, err)[0,1]:>9.4f}{_spearman(q, err):>10.4f}")
    print(f"{'learned query vs gain':<26}{np.corrcoef(q, gain)[0,1]:>9.4f}{_spearman(q, gain):>10.4f}"
          f"   <- what the head ACTUALLY needs to rank")

    # Oracle comparison: total realised gain when ranking by each signal. This is the ceiling
    # on the whole idea -- if oracle-gain barely beats oracle-error, retargeting buys nothing.
    err_s = err.reshape(-1, npl); gain_s = gain.reshape(-1, npl); q_s = q.reshape(-1, npl)
    print(f"\ntotal captured gain by selector, mean over {err_s.shape[0]} samples "
          f"(higher = better):")
    print(f"{'k':>6}{'random':>10}{'by err':>10}{'learned':>10}{'ORACLE gain':>13}"
          f"{'oracle/err':>12}")
    rng = np.random.default_rng(0)
    for k in (int(x) for x in args.ks.split(",")):
        if k > npl:
            continue
        def cap(sel):
            idx = np.argsort(-sel, axis=1)[:, :k]
            return float(np.mean(np.take_along_axis(gain_s, idx, axis=1).sum(1)))
        rnd = float(np.mean([gain_s[r, rng.choice(npl, k, replace=False)].sum()
                             for r in range(gain_s.shape[0])]))
        c_err, c_q, c_or = cap(err_s), cap(q_s), cap(gain_s)
        print(f"{k:>6}{rnd:>10.3f}{c_err:>10.3f}{c_q:>10.3f}{c_or:>13.3f}"
              f"{c_or/max(c_err,1e-9):>12.2f}x")
    print("\nRetargeting is worth it only if ORACLE gain clearly beats 'by err'.")


if __name__ == "__main__":
    main()
