"""Does mAnyUp's cross-attention entropy predict where the upsample is WRONG?

The premise of using entropy as a rule-based query selector: a high-res patch whose attention
is spread over many low-res patches is "uncertain" and therefore a good place to spend a
lookup. That only works if entropy actually correlates with reconstruction error. This
measures that on a TRAINED checkpoint, before any code is written to depend on it.

The attention matrix is already computed inside AnyUp (features = attn @ v), so entropy costs
nothing extra -- it is only not RETURNED by default. We capture it with a hook.

Reported per timestep-location, pooled over samples:
  pearson / spearman  entropy vs cosmse(F_lrhc_up, F_hrhc)
  overlap@k           how many of the true worst-k locations an entropy-ranked top-k finds,
                      against the learned query head and against chance -- the number that
                      decides whether entropy is a BETTER selector, not just a correlated one.

    source env_setup/env_olmo.sh
    python -m exp.upsamplers.check_attn_entropy --ckpt <timanyup ckpt> --n 64
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from exp.upsamplers.common import DATA_ROOT, FEATURES_ROOT, GUIDANCE_BANDS
from exp.upsamplers.data_timanyup import TriFeatureDataset, check_arms
from exp.upsamplers.train_timanyup import _import_timanyup, _flat_bt
from exp.common.paths import ANYUP_REPO


def _attach_attn_capture(upsampler, store):
    """AnyUp computes the attention weights and uses them (features = attn @ v) but returns
    None unless store_attn is set. Wrap both the block (to pass the flag down) and the inner
    CrossAttention (to grab the tensor) rather than editing the vendored repo."""
    blk = upsampler.cross_decode
    blk_fwd, attn_fwd = blk.forward, blk.cross_attn.forward

    def blk_wrap(q, k, v, **kw):
        kw["store_attn"] = True
        return blk_fwd(q, k, v, **kw)

    def attn_wrap(q, k, v, **kw):
        out, attn = attn_fwd(q, k, v, **kw)
        store["attn"] = attn
        return out, attn

    blk.forward, blk.cross_attn.forward = blk_wrap, attn_wrap


def _spearman(a, b):
    """Rank correlation without scipy: pearson on the ranks."""
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=64, help="test samples to pool over")
    p.add_argument("--split", default="test")
    p.add_argument("--ks", default="64,256,512,1024",
                   help="budgets at which to compare selector overlap with the true worst-k")
    p.add_argument("--features_root", default=str(FEATURES_ROOT))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--anyup_repo", default=str(ANYUP_REPO))
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

    store = {}
    _attach_attn_capture(model.upsampler, store)

    ent_all, err_all, q_all = [], [], []
    n = min(args.n, len(ds))
    print(f"scoring {n} {args.split} samples ({T} timesteps x {GH}x{GW} locations each)...")
    with torch.no_grad():
        for i in range(n):
            lrhc, hrlc, hrhc, guide = [x.unsqueeze(0).to(device) for x in ds[i]]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                up = model.upsample(lrhc, guide, (GH, GW))
                scores = model.query(lrhc, (GH, GW), lrhc_up=up)
                # project INSIDE autocast (the projector is fp32; a bf16 input errors out).
                # This mirrors _forward_losses: the error the mask must predict is measured
                # on the PROJECTED upsample, i.e. what the objective actually minimizes.
                up_proj = model.project(up)
            # entropy of the LAST captured attention == the final timestep chunk; upsample()
            # folds T into batch, so one call covers all T at once.
            a = store["attn"].float()                     # (B*T, GH*GW, n_key)
            ent = -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(-1)   # (B*T, GH*GW)
            ent = ent.reshape(1, T, GH, GW)
            err = A["cosmse_map"](_flat_bt(up_proj).float(),
                                  _flat_bt(hrhc).float()).reshape(1, T, GH, GW)
            ent_all.append(ent.cpu().numpy().ravel())
            err_all.append(err.cpu().numpy().ravel())
            q_all.append(scores.float().cpu().numpy().ravel())

    ent = np.concatenate(ent_all); err = np.concatenate(err_all); q = np.concatenate(q_all)
    print(f"\npooled {ent.size:,} locations")
    print(f"entropy  mean {ent.mean():.4f}  std {ent.std():.4f}  "
          f"range [{ent.min():.4f}, {ent.max():.4f}]  (max possible {np.log(lrhc0.shape[-1]*lrhc0.shape[-2]):.4f})")
    print(f"err      mean {err.mean():.4f}  std {err.std():.4f}")
    print(f"\n{'':<22}{'pearson':>9}{'spearman':>10}")
    print(f"{'entropy vs err':<22}{np.corrcoef(ent, err)[0,1]:>9.4f}{_spearman(ent, err):>10.4f}")
    print(f"{'learned query vs err':<22}{np.corrcoef(q, err)[0,1]:>9.4f}{_spearman(q, err):>10.4f}")

    # The decisive comparison: as a SELECTOR, per sample, at each budget.
    npl = T * GH * GW
    ent_s = ent.reshape(-1, npl); err_s = err.reshape(-1, npl); q_s = q.reshape(-1, npl)
    print(f"\noverlap with the true worst-k, averaged over {ent_s.shape[0]} samples:")
    print(f"{'k':>6}{'chance':>9}{'entropy':>9}{'learned':>9}")
    for k in (int(x) for x in args.ks.split(",")):
        if k > npl:
            continue
        true_k = np.argsort(-err_s, axis=1)[:, :k]
        def ov(sel):
            idx = np.argsort(-sel, axis=1)[:, :k]
            return np.mean([len(set(a) & set(b)) / k for a, b in zip(idx, true_k)])
        print(f"{k:>6}{k/npl:>9.3f}{ov(ent_s):>9.3f}{ov(q_s):>9.3f}")
    print("\n(entropy is a viable selector only if its overlap beats BOTH chance and learned)")


if __name__ == "__main__":
    main()
