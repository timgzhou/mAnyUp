"""Where would an attention-divergence heuristic spend high-res lookups? (seasonal timAnyUp probe)

The proposed method encodes a SEASONAL low-res series (months pooled into groups, OlmoEarth at
ps16) and upsamples every MONTH from its season's tokens, guided by that month's own image.
Within a season all months attend over the same value tokens and differ only in their
attention, so where the months' attention maps disagree the season token is a poor stand-in
for the month -- the candidate place to re-encode at high resolution.

This renders what that heuristic selects with an already-trained per-timestep mAnyUp
(manyup_t; GEOID's plain manyup is already per timestep), before any selector or re-encoder
exists, and checks it against the error it is meant to find:

    groups       PASTIS / BioMassters: 4 seasons of 3 consecutive months; GEOID: the pre/post
                 pair is one group
    lr           each group composited (mean over valid frames/pixels) and encoded by
                 OlmoEarth tiny, series, ps16 tile64 -- the lr cache's own setup; timestamp =
                 mean date of the group's months
    A_m          attention (16x16 hr patches x 4x4 lr tokens) upsampling month m from its
                 group's tokens with month m's guidance
    kl_mean      KL(A_m || mean of the group's A)   -- sums over the group to the generalized
                 Jensen-Shannon divergence, so each month's value is its share of the
                 disagreement ("this month is the outlier")
    kl_season    KL(A_m || A_season), A_season guided by the group's mean image (the image the
                 lr tokens were encoded from); one extra upsample per group
    img_diff     CONTROL: mean |guide_m - guide_season| pooled to the 16x16 grid. If the
                 attention scores do no better than this, they only measure image change.
    err          cosmse(season->month upsample, cached monthly ps4 tile64 target): what a
                 lookup would have to fix. err_own is the same upsample from the month's OWN
                 lr tokens (the cache), i.e. the error without seasonal pooling.

Stats, per sample then averaged (printed, and stats.csv next to the figures):
    rho          Spearman of score vs err over all (t, y, x)
    rho_month    Spearman of per-month MEANS: does the score pick the right month?
    rho_space    Spearman within each month, averaged: given the month, the right place?
    rho_dmg      as rho, vs err - err_own (the error that seasonal pooling ADDED)
    ov@f         overlap of the score's top-k with the true worst-k, k = f * T * 16 * 16
                 (chance = f)

A month is upsampled only if EVERY guidance modality has a frame within that month
(manyup.data.guidance.guidance_present, which the upsamplers now
apply themselves). build_guidance_batch's "nearest" would otherwise fill e.g. a month without
S2 with a NEIGHBOURING month's S2, so the month would be guided by another date's image -- its
attention then copies that month's, and its error (vs an S1-only target) swamps every score.
Such months are not upsampled, scored, selected or counted; the figures blank them.

    source env_setup/env_olmo.sh
    python -m exp.upsamplers.viz_attn_divergence --task pastis \\
        --ckpt checkpoints/upsamplers/manyup_t_pastis_tiny_ep20.pt
"""
from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers.task_data import OUT_GRID, UpsamplerData  # noqa: E402
from manyup.data.guidance import (GuidanceSpec, build_guidance_batch,  # noqa: E402
                                  guidance_present)
from manyup.data.sample import MTMMBatch  # noqa: E402
from manyup.data.time import from_days  # noqa: E402
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput  # noqa: E402
from manyup.loss import cosmse_map  # noqa: E402
from manyup.upsamplers import MAnyUp  # noqa: E402

SCORES = ("kl_mean", "kl_season", "img_diff")
EPS = 1e-8


# -- seasonal low-res input ---------------------------------------------------------------
def groups_for(T: int, size: int) -> list[list[int]]:
    if T % size:
        raise ValueError(f"T={T} does not split into groups of {size}")
    return [list(range(g, g + size)) for g in range(0, T, size)]


def composite(batch: MTMMBatch, groups: list[list[int]]) -> MTMMBatch:
    """Mean of each group's valid frames (and valid pixels), per modality -> G steps.

    Every modality gets the SAME timestamp per group -- the mean midpoint of the group's
    steps, whether or not they have data -- because OlmoEarth needs aligned time axes and a
    per-modality validity-weighted date would split them."""
    mods = {}
    for name, mb in batch.modalities.items():
        x = mb.data.float()                                                # (B,T,C,H,W)
        m = mb.frame_valid[:, :, None, None, None]
        if mb.pixel_valid is not None:
            m = m & mb.pixel_valid[:, :, None]
        m = m.expand_as(x)
        data, fv, pv = [], [], []
        for g in groups:
            xs, ms = x[:, g], m[:, g]
            cnt = ms.sum(1)
            data.append(torch.where(cnt > 0, (xs * ms).sum(1) / cnt.clamp(min=1),
                                    torch.zeros(())))
            fv.append(mb.frame_valid[:, g].any(1))
            pv.append(cnt[:, 0] > 0)
        mid = (mb.t_start + mb.t_end).double() / 2
        date = torch.stack([mid[:, g].mean(1).round().long() for g in groups], 1)
        mods[name] = replace(mb, data=torch.stack(data, 1), t_start=date, t_end=date.clone(),
                             frame_valid=torch.stack(fv, 1),
                             pixel_valid=(torch.stack(pv, 1) if mb.pixel_valid is not None
                                          else None),
                             aux={}, history=mb.history + ("season_mean",))
    return replace(batch, modalities=mods)


# -- attention capture --------------------------------------------------------------------
@contextmanager
def capture_attention(model):
    """AnyUp computes attn (features = attn @ v) but only returns it when store_attn is set,
    which CrossAttentionBlock never passes down. Force it with hooks rather than editing the
    vendored repo. Yields a dict whose "attn" is (N, H*W, h*w), head-averaged."""
    store = {}
    ca = model.cross_decode.cross_attn

    def pre(_, args, kwargs):
        return args, {**kwargs, "store_attn": True}

    def post(_, args, kwargs, out):
        store["attn"] = out[1]

    hs = [ca.register_forward_pre_hook(pre, with_kwargs=True),
          ca.register_forward_hook(post, with_kwargs=True)]
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


def upsample(model, feats, guide, size):
    """feats (N,D,h,w), guide (N,C,H,W) -> (out (N,D,*size), attn (N,prod(size),h*w))."""
    with capture_attention(model) as st:
        out = model(guide.float(), feats.float(), size)
    return out, st["attn"].float()


def kl(p, q):
    """KL(p || q) over the last dim."""
    p, q = p.clamp_min(EPS), q.clamp_min(EPS)
    return (p * (p.log() - q.log())).sum(-1)


# -- metrics ------------------------------------------------------------------------------
def spearman(a, b):
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_flat(x, k):
    """Bool mask of the k largest entries of x (any shape)."""
    idx = np.argsort(-x.ravel(), kind="stable")[:k]
    m = np.zeros(x.size, bool)
    m[idx] = True
    return m.reshape(x.shape)


# -- rendering ----------------------------------------------------------------------------
def step_label(days: int, T: int) -> str:
    """Month name for a monthly series; the full date for a short one (GEOID's pre and
    post often fall in the same month)."""
    return from_days(int(days)).strftime("%b" if T > 2 else "%Y-%m-%d")


def show_target(ax, task, tgt):
    t = tgt.numpy().astype(float)
    if task.kind == "classification":
        t = np.ma.masked_less(t, 0)
        n = task.num_outputs
        ax.imshow(t, cmap=plt.get_cmap("tab20", 20) if n > 3 else
                  matplotlib.colors.ListedColormap(["#d9d9d9", "#08306b", "#e31a1c"]),
                  vmin=0, vmax=(19 if n > 3 else 2), interpolation="nearest")
    else:
        ax.imshow(t, cmap="viridis", interpolation="nearest")
    ax.set_title("target", fontsize=8)


def overlay(ax, rgb, mask, color):
    ax.imshow(rgb)
    up = np.kron(mask, np.ones((rgb.shape[0] // mask.shape[0],) * 2))
    col = np.zeros((*up.shape, 4))
    col[..., :3] = matplotlib.colors.to_rgb(color)
    col[..., 3] = 0.55 * up
    ax.imshow(col)


def render(path, task, sid, target, rgb, maps, sel, worst, months, groups, present, missing,
           k):
    T = len(months)
    rows = [("guidance", None), ("kl_mean", "magma"), ("kl_season", "magma"),
            ("err (season→month)", "viridis"), (f"top-{k} by kl_mean", None),
            (f"true worst-{k} err", None)]
    fig, axes = plt.subplots(len(rows), T + 1, figsize=(1.35 * (T + 1), 1.4 * len(rows)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    show_target(axes[0, 0], task, target)
    for r, (name, _) in enumerate(rows):
        axes[r, 1].set_ylabel(name, fontsize=8)
    vmax = {n: np.nanpercentile(maps[n][present], 99) for n in ("kl_mean", "kl_season", "err")}
    first = {g[0] for g in groups}
    for t in range(T):
        c = t + 1
        axes[0, c].set_title(months[t] + ("  ‖" if t in first and t else ""), fontsize=7,
                             color="black" if present[t] else "#b2182b")
        if not present[t]:                  # not upsampled: nothing to show but the reason
            axes[0, c].text(0.5, 0.5, f"no {missing[t]}\nnot upsampled", ha="center",
                            va="center", fontsize=7, color="#b2182b",
                            transform=axes[0, c].transAxes)
            continue
        axes[0, c].imshow(rgb[t])
        for r, n in ((1, "kl_mean"), (2, "kl_season"), (3, "err")):
            axes[r, c].imshow(maps[n][t], cmap=rows[r][1], vmin=0, vmax=vmax[n],
                              interpolation="nearest")
        overlay(axes[4, c], rgb[t], sel[t], "#e31a1c")
        overlay(axes[5, c], rgb[t], worst[t], "#00bcd4")
        if t in first and t:                               # season boundary
            for r in range(len(rows)):
                axes[r, c].axvline(-0.5, color="black", lw=2, clip_on=False)
    fig.suptitle(f"{task.name}  {sid}   shared colour scale per row (99th pct)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# -- main ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--ckpt", required=True, help="a per-timestep mAnyUp checkpoint")
    ap.add_argument("--model_size", default="tiny")
    ap.add_argument("--group", type=int, default=None,
                    help="months per season (default 3; GEOID: the whole pre/post pair)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=128, help="samples the stats pool over")
    ap.add_argument("--n_fig", type=int, default=6, help="figures, spread over the --n")
    ap.add_argument("--k_fracs", default="0.05,0.1667,0.33",
                    help="lookup budgets (fraction of T x 16 x 16) for overlap@k; the "
                         "figures use the middle one")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_dir", default=None,
                    help="default: results/<task>/viz_attn_divergence/<ckpt stem>")
    a = ap.parse_args()

    task = TASKS[a.task]
    device = torch.device("cuda")
    up = MAnyUp.from_checkpoint(a.ckpt).to(device)
    if up.time_matching == "pooled":
        raise SystemExit(f"{a.ckpt} is time-pooled; the heuristic needs a per-timestep mAnyUp")
    model = up.model
    inputs = OlmoEarthInput(task.olmoearth.modalities, task.olmoearth.impute)
    enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID[a.model_size], inputs, 16, 64,
                           name=a.model_size).to(device).eval()
    rgb_spec = GuidanceSpec(task.rgb, None, norm="percentile")

    ds = UpsamplerData(task, a.split, a.model_size, ("lr", "target"))
    n = min(a.n, len(ds))
    fig_at = set(np.linspace(0, n - 1, a.n_fig).round().astype(int).tolist())
    loader = DataLoader(Subset(ds, range(n)), batch_size=a.batch_size, num_workers=4,
                        collate_fn=ds.collate)
    k_fracs = [float(x) for x in a.k_fracs.split(",")]
    out = Path(a.out_dir or f"results/{task.name}/viz_attn_divergence/{Path(a.ckpt).stem}")
    out.mkdir(parents=True, exist_ok=True)

    rows, idx, checked, n_skip, n_steps = [], 0, False, 0, 0
    size = (OUT_GRID, OUT_GRID)
    for batch in loader:
        lr, tgt, guid = batch.feats["lr"], batch.feats["target"], batch.guidance
        B, T = lr.tokens.shape[:2]
        groups = groups_for(T, a.group or (3 if T >= 3 else T))
        g_of = torch.tensor([gi for gi, g in enumerate(groups) for _ in g])
        span_s = torch.stack([lr.t_start[:, g].min(1).values for g in groups], 1)
        span_e = torch.stack([lr.t_end[:, g].max(1).values for g in groups], 1)

        with torch.no_grad():
            if not checked:        # our encoder must reproduce the lr cache it stands in for
                own = enc.encode(guid, "series").tokens.float().cpu()
                rel = ((own - lr.tokens.float()).norm(dim=-1)
                       / lr.tokens.float().norm(dim=-1)).mean()
                print(f"encoder vs lr cache: mean relative diff {rel:.2e}")
                if rel > 0.02:
                    raise SystemExit("encoder does not reproduce the lr cache; check setup")
                checked = True
            season = enc.encode(composite(guid, groups), "series").tokens.float()  # (B,G,..)
            G = season.shape[1]
            gm = build_guidance_batch(guid, up.guidance, match="nearest",
                                      f_start=lr.t_start, f_end=lr.t_end).to(device)
            gs = build_guidance_batch(guid, up.guidance, match="within_range",
                                      f_start=span_s, f_end=span_e).to(device)
            C, H, W = gm.shape[2:]
            feats = season[:, g_of].permute(0, 1, 4, 2, 3).reshape(B * T, -1, 4, 4)
            y_m, A_m = upsample(model, feats, gm.reshape(B * T, C, H, W), size)
            _, A_s = upsample(model, season.permute(0, 1, 4, 2, 3).reshape(B * G, -1, 4, 4),
                              gs.reshape(B * G, C, H, W), size)
            own_f = lr.tokens.float().to(device).permute(0, 1, 4, 2, 3).reshape(B * T, -1, 4, 4)
            y_own, _ = upsample(model, own_f, gm.reshape(B * T, C, H, W), size)

            A_m = A_m.reshape(B, T, *A_m.shape[1:])
            A_s = A_s.reshape(B, G, *A_s.shape[1:])
            present = lr.frame_valid & guidance_present(guid, up.guidance, lr.t_start,
                                                        lr.t_end)            # (B,T)
            valid = present.to(device)
            A_bar = torch.stack([(A_m[:, g] * valid[:, g, None, None]).sum(1)
                                 / valid[:, g].sum(1).clamp(min=1)[:, None, None]
                                 for g in groups], 1)
            grid = lambda x: x.reshape(B, T, *size).cpu().numpy()          # noqa: E731
            scores = {
                "kl_mean": grid(kl(A_m, A_bar[:, g_of])),
                "kl_season": grid(kl(A_m, A_s[:, g_of])),
                "img_diff": grid(F.adaptive_avg_pool2d(
                    (gm - gs[:, g_of]).abs().mean(2), size)),
            }
            want = tgt.tokens.float().to(device).permute(0, 1, 4, 2, 3).reshape(B * T, -1, *size)
            err = grid(cosmse_map(y_m.float(), want))
            err_own = grid(cosmse_map(y_own.float(), want))
            rgb = build_guidance_batch(guid, rgb_spec, match="nearest", f_start=lr.t_start,
                                       f_end=lr.t_end).permute(0, 1, 3, 4, 2).clamp(0, 1)
        vnp = present.numpy()
        lacks = {m: ~guid.modalities[m].frame_valid.numpy() for m in up.guidance.bands
                 if guid.modalities[m].t_start is not None}
        n_skip += int((~vnp).sum())
        n_steps += vnp.size

        for b in range(B):
            v = vnp[b]
            e, dmg = err[b][v], (err[b] - err_own[b])[v]
            nv = e.size
            row = {"sample": batch.ids[b], "err": float(e.mean()),
                   "err_own": float(err_own[b][v].mean())}
            for s in SCORES:
                x = scores[s][b][v]
                row[f"rho_{s}"] = spearman(x.ravel(), e.ravel())
                row[f"rho_month_{s}"] = spearman(x.mean((1, 2)), e.mean((1, 2)))
                row[f"rho_space_{s}"] = float(np.nanmean(
                    [spearman(x[t].ravel(), e[t].ravel()) for t in range(len(x))]))
                row[f"rho_dmg_{s}"] = spearman(x.ravel(), dmg.ravel())
                for f in k_fracs:
                    k = max(1, round(f * nv))
                    row[f"ov{f}_{s}"] = float((topk_flat(x, k) & topk_flat(e, k)).sum() / k)
            rows.append(row)

            if idx in fig_at:
                kf = k_fracs[len(k_fracs) // 2]
                k = max(1, round(kf * v.sum() * OUT_GRID * OUT_GRID))
                sc = np.where(vnp[b][:, None, None], scores["kl_mean"][b], -np.inf)
                eb = np.where(vnp[b][:, None, None], err[b], -np.inf)
                render(out / f"{idx:04d}_{batch.ids[b].replace('/', '_')}.png", task,
                       batch.ids[b], batch.target[b], rgb[b].numpy(),
                       {"kl_mean": scores["kl_mean"][b], "kl_season": scores["kl_season"][b],
                        "err": err[b]},
                       topk_flat(sc, k), topk_flat(eb, k),
                       [step_label(d, T) for d in lr.t_start[b]], groups, v,
                       ["+".join(m for m, x in lacks.items() if x[b, t]) for t in range(T)], k)
            idx += 1

    with open(out / "stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mean = lambda key: np.nanmean([r[key] for r in rows])                  # noqa: E731
    print(f"\n{task.name}: {len(rows)} {a.split} samples, groups of {len(groups[0])}; "
          f"{n_skip}/{n_steps} steps lack a guidance modality, not upsampled  ->  {out}")
    print(f"mean err  season->month {mean('err'):.4f}   own month lr {mean('err_own'):.4f}")
    cols = ("rho", "rho_month", "rho_space", "rho_dmg")
    print(f"{'score':>10} " + " ".join(f"{c:>9}" for c in cols) + " "
          + " ".join(f"{'ov@' + format(f, 'g'):>9}" for f in k_fracs))
    for s in SCORES:
        print(f"{s:>10} " + " ".join(f"{mean(f'{c}_{s}'):9.3f}" for c in cols) + " "
              + " ".join(f"{mean(f'ov{f}_{s}'):9.3f}" for f in k_fracs))
    print(f"{'chance':>10} " + " ".join(f"{0:9.3f}" for _ in cols) + " "
          + " ".join(f"{f:9.3f}" for f in k_fracs))


if __name__ == "__main__":
    main()
