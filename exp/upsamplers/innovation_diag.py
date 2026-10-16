"""Headroom diagnostic for a "where to sample next" selector (the GP / Bayesian-quadrature idea):
are the month-level INNOVATIONS -- how far a month's true value is from what the season tokens
already say -- concentrated in a few (month, cell) slots, and can cheap pre-look statistics find
them?

For every present (month t, ps4 cell p) of a sample:
    season   the crop-free pass (season tokens through all blocks) -> the native head's logits
             of t's season token over p's 4 x 4 px                       L_season(t, p)
    crop     a pass with EVERY present (month, cell) cropped at ps4 -> that crop's native
             logits (16 x 16 map area-downsampled to 4 x 4)             L_crop(t, p)
    innovation(t, p) = mean over the 4 x 4 px of |L_crop - L_season|^2 (summed over classes);
             classification also: mean |softmax(L_crop) - softmax(L_season)|_1 (bounded)

Concentration (per sample over its present slots, then the mean over samples):
    top-q share      share of the sample's total innovation in its top q of slots (uniform: q)
    cell top-q       the same over CELLS (innovation summed over a cell's months): where
    month top-k      within each cell, the share of its top-k months (uniform: k / n months): when

Predictability: each pre-look statistic ranks the slots; "capture@q" = share of the total
innovation inside that statistic's top q (random: q, perfect: the top-q share), and Spearman rho:
    month_dev     |e_t(p) - season mean of e(p)|: the month's raw ps4 patch embedding vs its
                  season's (the "month that differs"), cheap
    pixel_var     variance of the month's raw input pixels inside p (spatial diversity at ps1)
    season_change |F_s - F_{s-1}| + |F_{s+1} - F_s| of the contextual (crop-free, all blocks)
                  season tokens at p's ps16 cell (season-level change; constant within a season)
    entropy       predictive entropy of the crop-free prediction over p (classification)
    combined      mean of the per-sample ranks of the above

    python -m exp.upsamplers.innovation_diag --task pastis \
        --run inject_fft_pastis_tiny_random_L0_k1_native_set_lr0.001_ep50 [--split val]
Writes results/<task>/innovation_diag_<run>.json.
"""
import os

from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from manyup.encoders.olmoearth import _to_device, crop_patches  # noqa: E402
from manyup.seasonal import season_mean  # noqa: E402

QS = (0.1, 0.2, 0.3)


def ranks(v):
    """(n,) -> ranks in [0, 1] (ties broken by order)."""
    r = torch.empty_like(v)
    r[v.argsort()] = torch.arange(len(v), dtype=v.dtype, device=v.device)
    return r / max(len(v) - 1, 1)


def spearman(a, b):
    ra, rb = ranks(a) - 0.5, ranks(b) - 0.5
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-12))


def top_share(inn, key, q):
    """Share of sum(inn) in the top q fraction of entries ranked by key."""
    n = max(1, round(q * len(inn)))
    return float(inn[key.argsort(descending=True)[:n]].sum() / inn.sum().clamp(min=1e-12))


@torch.no_grad()
def prelook(m, ms, ss, season, present):
    """What a selector may see BEFORE looking, exactly as dumped for fit_innovation (one
    function for the dump and for test-time selection, so the two cannot drift apart).
    ms, ss, season, present already on the GPU. -> (lr: season tokens after the head blocks,
    for a later crop pass; dict of half-precision pre-look inputs):
        e          (B, T, 16, 16, D)  each month's raw ps4 patch embedding
        F0         (B, G, 4, 4, D)    crop-free contextual season tokens (all blocks)
        s_logits   (B, G, 16, 16, C)  crop-free native logits of each season, per ps4 cell
        pixel_var  (B, T, 16, 16)     each month's raw-pixel variance inside each ps4 cell
        season, present
    """
    mix, lin = m.mix, m.head.linear
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = mix.head_blocks(mix.embed(ss, 16))
        lr_x0, _ = mix.tail(lr, None, None)                                     # crop-free
        e = m.month_embedding(ms).float()                                       # (B,T,H,W,D)
    F0 = lr.pooled(lr_x0).float()                                               # (B,G,h,w,D)
    B, G = F0.shape[:2]
    if lin.out_features % 256 == 0 and m.native:    # native head: a ps16 token's 16 x 16 px
        C = lin.out_features // 256
        s_cell = rearrange(lin(F0).view(B, G, 4, 4, C, 4, 4, 4, 4),
                           "B G h w C i u j v -> B G (h i) (w j) C u v").mean((-1, -2))
    else:                         # feature-space head (4 x 4 px per ps4 cell): the season
        C = lin.out_features // 16   # token's mean logits, the same for its 16 ps4 cells
        s_cell = lin(F0).view(B, G, 4, 4, C, 16).mean(-1)
        s_cell = s_cell.repeat_interleave(4, 2).repeat_interleave(4, 3)
    pix = 0
    for k, val in ms.as_dict().items():
        if torch.is_tensor(val) and val.ndim == 5 and not k.endswith("_mask"):
            u = rearrange(val.float(), "B (Y u) (X w) T c -> B T Y X (u w) c", u=4, w=4)
            pix = pix + u.var(4).mean(-1)                                       # (B, T, H, W)
    return lr, {"e": e.half(), "F0": F0.half(), "s_logits": s_cell.half(), "season": season,
                "present": present, "pixel_var": pix.float()}


@torch.no_grad()
def innovation(m, lr, F0, ms, season, present, frac):
    """Crop EVERY present (month, ps4 cell) in one pass -> per-slot innovation (B, T, 16, 16):
    mean over the slot's 4 x 4 px of |crop logits - its season token's logits|^2 (summed over
    classes); plus (slot indices, crop logits (N, C, 16), season logits (N, C, 16))."""
    mix, lin = m.mix, m.head.linear
    B, T = present.shape
    H = fi.OUT_GRID
    where = present[:, :, None, None].expand(B, T, H, H)
    b, t, y, x = where.nonzero(as_tuple=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hr = mix.embed(crop_patches(ms, b, t, y, x, 4), 4,
                       slot=m._crop_slot(season[b, t], frac[b, t]), py=y * 4, px=x * 4)
        hr = mix.head_blocks(hr)
        _, hr_x = mix.tail(lr, hr, b)
    v = hr.pooled(hr_x).float()[:, 0, 0, 0]                                     # (N, D)
    N, C, G = len(b), lin.out_features // 256, F0.shape[1]
    L_crop = F.avg_pool2d(lin(v).view(N, C, 16, 16), 4).reshape(N, C, 16)
    s_map = lin(F0).view(B, G, 4, 4, C, 16, 16)[b, season[b, t].clamp(min=0), y // 4, x // 4]
    s_map = rearrange(s_map, "n c (i u) (j w) -> n i j c (u w)", u=4, w=4)
    L_seas = s_map[torch.arange(N, device=F0.device), y % 4, x % 4]             # (N, C, 16)
    inn = ((L_crop - L_seas) ** 2).sum(1).mean(-1)                              # (N,)
    g = torch.zeros(B, T, H, H, device=F0.device)
    g[b, t, y, x] = inn
    return g, (b, t, y, x), L_crop, L_seas


@torch.no_grad()
def main():
    olmo_bootstrap.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", required=True, help="weights at checkpoints/<run>.pt (native_set)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--dump", default="", help="also save per-batch inputs + innovations here "
                    "(for exp.upsamplers.fit_innovation)")
    ap.add_argument("--max_samples", type=int, default=0, help="stop after this many samples")
    a = ap.parse_args()
    task = TASKS[a.task]
    cls = task.kind == "classification"
    m = fi.InjectFT(task, "tiny", 0, "random", 1.0, "slot", readout="native_set").cuda().eval()
    m.load_state_dict(torch.load(f"checkpoints/{a.run}.pt", map_location="cpu",
                                 weights_only=False)["model"])
    lin, mix = m.head.linear, m.mix
    dl = fi.loader(task, a.split, a.batch_size, 8, False)
    names = ["month_dev", "pixel_var", "season_change"] + (["entropy"] if cls else []) + ["combined"]
    acc = {k: [] for k in ["top", "cell_top", "month_top1", "month_top2", "month_uniform1",
                           "month_uniform2"] + [f"cap_{n}" for n in names]
           + [f"rho_{n}" for n in names] + (["top_prob"] if cls else [])}
    if a.dump:
        Path(a.dump).mkdir(parents=True, exist_ok=True)
    seen = 0
    for i, (ms, ss, season, present, pairs, extra, _) in enumerate(dl):
        if a.limit_batches and i >= a.limit_batches:
            break
        if a.max_samples and seen >= a.max_samples:
            break
        seen += present.shape[0]
        dev = "cuda"
        ms, ss = _to_device(ms, dev, torch.bfloat16), _to_device(ss, dev, torch.bfloat16)
        season, present = season.to(dev), present.to(dev)
        B, T = present.shape
        G, H = ss.timestamps.shape[1], fi.OUT_GRID
        lr, pre = prelook(m, ms, ss, season, present)
        F0, e, pix = pre["F0"].float(), pre["e"].float(), pre["pixel_var"]
        inn_g, (b, t, y, x), L_crop, L_seas = innovation(m, lr, F0, ms, season, present,
                                                         extra["frac"].to(dev))
        N, C = len(b), lin.out_features // 256
        ar = torch.arange(N, device=dev)
        inn = inn_g[b, t, y, x]                                                 # (N,)
        inn_p = ((L_crop.softmax(1) - L_seas.softmax(1)).abs().sum(1).mean(-1)
                 if cls else None)
        # pre-look statistics per slot
        e_bar = season_mean(e, season, present, G)
        month_dev = (e - e_bar).norm(dim=-1)[b, t, y, x]
        pixel_var = pix[b, t, y, x]
        sidx = season[b, t].clamp(min=0)
        Fc = F0[b, :, y // 4, x // 4]                                           # (N, G, D)
        prev = (Fc[ar, sidx] - Fc[ar, (sidx - 1).clamp(min=0)]).norm(dim=-1)
        nxt = (Fc[ar, (sidx + 1).clamp(max=G - 1)] - Fc[ar, sidx]).norm(dim=-1)
        season_change = prev + nxt
        stats = {"month_dev": month_dev, "pixel_var": pixel_var, "season_change": season_change}
        if cls:
            p0 = L_seas.softmax(1)
            stats["entropy"] = -(p0 * p0.clamp(min=1e-8).log()).sum(1).mean(-1)
        if a.dump:   # everything a pre-look predictor may use, + the innovation it predicts
            def grid(v):
                g = torch.zeros(B, T, H, H, device=dev)
                g[b, t, y, x] = v.float()
                return g
            torch.save({**{k: v.cpu() for k, v in pre.items()},     # see prelook()
                        "inn": inn_g.cpu(),                          # (B,T,16,16) target
                        "inn_p": grid(inn_p).cpu() if cls else None},
                       Path(a.dump) / f"batch{i:05d}.pt")
        for bb in range(B):
            s = b == bb
            if s.sum() < 10:
                continue
            ib = inn[s]
            for q in QS:
                acc["top"].append((q, top_share(ib, ib, q)))
                if cls:
                    acc["top_prob"].append((q, top_share(inn_p[s], inn_p[s], q)))
            # cells: total over months; months: within each cell
            cell = y[s] * H + x[s]
            tot = torch.zeros(H * H, device=dev).index_add_(0, cell, ib)
            used = torch.zeros(H * H, device=dev).index_add_(0, cell, torch.ones_like(ib)) > 0
            for q in QS:
                acc["cell_top"].append((q, top_share(tot[used], tot[used], q)))
            grid = torch.zeros(H * H, T, device=dev)
            grid[cell, t[s]] = ib
            srt = grid[used].sort(1, descending=True).values
            n_m = (grid[used] > 0).sum(1).clamp(min=1).float()
            acc["month_top1"].append((0, float((srt[:, 0] / srt.sum(1).clamp(min=1e-12)).mean())))
            acc["month_top2"].append((0, float((srt[:, :2].sum(1) / srt.sum(1).clamp(min=1e-12)).mean())))
            acc["month_uniform1"].append((0, float((1 / n_m).mean())))
            acc["month_uniform2"].append((0, float((2 / n_m).clamp(max=1).mean())))
            rk = []
            for n, st in stats.items():
                sv = st[s].float()
                rk.append(ranks(sv))
                acc[f"rho_{n}"].append((0, spearman(sv, ib)))
                for q in QS:
                    acc[f"cap_{n}"].append((q, top_share(ib, sv, q)))
            comb = torch.stack(rk).mean(0)
            acc["rho_combined"].append((0, spearman(comb, ib)))
            for q in QS:
                acc["cap_combined"].append((q, top_share(ib, comb, q)))
    out = {}
    for k, vals in acc.items():
        by_q = {}
        for q, val in vals:
            by_q.setdefault(q, []).append(val)
        out[k] = {str(q): sum(v) / len(v) for q, v in by_q.items()}
    out["n_samples"] = len({1 for _ in acc["rho_month_dev"]}) and len(acc["rho_month_dev"])
    out["run"], out["split"] = a.run, a.split
    print(json.dumps(out, indent=1))
    if not a.limit_batches:
        tag = "" if a.split == "val" and not a.max_samples else (
            f"_{a.split}" + (f"_n{a.max_samples}" if a.max_samples else ""))
        p = Path(f"results/{a.task}/innovation_diag_{a.run}{tag}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("wrote", p, "job", os.environ.get("SLURM_JOB_ID", ""))


if __name__ == "__main__":
    main()
