"""Paper material, model: run the trained mixed-resolution PASTIS models on ONE test tile and
keep every intermediate the model / readout figures show.

    source env_setup/env_olmo.sh && python -m exp.viz.paper.collect [--tile 40091@r0c0]

-> figures/paper/model/collect_<tile>.pt (read by exp.viz.paper.model and .readout)

Two trained checkpoints are used, because no single run has a learned selector at both levels:
    --month_run   a --joint run: its predictor scores every (month, ps4 cell) from the crop-free
                  pre-look; the top k = k_frac x 12 x 256 are the month crops (eps = 0: no
                  random share)
    --date_run    a --date_level free --date_selector xattn run (months random in training): it
                  is given the month picks above (InjectFT.fixed_scores), scores every
                  (modality, date, ps4 cell) itself and takes the top 2k as date crops; all
                  features, the readout and the prediction come from this model
`--month_picks random` instead keeps the date run exactly as it was trained (random months).

Saved (one tile, so no batch axis; T = 12 months, G = 4 seasons, H = W = 16 ps4 cells):
    images     raw frames per modality with their dates, monthly and seasonal composites, as
               the dataset returns them (before OlmoEarth normalization)
    F_free     (G, 4, 4, D)   season tokens of the crop-free pass (what a selector sees)
    F_season   (G, 4, 4, D)   season tokens of the mixed pass (after joint attention with crops)
    tok        (N, D) final crop tokens; tok_level 0 = month crop, 1 = date crop; tok_mod
               (-1 for months), tok_step (month index / date index in its modality), tok_month,
               tok_y, tok_x (ps4 cell)
    month_score (T, H, W), month_mask (T, H, W); date_score[m] (T_m, H, W), date_cand[m]
    readout    wgt (Q, K), cand (Q, K, C, 16), cid (Q, M), season prior, logits (C, 64, 64)
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from manyup.encoders.olmoearth import _to_device  # noqa: E402

OUT = Path("figures/paper/model")
MONTH_RUN = "inject_fft_pastis_tiny_random_L0_k0.167_native_set_joint0.5-0.5a0.5_gfix_warm_lr0.001_ep20"
DATE_RUN = "inject_fft_pastis_tiny_random_L0_k0.167_native_set_frac_datefree2_dxattn_lr0.001_ep50"


def build(run: str):
    """The InjectFT of a finished run, with its best weights (as finetune_inject.main builds it)."""
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    g = lambda k, d: getattr(a, k, d)  # noqa: E731  (older runs lack newer flags)
    res_mix = tuple(float(v) for v in a.res_mix.split(",")) if g("res_mix", None) else None
    m = fi.InjectFT(TASKS[a.task], a.model_size, a.inject_layer, a.selector, a.k_frac,
                    a.select_unit, a.selector_hidden, a.pick, a.head, a.score_norm,
                    a.sel_temporal, a.readout, a.lr_patch, a.frac_slot, a.date_level,
                    a.date_per_month, not a.no_month_crops, a.date_selector, a.date_gate,
                    res_mix, g("joint", False), g("lr_tile", fi.TILE), g("joint_res", False),
                    g("pair_match_random", False), 0, g("pos", "corner"), g("doy", False),
                    g("attn", "joint"), g("head_attn", "alone")).cuda().eval()
    m.load_state_dict(ck["model"])
    return m, a


def frames(md):
    """A sample's modality -> what the figures draw."""
    return {"data": md.data.float(), "bands": md.bands, "start": md.time.start, "end": md.time.end,
            "valid": md.frame_valid,
            "pixel_valid": md.pixel_valid if md.pixel_valid is not None else None}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", default="40091@r0c0")
    ap.add_argument("--month_run", default=MONTH_RUN)
    ap.add_argument("--date_run", default=DATE_RUN)
    ap.add_argument("--month_picks", default="joint", choices=("joint", "random"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    task = TASKS["pastis"]
    lr_pol = fi.GroupSteps(task.lr_policy()) if isinstance(task.lr_policy(), int) else task.lr_policy()
    ds = fi.WithDates(task, "test", lr_pol)
    i = ds.raw.index_of(a.tile)
    item = ds[i]
    months, seasons, raw = item
    ms, ss, season, present, pairs, extra, target = fi.Collate(task, dates=True)([item])
    out = {"tile": a.tile, "month_run": a.month_run, "date_run": a.date_run,
           "month_picks": a.month_picks, "target": target[0],
           "images": {"raw": {m: frames(md) for m, md in raw.modalities.items()},
                      "month": {m: frames(md) for m, md in months.modalities.items()},
                      "season": {m: frames(md) for m, md in seasons.modalities.items()}},
           "season_of_month": season[0], "present": present[0], "frac": extra["frac"][0],
           "date_month": {m: v[2][0] for m, v in extra["dates"].items()},
           "date_valid": {m: v[1][0] for m, v in extra["dates"].items()}}

    # ---- month picks: the joint predictor, eps = 0
    B, T, H = 1, present.shape[1], fi.OUT_GRID
    if a.month_picks == "joint":
        mj, _ = build(a.month_run)
        mj.joint_eps = 0.0
        logits_j, _, n_j = mj(ms, ss, season, present, pairs, extra)
        mask, _, _ = mj.last_pick
        out["month_score"] = mj._joint_p[0].float().cpu()
        out["month_model_logits"] = logits_j[0].float().cpu()
        fixed = mask.float() + 1e-3 * torch.rand(mask.shape, device=mask.device)
        del mj
    else:
        fixed = torch.rand(B, T, H, H, device="cuda")
        out["month_score"] = fixed[0].cpu()

    # ---- the date model, with those month picks; everything else is its own
    md_, args = build(a.date_run)
    md_.fixed_scores = fixed
    rec = {"crops": [], "date_z": None, "readout": None}
    crop_patches = fi.crop_patches

    def rec_crops(sample, b, t, y, x, size):
        rec["crops"].append((sample.timestamps.shape[1], t.cpu(), y.cpu(), x.cpu()))
        return crop_patches(sample, b, t, y, x, size)

    date_scores = md_._date_scores

    def rec_date_scores(views, cands, *rest):
        zs, shapes = date_scores(views, cands, *rest)
        rec["date_z"] = {v[0]: z.view(sh)[0].float().cpu() for v, z, sh in zip(views, zs, shapes)}
        rec["date_cand"] = {v[0]: c[0].cpu() for v, c in zip(views, cands)}
        return zs, shapes

    readout = md_.readout.forward

    def rec_readout(linear, lr_F, season_, present_, hr, hr_idx, logw=None, size=None, parts=False):
        img, (wgt, cand, _, cid) = readout(linear, lr_F, season_, present_, hr, hr_idx, logw,
                                           size, parts=True)
        s_act = md_.readout.score(lr_F.float())[..., 0]
        rec["readout"] = dict(
            F_season=lr_F[0].float().cpu(), tok=hr.float().cpu(),
            idx=[v.cpu() for v in hr_idx], logw=None if logw is None else logw.float().cpu(),
            wgt=wgt[0].float().cpu(), cand=cand[0].float().cpu(), cid=cid[0].cpu(),
            tok_score=md_.readout.score(hr.float())[:, 0].float().cpu(),
            season_score=s_act[0].float().cpu())
        return img

    fi.crop_patches, md_._date_scores, md_.readout.forward = rec_crops, rec_date_scores, rec_readout
    logits, _, n_all = md_(ms, ss, season, present, pairs, extra)
    fi.crop_patches, md_._date_scores, md_.readout.forward = crop_patches, date_scores, readout
    mask, _, _ = md_.last_pick
    out["month_mask"] = mask[0].cpu()
    out["logits"] = logits[0].float().cpu()
    out.update({"date_score": rec["date_z"], "date_cand": rec["date_cand"]})
    r = rec["readout"]
    out.update({k: r[k] for k in ("F_season", "tok", "wgt", "cand", "cid", "logw", "tok_score",
                                  "season_score")})
    b, mo, y, x = r["idx"]
    # crop_patches calls: each date modality (extra["dates"] order), then the month crops; the
    # readout's token order: month crops, then each date modality
    mods = list(extra["dates"])
    calls = rec["crops"]
    n_month = int(mask.sum())
    date_calls = calls[:-1]
    assert len(calls[-1][1]) == n_month and sum(len(c[1]) for c in date_calls) == len(b) - n_month
    level = torch.cat([torch.zeros(n_month), torch.ones(len(b) - n_month)]).long()
    mod = torch.cat([torch.full((n_month,), -1)] + [torch.full((len(c[1]),), j)
                                                    for j, c in enumerate(date_calls)]).long()
    step = torch.cat([calls[-1][1]] + [c[1] for c in date_calls])
    assert torch.equal(torch.cat([calls[-1][2]] + [c[2] for c in date_calls]), y)
    out.update({"tok_level": level, "tok_mod": mod, "tok_step": step, "tok_month": mo,
                "tok_y": y, "tok_x": x, "mods": [m for m in mods][:len(date_calls)],
                "n_crops": n_all})

    # ---- the crop-free pass of the same model: what the selectors see
    dev = "cuda"
    msd, ssd = _to_device(ms, dev, torch.bfloat16), _to_device(ss, dev, torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = md_.mix.head_blocks(md_.mix.embed(ssd, 16))
        lr_x0, _ = md_.mix.tail(lr, None, None)
        out["E_month"] = md_.month_embedding(msd)[0].float().cpu()      # (T, H, W, D) raw ps4
    out["F_free"] = lr.pooled(lr_x0)[0].float().cpu()
    md_.fixed_scores = None
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"collect_{a.tile.replace('@', '_')}{'' if a.month_picks == 'joint' else '_randmonths'}.pt"
    torch.save(out, path)
    pred = out["logits"].argmax(0)
    ok = out["target"] >= 0
    print(f"{path}: {n_month} month crops, {len(b) - n_month} date crops "
          f"({', '.join(f'{m} {len(c[1])}' for m, c in zip(mods, date_calls))}), "
          f"pixel accuracy {(pred[ok] == out['target'][ok]).float().mean():.3f}")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:16s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
