"""Paper material, one-way attention figure: the tokens of one tile layer by layer, from the same
run and tile as the method figure (exp.viz.paper.method_parts).

    source env_setup/env_olmo.sh && python -m exp.viz.paper.oneway_layers --setup s8m4_qs0.5 \\
        --tile 30201@r64c0 [--layers 0 1 12]

Layer 0 = the tokens as tokenized; layer k = after k blocks. MixedResOE._cross_blocks runs one block
per trace entry, so a trace cut to k entries gives the crop tokens after k blocks, each block
reading the season tokens ENTERING it (the figure's KV arrows). The last layer through the final
norm must equal the model's own crop tokens (checked below).

Early tokens are dominated by their additive time and position encodings, which PCA then shows
instead of the image. So the panels show each token minus the same token computed from a blank
input (every image zero; same timestamps, masks and crop positions): what depends on the content.
The final layer (--layers 12) is the exception: raw tokens, so it is identical to method_parts'
lr_feature_* and month_feature_* slices.

Writes figures/paper/method/<setup>/<tile>/oneway/ (transparent PNGs, method_parts' stacks; one
PCA basis per layer, fit on that layer's season + crop tokens, each row centred on its own mean and
weighted equally (the final layer: neither, as method_parts), so a layer's two rows are comparable):
    stack_inputs_seasons.png, stack_inputs_sparse.png
    stack_seasons_L<k>.png, stack_crops_L<k>.png
    single/   every slice as its own PNG
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from manyup.encoders.olmoearth import _map  # noqa: E402
from exp.viz.paper.method_parts import (MONTHS, ORDERS, PCA, SEASON_STEP, SEASONS,  # noqa: E402
                                        median_composites, s2_stretch, stack, to_img)
from exp.viz.paper.method_scan import OUT, RUNS, build, run_batch  # noqa: E402
from manyup.data.temporal import All  # noqa: E402
from manyup.mixres.data import Collate, Paired  # noqa: E402


@torch.no_grad()
def blank(sample):
    """The sample with every image zeroed: timestamps and masks kept."""
    return _map(sample, lambda k, v: v if k == "timestamps" or "mask" in k else torch.zeros_like(v))


@torch.no_grad()
def layer_tokens(model, ctx, plan, batch, out, layers):
    """-> {k: (season tokens (NS, h, w, D), crop tokens (N, D))}, after k blocks, each minus the
    blank input's."""
    lr = ctx.seasons
    trace = ctx.lr_enc[0]
    ms, ss, season, _ = model._inputs(batch.ms, batch.ss, batch.season, batch.present)
    mix = model.mix
    n = len(trace)
    res = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hr = model.tokenize_crops(ms, plan, season, lr, None, ss.timestamps)
        sets = [(hr, plan.b)]
        lr0 = model.tokenize_seasons(blank(ss))
        trace0, x0 = mix.lr_trace(lr0)
        hr0 = model.tokenize_crops(blank(ms), plan, season, lr0, None, ss.timestamps)
        sets0 = [(hr0, plan.b)]

        def at(lr_, tr, end, h, s, k):
            xl = tr[k] if k < n else end
            xc = h.x if k == 0 else mix._cross_blocks(lr_, tr[:k], s, mix.L, True)[0]
            if k >= n:          # (the last layer's season tokens are past the final norm)
                xc = mix.encoder.norm(xc)
            return lr_.pooled(xl).float()[0], h.pooled(xc).float()[:, 0, 0, 0]

        for k in layers:
            a, b = at(lr, trace, ctx.lr_enc[1], hr, sets, k)
            a0, b0 = at(lr0, trace0, x0, hr0, sets0, k)
            # (the final layer stays raw: the method figure's features, method_parts' PCA)
            res[k] = ((a.cpu(), b.cpu()) if k >= n else ((a - a0).cpu(), (b - b0).cpu()))
        last = mix._cross_blocks(lr, trace, sets, mix.L, True)[0]
        last = hr.pooled(mix.encoder.norm(last)).float()[:, 0, 0, 0]
    gap = float((last - out.hr.float()).abs().max())
    if gap > 1e-3:
        raise RuntimeError(f"truncated-trace crop tokens differ from the model's by {gap}")
    print(f"crop tokens check: max gap to the model's final tokens {gap:.2e}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--setup", default="s8m4_qs0.5", choices=tuple(RUNS))
    ap.add_argument("--tile", default="30201@r64c0")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 12])
    ap.add_argument("--order", default="spring", choices=tuple(ORDERS))
    a = ap.parse_args()
    torch.manual_seed(0)         # (method_parts' seed: the same picks)
    task = TASKS["pastis"]
    ds = Paired(task, "test")
    raw_ds = task.open("test", {"S2": All()})
    b0 = Collate(task)([ds[0]])
    model, selector, cfg = build(RUNS[a.setup], (b0.ms, b0.ss))
    item = ds[ds.ds.index_of(a.tile)]
    months, seasons = item
    batch = Collate(task)([item])
    ctx, plan, out, _ = run_batch(model, selector, batch)
    if cfg.model.attn != "cross_hr" or cfg.model.crop_patch:
        raise ValueError("cross_hr runs with one token per window only")
    cp = cfg.select.crop_px
    g = 64 // cp
    res = layer_tokens(model, ctx, plan, batch, out, a.layers)

    mm, sm = months.modalities["S2"], seasons.modalities["S2"]
    raw = raw_ds[raw_ds.index_of(a.tile)].modalities["S2"]
    s2m = median_composites(raw, mm.time, mm.data.float())
    s2s = median_composites(raw, sm.time, sm.data.float())
    rgb = s2_stretch(s2m)
    T, NS = s2m.shape[0], s2s.shape[0]
    py, px, t = (v.cpu() for v in (plan.py, plan.px, plan.t))
    active = sorted(set(t.tolist()))
    d = OUT / a.setup / a.tile.replace("@", "_") / "oneway"
    one = d / "single"
    one.mkdir(parents=True, exist_ok=True)

    def save(name, im):
        im.save(one / f"{name}.png")
        return im

    mo, so = ORDERS[a.order]
    M = lambda ims: [ims[m] for m in mo]                                # noqa: E731
    S = lambda ims: [ims[s] for s in so]                                # noqa: E731

    # inputs: the season composites, the picked windows of each month
    stack(S([save(f"input_season_{s}", to_img(rgb(s2s[s]))) for s in range(NS)]),
          d / "stack_inputs_seasons.png", SEASON_STEP)
    sparse = []
    for m in range(T):
        x = np.zeros((64, 64, 4), np.float32)
        img = rgb(s2m[m])
        for j in (t == m).nonzero()[:, 0].tolist():
            y_, x_ = int(py[j]), int(px[j])
            x[y_:y_ + cp, x_:x_ + cp, :3] = img[y_:y_ + cp, x_:x_ + cp]
            x[y_:y_ + cp, x_:x_ + cp, 3] = 1
        sparse.append(save(f"input_sparse_{m:02d}_{MONTHS[m]}", to_img(x)))
    stack(M(sparse), d / "stack_inputs_sparse.png")

    # tokens per layer: one PCA basis over both rows
    # (final layer first: PCA draws random numbers, this way the same ones as method_parts)
    for k, (lr_F, hr) in sorted(res.items(), key=lambda kv: kv[0] < len(ctx.lr_enc[0])):
        flat = lr_F.reshape(-1, lr_F.shape[-1])
        if k < len(ctx.lr_enc[0]):
            # before the final layer: each row loses its own mean (else the constant offset
            # between seasons and crops takes a principal component), and both weigh the same
            # in the fit, whatever their token counts
            mf, mh = flat.mean(0), hr.mean(0)
            flat, lr_F, hr = flat - mf, lr_F - mf, hr - mh
            pca = PCA(torch.cat([flat.repeat(len(hr), 1), hr.repeat(len(flat), 1)]))
        else:
            pca = PCA(torch.cat([flat, hr]))
        lr_im = [save(f"L{k}_season_{s}_{SEASONS[s]}", to_img(pca(lr_F[s]))) for s in range(NS)]
        cr_im = []
        for m in range(T):
            f = np.zeros((g, g, 4), np.float32)
            j = (t == m).nonzero()[:, 0]
            if len(j):
                f[py[j] // cp, px[j] // cp, :3] = pca(hr[j])
                f[py[j] // cp, px[j] // cp, 3] = 1
            cr_im.append(save(f"L{k}_crops_{m:02d}_{MONTHS[m]}", to_img(f)))
        stack(S(lr_im), d / f"stack_seasons_L{k}.png", SEASON_STEP)
        stack(M(cr_im), d / f"stack_crops_L{k}.png")
    info = {"tile": a.tile, "run": RUNS[a.setup], "layers": a.layers, "crops": len(t),
            "blocks": len(model.mix.encoder.blocks),
            "months_with_picks": [MONTHS[m] for m in active],
            "pca": "3 principal components of one layer's season + crop tokens, per layer; "
                   "before the final layer each row is centred on its own mean, equal weight"}
    (d / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
