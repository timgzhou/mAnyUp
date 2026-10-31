"""Score trained euromine runs under the paper's evaluation protocols (Yu et al. 2025, Sec. 3.6.2 /
Table 4), from predictions stitched back into whole site rasters:

    python -u -m exp.euromine.eval_protocols <run name> [<run name> ...] [--split test]

Every tile's per-year mine probability is averaged where tiles overlap (the snapped edge
tiles), giving one map per site and year. Then
    tile_of1        pooled F1 over the tiles as trained (edge overlaps counted twice): the
                    of1 that exp.mixres.train reports
    site_of1        pooled F1 over every (pixel, year) of the stitched sites, each pixel once
    site_w_of1      per-site pooled F1, mean weighted by the site's pixel-years (the paper's
                    "averaged by different mining sites, with the weight given by the number
                    of pixels")
    p160_of1        mean F1 over the paper's samples: 160 px patches (preprocess.py's cropping,
                    non-overlapping + edge-snapped, repetition-padded if a site is < 160 px) x
                    years (Table 4's "average F1 score from all samples"); a patch-year with no
                    mine in label or prediction has no F1 and is left out (p160_undefined:
                    their share; p160_of1_undef1 counts them as 1)
    f1_<year>, lca_tiou, gca_tiou, cf1*, dyn_f1   StepMeter on the stitched sites
-> results/euromine/protocols.csv (one row per run and split).
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp.common.metrics import StepMeter  # noqa: E402
from exp.common.tasks import TASKS  # noqa: E402
from exp.mixres.train import step  # noqa: E402
from exp.olmoearth.train import _append  # noqa: E402
from manyup.data.datasets.euromine import YEARS, load_index, parse_id  # noqa: E402
from manyup.mixres import MixedResModel  # noqa: E402
from manyup.mixres.config import RunConfig, from_dict  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.steps import build_step_selector  # noqa: E402

PATCH = 160


def pad_repeat(a: np.ndarray, size: int) -> np.ndarray:
    """The release's padding_with_repetition on (..., H, W): rows / cols past the edge copy the
    raster's first ones."""
    H, W = a.shape[-2:]
    if H < size:
        a = np.concatenate([a, a[..., :size - H, :]], -2)
    if W < size:
        a = np.concatenate([a, a[..., :, :size - W]], -1)
    return a


def paper_patches(H: int, W: int, size: int = PATCH) -> list[tuple[int, int]]:
    """preprocess.py crop_and_save_patches(step = crop = size): the grid, then the bottom row,
    the right column and the corner snapped to the edge when the raster does not divide."""
    hp, wp = (H - size) // size, (W - size) // size
    o = [(h * size, w * size) for h in range(hp + 1) for w in range(wp + 1)]
    if H % size:
        o += [(H - size, w * size) for w in range(wp + 1)]
    if W % size:
        o += [(h * size, W - size) for h in range(hp + 1)]
    if H % size and W % size:
        o.append((H - size, W - size))
    return o


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else float("nan")


class SiteScores:
    """Accumulates the stitched sites' counts."""

    def __init__(self):
        self.meter = StepMeter(tuple(map(str, YEARS)), ("background", "mine"))
        self.site = []                  # (pixel-years, tp, fp, fn) per site
        self.patch = []                 # F1 per patch-year (nan: undefined)

    def add(self, prob: np.ndarray, lab: np.ndarray):
        """prob (T, H, W) mine probability, lab (T, H, W) {0, 1}."""
        p, g = prob > 0.5, lab.astype(bool)
        logits = torch.from_numpy(np.stack([1 - p, p], 1).astype(np.float32))[None]
        self.meter.update(logits, torch.from_numpy(lab.astype(np.int64))[None])
        self.site.append((g.size, (p & g).sum(), (p & ~g).sum(), (~p & g).sum()))
        pp, gg = pad_repeat(p, PATCH), pad_repeat(g, PATCH)
        for r, c in paper_patches(*pp.shape[-2:]):
            a, b = pp[:, r:r + PATCH, c:c + PATCH], gg[:, r:r + PATCH, c:c + PATCH]
            tp = (a & b).sum((1, 2))
            fp, fn = (a & ~b).sum((1, 2)), (~a & b).sum((1, 2))
            self.patch += [f1(*v) for v in zip(tp, fp, fn)]

    def compute(self) -> dict:
        out = {k: v for k, v in self.meter.compute().items() if k != "primary"}
        out["site_of1"] = out.pop("of1")
        n, tp, fp, fn = (np.array(v, dtype=np.float64) for v in zip(*self.site))
        per = np.array([f1(*v) for v in zip(tp, fp, fn)])
        ok = np.isfinite(per)
        out["site_w_of1"] = float((per[ok] * n[ok]).sum() / n[ok].sum())
        pf = np.array(self.patch)
        out["p160_of1"] = float(np.nanmean(pf))
        out["p160_of1_undef1"] = float(np.where(np.isnan(pf), 1.0, pf).mean())
        out["p160_undefined"] = float(np.isnan(pf).mean())
        out["p160_samples"] = len(pf)
        out["sites"] = len(self.site)
        return out


@torch.no_grad()
def evaluate(name: str, split: str, batch_size: int) -> dict:
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ck["config"])
    task = TASKS[cfg.task]
    model = MixedResModel(task, cfg.model, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    dl = loader(task, split, batch_size, 8, False)
    b0 = next(iter(dl))
    selector = build_step_selector(cfg.select, model, (b0.ms, b0.ss))
    torch.nn.ModuleList(selector.modules()).load_state_dict(ck["selector"])
    model.eval()
    for m in selector.modules():
        m.eval()
    ds = dl.dataset.ds
    S = ds.tile
    idx = load_index(str(ds.packed))
    scores, tile = SiteScores(), task.meter()
    cur, acc, cnt, i = None, None, None, 0

    def flush():
        lab = np.load(ds.packed / "label" / f"{cur}.npy")
        scores.add(acc / np.maximum(cnt, 1), lab)

    for batch in dl:
        logits, _, _ = step(model, selector, batch)
        tile.update(logits, batch.target.cuda())
        if task.label_steps is not None:                 # seasonal: the labelled quarters
            logits = logits[:, list(task.label_steps)]
        prob = logits.float().softmax(2)[:, :, 1].cpu().numpy()            # (B, T, S, S)
        for k in range(prob.shape[0]):
            site, r, c = parse_id(ds.sample_id(i))
            i += 1
            if site != cur:
                if cur is not None:
                    flush()
                cur = site
                H, W = idx["sites"][site]["shape"]
                acc = np.zeros((len(YEARS), H, W), np.float32)
                cnt = np.zeros((1, H, W), np.float32)
            h, w = min(S, H - r), min(S, W - c)          # a padded tile: its raster part
            acc[:, r:r + h, c:c + w] += prob[k][:, :h, :w]
            cnt[:, r:r + h, c:c + w] += 1
    flush()
    if i != len(ds):
        raise AssertionError(f"{i} tiles scored, {len(ds)} in the split")
    out = {"tile_of1": tile.compute()["of1"], **scores.compute()}
    return {"run": name, "split": split, "lr_patch": cfg.model.lr_patch,
            "independent_steps": cfg.model.independent_steps,
            "scorer": cfg.select.scorer, "allocator": cfg.select.allocator, "n": cfg.select.n,
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in out.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    for name in a.runs:
        row = evaluate(name, a.split, a.batch_size)
        print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
        _append(Path("results/euromine/protocols.csv"),
                {"job": os.environ.get("SLURM_JOB_ID", ""), **row})


if __name__ == "__main__":
    main()
