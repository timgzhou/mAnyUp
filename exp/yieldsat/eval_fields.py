"""Per-field predictions of a fine-tuned exp.olmoearth.train checkpoint on a YieldSAT task.

One CSV row per field of the split: its country, crop, year, labelled pixels, true and
predicted mean yield (t/ha), pixel RMSE, and the pixel RMSE left after removing the field's
mean error (within-field error). Tells a whole-field offset (e.g. a crop taken for another)
from within-field noise.

    python -u -m exp.yieldsat.eval_fields \
        --ckpt checkpoints/oe_yieldsat_ml24_base_series_ps8ts64_fft_ep50_lr0.001.pt
    -> results/yieldsat_ml24/fields_<ckpt stem>_<split>.csv
"""
from exp.common import olmo_bootstrap  # noqa: E402,F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import argparse  # noqa: E402
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.heads import Pa2PxHead  # noqa: E402
from exp.common.tasks import TASKS  # noqa: E402
from exp.olmoearth.train import loader  # noqa: E402
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, weights_only=False)
    c = ck["args"]
    task = TASKS[c["task"]]
    inputs = OlmoEarthInput(task.olmoearth.modalities, task.olmoearth.impute)
    encoder = OlmoEarthEncoder(MODEL_SIZE_TO_ID[c["model_size"]], inputs, c["patch_size"],
                               c["tile_size"]).cuda().eval()
    encoder.encoder.load_state_dict(ck["encoder"])
    head = Pa2PxHead(task.reduction, encoder.encoder.embedding_size, task.num_outputs,
                     c["patch_size"], norm=c["head_norm"]).cuda().eval()
    head.load_state_dict(ck["head"])
    idx = task.index
    mean, std = idx["target_mean"], idx["target_std"]
    names = sorted(idx["fields"])
    for split in a.splits.split(","):
        acc = torch.zeros(5, len(names), dtype=torch.float64)       # n, sum p, sum t, sse, sum e
        bs = task.batch_size[c["patch_size"]]
        for sample, fv, target in loader(task, inputs, split, bs, a.num_workers, False):
            tokens = encoder(sample, c["temporal_mode"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(tokens, fv.cuda()).float().cpu()[:, 0]
            y = target[:, 0]
            keep = torch.isfinite(y)
            p, t = pred[keep].double() * std + mean, y[keep].double() * std + mean
            f = target[:, 1][keep].long()
            acc.index_add_(1, f, torch.stack([torch.ones_like(t), p, t, (p - t) ** 2, p - t]))
        out = Path(f"results/{task.name}/fields_{Path(a.ckpt).stem}_{split}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["field", "country", "crop", "year", "pixels", "true_mean", "pred_mean",
                        "pixel_rmse", "within_field_rmse"])
            for i, name in enumerate(names):
                n, sp, st, sse, se = acc[:, i].tolist()
                if n == 0:
                    continue
                fl = idx["fields"][name]
                within = max(sse / n - (se / n) ** 2, 0.0) ** 0.5
                w.writerow([name, fl["country"], fl["crop"], fl["year"], int(n),
                            round(st / n, 4), round(sp / n, 4), round((sse / n) ** 0.5, 4),
                            round(within, 4)])
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
