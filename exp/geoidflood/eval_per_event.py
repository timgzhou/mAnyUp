"""Per-EVENT confusion matrices of saved GEOID fine-tuning checkpoints on val and test.

GEOID's val split is dominated by one event whose labels are not visible in its S1 pair
(EMSR342, Townsville 2019: the post image predates the flood peak; flood-labelled pixels do
not darken), so val-based model selection on GEOID is close to random. Storing a confusion
matrix per event lets any exclusion rule be applied afterwards without re-running models.

Checkpoints: exp.olmoearth.train fft ("oe_geoidflood_*_fft_*_lr*.pt": args, encoder, head) and
exp.upsamplers.finetune_seasonal ("seasonal_fft_geoidflood_*.pt": args, model). Output:
results/geoidflood/per_event/<ckpt stem>.pt = {split: {event: (3, 3) long confusion}}, rows =
target, cols = prediction, ignored pixels excluded.

    python -u -m exp.geoidflood.eval_per_event checkpoints/oe_geoidflood_tiny_series_ps4ts64_fft_ep10_lr0.001.pt
"""
from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import argparse  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.heads import Pa2PxHead  # noqa: E402
from exp.common.tasks import TASKS  # noqa: E402

OUT = Path("results/geoidflood/per_event")
N = 3


def event_of(sample_id: str) -> str:
    return sample_id.split("-")[0]


def baseline(ck, task, split, bs, workers):
    """(logits iterator) of an exp.olmoearth.train fft checkpoint."""
    from exp.olmoearth.train import loader
    from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput
    a = ck["args"]
    inputs = OlmoEarthInput(task.olmoearth.modalities, task.olmoearth.impute)
    enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID[a["model_size"]], inputs, a["patch_size"],
                           a["tile_size"]).cuda().eval()
    enc.encoder.load_state_dict(ck["encoder"])
    head = Pa2PxHead(task.reduction, enc.encoder.embedding_size, task.num_outputs,
                     a["patch_size"], norm=a.get("head_norm", False)).cuda().eval()
    head.load_state_dict(ck["head"])
    for sample, fv, target in loader(task, inputs, split, bs, workers, False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield head(enc(sample, a["temporal_mode"]), fv.cuda()).float(), target


def seasonal(ck, task, split, bs, workers):
    """(logits iterator) of an exp.upsamplers.finetune_seasonal checkpoint."""
    from exp.upsamplers import finetune_seasonal as FS
    a = ck["args"]
    model = FS.SeasonalFT(task, a["model_size"], a["upsampler_ckpt"], a["selector"], a["k_frac"],
                          a["fuser_hidden"], a.get("train_upsampler", False),
                          a.get("selector_ckpt"), a.get("train_selector", False),
                          a.get("selector_hidden", 64), a.get("select_unit", "slot")).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    for batch in FS.loader(task, model.spec, split, bs, workers, False):
        *x, target = FS.to_device(batch, "cuda")
        yield model(*x)[0], target


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    task = TASKS["geoidflood"]
    OUT.mkdir(parents=True, exist_ok=True)
    for path in map(Path, a.ckpts):
        out = OUT / f"{path.stem}.pt"
        if out.exists():
            print(f"{out} exists, skipping")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        run = seasonal if "model" in ck else baseline
        res = {}
        for split in a.splits.split(","):
            ds = task.dataset(split)
            events = [event_of(ds.sample_id(i)) for i in range(len(ds))]
            conf, i0 = defaultdict(lambda: torch.zeros(N, N, dtype=torch.long)), 0
            for logits, target in run(ck, task, split, a.batch_size, a.num_workers):
                pred, target = logits.argmax(1).cpu(), target.cpu()
                for b in range(len(target)):
                    t, p = target[b].flatten(), pred[b].flatten()
                    m = t >= 0
                    conf[events[i0 + b]] += torch.bincount(t[m] * N + p[m],
                                                           minlength=N * N).view(N, N)
                i0 += len(target)
            if i0 != len(events):
                raise RuntimeError(f"{path.name}/{split}: scored {i0} of {len(events)} chips")
            res[split] = dict(conf)
        torch.save(res, out)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
