"""APT's per-dataset calibration -> results/<task>/apt.json: the RGB stretch (2-98 percentiles per
band over the training images, the counterpart of an 8-bit image) and each scale's entropy
threshold, the quantile of the training entropies that keeps that share of the coarse patches.

    python -u -m exp.baselines.calibrate_apt --task pastis [--batches 200]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
from itertools import islice  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from manyup.mixres.baselines.apt import grayscale, patch_entropy  # noqa: E402
from manyup.mixres.baselines.base import rgb_frames  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402

PATCHES = (16, 14, 12, 10, 8, 6)
KEEPS = (0.25, 0.5, 0.75, 0.9)
N_MAX = 2_000_000


def frames(batch, tokens: str, patch: int, task):
    """(N, 3, P, P) the valid frames of one kind of a batch, cuda."""
    if tokens == "months":
        s, valid = batch.ms, batch.present & (batch.season >= 0)
    else:
        s, S = batch.ss, batch.ss.timestamps.shape[1]
        valid = (torch.nn.functional.one_hot(batch.season.clamp(min=0), S).bool()
                 & (batch.present & (batch.season >= 0))[..., None]).any(1)
    return rgb_frames(task, s, patch)[valid].cuda()


def sample(x, n: int):
    """n random rows of x, on the CPU."""
    return x[torch.randperm(len(x), device=x.device)[:n]].cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    task = TASKS[a.task]
    dl = loader(task, "train", 32, a.num_workers, True)
    n = min(a.batches, len(dl))
    each = N_MAX // n                   # samples per batch: a batch is far too big to keep
    x = torch.cat([sample(frames(b, "months", 16, task).permute(1, 0, 2, 3).flatten(1).T, each)
                   for b in islice(dl, n)]).T
    lo, hi = torch.quantile(x, 0.02, dim=1), torch.quantile(x, 0.98, dim=1)
    ents = {(tokens, p): [] for tokens in ("months", "seasons") for p in PATCHES}
    for b in islice(dl, n):
        for (tokens, p), e in ents.items():
            e.append(sample(patch_entropy(grayscale(frames(b, tokens, p, task), lo, hi), p).flatten(),
                            each))
    out = {"lo": lo.tolist(), "hi": hi.tolist(), "tau": {}}
    for (tokens, p), e in ents.items():
        ent = torch.cat(e)
        out["tau"][f"{tokens}_ps{p}"] = {f"{k:g}": round(torch.quantile(ent, k).item(), 4)
                                         for k in KEEPS}
        print(tokens, p, out["tau"][f"{tokens}_ps{p}"], flush=True)
    path = Path(f"results/{a.task}/apt.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
