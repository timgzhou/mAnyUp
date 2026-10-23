"""manyup.mixres.data.loader vs the legacy exp.upsamplers.finetune_inject.loader: the same batches
(every tensor of the monthly / season samples, season, present, target, frac, and the raw dates)
on test, on test with dates, and on the shuffled train split under the same seed.

    python -m exp.mixres.check_data --task pastis
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from manyup.mixres import data  # noqa: E402


def digest(*xs) -> str:
    h = hashlib.md5()

    def add(x):
        if torch.is_tensor(x):
            h.update(str(x.dtype).encode())
            h.update(x.float().nan_to_num(-9e9).contiguous().numpy().tobytes())
        elif hasattr(x, "as_dict"):
            for k, v in sorted(x.as_dict().items()):
                h.update(k.encode())
                add(v)
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                add(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x:
                add(v)
        elif x is not None:
            h.update(repr(x).encode())
    for x in xs:
        add(x)
    return h.hexdigest()


def compare(task, split, shuffle, dates, batches):
    torch.manual_seed(0)
    old = fi.loader(task, split, 8, 4, shuffle, dates=dates)
    old_it = iter(old)
    torch.manual_seed(0)
    new_it = iter(data.loader(task, split, 8, 4, shuffle, dates=dates))
    ok = True
    for i in range(batches):
        ms, ss, season, present, _, extra, target = next(old_it)
        b = next(new_it)
        pairs = {"ms": (ms, b.ms), "ss": (ss, b.ss), "season": (season, b.season),
                 "present": (present, b.present), "target": (target, b.target),
                 "frac": (extra["frac"], b.frac)}
        if dates:
            pairs["dates"] = (extra["dates"], b.dates)
        bad = [k for k, (o, n) in pairs.items() if digest(o) != digest(n)]
        ok &= not bad
        print(f"   {split}{' +dates' if dates else ''}{' shuffled' if shuffle else ''} batch {i}: "
              f"{'same' if not bad else 'DIFFERENT ' + ', '.join(bad)}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pastis")
    ap.add_argument("--batches", type=int, default=3)
    a = ap.parse_args()
    task = TASKS[a.task]
    ok = all([compare(task, "test", False, False, a.batches),
              compare(task, "test", False, True, a.batches),
              compare(task, "train", True, False, a.batches)])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
