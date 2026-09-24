"""Cache FROZEN OlmoEarth features of a task (exp/common/tasks.py) for LP and upsamplers.

    <FEATURES>/<name>/meta.json
    <FEATURES>/<name>/<split>/<idx>.pt        fp16 tokens, one file per sample
    <FEATURES>/<name>/<split>/ids.txt, frame_valid.pt, t_start.pt, t_end.pt   (FeatureCache)
    <FEATURES>/<name>/<split>/target.pt       (N, H, W) the task's target (Task.target):
                                              int16 classes (-1 ignored) or float32 standardized

--store per_timestep  (T, gH, gW, D): everything the encoder gives; what upsamplers train on.
--store reduced       (1, gH, gW, D): only what the task's head reads (time mean over valid
                      steps, or post - pre), 1/T the size. The pa2px head is linear AFTER the
                      reduction, so an LP on it equals the LP on per-timestep features.

name = oe_<task>_<model>_<temporal_mode>_ps<p>ts<t>[_reduced]

    python -u -m exp.olmoearth.extract --task pastis --temporal_mode series --patch_size 4 \
        --tile_size 64 --store per_timestep
"""
from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.heads import time_mean  # noqa: E402
from exp.common.paths import FEATURES  # noqa: E402
from exp.common.tasks import SPLITS, TASKS, Task  # noqa: E402
from manyup.data import mtmm_collate  # noqa: E402
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput  # noqa: E402

STORES = ("per_timestep", "reduced")


def cache_name(task: str, model_size: str, temporal_mode: str, ps: int, ts: int,
               store: str) -> str:
    return (f"oe_{task}_{model_size}_{temporal_mode}_ps{ps}ts{ts}"
            + ("_reduced" if store == "reduced" else ""))


def reduce(tokens: torch.Tensor, frame_valid: torch.Tensor, reduction: str) -> torch.Tensor:
    """(B, T, gH, gW, D) -> (B, 1, gH, gW, D): what a Pa2PxHead of this reduction reads."""
    if reduction == "diff":
        return (tokens[:, 1] - tokens[:, 0]).unsqueeze(1)
    return time_mean(tokens.float(), frame_valid.to(tokens.device)).unsqueeze(1)


def target_dtype(task: Task):
    return torch.int16 if task.kind == "classification" else torch.float32


def extract_split(encoder, task, split, out: Path, a) -> None:
    ds = task.dataset(split)
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    n = min(a.limit, len(ds)) if a.limit else len(ds)
    fields = ("ids.txt", "frame_valid.pt", "t_start.pt", "t_end.pt", "target.pt")
    if all((d / f).exists() for f in fields) and all((d / f"{i}.pt").exists() for i in range(n)):
        print(f"[{split}] complete, skipping")
        return
    loader = DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=a.batch_size,
                        num_workers=a.num_workers, collate_fn=mtmm_collate)
    ref = next(iter(task.olmoearth.modalities))
    ids, fv, ts, te, tg, idx = [], [], [], [], [], 0
    for k, batch in enumerate(loader):
        B = len(batch.ids)
        ids += batch.ids
        ts.append(batch.modalities[ref].t_start)
        te.append(batch.modalities[ref].t_end)
        tg.append(task.target(batch).to(target_dtype(task)))
        if all((d / f"{idx + b}.pt").exists() for b in range(B)):      # resume
            fv.append(torch.stack([batch.modalities[m].frame_valid
                                   for m in task.olmoearth.modalities]).any(0))
            idx += B
            continue
        f = encoder.encode(batch, a.temporal_mode)
        fv.append(f.frame_valid)
        tok = f.tokens if a.store == "per_timestep" else reduce(f.tokens, f.frame_valid,
                                                                  task.reduction)
        tok = tok.half().cpu()
        if not torch.isfinite(tok).all():
            raise ValueError(f"[{split}] non-finite features in batch {k}")
        for b in range(B):
            torch.save(tok[b].clone(), d / f"{idx + b}.pt")
        idx += B
        if k % 200 == 0:
            print(f"[{split}] {idx}/{n}", flush=True)
    torch.save(torch.cat(fv), d / "frame_valid.pt")
    torch.save(torch.cat(ts), d / "t_start.pt")
    torch.save(torch.cat(te), d / "t_end.pt")
    torch.save(torch.cat(tg), d / "target.pt")
    (d / "ids.txt").write_text("".join(s + "\n" for s in ids))       # written last
    print(f"[{split}] wrote {n} samples to {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--temporal_mode", required=True, choices=OlmoEarthEncoder.TEMPORAL_MODES)
    ap.add_argument("--patch_size", type=int, required=True)
    ap.add_argument("--tile_size", type=int, required=True)
    ap.add_argument("--store", required=True, choices=STORES)
    ap.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help=">0: first N samples per split")
    ap.add_argument("--out_root", default=str(FEATURES))
    a = ap.parse_args()

    task = TASKS[a.task]
    name = cache_name(a.task, a.model_size, a.temporal_mode, a.patch_size, a.tile_size, a.store)
    out = Path(a.out_root) / name
    inputs = OlmoEarthInput(task.olmoearth.modalities, task.olmoearth.impute)
    encoder = OlmoEarthEncoder(MODEL_SIZE_TO_ID[a.model_size], inputs, a.patch_size,
                               a.tile_size, name=a.model_size).cuda().eval()
    ds = task.dataset("train")
    encoder.check(ds, a.temporal_mode)
    meta = {"task": a.task, "dataset_key": ds.key, "model_size": a.model_size,
            "model_id": MODEL_SIZE_TO_ID[a.model_size], "patch_size": a.patch_size,
            "tile_size": a.tile_size, "temporal_mode": a.temporal_mode, "store": a.store,
            "reduction": task.reduction if a.store == "reduced" else None,
            "modalities": task.olmoearth.modalities, "impute": task.olmoearth.impute,
            "attention": encoder.attention, "embed_dim": encoder.encoder.embedding_size,
            "target": task.kind, "dtype": "float16"}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        have = json.loads((out / "meta.json").read_text())
        if have != meta:
            raise SystemExit(f"{out} was made with a different setup:\n{have}\nvs\n{meta}")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Cache: {out}")
    for split in a.splits.split(","):
        extract_split(encoder, task, split, out, a)


if __name__ == "__main__":
    main()
