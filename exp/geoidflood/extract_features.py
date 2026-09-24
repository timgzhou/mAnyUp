"""Extract & cache FROZEN OlmoEarth features for GEOID-Flood chips.

The patch-size study compares three (patch_size, chip size) configs with the SAME 16x16
token grid, so the probe has identical parameter and token counts; only the ground area per
token and the chip's footprint change:

    ps1 / chip 16    1 token per 10 m pixel, 160 m footprint
    ps4 / chip 64    1 token per 40 m,       640 m
    ps8 / chip 128   1 token per 80 m,       1.28 km

At ps1/chip16 a chip can sit entirely inside the flood (no dry reference); at ps8/chip128
each token averages 80 m, so thin channels are gone and the head upsamples 8x.

Chips come from GeoidFloodDataset (chip index of the selection: mapped >= 50%, >= 1 flood
pixel) and are encoded by OlmoEarthEncoder with both S1 timesteps (pre, post) on their own
dates, kept separate for the change-detection head:
    <out_root>/<cfg>/<split>/<idx>.pt = {"feat": (T=2, gH, gW, D) fp16, "label": (s, s) int16}
    <out_root>/<cfg>/<split>/ids.txt  = chip id of every idx
labels: 0 background, 1 permanent water, 2 flood, -1 = outside the mapped area.

    python -u -m exp.geoidflood.extract_features --splits train,val --tile_size 128 --patch_size 8
"""
import argparse
import json
from pathlib import Path

from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.paths import FEATURES  # noqa: E402
from manyup.data import All, mtmm_collate  # noqa: E402
from manyup.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset, UNMAPPED  # noqa: E402
from manyup.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput  # noqa: E402

IGNORE_LABEL = -1
NUM_CLASSES = 3
INPUT_RES = 10


def extract_split(encoder, ds, split_out: Path, limit: int, batch_size: int,
                  num_workers: int) -> int:
    split_out.mkdir(parents=True, exist_ok=True)
    n = min(limit, len(ds)) if limit > 0 else len(ds)
    (split_out / "ids.txt").write_text("".join(ds.sample_id(i) + "\n" for i in range(n)))
    if all((split_out / f"{i}.pt").exists() for i in range(n)):
        print(f"[{ds.split}] {n} feature files already present, skipping.")
        return n
    loader = DataLoader(Subset(ds, range(n)), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=mtmm_collate)
    idx, n_nonfinite = 0, 0
    for batch in tqdm(loader, desc=f"extract {ds.split}"):
        feats = encoder.encode(batch, "series").tokens.float()             # (B, 2, gH, gW, D)
        bad = ~torch.isfinite(feats).all(dim=(1, 2, 3, 4))
        if bad.any():                        # report, never silently keep NaNs in a cache
            n_nonfinite += int(bad.sum())
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        label = batch.target.clone()
        label[label == UNMAPPED] = IGNORE_LABEL
        for b in range(len(batch.ids)):
            torch.save({"feat": feats[b].half().cpu().clone(), "label": label[b].to(torch.int16)},
                       split_out / f"{idx}.pt")
            idx += 1
    if n_nonfinite:
        print(f"[{ds.split}] {n_nonfinite} chips had non-finite encoder output (zeroed).")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Cache OlmoEarth features for GEOID-Flood chips.")
    p.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--tile_size", type=int, default=128, help="chip size in pixels")
    p.add_argument("--out_root", default=str(FEATURES))
    p.add_argument("--splits", default="train,val")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help=">0: only the first N chips per split")
    args = p.parse_args()

    chips = ChipSelection(args.tile_size)
    name = f"geoid_{args.model_size}_s1_ps{args.patch_size}_res{INPUT_RES}_t{args.tile_size}"
    out_dir = Path(args.out_root) / name
    inputs = OlmoEarthInput({"S1GRD": "sentinel1"})
    encoder = OlmoEarthEncoder(MODEL_SIZE_TO_ID[args.model_size], inputs, args.patch_size,
                               args.tile_size, name=args.model_size)
    encoder = encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    print(f"Extraction config: {name} ({chips.key}, {args.patch_size * INPUT_RES} m per token)")

    counts, key = {}, None
    for split in [s for s in args.splits.split(",") if s]:
        ds = GeoidFloodDataset(split, {"S1GRD": All()}, chips)
        encoder.check(ds, "series")
        key = ds.key
        counts[split] = extract_split(encoder, ds, out_dir / split, args.limit,
                                      args.batch_size, args.num_workers)
    first = next(s for s in counts if counts[s])
    probe = torch.load(out_dir / first / "0.pt")["feat"]
    grid = args.tile_size // args.patch_size
    meta = {
        "dataset": "geoidflood", "dataset_key": key, "chip_selection": chips.key,
        "model_size": args.model_size, "model_id": MODEL_SIZE_TO_ID[args.model_size],
        "patch_size": args.patch_size, "input_res": INPUT_RES, "modalities": ["sentinel1"],
        "pooling": "mean over bandsets", "timesteps": probe.shape[0],
        "embed_dim": probe.shape[-1], "grid": grid, "tile_size": args.tile_size,
        "dtype": "float16", "counts": counts, "feature_shape": list(probe.shape),
        "num_classes": NUM_CLASSES, "label_size": args.tile_size,
        "metres_per_token": args.patch_size * INPUT_RES,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir}/meta.json")


if __name__ == "__main__":
    main()
