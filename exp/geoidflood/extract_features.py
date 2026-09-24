"""Extract & cache FROZEN OlmoEarth features for the tiled GEOID-Flood dataset.

The patch-size study. We compare three (patch_size, tile_size) configs:

    ps1tile16    patch 1, 16x16 px tile  -> 16x16 token grid, 1 token per  10 m pixel
    ps4tile64    patch 4, 64x64 px tile  -> 16x16 token grid, 1 token per  40 m
    ps8tile128   patch 8, 128x128 px tile-> 16x16 token grid, 1 token per  80 m

All three produce the SAME 16x16 token grid and the same D, so the probe has an identical
parameter count and an identical number of tokens in every config. What changes is the
ground area each token summarises and how much context the tile spans. That isolates the
question this study is about -- at fixed token budget, is it better to look finely at a
small area or coarsely at a large one? -- instead of confounding it with token count.

Consequences worth stating up front:
  - ps1tile16 sees a 160 m x 160 m footprint. Flood extent is often larger than that, so
    a chip can be entirely inside the flood and offer no un-flooded reference.
  - ps8tile128 sees 1.28 km and keeps the flood/dry boundary in view, but each token
    averages 80 m, so thin channels and field-scale detail are gone by construction.
  - The head upsamples token grid -> label resolution by a factor of patch_size, so
    ps8 must hallucinate 8x more spatial detail than ps1 from each token.

Per sub-tile we save PER-TIMESTEP features (T is kept, NOT mean-pooled) so the downstream
head can use pre/post separately -- that is the change-detection signal:
    features/<cfg>/<split>/<idx>.pt = {
        "feat":  (T=2, gH, gW, D) float16,   # t0 = pre-event, t1 = post-event
        "label": (tile, tile) int16,         # {0,1,2}, -1 = ignore
    }

Reuses pool_per_timestep from exp/pastis/finetune_olmoearth, the same per-timestep spatial
pooling used by the PASTIS extractor, so features are directly comparable across datasets.

Run (OlmoEarth venv, GPU):
    python -u -m exp.geoidflood.extract_features --splits train,val --tile_size 128 --patch_size 8
"""
import os
import sys

from exp.common import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.finetune.model import to_device
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample

from exp.common.config import MODEL_SIZE_TO_ID
from exp.pastis.finetune_olmoearth import pool_per_timestep

POOLING_TYPE = PoolingType.MEAN
INPUT_RES = 10                 # GEOID-Flood is 10 m
NUM_CLASSES = 3                # 0 background, 1 permanent water, 2 flood


def date_to_timestamps(dates: list[str]) -> torch.Tensor:
    """(T,3) [day, month(0-indexed), year] per timestep. OlmoEarth uses month for its
    seasonal encoding, and GEOID's pre/post are often months apart (and in GEOID's case
    frequently across a season boundary), so each timestep gets its OWN date rather than
    one shared event date."""
    rows = []
    for d in dates:
        if d and len(d) == 8 and d.isdigit():
            y, m, dd = int(d[:4]), int(d[4:6]), int(d[6:])
        else:
            y, m, dd = 2020, 6, 1
        rows.append([dd, m - 1, y])
    return torch.tensor(rows, dtype=torch.long)


def s1_to_olmoearth_sample(s1: np.ndarray, dates: list[str], normalizer):
    """(T=2, C=2[VV,VH], H, W) dB -> MaskedOlmoEarthSample.

    OlmoEarth's S1 modality wants (H, W, T, [vv, vh]) in dB, which is exactly what
    prep_tiles stored, so this is a transpose plus normalization -- no band juggling."""
    s1 = np.asarray(s1, dtype=np.float32)
    arr = np.transpose(s1, (2, 3, 0, 1))                 # (H, W, T, C[vv,vh])
    if normalizer is not None:
        arr = normalizer.normalize(Modality.SENTINEL1, arr)
    sample = OlmoEarthSample(
        timestamps=date_to_timestamps(dates),
        sentinel1=torch.from_numpy(np.ascontiguousarray(arr)).float(),
    )
    return MaskedOlmoEarthSample.from_olmoearthsample(sample)


class TiledGeoidDataset(torch.utils.data.Dataset):
    """Yields (MaskedOlmoEarthSample, label, idx) from the prepped .pt sub-tiles."""

    def __init__(self, tiles_dir: Path, split: str, normalizer):
        self.dir = tiles_dir / split
        self.n = len(list(self.dir.glob("*.pt")))
        self.normalizer = normalizer

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rec = torch.load(self.dir / f"{i}.pt")
        s1 = rec["s1"].float().numpy()                    # (2,2,tile,tile) dB
        masked = s1_to_olmoearth_sample(
            s1, [rec.get("date_pre", ""), rec.get("date_post", "")], self.normalizer)
        return masked, rec["label"].long(), i


def _collate(batch):
    masked, label = eval_collate_fn([(m, l) for m, l, _ in batch])
    return masked, label, [i for _, _, i in batch]


@torch.no_grad()
def extract_split(encoder, split, tiles_dir, out_dir, args, device, normalizer) -> int:
    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    ds = TiledGeoidDataset(tiles_dir, split, normalizer)
    if ds.n == 0:
        print(f"[{split}] no tiles in {tiles_dir / split}; run prep_tiles first.")
        return 0
    if len(list(split_out.glob("*.pt"))) == ds.n:
        print(f"[{split}] {ds.n} feature files already present, skipping.")
        return ds.n

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=_collate)
    # bf16 autocast on CUDA only: FlexiViT's patch-resize uses bicubic+antialias, which has
    # no bf16 CPU kernel.
    use_amp = device.type == "cuda"
    n_sanitized = 0
    for masked, label, idxs in tqdm(loader, desc=f"extract {split}"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            tam = encoder(to_device(masked, device), patch_size=args.patch_size,
                          input_res=INPUT_RES, fast_pass=True)["tokens_and_masks"]
            T = next(getattr(tam, m).shape[3] for m in tam.modalities)
            feats = torch.stack([pool_per_timestep(tam, t, POOLING_TYPE) for t in range(T)],
                                dim=1)                     # (B, T, gH, gW, D)
        feats = feats.float()
        if not torch.isfinite(feats).all():
            n_sanitized += int((~torch.isfinite(feats)).any(dim=(1, 2, 3, 4)).sum().item())
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = feats.half().cpu()
        label = label.to(torch.int16).cpu()
        for b, idx in enumerate(idxs):
            torch.save({"feat": feats[b].clone(), "label": label[b].clone()},
                       split_out / f"{idx}.pt")
    if n_sanitized:
        print(f"[{split}] sanitized {n_sanitized} tiles with non-finite encoder output.")
    return ds.n


def main() -> None:
    p = argparse.ArgumentParser(description="Cache OlmoEarth features for tiled GEOID-Flood.")
    p.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--tile_size", type=int, default=128)
    p.add_argument("--tiles_root", default="data/geoidflood_tiles")
    p.add_argument("--out_root", default="features")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    if args.tile_size % args.patch_size != 0:
        raise ValueError(f"tile_size {args.tile_size} must be divisible by "
                         f"patch_size {args.patch_size}")
    grid = args.tile_size // args.patch_size
    name = f"geoid_{args.model_size}_s1_ps{args.patch_size}_res{INPUT_RES}_t{args.tile_size}"
    out_dir = Path(args.out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = Path(f"{args.tiles_root}_t{args.tile_size}")
    print(f"Extraction config: {name} (patch_size={args.patch_size}, "
          f"tile_size={args.tile_size}, input_res={INPUT_RES}, grid={grid}x{grid}, "
          f"{args.patch_size * INPUT_RES} m per token) reading {tiles_dir}/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local = Path("/scratch/timz/OlmoEarth-v1-Base")
    model_ref = str(local) if (args.model_size == "base" and local.exists()) \
        else getattr(ModelID, MODEL_SIZE_TO_ID[args.model_size])
    model = load_model_from_id(model_ref, load_weights=True)
    encoder = cast(nn.Module, model.encoder if hasattr(model, "encoder") else model)
    encoder = encoder.to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad = False
    normalizer = Normalizer(Strategy.COMPUTED)

    counts, T_dim, embed_dim = {}, None, None
    for split in [s for s in args.splits.split(",") if s]:
        counts[split] = extract_split(encoder, split, tiles_dir, out_dir, args, device,
                                      normalizer)
        if embed_dim is None and counts[split] > 0:
            sample = torch.load(out_dir / split / "0.pt")["feat"]
            T_dim, embed_dim = sample.shape[0], sample.shape[-1]

    meta = {
        "dataset": "geoidflood", "model_size": args.model_size,
        "patch_size": args.patch_size, "input_res": INPUT_RES, "modalities": ["sentinel1"],
        "pooling": str(POOLING_TYPE), "timesteps": T_dim, "embed_dim": embed_dim,
        "grid": grid, "tile_size": args.tile_size, "dtype": "float16", "counts": counts,
        "feature_shape": [T_dim, grid, grid, embed_dim], "num_classes": NUM_CLASSES,
        "label_size": args.tile_size, "metres_per_token": args.patch_size * INPUT_RES,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir}/meta.json: {meta}")


if __name__ == "__main__":
    main()
