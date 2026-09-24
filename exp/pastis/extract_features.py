"""Extract & cache OlmoEarth encoder features for PASTIS (frozen backbone).

Reads raw PASTIS-R through manyup's PastisDataset and encodes with OlmoEarthEncoder, writing
one (T, gH, gW, D) fp16 tensor per sample:
    <out_root>/<cfg>/pastis_r_<split>/<idx>.pt     idx = dataset index (see ids.txt)
    <out_root>/<cfg>/pastis_r_<split>/ids.txt      the sample id of every idx, in order
    <out_root>/<cfg>/pastis_r_<split>/frame_valid.pt   (N, T) bool: step t has data (empty
                                                   months are False; their features are 0)
    <out_root>/<cfg>/pastis_r_<split>/t_start.pt, t_end.pt  (N, T) int64 days since 1970: the
                                                   dates each feature step covers
    <out_root>/<cfg>/meta.json                     everything that produced the cache

--temporal picks how each modality's time series is sampled (see manyup.data.temporal):
  monthly  the default setup: Oct 2018 - Sep 2019 calendar months, S2 median / S1A mean,
           empty months masked (the encoder passes them as MISSING tokens).
  legacy   OlmoEarth's own PASTIS prep (S2 month means, first 12 non-empty months). S2 only:
           that prep aligned S1 by S2's dates, so legacy S1 is not reproducible. Legacy cache
           names are unchanged, so old caches and new legacy runs share them.

--temporal_mode series: timesteps attend to each other; single: each timestep encoded alone.
--tile_size < sample size splits the image into independent encoder windows (approximate).
Resumable: batches whose files all exist are skipped. Sample order is the legacy order, so
old caches' indices refer to the same samples.

    python -u -m exp.pastis.extract_features --temporal monthly --modalities sentinel2_l2a,sentinel1 \
        --patch_size 4 --tile_size 64
"""
import argparse
import json
import re
from pathlib import Path

from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.paths import FEATURES  # noqa: E402
from exp.pastis.data import MODALITIES, TEMPORAL, make_dataset  # noqa: E402
from manyup.data import mtmm_collate  # noqa: E402
from manyup.data.datasets.pastis import PastisDataset  # noqa: E402
from manyup.encoders.olmoearth import (LEGACY_S2_IMPUTE, OlmoEarthEncoder,  # noqa: E402
                                       OlmoEarthInput)

SHORT = {"sentinel2_l2a": "s2", "sentinel1": "s1"}
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}   # legacy folder names
TEMPORAL_MODES = {"series": "series", "single": "per_timestep"}


def cfg_name(model_size: str, modalities: list[str], patch_size: int, tile_size: int,
             temporal_mode: str, image_size: int, temporal: str, ft_tag: str | None) -> str:
    """oe_<size>_<mods>_ps<P>_tile<T>[_single][_img<N>][_<temporal>][_ft<tag>]; legacy
    temporal adds nothing, so pre-existing cache names keep their meaning."""
    mods = "".join(SHORT[m] for m in modalities)
    return (f"oe_{model_size}_{mods}_ps{patch_size}_tile{tile_size}"
            + ("" if temporal_mode == "series" else f"_{temporal_mode}")
            + ("" if image_size == 64 else f"_img{image_size}")
            + ("" if temporal == "legacy" else f"_{temporal}")
            + (f"_ft{ft_tag}" if ft_tag else ""))


def extract_split(encoder, ds, out_dir: Path, args, mode: str) -> int:
    split_dir = out_dir / f"pastis_r_{SPLIT_DIRS[ds.split]}"
    split_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.limit, len(ds)) if args.limit > 0 else len(ds)
    (split_dir / "ids.txt").write_text("".join(ds.sample_id(i) + "\n" for i in range(len(ds))))
    done = all((split_dir / f"{i}.pt").exists() for i in range(n))
    if done and all((split_dir / f).exists() for f in ("frame_valid.pt", "t_start.pt", "t_end.pt")):
        print(f"[{ds.split}] {n} files already present, skipping.")
        return n
    subset = torch.utils.data.Subset(ds, range(n))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=mtmm_collate)
    idx, valid, t_start, t_end = 0, [], [], []
    mapped = list(encoder.inputs.modality_map)
    for batch in tqdm(loader, desc=f"extract {ds.split}"):
        B = len(batch.ids)
        valid.append(torch.stack([batch.modalities[m].frame_valid for m in mapped]).any(0))
        t_start.append(batch.modalities[mapped[0]].t_start)     # aligned across mapped mods
        t_end.append(batch.modalities[mapped[0]].t_end)
        if all((split_dir / f"{idx + b}.pt").exists() for b in range(B)):   # resume
            idx += B
            continue
        feats = encoder.encode(batch, mode).tokens.half().cpu()             # (B,T,gH,gW,D)
        for b in range(B):
            torch.save(feats[b].clone(), split_dir / f"{idx + b}.pt")
        idx += B
    torch.save(torch.cat(valid), split_dir / "frame_valid.pt")
    torch.save(torch.cat(t_start), split_dir / "t_start.pt")
    torch.save(torch.cat(t_end), split_dir / "t_end.pt")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Cache OlmoEarth per-timestep features for PASTIS.")
    p.add_argument("--temporal", required=True, choices=TEMPORAL)
    p.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    p.add_argument("--modalities", default="sentinel2_l2a",
                   help="comma-separated subset of " + ",".join(MODALITIES))
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--tile_size", type=int, default=None, help="encoder window; default = image")
    p.add_argument("--image_size", type=int, default=64, choices=(64, 128),
                   help="64: 64 px tiles of each patch (legacy order); 128: whole patches")
    p.add_argument("--temporal_mode", default="series", choices=list(TEMPORAL_MODES))
    p.add_argument("--attention", default="flash", choices=("flash", "sdpa"),
                   help="flash: exact masking of empty months, batch-independent (default). "
                        "sdpa: bit-identical to the legacy caches; no missing data allowed")
    p.add_argument("--init_ckpt", default=None,
                   help="finetune_olmoearth checkpoint whose backbone replaces the pretrained "
                        "weights; the cache name gets _ft<tag>")
    p.add_argument("--ft_tag", default=None, help="tag for --init_ckpt (default: from filename)")
    p.add_argument("--out_root", default=str(FEATURES))
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help=">0: only the first N samples per split")
    args = p.parse_args()

    modalities = [m for m in args.modalities.split(",") if m]
    if not modalities or set(modalities) - set(MODALITIES):
        raise ValueError(f"--modalities must be a subset of {list(MODALITIES)}")
    tile = args.tile_size or args.image_size
    ft_tag = None
    if args.init_ckpt:
        stem = Path(args.init_ckpt).stem
        m = re.search(r"_p(\d+)(?:_img\d+)?_lr[\d.e-]+_ep(\d+)", stem)
        ft_tag = re.sub(r"[^A-Za-z0-9]", "", args.ft_tag or (f"p{m.group(1)}ep{m.group(2)}"
                                                             if m else stem[:16]))
    name = cfg_name(args.model_size, modalities, args.patch_size, tile, args.temporal_mode,
                    args.image_size, args.temporal, ft_tag)
    splits = [s for s in args.splits.split(",") if s]
    datasets = {s: make_dataset(s, modalities, args.temporal, args.image_size) for s in splits}
    out_dir = Path(args.out_root) / name              # created only once the setup is valid
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = OlmoEarthInput({MODALITIES[m][0]: MODALITIES[m][1] for m in modalities},
                            impute={"S2": LEGACY_S2_IMPUTE})
    encoder = OlmoEarthEncoder(MODEL_SIZE_TO_ID[args.model_size], inputs, args.patch_size, tile,
                               name=args.model_size, attention=args.attention)
    if args.init_ckpt:
        encoder.load_finetuned(args.init_ckpt)
    encoder = encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    mode = TEMPORAL_MODES[args.temporal_mode]
    print(f"Extraction config: {name}")

    counts = {}
    for split, ds in datasets.items():
        encoder.check(ds, mode)
        counts[SPLIT_DIRS[split]] = extract_split(encoder, ds, out_dir, args, mode)

    first = next(iter(datasets.values()))
    probe = torch.load(out_dir / f"pastis_r_{SPLIT_DIRS[first.split]}" / "0.pt")
    meta = {
        "model_size": args.model_size, "model_id": MODEL_SIZE_TO_ID[args.model_size],
        "patch_size": args.patch_size, "tile_size": tile, "input_res": 10,
        "image_size": args.image_size, "grid": args.image_size // args.patch_size,
        "modalities": modalities, "temporal": args.temporal,
        "temporal_keys": {m: pol.resolved_key(PastisDataset.SPEC.modalities[m])
                          for m, pol in first.modalities.items()},
        "dataset_key": first.key, "impute": inputs.impute, "attention": args.attention,
        "temporal_mode": args.temporal_mode,
        "time_precision": "month" if args.temporal == "monthly" else "month (legacy: first 12 non-empty)",
        "cross_timestep_attention": args.temporal_mode == "series",
        "pooling": "mean over bandsets, then over the modalities present",
        "timesteps": probe.shape[0], "embed_dim": probe.shape[-1], "dtype": "float16",
        "counts": counts, "feature_shape": list(probe.shape),
        "exact": tile == args.image_size, "init_ckpt": args.init_ckpt,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir}/meta.json")


if __name__ == "__main__":
    main()
