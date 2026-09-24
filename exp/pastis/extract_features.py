"""Extract & cache OlmoEarth encoder features for PASTIS (frozen backbone).

With the backbone frozen, its output is identical across epochs and across head
variants (lp / anyup / anyup_t1 / anyup_t2). Computing it once and caching to disk lets
head experiments (see exp/pastis/lp_cached_features.py) skip the encoder entirely.

What we cache, per sample: the encoder's PER-TIMESTEP spatial token map, i.e.
    feats[t] = pool over bandsets and across modalities (S2+S1) of timestep t's tokens
        -> (T, gH, gW, D)   with gH = gW = input_size / patch_size, D = embed dim.
Time is kept (not mean-pooled) so temporal heads (anyup_t1/t2) can use it; lp/anyup just
mean over T. This reuses pool_per_timestep() from exp/pastis/finetune_olmoearth.py so the
cached features are bit-for-bit the same reduction the live heads use.

Sample size: --image_size (64 default, or 128 for a prepare_data.py --image_size 128 prep,
i.e. raw PASTIS patches kept whole instead of quartered). Caches for sizes other than 64 get
an _img<N> suffix, since tile_size alone does not disambiguate them.

Spatial handling: we operate on the existing PASTIS tiles (no resize). input_res is
FIXED at BASE_GSD = 10m (the true physical pixel size). patch_size is the resolution knob
(token grid = 64/patch_size). Smaller patch_size => finer grid => quadratically more tokens
=> quadratically more attention compute; to bound peak memory/compute we optionally split
each 64x64 sample into tile_size x tile_size sub-tiles, encode each independently, and
stitch the token grids back together.

  NOTE: transformer attention is global, so independent sub-tiles do NOT cross-attend.
  tile_size < 64 therefore yields an APPROXIMATION of the full-image features (accepted
  for the compute savings). tile_size == 64 is the exact, non-approximated reference.

Temporal handling (--temporal_mode):
  series  (default) feed the whole T-step series in one encoder call. Attention is global
          over all (H, W, T, bandset) tokens, so a timestep's tokens DO attend to other
          timesteps -- the per-timestep feature map is temporally contextualized.
  single  feed each timestep separately (T encoder calls on T=1 slices). Tokens from
          different timesteps cannot attend to each other, so feats[t] depends only on
          timestep t. Output shape/dtype/layout are identical to `series`, so downstream
          heads (lp / anyup / anyup_t1 / anyup_t2) consume either cache unchanged.

  The encoder has no flag for this: HeliosEncoder.forward flattens H,W,T,bandsets into one
  sequence and (under fast_pass) attends with attn_mask=None. Slicing the sample to T=1 is
  how the eval baselines (dinov3/clay/croma) do single-timestep encoding, and it is exact
  rather than an approximation -- with one timestep present there is nothing to attend to.
  The slice includes `timestamps` ([B, T, 3]), which drives the month/year positional
  encoding (flexi_vit uses timestamps[:, :, 1]); leaving it unsliced would encode every
  timestep as t=0's date.

Run inside a GPU salloc:
    source env_setup/env_olmo.sh
    python -u -m exp.pastis.extract_features --model_size base --patch_size 1 --tile_size 32
    python -u -m exp.pastis.extract_features --model_size base --patch_size 1 --tile_size 32 --temporal_mode single
"""
import os
import sys

# Must run before any olmoearth_pretrain.evals import (see exp/common/olmo_bootstrap.py).
from exp.common import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import json
import re
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.evals.datasets.pastis_dataset import PASTISRDataset
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.finetune.model import to_device
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.data.constants import BASE_GSD

from exp.common.config import MODEL_SIZE_TO_ID, ALLOWED_MODALITIES
from exp.pastis.finetune_olmoearth import pool_per_timestep
from exp.common.paths import FEATURES

POOLING_TYPE = PoolingType.MEAN          # matches finetune default
IMAGE_SIZE = 64                          # DEFAULT sample size; override with --image_size
SPLITS = ("train", "valid", "test")
# Upper bound on batch items in ONE folded encoder call. Past roughly this, CUDA's attention
# kernels hit a grid-dimension limit and raise "invalid configuration argument". Measured to
# fail at 98,304 (1024 tiles x batch 8 x T 12); 16384 leaves comfortable headroom and still
# collapses 1024 one-token tiles into a handful of calls instead of 1024.
MAX_FOLDED_ITEMS = 16384


def cfg_name(model_size: str, modalities: list[str], patch_size: int, tile_size: int,
             temporal_mode: str = "series", image_size: int = IMAGE_SIZE,
             ft_tag: str | None = None) -> str:
    """Stable folder name encoding the extraction params, so different settings never
    collide on disk, e.g. oe_base_s2s1_ps4_tile64.

    temporal_mode="series" adds no suffix, so existing caches keep their current paths;
    "single" appends _single.

    image_size=64 adds no suffix (every existing cache is 64); other sizes append _img<N>.
    This suffix is REQUIRED for correctness, not just tidiness: tile_size alone is ambiguous
    across image sizes -- ps4_tile64 is a whole 64x64 sample encoded in one pass, but on a
    128x128 sample it is a 2x2-tiled approximation with a 32x32 token grid. Without the
    suffix the two would share a directory and the second run would look 'already extracted'."""
    mods = "".join({"sentinel2_l2a": "s2", "sentinel1": "s1"}[m] for m in modalities)
    suffix = "" if temporal_mode == "series" else f"_{temporal_mode}"
    img = "" if image_size == 64 else f"_img{image_size}"
    # Fine-tuned features come from DIFFERENT weights, so they must never share a directory
    # with pretrained ones -- otherwise a second run looks "already extracted" and silently
    # mixes two encoders in one cache.
    ft = f"_ft{ft_tag}" if ft_tag else ""
    return f"oe_{model_size}_{mods}_ps{patch_size}_tile{tile_size}{suffix}{img}{ft}"


def make_loader(split: str, data_splits: str, modalities: list[str],
                batch_size: int, num_workers: int) -> DataLoader:
    """PASTIS loader (non-anyup branch of finetune's make_loader). shuffle is always
    False here: we write per-sample files keyed by dataset index, so order must be stable."""
    ds = PASTISRDataset(
        path_to_splits=Path(data_splits),
        split=split,
        norm_stats_from_pretrained=True,
        input_modalities=modalities,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, collate_fn=eval_collate_fn)


def _slice_tile(masked, h0: int, h1: int, w0: int, w1: int):
    """Slice a spatial sub-tile [h0:h1, w0:w1] out of a batched MaskedOlmoEarthSample.
    Spatial modalities and their masks are [B, H, W, T, ...]; non-spatial fields
    (timestamps, latlon) pass through unchanged."""
    repl = {}
    for field in masked._fields:
        val = getattr(masked, field)
        if val is None or field in ("timestamps", "latlon", "latlon_mask"):
            continue
        if val.dim() >= 3:              # [B, H, W, ...]
            repl[field] = val[:, h0:h1, w0:w1]
    return masked._replace(**repl)


def _slice_timestep(masked, t: int):
    """Slice a single timestep t out of a batched MaskedOlmoEarthSample, keeping a length-1
    time axis so the sample stays shape-compatible with the encoder.

    Spatial modalities and their masks are [B, H, W, T, ...] (time is dim 3); timestamps are
    [B, T, 3] (time is dim 1) and MUST be sliced too -- they drive the month/year encoding,
    so an unsliced copy would label every timestep with t=0's date. latlon ([B, 2]) has no
    time axis and passes through."""
    repl = {}
    for field in masked._fields:
        val = getattr(masked, field)
        if val is None or field in ("latlon", "latlon_mask"):
            continue
        if field == "timestamps":
            repl[field] = val[:, t:t + 1]
        elif val.dim() >= 4:            # [B, H, W, T, ...]
            repl[field] = val[:, :, :, t:t + 1]
    return masked._replace(**repl)


def _num_input_timesteps(masked) -> int:
    """T from the batched sample's timestamps ([B, T, 3])."""
    return masked.timestamps.shape[1]


def _fold_time_into_batch(masked, T: int):
    """Stack the T single-timestep slices along the BATCH dim -> one sample of batch B*T,
    each with T=1. Layout is [all B @ t0, all B @ t1, ...], so a later chunk(T, dim=0)
    recovers per-timestep blocks in order.

    This is what makes `single` mode fast: the T slices are independent, so instead of T
    small encoder calls (which leave the GPU idle -- measured ~3x slower than one series
    call) we issue ONE call on a B*T batch. Same math, no cross-timestep attention (each
    batch item holds a single timestep), but full GPU utilization."""
    slices = [_slice_timestep(masked, t) for t in range(T)]
    merged = {}
    for field in slices[0]._fields:
        vals = [getattr(s, field) for s in slices]
        merged[field] = None if vals[0] is None else torch.cat(vals, dim=0)
    return slices[0]._replace(**merged)


def _cat_samples(slices):
    """Concatenate a list of equally-shaped MaskedOlmoEarthSamples along the BATCH dim.

    Group-major layout ([all B of slice0, all B of slice1, ...]) so a later slice of dim 0
    recovers each input's block in order -- same convention as _fold_time_into_batch."""
    merged = {}
    for field in slices[0]._fields:
        vals = [getattr(s, field) for s in slices]
        merged[field] = None if vals[0] is None else torch.cat(vals, dim=0)
    return slices[0]._replace(**merged)


@torch.no_grad()
def _encode_tile(encoder, tile, patch_size: int, temporal_mode: str) -> torch.Tensor:
    """Encode one spatial tile -> (B, tg, tg, T, D).

    series: one encoder call on the full series; attention is global over all timesteps, so
        each timestep's tokens are contextualized by the others.
    single: one encoder call on a B*T batch of T=1 slices (see _fold_time_into_batch); no
        cross-timestep attention is possible. Output layout matches the series path."""
    if temporal_mode == "single":
        T = _num_input_timesteps(tile)
        tam = encoder(_fold_time_into_batch(tile, T), patch_size=patch_size,
                      input_res=BASE_GSD, fast_pass=True)["tokens_and_masks"]
        # every batch item has a single timestep, so index 0 pools it
        pooled = pool_per_timestep(tam, 0, POOLING_TYPE)   # (B*T, tg, tg, D)
        per_t = list(pooled.chunk(T, dim=0))               # T x (B, tg, tg, D)
    else:
        tam = encoder(tile, patch_size=patch_size, input_res=BASE_GSD,
                      fast_pass=True)["tokens_and_masks"]
        per_t = [pool_per_timestep(tam, t, POOLING_TYPE)
                 for t in range(_num_timesteps(tam))]
    return torch.stack(per_t, dim=-2)                   # (B, tg, tg, T, D)


@torch.no_grad()
def encode_batch(encoder, masked, patch_size: int, tile_size: int, device,
                 temporal_mode: str = "series", image_size: int = IMAGE_SIZE,
                 tiles_per_call: int = 1) -> torch.Tensor:
    """Encode one batch -> (B, T, gH, gW, D), tiling spatially if tile_size < image_size.

    For each tile we run the encoder (see _encode_tile for the temporal_mode split) and pool
    per timestep (pool_per_timestep), then place the tile's (B, tg, tg, T, D) token block
    into the full (B, gH, gW, T, D) grid and finally move time to dim 1."""
    masked = to_device(masked, device)
    tg = tile_size // patch_size                       # tokens per tile side
    n = image_size // tile_size                         # tiles per side
    grid = image_size // patch_size                     # full token grid side

    # Tiles are INDEPENDENT (attention never crosses a tile), identically shaped, and there
    # can be very many of them: tile_size == patch_size gives (image_size/tile_size)^2 tiles
    # of ONE token each -- 1024 per sample at ps2/tile2. Encoding those one at a time is
    # 1024 serial launches of a 1-token transformer, which is pure overhead: the GPU sits
    # idle and the theoretical FLOP saving of `single` mode never materializes (measured:
    # ps2_tile2_single ran SLOWER than ps2_tile32 despite doing far less attention work).
    # So fold the tile axis into the batch, exactly as _fold_time_into_batch does for time,
    # and issue ONE call. tiles_per_call caps the fold so memory stays bounded.
    tiles = [(ti, tj) for ti in range(n) for tj in range(n)]
    full = None
    if tiles_per_call <= 0:
        # Auto: cap the folded call at ~4096 tokens per batch item, so a tiny-tile config
        # (1 token/tile) folds many tiles into one call while a big-tile config keeps the
        # old one-tile-at-a-time behaviour and its memory profile.
        tok_per_tile = (tile_size // patch_size) ** 2
        tiles_per_call = max(1, min(len(tiles), 4096 // max(tok_per_tile, 1)))
        # Also cap the resulting BATCH WIDTH. `single` mode folds T into the batch on top of
        # the tiles, so tiles x B x T can reach ~98k items at tile2/batch8 -- past that
        # scaled_dot_product_attention fails with "CUDA error: invalid configuration
        # argument" (a kernel grid-dimension limit, not OOM). Bounding items keeps the fold
        # legal for any batch_size instead of only the small ones.
        B0 = masked.timestamps.shape[0]
        fold = B0 * (_num_input_timesteps(masked) if temporal_mode == "single" else 1)
        tiles_per_call = max(1, min(tiles_per_call, MAX_FOLDED_ITEMS // max(fold, 1)))
    chunk = max(1, tiles_per_call)
    for c0 in range(0, len(tiles), chunk):
        group = tiles[c0:c0 + chunk]
        sl = [_slice_tile(masked, ti * tile_size, (ti + 1) * tile_size,
                          tj * tile_size, (tj + 1) * tile_size) for ti, tj in group]
        merged = _cat_samples(sl) if len(sl) > 1 else sl[0]
        block = _encode_tile(encoder, merged, patch_size, temporal_mode)   # (G*B, tg,tg,T,D)
        if full is None:
            GB, _, _, T, D = block.shape
            B = GB // len(group)
            full = block.new_zeros((B, grid, grid, T, D))
        else:
            B = full.shape[0]
        # _cat_samples stacks group-major ([all B of tile0, all B of tile1, ...]), so chunking
        # on dim 0 recovers each tile's own (B, tg, tg, T, D) block in order.
        for gi, (ti, tj) in enumerate(group):
            full[:, ti * tg:(ti + 1) * tg, tj * tg:(tj + 1) * tg] = block[gi * B:(gi + 1) * B]
    return full.permute(0, 3, 1, 2, 4).contiguous()     # (B, T, gH, gW, D)


def _num_timesteps(tam) -> int:
    """T from the first spatial modality token tensor (B, gH, gW, T, BandSets, D)."""
    for m in tam.modalities:
        return getattr(tam, m).shape[3]
    raise ValueError("no modalities in TokensAndMasks")


def _dir_size_bytes(path: Path) -> int:
    """Total bytes of all files under path (recursive)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(nbytes: int) -> str:
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def extract_split(encoder, split: str, out_dir: Path, args, device) -> int:
    """Write one .pt per sample: features/<cfg>/pastis_r_<split>/<idx>.pt -> (T,gH,gW,D) fp16.
    Idempotent: if the folder already has the expected file count, skip the split.

    args.limit (>0) caps the split to its first N samples (indices 0..N-1). The loader has
    shuffle=False, so those are the SAME samples/files the full run writes -- a limited run is a
    strict prefix of the full one, letting a later unlimited run just fill in the rest. Handy for
    getting the first few test images extracted for visualization before the full job finishes."""
    split_dir = out_dir / f"pastis_r_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(split, args.data_splits, args.modalities,
                         args.batch_size, args.num_workers)
    full_n = len(loader.dataset)
    n_samples = min(args.limit, full_n) if args.limit and args.limit > 0 else full_n
    existing = len(list(split_dir.glob("*.pt")))
    if existing >= n_samples:
        # >= (not ==) so a limited run is a no-op once the full extraction already covers it.
        print(f"[{split}] {existing} files already present (need {n_samples}), skipping.")
        return n_samples

    idx = 0
    for batch in tqdm(loader, desc=f"extract {split}"):
        masked, _label = batch

        # RESUME: skip the encoder entirely when every sample this batch would write is
        # already on disk. Without this a partially-extracted split restarts at index 0 and
        # re-encodes work it already has -- for the very slow configs (ps1 tile128 is ~75 s
        # per sample) that means a resubmitted job can spend its whole walltime redoing the
        # prefix and never advance, so a chain of jobs never converges. Indices are stable
        # because the loader is shuffle=False.
        batch_n = min(masked.timestamps.shape[0], max(0, n_samples - idx))
        if batch_n and all((split_dir / f"{idx + b}.pt").exists() for b in range(batch_n)):
            idx += batch_n
            continue

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats = encode_batch(encoder, masked, args.patch_size, args.tile_size, device,
                                 args.temporal_mode, args.image_size,
                                 tiles_per_call=args.tiles_per_call)
        feats = feats.half().cpu()                       # (B, T, gH, gW, D)
        for b in range(feats.shape[0]):
            if idx >= n_samples:                         # hit the limit mid-batch: stop writing
                return n_samples
            torch.save(feats[b].clone(), split_dir / f"{idx}.pt")
            idx += 1
    assert idx == n_samples, f"wrote {idx} != {n_samples} for {split}"
    return n_samples


def main() -> None:
    p = argparse.ArgumentParser(description="Cache OlmoEarth per-timestep features for PASTIS.")
    p.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    p.add_argument("--patch_size", type=int, default=4)
    # Default resolved after parsing, since it depends on --image_size.
    p.add_argument("--tile_size", type=int, default=None,
                   help="spatial sub-tile size (<= image_size); equal to image_size means no "
                        "tiling (exact). Default: image_size.")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE, choices=(64, 128),
                   help=f"PASTIS sample size on disk (default {IMAGE_SIZE}). Use 128 with a "
                        f"--data_splits prepared by prepare_data.py --image_size 128; caches "
                        f"for sizes other than 64 get an _img<N> suffix.")
    p.add_argument("--modalities", default="sentinel2_l2a", # sentinel1
                   help="comma-separated; subset of " + ",".join(ALLOWED_MODALITIES))
    p.add_argument("--temporal_mode", default="series", choices=("series", "single"),
                   help="series: one encoder call per sample, tokens attend across time. "
                        "single: one encoder call per timestep, no cross-timestep attention "
                        "(T x more calls; written to a separate _single cache dir)")
    p.add_argument("--tiles_per_call", type=int, default=0,
                   help="how many spatial tiles to fold into ONE encoder call. Tiles never "
                        "cross-attend, so folding them is EXACT -- it changes only GPU "
                        "utilization. 0 (default) auto-picks ~4096 tokens per call: a tiny-"
                        "tile config (ps2/tile2 = 1024 one-token tiles) folds them all into "
                        "one call, while existing big-tile configs keep the old serial "
                        "behaviour and their memory profile. Lower it if a config OOMs.")
    p.add_argument("--init_ckpt", default=None,
                   help="finetune checkpoint whose BACKBONE weights replace the pretrained "
                        "ones (from finetune_olmoearth.py, keys prefixed 'backbone.'). The "
                        "cache name gets an _ft<tag> suffix so fine-tuned features never "
                        "share a directory with pretrained ones.")
    p.add_argument("--ft_tag", default=None,
                   help="short tag for the --init_ckpt cache suffix; defaults to the "
                        "checkpoint's patch size + epochs (e.g. p16ep64)")
    p.add_argument("--data_splits", default="data/pastis_olmoearth")
    p.add_argument("--out_root", default=str(FEATURES))
    p.add_argument("--splits", default=",".join(SPLITS),
                   help="comma-separated subset of train,valid,test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only extract the first N samples per split (a prefix of the full "
                        "run; e.g. --splits test --limit 4 for quick visualization)")
    args = p.parse_args()

    args.modalities = [m for m in args.modalities.split(",") if m]
    bad = [m for m in args.modalities if m not in ALLOWED_MODALITIES]
    if bad:
        raise ValueError(f"modalities {bad} not in {list(ALLOWED_MODALITIES)}")
    splits = [s for s in args.splits.split(",") if s]
    if args.tile_size is None:
        args.tile_size = args.image_size          # no tiling: the exact, full-image reference
    if args.tile_size > args.image_size:
        raise ValueError(f"tile_size {args.tile_size} exceeds image_size {args.image_size}")
    if args.image_size % args.tile_size != 0:
        raise ValueError(f"tile_size {args.tile_size} must divide image_size {args.image_size}")
    if args.tile_size % args.patch_size != 0:
        raise ValueError(f"tile_size {args.tile_size} must be divisible by patch_size {args.patch_size}")

    # Derive the fine-tune tag from the checkpoint filename when not given, so the cache
    # name records WHICH finetune it came from rather than just "some finetune".
    ft_tag = None
    if args.init_ckpt:
        ft_tag = args.ft_tag
        if not ft_tag:
            stem = Path(args.init_ckpt).stem
            m = re.search(r"_p(\d+)(?:_img\d+)?_lr[\d.e-]+_ep(\d+)", stem)
            ft_tag = f"p{m.group(1)}ep{m.group(2)}" if m else stem[:16]
        ft_tag = re.sub(r"[^A-Za-z0-9]", "", ft_tag)

    name = cfg_name(args.model_size, args.modalities, args.patch_size, args.tile_size,
                    args.temporal_mode, args.image_size, ft_tag)
    out_dir = Path(args.out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extraction config: {name}")
    print(f"  image_size={args.image_size} patch_size={args.patch_size} "
          f"tile_size={args.tile_size} input_res={BASE_GSD} "
          f"grid={args.image_size // args.patch_size} modalities={args.modalities} "
          f"temporal_mode={args.temporal_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_id(getattr(ModelID, MODEL_SIZE_TO_ID[args.model_size]),
                               load_weights=True)
    encoder = cast(nn.Module, model.encoder if hasattr(model, "encoder") else model)
    if args.init_ckpt:
        # finetune_olmoearth saves the whole task model; the encoder lives under "backbone."
        # and "_head.*" is the task head we do not want. Load STRICTLY (after stripping the
        # prefix) so a key mismatch is an error, not a silently half-pretrained encoder.
        sd = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not bb:
            raise ValueError(f"--init_ckpt {args.init_ckpt} has no 'backbone.' keys "
                             f"(got prefixes {sorted({k.split('.')[0] for k in sd})})")
        missing, unexpected = encoder.load_state_dict(bb, strict=False)
        if missing or unexpected:
            raise ValueError(f"--init_ckpt backbone does not match the {args.model_size} "
                             f"encoder: {len(missing)} missing, {len(unexpected)} unexpected "
                             f"(first missing {missing[:3]}, first unexpected {unexpected[:3]})")
        print(f"  loaded FINETUNED backbone from {args.init_ckpt} ({len(bb)} tensors)")
    encoder = encoder.to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad = False

    counts = {}
    grid = args.image_size // args.patch_size
    embed_dim = None
    for split in splits:
        counts[split] = extract_split(encoder, split, out_dir, args, device)
        if embed_dim is None:
            # peek one file to record D / T for meta
            sample = torch.load(out_dir / f"pastis_r_{split}" / "0.pt")
            T_dim, embed_dim = sample.shape[0], sample.shape[-1]

    meta = {
        "model_size": args.model_size,
        "model_id": MODEL_SIZE_TO_ID[args.model_size],
        "patch_size": args.patch_size,
        "tile_size": args.tile_size,
        "input_res": BASE_GSD,
        "image_size": args.image_size,
        "grid": grid,
        "modalities": args.modalities,
        "temporal_mode": args.temporal_mode,
        "cross_timestep_attention": args.temporal_mode == "series",
        "pooling": str(POOLING_TYPE),
        "timesteps": T_dim,
        "embed_dim": embed_dim,
        "dtype": "float16",
        "counts": counts,
        "feature_shape": [T_dim, grid, grid, embed_dim],
        "exact": args.tile_size == args.image_size,
        "size_bytes": _dir_size_bytes(out_dir),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_dir}/meta.json: {meta}")
    print(f"Total size of {out_dir}: {_human(meta['size_bytes'])}")


if __name__ == "__main__":
    main()
