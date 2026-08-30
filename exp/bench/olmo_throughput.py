"""Measure OlmoEarth feature-extraction throughput across (patch_size, tile_size, modality).

Answers: how fast can we extract frozen encoder features at each resolution setting, and
how does that scale with the token grid? Uses DUMMY tensors shaped exactly like PASTIS so
no dataset/IO is in the loop -- this measures the ENCODER, which is what the patch/tile
knobs actually change.

PASTIS geometry (verified against data/pastis_olmoearth/*/s{1,2}_images/0.pt):
    64 x 64 px @ 10 m (BASE_GSD),  T = 12 monthly composites
    sentinel2_l2a : 13 bands       sentinel1 : 2 bands (VV, VH) in dB

Arms: s2 only, s1 only, s2+s1 -- so the multi-modal cost is separable from the
per-modality cost. OlmoEarth splits each modality into bandsets, so adding S1 adds tokens
to the SAME sequence; cost is superlinear in total tokens, not additive across modalities.

Batch size: auto-tuned per config (--batch_size 0, the default). We probe upward by
doubling from 1 while a step fits in GPU memory, then binary-search the gap, and finally
back off by --bs_safety (default 10%) so the reported number is not sitting on the OOM
edge. This matters because the whole point is peak throughput, and small patch sizes have
quadratically larger activations -- a batch size that fits ps8 will OOM ps1 and a batch
size that fits ps1 leaves ps8 at a fraction of its achievable rate.

Timing: torch.cuda.synchronize around a warmup + --iters timed steps, reporting median
step time (robust to a stray clock/thermal blip) alongside mean +- std. Throughput is
reported as samples/s where a "sample" is one full 64x64x12 PASTIS chip -- so tiled
configs (tile_size < 64) are charged for ALL their tiles, which is the honest comparison.

Runs in the OlmoEarth venv, on a GPU:
    source env_setup/env_olmo.sh
    python -u -m exp.bench.olmo_throughput --out results/bench/olmo_throughput.csv
"""
import os
import sys

# Bootstrap MUST run before any olmoearth_pretrain import (HDF5/rasterio ABI).
from exp.common import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from torch.utils.flop_counter import FlopCounterMode

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.data.constants import BASE_GSD
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample

from exp.common.config import MODEL_SIZE_TO_ID
from exp.pastis.finetune_olmoearth import pool_per_timestep

POOLING_TYPE = PoolingType.MEAN

# --- PASTIS geometry (see module docstring) ---
IMAGE_SIZE = 64          # default PASTIS sample side; --image_size 128 for the 128 prep
NUM_TIMESTEPS = 12
BANDS = {"sentinel2_l2a": 13, "sentinel1": 2}

# Batch x tokens-per-call budget used to SEED the batch-size search (see find_max_batch).
# Calibrated on a 46 GB L40S, where ps1/tile16 (3072 tokens/call) fits ~64 and OOMs at 256:
# 64 x 3072 ~= 200k. It is only a starting guess -- the probe still verifies and expands.
TUNER_TOKEN_BUDGET = 200_000

# Found batch sizes are cached here keyed by (gpu, model, arm, patch, tile, image_size), so a
# rerun on the same card skips the search entirely -- the probes are the slow part of this
# benchmark (each one is a full encode, minutes at ps1). Delete the file to re-probe.
BATCH_CACHE = Path("results/bench/max_batch_cache.json")


# FLOPs depend only on the GRAPH SHAPE -- (arm, patch, tile, image_size) -- never on batch
# size or GPU, so they are cached separately from batch sizes and are portable across cards.
FLOP_CACHE = Path("results/bench/flops_cache.json")


def _flop_key(model_size: str, arm: str, patch_size: int, tile: int, image_size: int) -> str:
    return f"{model_size}|{arm}|ps{patch_size}|tile{tile}|img{image_size}"


def load_flop_cache() -> dict:
    if FLOP_CACHE.exists():
        try:
            return json.loads(FLOP_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


@torch.no_grad()
def measure_flops(encoder, arm: str, patch_size: int, tile: int, image_size: int,
                  device, cache: dict, model_size: str) -> float | None:
    """FLOPs for ONE sample at this config, measured on a single batch-1 dummy.

    Done in its own pass BEFORE any batch-size tuning or timing, because:
      - FLOPs per sample are independent of batch size (the graph is the same, just replayed
        B times), so there is nothing to gain from measuring at the tuned batch size, and
      - tracing while a card-filling batch is resident is what OOM-killed the first
        FLOP-enabled sweep at 4/15 configs. At batch 1, before anything large is allocated,
        the probe costs a few hundred MB and cannot disturb the measurement that follows.

    FlopCounterMode traces the ops actually dispatched (SDPA attention included), so we do
    not re-derive the model's arithmetic by hand and risk missing a term. One tile is traced
    and multiplied by n_tiles -- every tile of a sample runs the identical graph.
    """
    key = _flop_key(model_size, arm, patch_size, tile, image_size)
    if key in cache:
        return cache[key]
    n_tiles = (image_size // tile) ** 2
    probe = None
    try:
        probe = make_dummy(1, tile, ARMS[arm], device)
        counter = FlopCounterMode(display=False)
        with counter:
            run_step(encoder, probe, patch_size, tile, 1, device)
        total = float(counter.get_total_flops()) * n_tiles
        cache[key] = total
        FLOP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FLOP_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
        return total
    except Exception as e:
        print(f"    (flop count failed for {key}: {type(e).__name__})", flush=True)
        return None
    finally:
        del probe
        gc.collect()
        torch.cuda.empty_cache()


def _cache_key(gpu: str, model_size: str, arm: str, patch_size: int, tile: int,
               image_size: int) -> str:
    return f"{gpu}|{model_size}|{arm}|ps{patch_size}|tile{tile}|img{image_size}"


def load_batch_cache() -> dict:
    if BATCH_CACHE.exists():
        try:
            return json.loads(BATCH_CACHE.read_text())
        except json.JSONDecodeError:      # a truncated write should not kill the run
            return {}
    return {}


def save_batch_cache(cache: dict) -> None:
    BATCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))

# The modality arms. Order is the report order.
ARMS = {
    "s2":   ["sentinel2_l2a"],
    "s1":   ["sentinel1"],
    "s2s1": ["sentinel2_l2a", "sentinel1"],
}


def make_dummy(batch: int, tile: int, modalities: list[str], device) -> MaskedOlmoEarthSample:
    """A batch of PASTIS-shaped random samples: (B, H, W, T, C) per modality.

    Values are standard-normal because the real loader hands the encoder NORMALIZED data
    (PASTISRDataset(norm_stats_from_pretrained=True)); the encoder's cost is
    data-independent anyway, but keeping the scale realistic avoids inf/nan in autocast
    that could change kernel timing.

    timestamps is (B, T, 3) = [day, month0, year], driving the seasonal positional
    encoding -- we walk 12 distinct months so the time encoding is exercised the way a
    real monthly PASTIS series exercises it.
    """
    months = torch.tensor([[15, m, 2019] for m in range(NUM_TIMESTEPS)], dtype=torch.long)
    kw = {
        "timestamps": months.unsqueeze(0).repeat(batch, 1, 1),
        "latlon": torch.tensor([[43.5, 1.5]], dtype=torch.float32).repeat(batch, 1),
    }
    for m in modalities:
        kw[m] = torch.randn(batch, tile, tile, NUM_TIMESTEPS, BANDS[m], dtype=torch.float32)

    # from_olmoearthsample builds the *_mask fields (all-visible) for us, but it is written
    # for a SINGLE unbatched sample. Build one and re-batch by broadcasting the masks.
    single = OlmoEarthSample(**{k: (v[0] if torch.is_tensor(v) else v) for k, v in kw.items()})
    masked = MaskedOlmoEarthSample.from_olmoearthsample(single)
    repl = {}
    for field in masked._fields:
        val = getattr(masked, field)
        if val is None:
            continue
        if field in kw:
            repl[field] = kw[field].to(device)
        else:                                   # a derived mask: add the batch dim
            repl[field] = val.unsqueeze(0).repeat(
                *( [batch] + [1] * val.dim() )).to(device)
    return masked._replace(**repl)


@torch.no_grad()
def run_step(encoder, sample, patch_size: int, tile: int, n_tiles: int, device) -> None:
    """One full extraction step for a batch: encode every spatial sub-tile and pool per
    timestep, exactly as exp/pastis/extract_features.encode_batch does.

    We re-encode the SAME dummy tile n_tiles times rather than slicing n_tiles distinct
    tiles out of a 64x64 sample: the encoder's cost depends only on the tile's shape, so
    this is the same compute, and it keeps the dummy allocation small for tiny tiles.
    """
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in range(n_tiles):
            tam = encoder(sample, patch_size=patch_size, input_res=BASE_GSD,
                          fast_pass=True)["tokens_and_masks"]
            per_t = [pool_per_timestep(tam, t, POOLING_TYPE) for t in range(NUM_TIMESTEPS)]
            torch.stack(per_t, dim=-2)


def _fits(encoder, batch: int, patch_size: int, tile: int, n_tiles: int, device) -> bool:
    """True if one step at this batch size runs without exhausting GPU memory."""
    try:
        sample = make_dummy(batch, tile, _fits.modalities, device)
        run_step(encoder, sample, patch_size, tile, n_tiles, device)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    except RuntimeError as e:                    # some paths raise plain RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        return False
    finally:
        # Free the failed step's fragments before the next probe, or a later (smaller)
        # batch can spuriously OOM on leftover cached blocks.
        sample = None
        torch.cuda.empty_cache()


def find_max_batch(encoder, patch_size: int, tile: int, n_tiles: int, modalities: list[str],
                   device, cap: int, safety: float, image_size: int = IMAGE_SIZE) -> int:
    """Largest batch that fits, minus a safety margin.

    Doubling probe to bracket the limit, then binary search the (lo, hi) gap. Cheaper than
    a linear scan and, more importantly, never leaves us reporting a number that only fit
    because the allocator happened to have the right cached block.
    """
    _fits.modalities = modalities
    # Seed the probe near the expected limit rather than doubling up from 1. Activation
    # memory scales with (batch x tokens_per_tile), so a fixed token budget is a far better
    # first guess than a constant: without this, a config whose limit is ~64 still pays for
    # probes at 1,2,4,...,64 and a config whose limit is ~1500 pays for ten doublings, and
    # at multi-second steps that dominates the whole benchmark.
    tok = (tile // patch_size) ** 2 * NUM_TIMESTEPS
    # Seed at a power of two near the expected limit, then walk by powers of two only.
    # Reporting bs=1827 (a binary search resolved to +-1, then scaled by 0.9) implies a
    # precision the measurement does not have -- rerun on a differently-fragmented card and
    # it lands elsewhere. A 2^n batch is a knob we CHOSE, is stable across runs, and avoids
    # ragged tensor-core tile quantization. Costs up to half the headroom, so we also record
    # the raw largest-fitting 2^n as batch_size_max_fit.
    seed = 1 << max(0, (TUNER_TOKEN_BUDGET // max(tok, 1)).bit_length() - 1)
    seed = max(1, min(cap, seed))
    lo = 0
    if _fits(encoder, seed, patch_size, tile, n_tiles, device):
        lo = seed
        nxt = seed * 2
        while nxt <= cap and _fits(encoder, nxt, patch_size, tile, n_tiles, device):
            lo = nxt
            nxt *= 2
    else:
        probe = seed // 2
        while probe >= 1 and not _fits(encoder, probe, patch_size, tile, n_tiles, device):
            probe //= 2
        lo = probe
    return lo                                    # a power of two, or 0 if nothing fit


def bench_config(encoder, arm: str, patch_size: int, tile: int, device, args,
                 gpu_name: str, cache: dict, flops_per_sample: float | None = None,
                 retry: bool = False) -> dict | None:
    """Time one (arm, patch_size, tile_size) config and return a result row.

    Throughput is averaged over args.iters INDEPENDENT batches rather than re-timing one
    resident batch: a single batch measures the steady-state kernel path with everything
    already hot, which flatters the number. Re-allocating each batch keeps allocator churn
    and the first-touch cost in the average, which is what a real extraction pays."""
    modalities = ARMS[arm]
    img = args.image_size
    n_tiles = (img // tile) ** 2                  # sub-tiles per sample
    grid = img // patch_size                      # full-sample token grid side
    tok_per_tile = (tile // patch_size) ** 2 * NUM_TIMESTEPS

    key = _cache_key(gpu_name, args.model_size, arm, patch_size, tile, img)
    if args.batch_size > 0:
        batch = args.batch_size
    elif key in cache and not args.refind_batch:
        batch = cache[key]
        print(f"  [{arm:4s} ps{patch_size} tile{tile:3d}] batch {batch} from cache")
    else:
        batch = find_max_batch(encoder, patch_size, tile, n_tiles, modalities, device,
                               args.max_batch, args.bs_safety, img)
        cache[key] = batch
        save_batch_cache(cache)                   # persist immediately; probes are expensive
    if batch == 0:
        print(f"  [{arm} ps{patch_size} tile{tile}] OOM at batch 1 -- skipped", flush=True)
        return None

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    try:
        # Warmup is INSIDE the guard: a stale cached batch size blows up here, before the
        # timing loop, and an unguarded warmup takes the whole job down with it.
        for _ in range(args.warmup):              # warm up kernels/autotune before timing
            sample = make_dummy(batch, tile, modalities, device)
            run_step(encoder, sample, patch_size, tile, n_tiles, device)
            del sample
        torch.cuda.synchronize()

        for _ in range(args.iters):
            sample = make_dummy(batch, tile, modalities, device)  # fresh batch each iteration
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_step(encoder, sample, patch_size, tile, n_tiles, device)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            del sample
    except torch.cuda.OutOfMemoryError:
        # A cached batch size stopped fitting. Rather than skip the config, DROP the stale
        # entry and re-probe once -- the cache is a speed optimisation, not ground truth, and
        # a wrong entry should self-heal instead of silently costing a data point.
        print(f"  [{arm} ps{patch_size} tile{tile}] OOM at cached batch {batch}; "
              f"dropping cache entry and re-probing", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        cache.pop(key, None)
        save_batch_cache(cache)
        if args.batch_size > 0 or retry:
            return None                            # explicit batch, or already retried once
        return bench_config(encoder, arm, patch_size, tile, device, args, gpu_name, cache,
                            flops_per_sample, retry=True)
    if not times:
        return None

    med = statistics.median(times)
    mean = statistics.mean(times)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    row = {
        "arm": arm,
        "modalities": "+".join(modalities),
        "image_size": img,
        "patch_size": patch_size,
        "tile_size": tile,
        "grid": grid,
        "tiles_per_sample": n_tiles,
        "tokens_per_tile": tok_per_tile,
        "batch_size": batch,
        "n_batches": args.iters,
        # headline throughput uses the MEAN over independent batches (see docstring)
        "samples_per_s": batch / mean,
        "ms_per_sample": 1000 * mean / batch,
        "samples_per_s_median": batch / med,
        "step_s_mean": mean,
        "step_s_median": med,
        "step_s_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "peak_mem_gb": peak_gb,
        # GMACs = GFLOPs/2, the convention most vision papers report.
        "gflops_per_sample": (flops_per_sample / 1e9) if flops_per_sample else "",
        "gmacs_per_sample": (flops_per_sample / 2e9) if flops_per_sample else "",
        "tflops_per_s": ((flops_per_sample * batch / mean) / 1e12) if flops_per_sample else "",
    }
    gm = f"{row['gmacs_per_sample']:9.1f} GMACs" if flops_per_sample else "        ? GMACs"
    print(f"  [{arm:4s} ps{patch_size} tile{tile:3d}] grid={grid:3d} bs={batch:5d} "
          f"{row['samples_per_s']:8.1f} samp/s  {row['ms_per_sample']:8.2f} ms/samp  "
          f"peak {peak_gb:5.2f} GB {gm}", flush=True)
    torch.cuda.empty_cache()
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma-separated subset of " + ",".join(ARMS))
    p.add_argument("--configs", default=None,
                   help="comma-separated patch_size:tile_size pairs; default is the full "
                        "ps{1,2,4,8,16} x tile{16,64,128} grid valid for --image_size")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE, choices=(64, 128),
                   help="PASTIS sample side. tile 128 requires 128.")
    p.add_argument("--no_flops", action="store_true",
                   help="skip the FLOP count (it adds one traced batch-1 step per config)")
    p.add_argument("--refind_batch", action="store_true",
                   help="ignore the cached batch sizes and re-probe")
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 (default) = auto-tune the largest batch that fits")
    p.add_argument("--max_batch", type=int, default=2048, help="cap for the auto-tuner")
    p.add_argument("--bs_safety", type=float, default=0.10,
                   help="back off this fraction from the max fitting batch (default 0.10)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out", default="results/bench/olmo_throughput.csv")
    args = p.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; allowed {list(ARMS)}")

    if args.configs is None:
        # Full grid, restricted to tiles that fit the sample and divide by the patch.
        args.configs = ",".join(
            f"{ps}:{ts}" for ts in (16, 64, 128) for ps in (1, 2, 4, 8, 16)
            if ts <= args.image_size and args.image_size % ts == 0 and ts % ps == 0)
    configs = []
    for item in args.configs.split(","):
        if not item:
            continue
        ps, ts = item.split(":")
        ps, ts = int(ps), int(ts)
        if args.image_size % ts or ts % ps:
            raise SystemExit(f"bad config {item}: need tile|{args.image_size} and patch|tile")
        configs.append((ps, ts))
    # Cheapest first: cost ~ (tokens per call)^2 x (tiles per sample). The expensive
    # configs (ps1) take minutes per config, so running them last means an interrupted
    # sweep still produced every affordable row instead of nothing.
    configs.sort(key=lambda c: ((c[1] // c[0]) ** 2 * NUM_TIMESTEPS) ** 2
                 * (args.image_size // c[1]) ** 2)

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- this benchmark is meaningless on CPU")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}  torch {torch.__version__}")
    print(f"PASTIS geometry: {args.image_size}x{args.image_size} @ {BASE_GSD} m, "
          f"T={NUM_TIMESTEPS}, bands {BANDS}")

    model = load_model_from_id(getattr(ModelID, MODEL_SIZE_TO_ID[args.model_size]),
                               load_weights=True)
    encoder = cast(nn.Module, model.encoder if hasattr(model, "encoder") else model)
    encoder = encoder.to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad = False

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush(rows):
        """Rewrite the CSV after every config. The expensive arms take minutes per row,
        so checkpointing means a killed or timed-out sweep still leaves usable results."""
        if not rows:
            return
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    batch_cache = load_batch_cache()

    # PASS 1: FLOPs for every (arm, config), on batch-1 dummies, before any large allocation
    # exists. Cheap, order-independent, and cached to disk so later runs skip it entirely.
    flop_cache = load_flop_cache()
    flops = {}
    if not args.no_flops:
        print("\n=== measuring FLOPs (batch 1, one pass) ===", flush=True)
        for arm in arms:
            for ps, ts in configs:
                f = measure_flops(encoder, arm, ps, ts, args.image_size, device,
                                  flop_cache, args.model_size)
                flops[(arm, ps, ts)] = f
                if f:
                    print(f"  [{arm:4s} ps{ps} tile{ts:3d}] {f / 2e9:10.1f} GMACs/sample",
                          flush=True)
        torch.cuda.empty_cache()

    # PASS 2: batch-size tuning + timing.
    rows = []
    for arm in arms:
        print(f"\n=== arm {arm} ({'+'.join(ARMS[arm])}) ===", flush=True)
        for ps, ts in configs:
            row = bench_config(encoder, arm, ps, ts, device, args, gpu_name, batch_cache,
                               flops.get((arm, ps, ts)))
            if row:
                row["gpu"] = gpu_name
                row["model_size"] = args.model_size
                rows.append(row)
                flush(rows)
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
