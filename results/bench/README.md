# OlmoEarth ps x tile: extraction speed and downstream accuracy

Two independent measurements over the same (patch_size, tile_size) grid:

- **Speed** — `exp/bench/olmo_throughput.py`, DUMMY inputs shaped like PASTIS, on an L40S.
  Batch size is auto-tuned to the largest power of two that fits (cached in
  `max_batch_cache.json`), and throughput is averaged over 5 independent batches.
  All three modality arms (s2 / s1 / s2s1) are in `olmoearth_ps-tile_speed.csv`.
- **Accuracy** — `exp/pastis/lp_cached_features.py` `lp_pa2px` linear probe on real cached
  PASTIS features, appended to `../pastis/lp_olmoearth_pastis.csv`.

![miou vs throughput](miou_vs_throughput.png)

## S2 arm: speed and accuracy together

| patch | tile | img | grid | batch | samples/s | Mpix/s | mIoU (lp_pa2px) |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 64 | 64 | 128 | 1.8 | 0.007 | 0.520 |
| 2 | 16 | 64 | 32 | 512 | 10.8 | 0.044 | 0.510 |
| 4 | 16 | 64 | 16 | 2048 | 49.4 | 0.202 | 0.490 |
| 8 | 16 | 64 | 8 | 2048 | 195.8 | 0.802 | 0.440 |
| 16 | 16 | 64 | 4 | 2048 | 740.6 | 3.033 | 0.370 |
| 1 | 64 | 64 | 64 | 8 | 0.2 | 0.001 | 0.510 |
| 2 | 64 | 64 | 32 | 32 | 3.1 | 0.013 | 0.510 |
| 4 | 64 | 64 | 16 | 128 | 28.6 | 0.117 | 0.500 |
| 8 | 64 | 64 | 8 | 512 | 173.6 | 0.711 | 0.460 |
| 16 | 64 | 64 | 4 | 1024 | 750.2 | 3.073 | 0.370 |
| 1 | 128 | 128 | 128 | 2 | 0.0 | 0.000 | — |
| 2 | 128 | 128 | 64 | 8 | 0.2 | 0.004 | 0.420 |
| 4 | 128 | 128 | 32 | 32 | 3.0 | 0.050 | 0.430 |
| 8 | 128 | 128 | 16 | 128 | 28.5 | 0.467 | 0.420 |
| 16 | 128 | 128 | 8 | 256 | 169.7 | 2.781 | 0.360 |

**Reading the throughput columns.** `samples/s` is not comparable across image sizes: the 64
prep quarters each raw 128x128 PASTIS patch, so an img128 "sample" covers 4x the ground of an
img64 one. `Mpix/s` is the common unit, and at fixed tile_size it matches across image sizes
to within noise -- image_size costs nothing per pixel, it only decides which tile sizes are
reachable (tile128 needs img128).

## What the numbers say

- **Accuracy is driven by patch size, not tiling.** mIoU climbs 0.37 -> 0.52 as patch goes
  16 -> 1, and at every patch size tile16 and tile64 land within 0.02 of each other
  (0.52/0.51, 0.51/0.51, 0.49/0.50, 0.44/0.46, 0.37/0.37).
- **So the cross-tile attention that tiling gives up is nearly free on PASTIS** -- while
  buying a 1.2-2.6x speedup. Small tiles dominate: ps2:tile16 matches ps2:tile64's 0.51 at
  ~3.4x the throughput.
- **tile128 is worse, not better,** despite seeing 1.28 km of context (0.42-0.43 vs 0.49-0.51
  for the same patch sizes at tile16/64). More context per attention call did not help here.
- **The knee is around ps2-ps4.** ps4:tile16 gives 0.49 at 0.20 Mpix/s; ps1:tile16 buys
  +0.03 mIoU for a 29x throughput cost.
- **Cost is superlinear in the token grid** -- log-log slope of throughput vs sequence length
  is -1.16 (S1), -1.32 (S2), -1.38 (S2+S1); steeper than -1 means attention dominates.
- **S1 is ~3x cheaper than S2** (2 bands vs 13); adding S1 to S2 costs 1.3-1.7x, not 2x.

## Known gap

`ps1:tile128` has speed numbers (dummy input needs no dataset) but no mIoU: a 128x128 token
grid is 196,608 tokens in ONE attention call. It OOMs above batch 1, and at batch 1 runs
~75 s/sample, so its 2433 samples need ~51 GPU-hours and ~685 GB. It is being finished by a
self-chaining job (`scripts/slurm/pastis/pstile_resume_chain.sh`); the extractor skips
batches already on disk, so each link resumes rather than restarts.

## Reproduce

    # speed (one job per arm x image_size)
    sbatch --export=ALL,ARM=s2,IMAGE_SIZE=64  scripts/slurm/bench/throughput.sh
    sbatch --export=ALL,ARM=s2,IMAGE_SIZE=128 scripts/slurm/bench/throughput.sh
    python -m exp.bench.merge_speed_csvs

    # extract + LP for one arm of the grid (features kept only for ps1/ps8)
    sbatch --export=ALL,PATCH_SIZE=4,TILE_SIZE=16 scripts/slurm/pastis/pstile_lp.sh

    python -m exp.viz.plot_miou_vs_speed
