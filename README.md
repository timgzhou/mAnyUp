# mAnyUp

Feature upsampling for remote-sensing foundation models. OlmoEarth features are cheap at
coarse patch sizes and expensive at fine ones; mAnyUp and timAnyUp learn to recover the
fine features from coarse ones (plus cheap guidance), and are evaluated by linear probing
on three datasets:

- **PASTIS**: crop-type segmentation from S2/S1 time series (UTAE is the baseline)
- **GEOID-Flood**: flood segmentation from S1 pre/post pairs
- **Biomassters**: forest biomass regression (not yet ported)

## Layout

```
manyup/                   the upsamplers: mAnyUp, timAnyUp, loss
exp/                      experiments, run as `python -m exp.<pkg>.<module>`
  common/                 run config, paths (data / feature caches), OlmoEarth import shims
  pastis/                 prepare -> extract features -> linear probe / fine-tune -> visualize
  geoidflood/             prep tiles -> extract features -> linear probe -> visualize
  upsamplers/             train + evaluate mAnyUp, timAnyUp, UPA/UPMA
  utae/                   UTAE baseline on PASTIS (+ our early/late fusion)
  bench/                  throughput / FLOPs benchmarks
  viz/                    results plots
third_party/              vendored upstream code: anyup (wimmerth/anyup), utae (utae-paps)
scripts/slurm/            run.sh (generic launcher) + multi-step pipelines
tests/                    unit tests for the upsampler building blocks
results/<dataset>/        CSVs, tracked; figures, untracked
dataset_visualization/    dataset sample figures (untracked)
docs/RUNBOOK.md           the command behind every experiment

data/ checkpoints/ logs/  gitignored; feature caches live in project space (exp/common/paths.py)
```

## Setup

```shell
source env_setup/env_olmo.sh
```

Builds the one venv everything uses: torch 2.9.1 + `olmoearth-pretrain` 0.1.2, on
node-local disk (`$SLURM_TMPDIR`, else `$TMPDIR`), never inside the repo. Everything runs
**from the repo root**, interactively or through the generic launcher:

```shell
python -u -m exp.pastis.extract_features --patch_size 4 --tile_size 64
sbatch -J extract --time=9:00:00 scripts/slurm/run.sh exp.pastis.extract_features --patch_size 4 --tile_size 64
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for every experiment.

## Tests

```shell
python tests/test_timanyup.py
python tests/test_cosmse_map.py
```
