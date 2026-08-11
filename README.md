# rs-change-detection

Remote-sensing change detection and segmentation research. The repo holds two independent
bodies of code:

- **`exp/`** — OlmoEarth research: feature extraction, linear probing, and upsampler
  (UPA / AnyUp) experiments on PASTIS and UrbanSARFloods. This is the active work.
- **`src/`** — a config-driven benchmark framework (~15 datasets × ~40 model
  architectures × seg/cd/scd tasks), driven by `configs/` and entered via `train.py` /
  `test.py`.

They share no code. Start from [docs/RUNBOOK.md](docs/RUNBOOK.md) for `exp/`, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for `src/`.

## Layout

```
exp/                    OlmoEarth research code (run as `python -m exp.<pkg>.<module>`)
  common/               shared config + OlmoEarth import bootstrap
  pastis/               PASTIS: prepare -> finetune -> extract features -> linear probe
  urbansarfloods/       UrbanSARFloods: tile -> extract features -> linear probe
  upsamplers/           UPA / AnyUp upsampler library, training, and evals
  utae/                 UTAE baseline runner + visualization
  viz/                  dataset sample plots and results plots
  notebooks/            exploratory notebooks

src/                    benchmark framework (core / datasets / models / tasks)
configs/                YAML configs for both: defaults for exp/, full matrix for src/
scripts/
  slurm/                sbatch launchers: pastis/ urbansarfloods/ upsamplers/ utae/
  data_prep/            dataset preparation for the src/ framework
env_setup/              cluster environment setup (env.sh, env_olmo.sh, env_login.sh)
docs/                   RUNBOOK, ARCHITECTURE, EXTENSION_GUIDE
results/                CSVs and figures, grouped by experiment line
reference/              kept for reference, not actively maintained (see below)

data/ features/ checkpoints/ logs/     gitignored artifacts
```

## Setup

```shell
source env_setup/env_olmo.sh    # OlmoEarth venv (PASTIS, UrbanSARFloods, upsamplers)
source env_setup/env.sh         # torchgeo venv (UTAE, src/ framework)
```

The two venvs are deliberately separate: `olmoearth-pretrain` pins `torch<2.8`, which
conflicts with the torchgeo stack.

Run everything **from the repo root**, in module form (so `exp.*` imports resolve) or via
an sbatch launcher:

```shell
python -u -m exp.pastis.finetune_olmoearth --set model_size=base modalities=sentinel1 head_mode=lp freeze_backbone=true
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel1 head_mode=lp freeze_backbone=true
```

Optionally `pip install -e .` to make the packages importable from any directory.
Note `zarr<3` is required — the ImpactMesh `.zarr.zip` archives are zarr v2 format.

## `reference/`

Kept because it may be useful, but not part of the active codebase:

| Path | What |
|---|---|
| `reference/utae_vendored/utae/` | Third-party UTAE implementation, unmodified. Used by the `exp/utae/` baseline. |
| `reference/dead_code/` | `pastis.py`, `olmoearth_utils.py`, `test_checkpoint.py` — an earlier torchgeo/Lightning pipeline that imports `olmoearth_pretrain_minimal`, a module no longer installed. Superseded by `exp/pastis/finetune_olmoearth.py`. |
| `reference/debug_probes/` | One-off probes (`debug_nan_tile.py`, `gpu_nan_probe.py`) from a since-resolved NaN investigation. |
