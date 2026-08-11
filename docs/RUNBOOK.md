# Runbook

Single source of truth for experiment commands. Merged from the old `runs.txt` and
`urbansarfloods_RUN.txt`.

All commands run **from the repo root** and use module form (`python -m exp.pkg.module`),
which is what makes the `exp.*` imports resolve. Activate the environment first:

```shell
source env_setup/env_olmo.sh     # OlmoEarth work (PASTIS, UrbanSARFloods, upsamplers)
source env_setup/env.sh          # torchgeo / UTAE work
```

## Config model

- `configs/defaults.yaml` — OlmoEarth tuning defaults (epochs, lr, batch_size, ...)
- `configs/utae_defaults.yaml` — UTAE tuning defaults
- Architecture fields are **required** via `--set` (no defaults) so you can't accidentally
  launch the wrong model. Tuning knobs default from the `*_defaults.yaml`; override any
  with `--set key=value`.

Run interactively: `python -u -m <module> --set <fields...>`
Run as a batch job: `sbatch <script>.sh --set <fields...>` (forwards all args; emails start/done)

| Model | Required fields |
|---|---|
| OlmoEarth | `model_size` (nano\|tiny\|base\|large), `modalities` (sentinel2_l2a\|sentinel1\|both), `head_mode` (lp\|anyup\|anyup_t2\|anyup_t1), `freeze_backbone` (true\|false) |
| UTAE | `modalities` (S2\|S1A\|S1D, comma-joined), `fusion` (early\|late, if multimodal) |

---

## PASTIS — OlmoEarth

Code: `exp/pastis/` · Launchers: `scripts/slurm/pastis/` · Results: `results/pastis/`

### Data prep (once)

```shell
sbatch scripts/slurm/pastis/prepare_data.sh
# or: python -u -m exp.pastis.prepare_data
```

### Fine-tuned (freeze-then-unfreeze warmup)

```shell
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=lp       freeze_backbone=false
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup    freeze_backbone=false
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t2 freeze_backbone=false
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t1 freeze_backbone=false
```

### Frozen backbone + frozen AnyUp (only head/probe trains)

```shell
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=lp       freeze_backbone=true
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup    freeze_backbone=true
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t2 freeze_backbone=true
sbatch scripts/slurm/pastis/finetune_olmoearth.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t1 freeze_backbone=true
```

### Quick smoke (tiny, 2 epochs)

```shell
python -u -m exp.pastis.finetune_olmoearth --set model_size=tiny modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t2 freeze_backbone=false epochs=2 batch_size=4
```

### Cached-feature linear probing

```shell
sbatch scripts/slurm/pastis/extract_features.sh          # -> features/
sbatch scripts/slurm/pastis/lp_heads.sh                  # sweeps head modes
```

---

## UrbanSARFloods

Code: `exp/urbansarfloods/` · Launcher: `scripts/slurm/urbansarfloods/sweep.sh` · Results: `results/urbansarfloods/`

Run inside the OlmoEarth venv (`source env_setup/env_olmo.sh`). HF rate-limits anonymous
downloads — set a token first: `hf auth login` (or `export HF_TOKEN=...`).

```shell
# 1. Download train/val tar (~38GB) + inspect. Skip the 414GB raw SLC.
HF_HUB_DISABLE_XET=1 hf download S1Floodbenchmark/UrbanSARFloods_v1 urban_sar_floods.tar.gz \
    --repo-type dataset --local-dir data

# 2. Extract (-> data/urban_sar_floods/ with 01_NF 02_FO 03_FU + Train/Valid_dataset.txt)
tar xzf data/urban_sar_floods.tar.gz -C data
# (the tarball can be deleted once extracted; it is ~38GB)

# 3. Tile 512 -> tile_size sub-tiles, flood-only, drop NaN-nodata tiles
#    (-> data/urbansarfloods_tiles_t64/). --tile_size must divide 512.
python -u -m exp.urbansarfloods.prep_tiles --splits train,valid --tile_size 64

# 4. Extract frozen OlmoEarth features (patch_size=4, input_res=20, keeps T=2 pre/post).
#    --tile_size must match step 3; feature folder -> features/usf_base_s1_ps4_res20_t64/
python -u -m exp.urbansarfloods.extract_features --splits train,valid --tile_size 64
python -u -m exp.urbansarfloods.extract_features --splits train,valid --tile_size 32 --patch_size 2

# 5. Linear-probe: concat pre+post features -> per-pixel head; reports val(=test) mIoU(NF,FO)
python -u -m exp.urbansarfloods.lp --features usf_base_s1_ps4_res20_t64 --weighted_ce
python -u -m exp.urbansarfloods.lp --features usf_base_s1_ps2_res20_t32 --weighted_ce
```

Full sweep over tile sizes / patch sizes / heads:

```shell
sbatch scripts/slurm/urbansarfloods/sweep.sh
```

---

## Upsamplers (UPA / AnyUp)

Code: `exp/upsamplers/` · Launchers: `scripts/slurm/upsamplers/` · Results: `results/upsamplers/`

`exp/upsamplers/upa_anyup.py` is the shared library (`UPA`, `UPMA`, `time_pool`,
`TIME_POOLS`, `percentile_stretch`) imported by the eval scripts and by
`exp/pastis/finetune_olmoearth.py`.

```shell
sbatch scripts/slurm/upsamplers/train_manyup.sh            # train the many-up upsampler
sbatch scripts/slurm/upsamplers/upsampler_ps16.sh         # coarser-grid eval (ps16 vs ps4)

python -u -m exp.upsamplers.eval_pa2pa --help              # PASTIS -> PASTIS eval
python -u -m exp.upsamplers.eval_usf   --help              # UrbanSARFloods eval
```

---

## UTAE baseline

Code: `exp/utae/` · Vendored model: `reference/utae_vendored/utae/` · Results: `results/utae/`

Uses the torchgeo venv (`source env_setup/env.sh`), not the OlmoEarth one.

```shell
sbatch scripts/slurm/utae/run_pastis.sh --set modalities=S2
sbatch scripts/slurm/utae/run_pastis.sh --set modalities=S1A
sbatch scripts/slurm/utae/run_pastis.sh --set modalities=S2,S1A fusion=early
sbatch scripts/slurm/utae/run_pastis.sh --set modalities=S2,S1A fusion=late

# smoke:
python -u -m exp.utae.run_pastis --set modalities=S2 epochs=2
```

---

## Visualization

Code: `exp/viz/` — dataset samplers and results plots, plus per-experiment
`visualize.py` under `exp/pastis/` and `exp/utae/`.

```shell
python -u -m exp.viz.visualize_lp_results          # LP result curves from results/*.csv
python -u -m exp.viz.plot_urbansarfloods_csv       # UrbanSARFloods sweep plot
python -u -m exp.viz.visualize_impactmesh          # ImpactMesh dataset samples
python -u -m exp.viz.visualize_sen12flood
python -u -m exp.viz.visualize_s1s2_landslide
python -u -m exp.pastis.visualize                  # PASTIS predictions
python -u -m exp.utae.visualize                    # UTAE predictions
```

Note: `exp/urbansarfloods/viz_features.py` loads a model from a hardcoded absolute path
(`/scratch/timz/OlmoEarth-v1-Base`) at import time. That path does not currently exist;
point it at a real checkpoint before running.

---

## The `src/` benchmark framework

`src/` is a separate, config-driven benchmark (datasets × models × tasks) with its own
entrypoints, unrelated to the `exp/` research code above. See
[ARCHITECTURE.md](ARCHITECTURE.md) and [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md).

```shell
python train.py --config configs/flood/urbansar/seg_unet.yaml
python test.py  --exp_path <run dir>
```
