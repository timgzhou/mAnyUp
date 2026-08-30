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

### Cloud-aware guidance (feasibility probe)

`exp/upsamplers/viz_cloud_masked_upsample.py` asks whether OmniCloudMask can make the
guided upsamplers robust to cloud. UPA/UPMA weight each contributing LR pixel by
*guide-image* similarity, so where the guide is cloud the kernel keys on cloud-top
reflectance and imprints cloud edges into the feature map. The probe neutralizes the guide
under cloud and drops cloudy pixels from the self-supervised fitting loss, then plots
plain vs cloud-masked side by side with a difference panel.

```shell
python -u -m exp.upsamplers.viz_cloud_masked_upsample \
    --tiles EMSR650-1-26,EMSR650-1-18,EMSR650-1-195 \
    --roots data/GEOID-Flood-aux/geoid-flood-extracted,data/GEOID-Flood-full/geoid-flood
```

Result on GEOID: the change is **3-4x larger inside the cloud than outside** (UPMA
localizes better than UPA in every tile tested), so the intervention is well-behaved.
But GEOID's S2 is a cloud-FILTERED composite — median `cloud_cover` is 0.000 across all
502k chips and only 11.7% have any cloud — so this is a fix for a small subpopulation
here, not a general win. It matters far more where the optical input is a single
acquisition. Pick cloudy tiles from `data_tiles_s256_st128.csv` (`cloud_cover` 0.15-0.75);
`s2l2a`/`cloudmask` are NOT in the main download and come from `data/GEOID-Flood-aux/`.

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
python -u -m exp.viz.visualize_geoid_flood         # GEOID-Flood dataset samples
python -u -m exp.viz.visualize_sen12flood
python -u -m exp.viz.visualize_s1s2_landslide
python -u -m exp.pastis.visualize                  # PASTIS predictions
python -u -m exp.utae.visualize                    # UTAE predictions
```

Note: `exp/urbansarfloods/viz_features.py` loads a model from a hardcoded absolute path
(`/scratch/timz/OlmoEarth-v1-Base`) at import time. That path does not currently exist;
point it at a real checkpoint before running.

---

## GEOID-Flood — OlmoEarth patch-size study

Code: `exp/geoidflood/` · Launcher: `scripts/slurm/geoidflood/sweep.sh` · Results: `results/geoidflood/`

[GEOID-Flood](https://huggingface.co/datasets/links-ads/geoid-flood) (arXiv:2608.02315) is
a flood-segmentation benchmark from 219 Copernicus EMS Rapid Mapping activations across 65
countries. **It is a different dataset from ImpactMesh-Flood** (`ibm-esa-geospatial`, in
`data/ImpactMesh-Flood/`) — different publisher, tiling, and label scheme. Easy to confuse
because both are CEMS-derived multimodal flood sets.

Layout: 1024x1024 tiles @10 m; `s1grd`/`s1rtc` pre+post (VV,VH), `s2l2a` **pre-only**,
`dem`, plus `label` / `floodmask` / `permwater` / `validity` / `cloudmask`.
Labels: `0` background, `1` permanent water, `2` flood, **`255` = ignore (outside the
mapped AOI, NOT background)** — prep remaps it to `-1` and the probe passes
`ignore_index=-1`.

### The study

Three configs, each giving the **same 16x16 token grid** so the probe has an identical
token count and parameter count in every arm. Only the ground area per token and the chip
footprint change:

| Config | patch | tile | m / token | chip footprint |
|---|---|---|---|---|
| `ps8tile128` | 8 | 128 | 80 m | 1.28 km |
| `ps4tile64`  | 4 | 64  | 40 m | 640 m |
| `ps1tile16`  | 1 | 16  | 10 m | 160 m |

The question: at a fixed token budget, is it better to look finely at a small area or
coarsely at a large one?

### Data (once, ~205 GB)

`s1grd` + `label` + `validity` only — the full 584 GB (with `s1rtc`, `s2l2a`) does not fit,
and S1-only keeps this comparable to the UrbanSARFloods run.

```shell
bash scripts/download_geoid_flood.sh            # -> data/GEOID-Flood-full/
```

### Run

```shell
sbatch scripts/slurm/geoidflood/sweep.sh        # all three configs x {concat, diff}
```

Or one arm at a time:

```shell
python -u -m exp.geoidflood.prep_tiles       --splits train,val --tile_size 128
python -u -m exp.geoidflood.extract_features --splits train,val --tile_size 128 --patch_size 8
python -u -m exp.geoidflood.lp --features geoid_base_s1_ps8_res10_t128 --weighted_ce
```

Notes:
- `prep_tiles` converts `s1grd` from linear sigma0 to **dB** (OlmoEarth's S1 pretraining
  units) and keeps only chips with >=1 flood pixel and >=50% mapped pixels. The flood
  filter makes val numbers flood-conditional, but applies identically to all three arms.
- The split is per-TILE and lives only in `data_tiles_s256_st128.csv` (51 of 210 events
  span multiple splits), so `prep_tiles` reads it from there — the download unpacks every
  split into one `<event>/` namespace.
- Headline metric is mIoU over {background, flood}; permanent water is reported separately
  because it is dark in both timesteps and so is not a change signal.

---

## The `src/` benchmark framework

`src/` is a separate, config-driven benchmark (datasets × models × tasks) with its own
entrypoints, unrelated to the `exp/` research code above. See
[ARCHITECTURE.md](ARCHITECTURE.md) and [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md).

```shell
python train.py --config configs/flood/urbansar/seg_unet.yaml
python test.py  --exp_path <run dir>
```
