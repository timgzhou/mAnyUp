# Runbook

The command behind every experiment. Everything runs **from the repo root** in module form
(`python -m exp.<pkg>.<module>`), which is what makes the `exp.*` imports resolve.

- **Interactively** (inside a GPU `salloc`): `source env_setup/env_olmo.sh`, then
  `python -u -m <module> [args]`.
- **As a batch job**: `sbatch [sbatch options] scripts/slurm/run.sh <module> [args]`.
  `run.sh` builds the env, forwards every argument, logs to `logs/<job-name>_<jobid>.out`
  and emails at start and end (`EMAIL=` to skip). Its default resources are 1 L40S, 8 CPUs,
  64 GB and 3 h. Override them on the `sbatch` line and name the job with `-J`.

Only four jobs have their own launcher, because they chain several steps:
`scripts/slurm/geoidflood/sweep.sh`, `scripts/slurm/pastis/pstile_arm.sh`,
`scripts/slurm/pastis/extract_chain.sh` and `scripts/slurm/upsamplers/lp_timanyup.sh`.
The one-off sweep launchers these replace are kept at git tag `pre-cleanup`.

Paths: data under `data/`, results under `results/<dataset>/`, feature caches under `exp.common.paths.FEATURES` (project
space, override with `$MANYUP_FEATURES`), checkpoints under `checkpoints/`, CSVs under
`results/`.

---

## PASTIS: OlmoEarth

Code: `exp/pastis/` · Results: `results/pastis/`

### Data prep (once)

Needs about 256 GB RAM and no GPU. `PASTISRProcessor` holds every fold in memory before
saving.

```shell
sbatch -J prep --gres=none --mem=256G scripts/slurm/run.sh exp.pastis.prepare_data
```

### Fine-tuning

`exp.common.config.Config` requires the architecture fields (`model_size`, `modalities`,
`head_mode`, `freeze_backbone`), so a run can't silently fall back to a default model.
Tuning knobs default on the dataclass. Override any of them with `--set key=value`.

```shell
sbatch -J ft --time=9:00:00 scripts/slurm/run.sh exp.pastis.finetune_olmoearth \
    --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=lp freeze_backbone=false
# head_mode: lp | lp_tcat | anyup | anyup_t2 | anyup_t1;  freeze_backbone=true trains the head only

# smoke
python -u -m exp.pastis.finetune_olmoearth --set model_size=tiny modalities=sentinel2_l2a \
    head_mode=lp freeze_backbone=true epochs=2 batch_size=4
```

### Cached features + linear probe

Feature caches are named `oe_<size>_<mods>_ps<P>_tile<T>[_img128][_single][_ft<tag>]`.

```shell
# extract (tile 128 needs --image_size 128 --data_splits data/pastis128_olmoearth)
sbatch -J extract --time=9:00:00 scripts/slurm/run.sh exp.pastis.extract_features \
    --modalities sentinel2_l2a --patch_size 4 --tile_size 64
#   --temporal_mode single   T independent T=1 calls (no cross-timestep attention)
#   --init_ckpt <ckpt.pt>    features from a FINE-TUNED encoder (separate _ft cache)

# probe (head modes: lp_pa2px lp_pa2pa_bu lp_pa2px_ens lp_bu_px2px anyup* manyup* timanyup*)
sbatch -J lp --mem=96G scripts/slurm/run.sh exp.pastis.lp_cached_features \
    --features oe_base_s2_ps4_tile64 --head_mode lp_pa2px
#   --knn                    probe-free KNN on the head's features instead of a linear head
#   --id_half first|second   fit on half the train ids (leakage controls)
#   --max_ram_gb 0           stream from disk (the 684 GB ps1 cache cannot be preloaded)
```

One arm of the patch-size × tile-size study runs extract, then probe, then deletes the cache
unless it is kept (ps1 and ps8 by default). A cache too slow for one job's walltime uses a
self-resubmitting chain:

```shell
sbatch --export=ALL,PATCH_SIZE=8,TILE_SIZE=16 scripts/slurm/pastis/pstile_arm.sh
sbatch --export=ALL,PATCH_SIZE=1,TILE_SIZE=128,IMAGE_SIZE=128 scripts/slurm/pastis/extract_chain.sh
```

### Visualization

```shell
python -u -m exp.pastis.visualize        # fine-tuned predictions -> results/pastis/predictions/
python -u -m exp.pastis.viz_ps_sweep     # feature maps across the ps sweep
python -u -m exp.pastis.make_rs_video --patches 20013 --gif   # -> dataset_visualization/pastis/
```

---

## Upsamplers: mAnyUp / timAnyUp / UPA

Code: `manyup/` (models, loss) and `exp/upsamplers/` (training, evals) · Results:
`results/pastis/` (`upsampler_pa2pa.csv`, rows in `lp_olmoearth_pastis.csv`) and
`results/geoidflood/cloud_upsample/`

```shell
# mAnyUp: LR feature cache -> HR feature cache  (ckpts -> checkpoints/manyup/<lr>__to__<hr>/)
sbatch -J mu --time=12:00:00 scripts/slurm/run.sh exp.upsamplers.train_manyup \
    --lr_cfg oe_base_s2_ps16_tile64 --hr_cfg oe_base_s2_ps4_tile64 --stage_to_tmpdir
#   shared-probe-eligible variant: --arch manyup --transform_depth 0 --window_ratio 1.0 --down_reg 0 --no-proj_head

# probe the upsampled map
sbatch -J lp_mu scripts/slurm/run.sh exp.pastis.lp_cached_features \
    --features oe_base_s2_ps16_tile64 --manyup_ckpt <ckpt.pth> --manyup_native_out

# timAnyUp: LR/high-context + HR/low-context -> HR/high-context  (k = lookup budget)
sbatch -J ta --time=12:00:00 scripts/slurm/run.sh exp.upsamplers.train_timanyup \
    --lrhc_cfg oe_base_s2_ps16_tile64 --hrlc_cfg oe_base_s2_ps4_tile4_single \
    --hrhc_cfg oe_base_s2_ps4_tile64 --k 512 --batch_size 4 --epochs 20 --stage_to_tmpdir

# probe timAnyUp checkpoints (both probe modes); one job per checkpoint
sbatch --export=ALL,CKPTS=<ckpt.pth> scripts/slurm/upsamplers/lp_timanyup.sh
#   k sweep: EXTRA="--timanyup_k <k>";  FFT caches: run lp_cached_features with
#   --hrlc_features <fft cache> --timanyup_train_transform

# UPA / UPMA / AnyUp comparison on frozen features (--retrain whenever the grid changes)
sbatch -J pa2pa --time=4:00:00 scripts/slurm/run.sh exp.upsamplers.eval_pa2pa \
    --features oe_base_s2_ps16_tile64 --head_ckpt checkpoints/pa2pa_head_ps16.pt --retrain
```

### Cloud-aware guidance (feasibility probe)

`exp/upsamplers/viz_cloud_masked_upsample.py` asks whether OmniCloudMask can make the
guided upsamplers robust to cloud. UPA/UPMA weight each LR pixel by *guide-image*
similarity, so under cloud the kernel keys on cloud-top reflectance. The probe neutralizes
the guide under cloud and drops cloudy pixels from the fitting loss.

```shell
python -u -m exp.upsamplers.viz_cloud_masked_upsample \
    --tiles EMSR650-1-26,EMSR650-1-18,EMSR650-1-195
```

On GEOID the change is 3–4× larger inside cloud than outside. GEOID's S2 is a
cloud-filtered composite, though: the median `cloud_cover` is 0 and only 11.7% of chips have
any cloud. So this fixes a small subpopulation here. `s2l2a`/`cloudmask` come from
`data/GEOID-Flood-aux/`.

---

## UTAE baseline (PASTIS)

Code: `exp/utae/` (runner, early/late fusion) · Upstream model: `third_party/utae/` ·
Results: `results/pastis/utae_pastis.csv`

```shell
sbatch -J utae --time=12:00:00 scripts/slurm/run.sh exp.utae.run_pastis --set modalities=S2
sbatch -J utae --time=12:00:00 scripts/slurm/run.sh exp.utae.run_pastis --set modalities=S2,S1A fusion=late
python -u -m exp.utae.run_pastis --set modalities=S2 epochs=2     # smoke
python -u -m exp.utae.visualize                                   # -> results/pastis/predictions/
```

---

## GEOID-Flood: OlmoEarth patch-size study

Code: `exp/geoidflood/` · Results: `results/geoidflood/`

[GEOID-Flood](https://huggingface.co/datasets/links-ads/geoid-flood) (arXiv:2608.02315) is
a flood-segmentation benchmark from 219 Copernicus EMS activations across 65 countries.
**It is not ImpactMesh-Flood** (`ibm-esa-geospatial`): the publisher, tiling and label
scheme all differ.

Layout: 1024×1024 tiles at 10 m. `s1grd`/`s1rtc` pre+post (VV, VH), `s2l2a` **pre-only**,
`dem`, plus `label`/`floodmask`/`permwater`/`validity`/`cloudmask`. Labels: `0` background,
`1` permanent water, `2` flood, **`255` = outside the mapped AOI, NOT background**. Prep
remaps it to `-1`, and the probe passes `ignore_index=-1`.

Data folders (all kept):

| Folder | What |
|---|---|
| `data/GEOID-Flood-full/` | the `s1grd`+`label`+`validity` download (~205 GB), source for `prep_tiles` |
| `data/geoidflood_tiles_t<T>/` | prepped sub-tiles the pipeline reads |
| `data/GEOID-Flood/` | the small sample split, used by `visualize_samples` |
| `data/GEOID-Flood-aux/` | `s2l2a`/`cloudmask` for the cloud-guidance probe |

### The study

Three configs give the **same 16×16 token grid**, so the probe sees the same token count
and parameter count in every arm. Only the ground per token and the chip footprint change:

| Config | patch | tile | m / token | chip footprint |
|---|---|---|---|---|
| `ps8tile128` | 8 | 128 | 80 m | 1.28 km |
| `ps4tile64`  | 4 | 64  | 40 m | 640 m |
| `ps1tile16`  | 1 | 16  | 10 m | 160 m |

```shell
bash scripts/download_geoid_flood.sh                  # once -> data/GEOID-Flood-full/
sbatch scripts/slurm/geoidflood/sweep.sh              # prep -> extract -> LP, 3 arms x {concat, diff}
SMOKE=1 bash scripts/slurm/geoidflood/sweep.sh        # minutes, on a slice; separate CSV

# one arm by hand
python -u -m exp.geoidflood.prep_tiles       --splits train,val --tile_size 128
python -u -m exp.geoidflood.extract_features --splits train,val --tile_size 128 --patch_size 8
python -u -m exp.geoidflood.lp --features geoid_base_s1_ps8_res10_t128 --weighted_ce

python -u -m exp.geoidflood.visualize_samples         # -> dataset_visualization/geoid_flood/
```

Notes:
- `prep_tiles` converts `s1grd` from linear sigma0 to **dB** (OlmoEarth's S1 pretraining
  units). It keeps chips with at least one flood pixel and at least 50% mapped pixels, so
  val numbers are flood-conditional in all three arms alike.
- The split is per-TILE and lives only in `data_tiles_s256_st128.csv`; 51 of 210 events
  span several splits.
- The headline metric is mIoU over {background, flood}. Permanent water is reported
  separately because it is dark at both timesteps, so it carries no change signal.

---

## Benchmarks and results plots

```shell
sbatch -J bench --time=3:00:00 scripts/slurm/run.sh exp.bench.olmo_throughput \
    --arms s2 --image_size 64 --out results/pastis/bench/olmoearth_ps-tile_speed_s2_img64.csv
python -u -m exp.bench.manyup_throughput
python -u -m exp.viz.plot_miou_vs_speed
python -u -m exp.viz.visualize_lp_results
```

The benchmark auto-tunes the batch size to fill GPU memory, so it needs an otherwise idle
GPU.
