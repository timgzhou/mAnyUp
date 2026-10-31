# Baselines

MiSTR against published methods, compute-controlled, on PASTIS and BioMassters. Code:
`manyup/mixres/baselines/`, `exp/baselines/`; launcher: `scripts/slurm/baselines.sh`; figure:
`python -m exp.viz.baselines_tradeoff`. Upstream code is cloned into the gitignored `references/`:

| method | repository | commit |
|---|---|---|
| AnyUp (ICLR 2026) | https://github.com/wimmerth/anyup (vendored: `third_party/anyup`, identical) | 351807a |
| Upsample Anything (CVPR 2026) | https://github.com/seominseok0429/Upsample-Anything-A-Simple-and-Hard-to-Beat-Baseline-for-Feature-Upsampling | 3925146 |
| vid-TLDR (CVPR 2024) | https://github.com/mlvlab/vid-TLDR | 4a89723 |
| APT (ICLR 2026) | https://github.com/rccchoudhury/apt | 707337e |

Every baseline is `RunConfig.baseline` + a model subclass, so `exp.mixres.train` gives it the same
training, resume, test and MAC tracing as MiSTR. `python -u -m exp.baselines.check` compares each
against upstream's code and against the fixed checkpoint it must reduce to.

## Inputs and protocol

All start from the fine-tuned fixed checkpoints the figure already plots (`none_native_set`,
seasons and `_lrall` months, every patch size). Inputs are MiSTR's: the same composites, 64 px
tiles and loaders. Models, thresholds and learning rates are chosen on val only.

## What differs from upstream, and why

**AnyUp.** The multi-backbone checkpoint without NATTEN, as published. Each step upsamples alone
with its own RGB composite, to image resolution; a per-pixel linear head and the fixed model's
time attention follow. Guidance is percentile-stretched instead of min-max (PASTIS nodata frames
reach -10000), then ImageNet-standardized. Also run with the backbone fine-tuned through it.

**Upsample Anything.** The code's settings, where code and paper disagree: bicubic, Adam 1e-1,
51 updates. Range term summed over the 3 channels (`JBU(range_sum=True)`; the legacy UPA/UPMA
runs average). RGB in [0, 1] comes from the percentile stretch rather than 8-bit images. Fitted at
test time on the bilinear run's head. Output tokens are convex combinations for both, so the head
transfers. The fit is elementwise arithmetic the tracer cannot see: `upsample.fit_macs` adds it
(column `gmacs_analytic`, part of `gmacs`).

**vid-TLDR.** One sequence over the months in (t, row, col) order. `get_objective_score`, `merging`
and the size weighting are upstream's; `exp.baselines.check` compares them. r is a share of the
input tokens, since sequence lengths vary. A dense task needs every location back, and saliency
weights background tokens to 0, so unweighted cluster membership is tracked and final tokens are
copied to the grid. Training-free, as upstream.

**APT.** Two scales at the MiSTR pairs (coarse = season patch, fine = half), one partition per
image of the series, every token in one sequence. Grayscale entropy in 512 bins as upstream, on a
fixed per-dataset stretch standing in for 8-bit images. tau is the training quantile keeping a
`baseline.keep` share of patches coarse, since upstream's values are tuned per model and dataset.
Coarse tokens use FlexiViT at the coarse patch plus upstream's zero-init sub-patch path. Dense
output writes each token's footprint through the fixed readout. A month missing a modality has no
tokens, as with MiSTR crops. Warm from the fixed model at lr x 0.1 for the fixed run's epochs,
matching MiSTR's total. Fixed 8 -> 4 costs ~400 GMACs on PASTIS, outside the compared range, so
the launcher stops at 10 -> 5.

The legacy `--selector apt` in `finetune_inject.py` is a different method (top-k added crops), not
this baseline.

## Reading the numbers

MiSTR arms with one seed need three before a claim. Adaptive arms' GMACs are traced on test
batches (`train.compute_split=test`).
