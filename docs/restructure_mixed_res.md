# Restructuring `finetune_inject.py` into modules

Status: design draft, 2026-10-22. Nothing ported yet.

## Why

`exp/upsamplers/finetune_inject.py` is 5,356 lines with 101 CLI flags; `manyup/inject.py` adds
1,026. `InjectFT` alone is ~3,000 lines and its `forward` is 1,523. Which data a selector reads
is decided by several interacting flags (`--selector`, `--score_src`, `--season_rule`,
`--res_mode`) inside that `forward`. That is how the `_lrent_` arms ended up ranking windows by
a frozen LR-only checkpoint, not by the model being trained, without anyone noticing.

## The model in one line

```
plan0  = uniform(sample)                      # every region, every season, LR patch size
out0   = decode(encode(tokenize(sample, plan0)))
plan1  = selector(sample, out0, labels if training else None)
out1   = decode(encode(tokenize(sample, plan1)))
```

There is no separate re-encoder. An LR-only pass is the mixed-resolution encoder called on a
token set whose tokens all share one patch size. The selector does not "pick crops"; it returns
a tokenization plan: which regions and which months to tokenize at which patch size, on top of
the uniform plan. A two-level selector (`--level2`) is the same step applied again to `out1`.

Because `out0` is computed by the model in training, a selector can only read a frozen model
when one is explicitly constructed and passed to it.

## Modules

### `TokenSet`
The only thing the encoder and decoder see. Per token:

| field | meaning |
|---|---|
| `x` | embedding (N, D) |
| `sample` | owning sample index |
| `patch` | patch size in pixels |
| `yx` | centre in pixels, in the tile frame |
| `t` | time, in season-slot units (season index, + fraction inside the season) |
| `month` | calendar month (OlmoEarth's month embedding) |
| `modality`, `bandset` | as in OlmoEarth |

Packed per sample for varlen FlashAttention (today's `Packed`, plus the fields above as
per-token tensors instead of crop-only side channels like `yx`, `owner`, `typed`, `trace`).

### `Tokenizer`
`tokenize(sample, plan) -> TokenSet`. Patch-embeds each requested (region, months, patch size)
with OlmoEarth's flexible patch embedding and adds the encodings from the token's own
`yx` / `t` / `month`.

### `Encoder`
`encode(tokens, layers=(lo, hi), state=None) -> TokenSet`. Joint attention within each sample.
A layer range plus cached state keeps the injection layer L: tokens added by `plan1` enter at
layer L, and layers [0, L) of the uniform tokens are reused from the first pass. At L = 0 (every
current run) nothing is reused, since the LR tokens attend to the crops.

### `Decoder`
`decode(tokens) -> logits (B, C, 64, 64)`. Today's `NativeSetReadout`: each token's head map at
its own patch size, pooled per pixel. Already resolution-agnostic.

### `Selector = Scorer + Allocator`
- `Scorer(ctx) -> ScoreMaps`: a window score map (B, H, W) and per-season maps (B, NS, H, W).
  `ctx` holds the raw sample, `out0` (tokens + logits) and the labels in training only.
  Implementations: `LabelLoss` (oracle), `OwnEntropy`, `Residual(base, arch)`, `Random`, `Grid`,
  `APT`.
- `Allocator(scores, budget) -> Plan`: `AllMonths` (top-n windows, every present month),
  `BetterHalf` (top windows, each its better-half seasons), `SeasonPairs`, `Level2(inner)`.
- A selector may return an auxiliary loss (residual ranking loss) and a gate log-weight per
  token. The training loop sums every module's auxiliary losses generically.

Example: the arm queued 2026-10-22 as `_sent_rese2exattn_srall_warm_` is
`AllMonths(Residual(OwnEntropy(), arch="xattn", train="e2e"))`.

## Position and time encodings (checked 2026-10-22)

**Space: already consistent, in pixels.**
- OlmoEarth encodes an LR token at grid index i at `i * patch * input_res / BASE_GSD`. At 10 m
  input that is `i * patch` pixels, its top-left corner.
- `MixedResOE.embed` encodes a crop as `sincos_2d(py, px)` in pixels. With `pos center` it is
  shifted by `(P - lr_ps) / 2`, so that a crop and an LR token with the same centre get the same
  encoding.
- The general rule for the `Tokenizer`: encode `centre - ref / 2`, where `ref` is the LR patch
  size. This reproduces both cases.

**Time: consistent, in season-slot units.**
- LR season tokens sit at slots 0..G-1.
- A month crop gets its season's slot (`_crop_slot`), optionally plus its fraction inside the
  season (`--frac_slot`).
- Its calendar month enters only through OlmoEarth's month embedding.
- The `Tokenizer` keeps both, as `t` and `month`.

## What gets ported

Live (every result row since job ~6.3M, early October, plus arms named only in run names):

| | kept |
|---|---|
| readout | `native_set` |
| attention | `joint`, `head_attn alone`, `pos center` |
| LR steps | `season` |
| scorers | label loss (location / per season), `season_grid`, `season_random`, `random`, `grid`, own entropy, residual |
| allocators | location top-n, better half, season pairs, all months (`--season_rule all`), level 2 |
| other | warm start (`--init_ckpt`), crop-content ablation |

Archived, not ported (still runnable from `finetune_inject_legacy.py`):
- **Readouts:** the other 14: bilinear, splat\*, cellattn\*, tsplat / tquery, native,
  native_attn, native_mean.
- **Selectors:** `learned`, `xattn` and their rank heads (`learned_select.py`); `apt` and all its
  octree / quadtree / pool / res variants; `PerturbedTopK` / DPS; the frozen-scorer
  `lr_entropy` sources.
- **Machinery:** joint picks, date crops (`--date_level`), `--self_teach`, `--acq_pool`.
- **Encoder variants:** `--lr_steps` all / year / raw; `attn` oneway / overlap; the `head_attn`
  variants; `pos` corner / box; the `hr_type` embedding.

A legacy arm is ported only when it is needed again. A different attention pattern for crops
would come back as a mask policy argument to `encode`, not as a second encoder.

## Config and run names

- One dataclass per module; a run is their composition, saved as JSON next to its checkpoint.
- The run name is derived from the config (a short readable tag plus a hash), replacing the
  ~60-line suffix chain in `main`.
- Results rows carry the config fields as columns, so options are never visible only in run
  names.

## Migration order

1. `git init` and commit the current tree (`/scratch/timz/mAnyUp` is not a repository).
2. Freeze `finetune_inject.py` as `finetune_inject_legacy.py`.
3. Regression set: about six trained checkpoints, namely location oracle, `oracle_bothalf`,
   season grid, level 2, self-entropy and one residual. For each, record its test metrics and its
   picks on a fixed batch, using the legacy code.
4. Port `TokenSet`, `Tokenizer`, `Encoder` and `Decoder`; the uniform-plan pass must match
   `_crop_free` exactly.
5. Port the scorers and allocators one at a time. Each must load the old state dict and
   reproduce its regression entry: identical picks, and test metrics within numerical noise.
6. Move the training loop, evaluation and config out of `main`; switch the Slurm scripts over.

## Open questions

- Package layout: `manyup/mixres/{tokens,encoder,decoder,selectors/}.py` plus
  `exp/mixres/train.py`
- Config system: plain dataclasses + a small CLI
- Do the YieldSAT / BioMassters regression paths (entropy is classification-only) need their own
  scorer later, for now implement the things that make sense (for example, the oracle loss map based selection is still valid) and leave the others empty (dont fall back to an arbitrary decision, if things are called with wrong composition, throw error.)

## Progress (2026-10-22, branch `mixres-restructure`)

| step | state |
|---|---|
| 1 git | the repo existed; snapshot `eea740c`, tag `legacy-finetune-inject` |
| 2 freeze | `finetune_inject.py` kept in place (queued jobs import it); the tag is the frozen reference |
| 3 regression set | `scripts/slurm/regress_record.sh` -> `cache/regress/` (9 runs) |
| 4 model | `manyup/mixres/{plan,model}.py`; `exp.mixres.check_equiv`: bit-exact |
| 5 selectors | `manyup/mixres/selectors.py` incl. `Level2` (m4 / m2); `exp.mixres.check_selectors`: 10 runs bit-exact (picks, logits, full-test metrics) |
| 6 config + training | `manyup/mixres/config.py`, `exp/mixres/train.py`; `exp.mixres.check_step`: first update within legacy run-to-run noise; end-to-end runs `scripts/slurm/mixres_validate.sh` |

Findings on the way:
- The legacy forward is not reproducible across processes: its bf16 numerics land in one of two
  states per process (same picks with logits ±0.08, or near-tied oracle windows flipping). Within
  a process it is stable, so the checks record the legacy reference in the same process.
- Training is not reproducible even within a process (backward nondeterminism): the same legacy
  run twice ends at val .396 vs .422 after 6 steps. Training equivalence is therefore checked
  against that noise floor, not bit for bit.

Still to do: raw-date level-2 arms (d8 / d4 / d2); moving `NativeSetReadout`, `ResidualScorer`,
`listnet_loss`, `window_pixels`, `location_grid_scores`, `loader` out of the legacy file; GMAC
measurement in `exp.mixres.train`; switching the experiment scripts over.

