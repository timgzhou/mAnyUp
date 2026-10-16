# Paper figures and tables

Regenerate (from the repo root, `source env_setup/env_olmo.sh`):

    python -m exp.viz.paper.datasets            # table + dataset examples
    python -m exp.viz.paper.collect             # one PASTIS test tile through the trained models (GPU)
    python -m exp.viz.paper.model               # pipeline + token volume + materials
    python -m exp.viz.paper.readout_block --tiles 40091@r0c0 40091@r64c0 40091@r0c64 40091@r64c64   # whole low-res cell
    python -m exp.viz.paper.readout --tiles 40091@r0c0 40091@r64c0 40091@r0c64 40091@r64c64   # ranked readout candidates

| File | What it is |
|---|---|
| `datasets/dataset_table.md`, `.tex` | one row per modality of each release; bold = read by the models |
| `datasets/dataset_stats.json` | every count in the table, as measured from the data |
| `datasets/examples.png/.pdf`, `example_<dataset>.png/.pdf` | input time series + label, one sample per dataset |
| `datasets/frames/<dataset>/` | each panel as its own PNG |
| `model/model.png/.pdf` | block diagram: components, what each reads, the three token paths (`python -m exp.viz.paper.model_diagram`) |
| `model/<tile>/pipeline.png/.pdf` | acquisitions → monthly → seasonal composites → season features → month query → date query → mixed-resolution feature map → prediction |
| `model/<tile>/volume.png/.pdf` | the output tokens as three stacks of time slices |
| `model/readout/cand<rank>_<tile>_y<y>x<x>.png/.pdf` | readout candidates at class borders: each token's per-pixel class map and weight, season-only vs all-token pool vs label; ranking in `candidates.json` |
| `model/readout_block/cand<rank>_<tile>_b<by><bx>.png/.pdf` | readout over a whole ps16 block: crops per ps4 cell, weight on season vs ps4 tokens, season-only vs all-token pool; `summary.json` has the with / without-crop cell statistics |
| `model/<tile>/materials/`, `readout_materials/` | every panel as a PNG, described in `materials.json` / `readout.json` |
| `model/collect_<tile>.pt` | the tensors behind the model figures |

The model figures use two trained PASTIS checkpoints (see `exp/viz/paper/collect.py`): the month
picks come from a jointly trained month selector, the date picks, features, readout and prediction
from a run with a learned date selector that was trained with random month picks.
`python -m exp.viz.paper.collect --month_picks random` keeps that run exactly as trained.
