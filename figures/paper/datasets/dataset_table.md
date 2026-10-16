| Dataset | Label (metric) | Samples (train / val / test) | Modality | Bands | Timesteps per sample | Cadence | Given to the model as |
|---|---|---|---|---|---|---|---|
| **PASTIS-R** | crop type, 19 classes (mIoU) | 2,433 patches, 128 px; 5,820 / 1,928 / 1,984 tiles of 64 px | **Sentinel-2 L2A** | 10 bands | 38–61 (median 46) dates | irregular, own dates | 4 seasons → 12 monthly means → single dates |
|  |  |  | **Sentinel-1 ascending** | VV, VH, VV−VH | 65–71 (median 71) dates | irregular, own dates | 4 seasons → 12 monthly means → single dates |
|  |  |  | Sentinel-1 descending | VV, VH, VV−VH | 69–70 (median 70) dates | irregular, own dates | — |
| **BioMassters** | above-ground biomass, t/ha (RMSE) | 8,513 chips, 256 px; 64,032 / 27,808 / 44,368 tiles of 64 px | **Sentinel-2 L2A** | 10 bands | 5–12 (median 11) months | monthly composites, Sep–Aug | 4 seasons → 12 months (no dates in the release) |
|  |  |  | **Sentinel-1 ascending** | VV, VH | 12 months | monthly composites, Sep–Aug | 4 seasons → 12 months |
|  |  |  | Sentinel-1 descending | VV, VH | ≤ 12 months | monthly composites | — |
|  |  |  | S2 cloud probability | 1 | as Sentinel-2 | monthly | — |
| **GEOID-Flood** | flood / permanent water / background (mIoU of background + flood; flood IoU) | 211,725 / 32,259 / 46,048 chips of 64 px (≥ 50% mapped, with flood) | **Sentinel-1 RTC year series (ours, Planetary Computer)** | VV, VH | 20–124 (median 58) dates per year stored; 7–64 (median 29) in the 6-month window read | irregular, every pass of GEOID's orbits | calendar months → single dates (consecutive-date pairs) |
|  |  |  | Sentinel-1 GRD | VV, VH | 2 (pre, post) | event pair | — (earlier pre/post baseline only) |
|  |  |  | Sentinel-1 RTC | VV, VH | 2 (pre, post) | event pair | — |
|  |  |  | Sentinel-2 L2A | multispectral | 1 (pre-event) | composite | — |
|  |  |  | Copernicus GLO-30 DEM | 1 | static | — | — |
| **YieldSAT** | crop yield, t/ha (pixel RMSE) | 2,173 field-seasons; 8,703 / 1,088 / 1,081 tiles of 64 px | **Sentinel-2 L2A, every acquisition** | 12 bands (+ SCL) | 11–153 (median 41) dates | irregular, seeding → harvest | 8 seasons → 24 monthly clear-pixel means → single dates |
|  |  |  | Sentinel-2 L2A, ML-ready | 12 bands | 24 month slots, 3–13 (median 7) filled | one date per month slot | — (fixed-resolution baselines only) |
|  |  |  | Soil properties | 8 layers | static | — | — |
|  |  |  | DEM + terrain derivatives | 5 layers | static | — | — |
|  |  |  | Weather (per field) | 4 variables | daily | daily | — |

Bold: the modalities the models read. Timesteps are per sample, counted from the data (minimum–maximum, median). "Given to the model as": the coarse ps16 level → the finer levels crops are taken from.
