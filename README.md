---
license: cc-by-4.0
pretty_name: ImpactMesh-Flood
size_categories:
- 10K<n<100K
viewer: false
task_categories:
- image-feature-extraction
---

[![arXiv](https://img.shields.io/badge/arXiv-comming_soon-b31b1b?logo=arxiv)](https://arxiv.org/abs/todo)
[![Code](https://img.shields.io/badge/GitHub-ImpactMesh-EE4B2B?logo=github)](https://github.com/IBM/ImpactMesh)
[![IBMblog](https://img.shields.io/badge/Blog-IBM-0F62FE)](https://research.ibm.com/blog/todo)


# ImpactMesh-Flood

ImpactMesh is a large-scale multimodal, multitemporal dataset for flood and wildfire mapping, released by IBM, DLR, and the ESA Φ-lab. 
It integrates **Sentinel-1 SAR**, **Sentinel-2 optical**, **Copernicus DEM**, and high-quality annotations from Copernicus EMS.
The technical report is released soon. You find the wildfire subset here: https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Fire.

![events_world](https://github.com/IBM/ImpactMesh/raw/main/assets/events_world_light.png)

---
## Features
- Multimodal: SAR, optical, DEM
- Multitemporal: Four time steps (pre-month, pre-event, event, post-event)
- Global coverage: 200+ flood events
- Scale: 80K samples
- License: CC-BY 4.0

## Quick Start

Download the dataset:
```shell
hf download ibm-esa-geospatial/ImpactMesh-Flood --repo-type dataset --local-dir data/ImpactMesh-Flood

# Only download a single modality (e.g., S2L2A)
hf download ibm-esa-geospatial/ImpactMesh-Flood --repo-type dataset --include "*/S2L2A.tar" --local-dir data/ImpactMesh-Flood

# Only download a single split (e.g., validation)
hf download ibm-esa-geospatial/ImpactMesh-Flood --repo-type dataset --include "val/*" --local-dir data/ImpactMesh-Flood
```

Untar the samples:
```shell
mkdir data/ImpactMesh-Flood/data
for f in data/ImpactMesh-Flood/*/*.tar; do
  echo "Extracting $f"
  tar -xf "$f" -C data/ImpactMesh-Flood/data
done
```

The samples from all splits are saved in shared folders `data/ImpactMesh-Flood/data/{modality}`. After extracting, you can delete the tars:
```shell
rm -r data/ImpactMesh-Flood/train
rm -r data/ImpactMesh-Flood/val
rm -r data/ImpactMesh-Flood/test
```

We use [TerraTorch](https://terrastackai.github.io/terratorch/stable/) for the model fine-tuning and provide data modules for ImpactMesh. You can download the code and configs for the fine-tuning from https://github.com/IBM/ImpactMesh.

Alternatively, you can install the data loading code with:

```shell
pip install impactmesh
```

```shell
terratorch fit --config configs/terramind_v1_tiny_impactmesh_flood.yaml
```

## Citation

Our technical report is released soon!

## Acknowledgement

ImpactMesh was developed as part of the FAST‑EO project funded by the European Space Agency Φ‑Lab (contract #4000143501/23/I‑DT).

Sentinel-2 Level-2A data were downloaded from Microsoft Planetary Computer and are provided under Copernicus Sentinel license conditions (© European Union 2015–2025, ESA) (https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).

Sentinel-1 Radiometrically Terrain Corrected (RTC) SAR data were retrieved from Microsoft Planetary Computer (calibrated to GRD and terrain-corrected using PlanetDEM) under Copernicus Sentinel license terms (© European Union 2014–2025) (https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc).

The DEM data is produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

Annotations were sourced from the Copernicus Emergency Management Service (© European Union, 2012–2025), available at https://emergency.copernicus.eu/.