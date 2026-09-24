# OlmoEarth venv for everything in this repo:  source env_setup/env_olmo.sh
#
# Built fresh on node-local disk, never inside the repo: $SLURM_TMPDIR in a job or salloc
# (private per job, so concurrent jobs never pip-install over each other), else $TMPDIR.
#
# olmoearth-pretrain 0.1.2 ships OlmoEarth v1, v1.1 and v1.2 (see MODEL_SIZE_TO_ID in
# exp/common/config.py); it needs torch 2.9. The [training] extra brings olmo-core, which
# olmoearth_pretrain.evals imports. The two eval modules that still cannot import here are
# stubbed by exp/common/olmo_bootstrap.py. `arrow` provides pyarrow (the cluster ships it
# as a module, not a wheel), which the GeoBench-v2 eval loader imports.
module load python/3.12 scipy-stack opencv libspatialindex proj arrow
export PROJ_DATA=$EBROOTPROJ/share/proj

OLMOEARTH_VERSION=0.1.2
TORCH_VERSION=2.9.1
TORCHVISION_VERSION=0.24.1
# Versions in the dir name: a bump builds a fresh venv instead of upgrading one in place.
VENV_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/env_olmo-oe${OLMOEARTH_VERSION}-torch${TORCH_VERSION}"
PIP="pip install -q --no-warn-conflicts"

virtualenv -q --no-download --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
$PIP --no-index --upgrade pip
$PIP --no-index torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION   # cluster CUDA wheels
$PIP "olmoearth-pretrain[training]==$OLMOEARTH_VERSION" geopandas torchmetrics
