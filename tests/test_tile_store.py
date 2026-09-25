"""Tile stores reproduce the raw datasets exactly: a small store is built from the first few
native samples of each task's source, then every stored tile, read with the task's time
setup (and with All()), is compared field by field with the raw dataset's sample."""
import pytest
import torch

from exp.common.build_tile_store import same
from exp.common.tasks import TASKS
from manyup.data import All
from manyup.data.tile_store import TileStoreDataset, build_tile_store, store_name

NATIVES = {"pastis": 2, "biomassters": 1, "geoidflood": 40}


def _raw_or_skip(task, split):
    try:
        return task.source(split)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"{task.name} raw data not available: {e}")


@pytest.mark.parametrize("name", list(TASKS))
def test_store_samples_equal_raw_samples(name, tmp_path):
    task = TASKS[name]
    src = _raw_or_skip(task, "val")
    build_tile_store(src, tmp_path, task.store_dtypes, natives_per_shard=1, num_workers=0,
                     limit_natives=NATIVES[name])
    d = tmp_path / store_name(src)
    for policies in (task.policies(), {m: All() for m in task.stored_modalities}):
        store = TileStoreDataset(d, "val", policies)
        raw = task.raw("val", policies)
        assert store.key == raw.key
        for i in range(len(store)):
            same(store[i], raw[i], f"{name}/{i}")
            assert torch.equal(store.target(i), raw.target(i))
        nid, off = store.native_of(len(store) - 1)
        assert raw.sample_id(len(store) - 1).startswith(nid)


def test_lossy_storage_is_refused(tmp_path):
    task = TASKS["pastis"]
    src = _raw_or_skip(task, "val")
    with pytest.raises(ValueError, match="lossy"):
        build_tile_store(src, tmp_path, {"S1A": "int16"}, num_workers=0, limit_natives=1)
