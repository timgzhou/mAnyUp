"""TileStore: a dataset's samples, pre-cut to fixed-size tiles, as memory-mappable arrays.

Raw datasets are slow to read per tile: a BioMassters chip is 24 GeoTIFFs (read again for
each of its 16 tiles when tiles are shuffled), a PASTIS patch is re-pooled for each of its 4
tiles. A store keeps every tile's RAW frames (the source read with All(), i.e. before any
temporal policy) so one tile is a few contiguous reads:

    <root>/<name>/meta.json                    source class + key, layout, dtypes
    <root>/<name>/<split>/index.json           tile ids (dataset order) + shard sizes
    <root>/<name>/<split>/shard_<k>/
        samples.json                           per tile: id, native id, row/col, geometry,
                                               target geometry, meta, time precision/sources
        <mod>.data.npy       (F, C, S, S)      the frames of all tiles in the shard, in order
        <mod>.offsets.npy    (N + 1,)          tile i owns frames offsets[i]:offsets[i+1]
        <mod>.t_start.npy / t_end.npy (F,)     days (timed modalities)
        <mod>.frame_valid.npy (F,)  <mod>.pixel_valid.npy (F, S, S)  <mod>.aux.<k>.npy
        target.npy           (N, S, S)
        done                                   written last (a shard is complete)

Each array is stored in a compact dtype and restored to the dtype the source produced; the
build fails if that round trip is not exact. Tile ids are the source's sample ids
("<native>@r<row>c<col>"), and each tile records its native sample and pixel offset.

TileStoreDataset reads a store as an MTMMDataset whose samples are the tiles, so temporal
policies are applied at load, per tile. Its `key` equals the source dataset's key for the
same policies, so caches keyed by dataset configuration stay valid. Every policy is
per-pixel except Pool's frame_valid ("any valid pixel in the month"), which is decided over
the tile here but over the whole native sample in the source; the two differ only when a
modality has pixel_valid AND a pooled month has valid pixels elsewhere in the native sample
but none in the tile. Samples are otherwise identical (tests/test_tile_store.py).

    build_tile_store(source_dataset, root, storage={"S2": "uint16"}, ...)
    TileStoreDataset(root / name, "train", {"S2": Pool("month", "mean", window=...)})
"""
from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from manyup.data.dataset import MTMMDataset
from manyup.data.geometry import Geometry
from manyup.data.sample import ModalityData, MTMMSample
from manyup.data.temporal import All, TemporalPolicy
from manyup.data.time import TimeAxis

_NP = {torch.float32: "float32", torch.float16: "float16", torch.int16: "int16",
       torch.int64: "int64", torch.uint8: "uint8", torch.int32: "int32", torch.bool: "bool"}


def store_name(source: MTMMDataset) -> str:
    return f"{source.SPEC.name}_{layout(source)}"


def layout(source: MTMMDataset) -> str:
    """How samples were cut: the source key after name and modalities (tile64,
    c64_mapped0.5_class2)."""
    mods = "_".join(f"{m}-{p.resolved_key(source.SPEC.modalities[m])}"
                    for m, p in source.modalities.items())
    prefix = f"{source.SPEC.name}_{mods}_"
    if not source.key.startswith(prefix):
        raise ValueError(f"unexpected dataset key {source.key!r}")
    return source.key[len(prefix):]


def _geo(g: Geometry | None):
    return None if g is None else [g.crs, None if g.transform is None else list(g.transform),
                                   list(g.shape), g.approximate]


def _ungeo(v) -> Geometry | None:
    if v is None:
        return None
    crs, tr, shape, approx = v
    return Geometry(crs, None if tr is None else tuple(tr), tuple(shape), approx)


def _pack(x: torch.Tensor, storage: str, what: str) -> np.ndarray:
    a = x.numpy()
    y = a.astype(storage)
    if not np.array_equal(y.astype(a.dtype), a, equal_nan=a.dtype.kind == "f"):
        raise ValueError(f"{what}: storing {a.dtype} as {storage} is lossy")
    return y


# ---- build ----------------------------------------------------------------------------------
class _Shards(Dataset):
    """Item k writes shard k (a run of whole native samples) and returns its tile ids."""

    def __init__(self, source, runs, out, fmt):
        self.source, self.runs, self.out, self.fmt = source, runs, out, fmt

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, k):
        d = self.out / f"shard_{k:05d}"
        if (d / "done").exists():
            return json.loads((d / "ids.json").read_text())
        d.mkdir(parents=True, exist_ok=True)
        samples = [self.source[i] for i in self.runs[k]]
        _write_shard(d, samples, [self.source._index[i] for i in self.runs[k]], self.fmt)
        ids = [s.id for s in samples]
        (d / "ids.json").write_text(json.dumps(ids))
        (d / "done").touch()
        return ids


def _write_shard(d: Path, samples: list[MTMMSample], index, fmt: dict) -> None:
    rows = []
    for s, (nid, off) in zip(samples, index):
        rows.append({"id": s.id, "native": nid, "offset": off, "geometry": {},
                     "target_geometry": _geo(s.target_geometry), "meta": s.meta, "time": {}})
    for m, mf in fmt["modalities"].items():
        mds = [s.modalities[m] for s in samples]
        cat = lambda f: torch.cat([f(md) for md in mds])                       # noqa: E731
        np.save(d / f"{m}.data.npy", _pack(cat(lambda md: md.data), mf["storage"], f"{m} data"))
        np.save(d / f"{m}.offsets.npy",
                np.cumsum([0] + [md.num_frames for md in mds]).astype(np.int64))
        np.save(d / f"{m}.frame_valid.npy", cat(lambda md: md.frame_valid).numpy())
        if mf["timed"]:
            np.save(d / f"{m}.t_start.npy", cat(lambda md: md.time.start).numpy())
            np.save(d / f"{m}.t_end.npy", cat(lambda md: md.time.end).numpy())
        if mf["pixel_valid"]:
            np.save(d / f"{m}.pixel_valid.npy", cat(lambda md: md.pixel_valid).numpy())
        for k, (st, _) in mf["aux"].items():
            np.save(d / f"{m}.aux.{k}.npy", _pack(cat(lambda md: md.aux[k]), st, f"{m} aux {k}"))
        for r, md in zip(rows, mds):
            r["geometry"][m] = _geo(md.geometry)
            r["time"][m] = None if md.time is None else [md.time.precision, md.time.sources]
    if fmt["target"] is not None:
        np.save(d / "target.npy", _pack(torch.stack([s.target for s in samples]),
                                        fmt["target"][0], "target"))
    (d / "samples.json").write_text(json.dumps(rows))


def _format(source: MTMMDataset, storage: dict[str, str]) -> dict:
    """Storage / output dtype of every array, from the first sample of the source."""
    s = source[0]
    mods = {}
    for m, md in s.modalities.items():
        out = _NP[md.data.dtype]
        mods[m] = {"storage": storage.get(m, out), "dtype": out, "timed": md.time is not None,
                   "pixel_valid": md.pixel_valid is not None,
                   "aux": {k: [storage.get(f"{m}.{k}", _NP[v.dtype]), _NP[v.dtype]]
                           for k, v in md.aux.items()}}
    tgt = None
    if s.target is not None:
        out = _NP[s.target.dtype]
        tgt = [storage.get("target", out), out]
    return {"modalities": mods, "target": tgt}


def build_tile_store(source: MTMMDataset, root: str | Path, storage: dict[str, str] | None = None,
                     natives_per_shard: int = 16, num_workers: int = 8,
                     limit_natives: int | None = None) -> Path:
    """Write `source` (every modality read with All()) to <root>/<store_name>/<split>.

    storage: compact dtype per array ("<mod>", "<mod>.<aux>", "target"); default: the
    source dtype. Resumable: finished shards are kept. Returns the split directory."""
    bad = {m: p.key for m, p in source.modalities.items() if not isinstance(p, All)}
    if bad:
        raise ValueError(f"a store keeps raw frames; read the source with All(), got {bad}")
    storage = storage or {}
    top = Path(root) / store_name(source)
    fmt = _format(source, storage)
    meta = {"source_class": f"{type(source).__module__}.{type(source).__qualname__}",
            "source_key": source.key, "name": store_name(source), "layout": layout(source),
            "size": list(source.SPEC.native_shape if source.tiling is None
                         else (source.tiling.size,) * 2),
            **fmt}
    meta.update(natives_per_shard=natives_per_shard, limit_natives=limit_natives)
    top.mkdir(parents=True, exist_ok=True)
    if (top / "meta.json").exists():
        have = json.loads((top / "meta.json").read_text())
        if have != meta:
            raise ValueError(f"{top} was built with\n{have}\nnot\n{meta}")
    (top / "meta.json").write_text(json.dumps(meta, indent=2))

    groups: list[list[int]] = []                 # source indices of each native sample
    for i, (nid, _) in enumerate(source._index):
        if not groups or source._index[groups[-1][0]][0] != nid:
            groups.append([])
        groups[-1].append(i)
    groups = groups[:limit_natives] if limit_natives else groups
    runs = [sum(groups[a:a + natives_per_shard], [])
            for a in range(0, len(groups), natives_per_shard)]
    out = top / source.split
    out.mkdir(exist_ok=True)
    loader = DataLoader(_Shards(source, runs, out, fmt), batch_size=None, num_workers=num_workers,
                        multiprocessing_context="fork" if num_workers else None)
    ids = []
    for k, shard_ids in enumerate(loader):
        ids += shard_ids
        if k % 20 == 0:
            print(f"[{source.split}] shard {k + 1}/{len(runs)}", flush=True)
    want = [source.sample_id(i) for r in runs for i in r]
    if ids != want:
        raise AssertionError(f"{out}: stored ids differ from the source order")
    (out / "index.json").write_text(json.dumps({"ids": ids,
                                                "shards": [len(r) for r in runs]}))
    print(f"[{source.split}] {len(ids)} tiles in {len(runs)} shards -> {out}")
    return out


# ---- read -----------------------------------------------------------------------------------
class TileStore:
    """Random access to one split of a store; arrays are memory-mapped on first use."""

    def __init__(self, directory: str | Path, split: str):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        idx = json.loads((self.dir / split / "index.json").read_text())
        self.ids = idx["ids"]
        self.split_dir = self.dir / split
        self._where = [(k, j) for k, n in enumerate(idx["shards"]) for j in range(n)]
        self._arrays: dict = {}
        self._rows: dict = {}

    def __len__(self):
        return len(self.ids)

    def _arr(self, k: int, name: str) -> np.ndarray:
        key = (k, name)
        if key not in self._arrays:
            self._arrays[key] = np.load(self.split_dir / f"shard_{k:05d}" / f"{name}.npy",
                                        mmap_mode="r")
        return self._arrays[key]

    def _row(self, k: int, j: int) -> dict:
        if k not in self._rows:
            if len(self._rows) > 64:                 # bound memory; shards are read in runs
                self._rows.clear()
            self._rows[k] = json.loads((self.split_dir / f"shard_{k:05d}" / "samples.json")
                                       .read_text())
        return self._rows[k][j]

    def modality(self, i: int, m: str, bands: tuple[str, ...]) -> ModalityData:
        k, j = self._where[i]
        mf, row = self.meta["modalities"][m], self._row(k, j)
        a, b = self._arr(k, f"{m}.offsets")[j:j + 2]
        sl = slice(int(a), int(b))
        data = torch.from_numpy(np.array(self._arr(k, f"{m}.data")[sl]).astype(mf["dtype"]))
        time = None
        if mf["timed"]:
            precision, sources = row["time"][m]
            time = TimeAxis(torch.from_numpy(np.array(self._arr(k, f"{m}.t_start")[sl])),
                            torch.from_numpy(np.array(self._arr(k, f"{m}.t_end")[sl])),
                            precision, sources)
        pv = (torch.from_numpy(np.array(self._arr(k, f"{m}.pixel_valid")[sl]))
              if mf["pixel_valid"] else None)
        aux = {n: torch.from_numpy(np.array(self._arr(k, f"{m}.aux.{n}")[sl]).astype(out))
               for n, (_, out) in mf["aux"].items()}
        return ModalityData(data, bands, time,
                            torch.from_numpy(np.array(self._arr(k, f"{m}.frame_valid")[sl])),
                            _ungeo(row["geometry"][m]), pixel_valid=pv, aux=aux)

    def target(self, i: int) -> torch.Tensor | None:
        if self.meta["target"] is None:
            return None
        k, j = self._where[i]
        return torch.from_numpy(np.array(self._arr(k, "target")[j])
                                .astype(self.meta["target"][1]))

    def row(self, i: int) -> dict:
        return self._row(*self._where[i])


class TileStoreDataset(MTMMDataset):
    """A store read as an MTMMDataset: each stored tile is one sample (tiling=None); temporal
    policies are applied per tile at load. Same sample ids, order and key as the source."""

    def __init__(self, directory: str | Path, split: str, modalities: dict[str, TemporalPolicy],
                 validate: bool = True):
        self.store = TileStore(directory, split)
        meta = self.store.meta
        mod, _, cls = meta["source_class"].rpartition(".")
        spec = getattr(importlib.import_module(mod), cls).SPEC
        sizes = {s: len(json.loads((Path(directory) / s / "index.json").read_text())["ids"])
                 for s in ("train", "val", "test") if (Path(directory) / s / "index.json").exists()}
        self.SPEC = replace(spec, native_shape=tuple(meta["size"]), split_sizes=sizes)
        missing = set(modalities) - set(meta["modalities"])
        if missing:
            raise ValueError(f"{meta['name']} does not store {sorted(missing)}")
        super().__init__(split, modalities, tiling=None, validate=validate)
        self._pos = {sid: i for i, sid in enumerate(self.store.ids)}

    @property
    def key(self) -> str:
        return super().key.rsplit("_native", 1)[0] + "_" + self.store.meta["layout"]

    def _native_ids(self, split):
        return list(self.store.ids)

    def _load(self, sample_id, modalities):
        i = self._pos[sample_id]
        row = self.store.row(i)
        mods = {m: self.store.modality(i, m, self.SPEC.modalities[m].bands) for m in modalities}
        return MTMMSample(sample_id, self.split, mods, self.store.target(i),
                          _ungeo(row["target_geometry"]), row["meta"])

    def _load_target(self, sample_id):
        return self.store.target(self._pos[sample_id])

    def native_of(self, i: int) -> tuple[str, tuple[int, int] | None]:
        """(native sample id, (row, col) pixel offset in it or None) of sample i."""
        row = self.store.row(i)
        return row["native"], None if row["offset"] is None else tuple(row["offset"])


def open_store(directory: str | Path, split: str,
               modalities: dict[str, TemporalPolicy]) -> TileStoreDataset | None:
    """The store at `directory` read with these policies; None if this split is not built."""
    if not (Path(directory) / split / "index.json").exists():
        return None
    return TileStoreDataset(directory, split, modalities)
