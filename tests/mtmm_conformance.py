"""Checks every MTMMDataset subclass must pass. A dataset's own test file calls
`check_dataset(make)` with a factory `make(split, modalities, tiling) -> dataset`.

What it verifies, on a few samples per split:
  - split sizes match the spec and ids are unique and stable across instantiations;
  - every returned sample passes validation and carries exactly the requested modalities;
  - All() on every modality keeps native dates; Pool / Every / MaxFrames keep timing intact;
  - tiling covers each native sample exactly once and moves the geometry with the pixels;
  - the shared collate stacks a batch with per-modality padding marked invalid.
"""
from __future__ import annotations

from manyup.data import All, Every, MaxFrames, Pool, mtmm_collate

N = 3  # samples examined per check


def _all(spec):
    return {m: All() for m in spec.modalities}


def check_dataset(make, spec, tiling, pool_window) -> None:
    for split, size in spec.split_sizes.items():
        ds = make(split, _all(spec), None)
        assert len(ds) == size, f"{split}: {len(ds)} != {size}"
        ids = [ds.sample_id(i) for i in range(len(ds))]
        assert len(set(ids)) == len(ids), f"{split}: duplicate ids"
        assert ids == [make(split, _all(spec), None).sample_id(i) for i in range(len(ds))]
        for i in range(min(N, len(ds))):
            s = ds[i]
            assert s.id == ids[i] and s.split == split
            assert list(s.modalities) == list(spec.modalities)

    split = next(iter(spec.split_sizes))
    timed = [m for m, ms in spec.modalities.items() if not ms.is_static]

    # temporal policies keep timing intact
    for policy in (Every(2), MaxFrames(2), Pool("month", window=pool_window)):
        mods = {m: (policy if m in timed else All()) for m in spec.modalities}
        ds = make(split, mods, None)
        native = make(split, _all(spec), None)
        for i in range(min(N, len(ds))):
            s, n = ds[i], native[i]
            for m in timed:
                md, nd = s.modalities[m], n.modalities[m]
                assert md.history[-1] == policy.resolved_key(spec.modalities[m])
                if isinstance(policy, Pool):
                    kept = sorted(d for src in md.time.sources for d in src)
                    valid_native = sorted(nd.time.start[nd.frame_valid].tolist())
                    assert kept == valid_native, f"{m}: pooling lost or invented source dates"
                else:
                    assert set(md.time.start.tolist()) <= set(nd.time.start.tolist())

    # tiling covers every pixel once and carries geometry
    if tiling is not None:
        full, tiled = make(split, _all(spec), None), make(split, _all(spec), tiling)
        per = len(tiled) // len(full)
        assert len(tiled) == per * len(full)
        s0 = full[0]
        seen = 0
        for k in range(per):
            t = tiled[k]
            assert t.id.startswith(s0.id + "@r")
            for m, md in t.modalities.items():
                g0 = s0.modalities[m].geometry
                if g0.known:
                    _, r, c = tiling.parse(t.id)
                    assert md.geometry == g0.crop(r, c, tiling.size, tiling.size)
            seen += t.target.numel() if t.target is not None else 0
        if s0.target is not None:
            assert seen == s0.target.numel()

    # collate
    ds = make(split, _all(spec), tiling)
    batch = mtmm_collate([ds[i] for i in range(min(N, len(ds)))])
    for m, mb in batch.modalities.items():
        B, T = mb.frame_valid.shape
        assert mb.data.shape[:2] == (B, T)
        if mb.t_start is not None:   # padded slots (no date) must be marked invalid
            assert not mb.frame_valid[mb.t_start == -1].any()
