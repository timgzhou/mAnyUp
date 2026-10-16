"""OlmoEarth as an Encoder: MTMMBatch -> FeatureSeries (B, T, gH, gW, D).

Packing into OlmoEarth's input is explicit and by name:
  - each dataset modality maps to an OlmoEarth modality (`modality_map`); bands are matched
    to OlmoEarth's band_order by name (case-insensitive), and bands the data lacks must be
    listed in `impute` (target band -> source band) or setup fails;
  - values are normalized with OlmoEarth's pretrain ("computed") statistics;
  - OlmoEarth takes ONE timestamp per step for all modalities, so the mapped modalities
    must share a time axis (e.g. the same Pool window) -- checked per batch;
  - invalid frames are passed as MISSING tokens and excluded from pooling.

Attention ("attention" argument):
  "flash"  OlmoEarth's varlen FlashAttention: each sample's valid tokens are packed back to
           back, so MISSING tokens are excluded exactly, a sample's features do not depend on
           its batch, and memory stays at the fast path's. The default.
  "sdpa"   PyTorch SDPA, OlmoEarth's default. Without missing tokens it reproduces the legacy
           caches bit for bit. With missing tokens it cannot be batched: OlmoEarth's masked
           path drops the attention mask in eval mode (padding becomes visible, features
           depend on the batch) and in training builds a (B, heads, N, N) mask (~9x memory).
           So a batch with missing tokens is refused under "sdpa".

Output: per timestep, tokens are averaged over bandsets, then over the modalities present at
that step. temporal_mode "series" encodes the whole series in one pass (timesteps attend to
each other); "per_timestep" encodes every frame on its own (folded into the batch).
`tile_size` < sample size splits the image into independent context windows (an
approximation: attention does not cross windows).
"""
from __future__ import annotations

import numpy as np
import torch
from olmoearth_pretrain.data.constants import BASE_GSD, Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

from manyup.data.dataset import MTMMDataset
from manyup.data.features import FeatureSeries
from manyup.data.sample import MTMMBatch
from manyup.data.time import from_days
from manyup.encoders.base import Encoder, Requirements

# What OlmoEarth's own PASTIS eval did for S2 (PASTIS has no B01/B09): copy neighbours.
LEGACY_S2_IMPUTE = {"B01": "B02", "B09": "B8A"}
MAX_FOLDED_ITEMS = 16384   # cap on batch width when folding tiles/time into the batch


class OlmoEarthInput:
    """Packs an MTMMBatch into OlmoEarth's MaskedOlmoEarthSample. No model: fine-tuning code
    that needs gradients through OlmoEarth uses this directly; OlmoEarthEncoder wraps it.

    timestamp: the date OlmoEarth sees for a step -- its "start" (every cache so far), or the
    "mid"dle of [start, end], for steps that span a season (GroupSteps)."""

    def __init__(self, modality_map: dict[str, str],
                 impute: dict[str, dict[str, str]] | None = None, timestamp: str = "start"):
        if timestamp not in ("start", "mid"):
            raise ValueError(f"timestamp {timestamp!r} not in ('start', 'mid')")
        self.timestamp = timestamp
        self.modality_map = dict(modality_map)
        self.olmo = {m: Modality.get(o) for m, o in self.modality_map.items()}
        self.impute = {m: dict((impute or {}).get(m, {})) for m in self.modality_map}
        self.normalizer = Normalizer(Strategy.COMPUTED)

    def bands(self) -> dict[str, tuple[str, ...]]:
        """Dataset bands read per modality (OlmoEarth order, imputation sources last)."""
        return {m: tuple(b for b in om.band_order if b not in self.impute[m])
                + tuple(self.impute[m].values()) for m, om in self.olmo.items()}

    def check(self, dataset: MTMMDataset) -> None:
        for m, need in self.bands().items():
            if m not in dataset.modalities:
                raise ValueError(f"OlmoEarth input needs modality {m!r}; dataset gives "
                                 f"{sorted(dataset.modalities)}")
            have = {b.upper() for b in dataset.SPEC.modalities[m].bands}
            lacking = {b.upper() for b in need} - have
            if lacking:
                raise ValueError(f"{dataset.SPEC.name}/{m} lacks bands {sorted(lacking)}; "
                                 "add them to `impute`")

    def _band_index(self, m: str, bands: tuple[str, ...]) -> list[int]:
        by_name = {b.upper(): i for i, b in enumerate(bands)}
        idx = []
        for b in self.olmo[m].band_order:
            src = self.impute[m].get(b, b).upper()
            if src not in by_name:
                raise ValueError(f"{m}: band {b} not in data {bands} and not imputed")
            idx.append(by_name[src])
        return idx

    def to_olmoearth(self, batch: MTMMBatch) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Returns the packed sample and frame_valid (B, T): a step is valid if any mapped
        modality has data there."""
        mods = [batch.modalities[m] for m in self.modality_map]
        ref = mods[0]
        for m, mb in zip(self.modality_map, mods):
            if mb.t_start is None:
                raise ValueError(f"{m}: OlmoEarth needs timed modalities")
            if not torch.equal(mb.t_start, ref.t_start):
                raise ValueError(f"{m}: time axis differs from {list(self.modality_map)[0]}; "
                                 "OlmoEarth needs aligned modalities (e.g. one Pool window)")
        day = ref.t_start if self.timestamp == "start" else torch.where(
            ref.t_start >= 0, (ref.t_start + ref.t_end) // 2, ref.t_start)   # keep padding -1
        d = {"timestamps": _timestamps(day)}
        for m, mb in zip(self.modality_map, mods):
            name = self.olmo[m].name
            x = mb.data.float()[:, :, self._band_index(m, mb.bands)]          # (B,T,C,H,W)
            x = x.permute(0, 3, 4, 1, 2).contiguous().numpy()                  # (B,H,W,T,C)
            d[name] = torch.from_numpy(self.normalizer.normalize(self.olmo[m], x)).float()
            # One mask entry per value, same shape as the data (what OlmoEarth's own eval
            # builds): ONLINE_ENCODER except on the frames the data marks invalid.
            mask = torch.full_like(d[name], MaskValue.ONLINE_ENCODER.value)
            b_idx, t_idx = (~mb.frame_valid).nonzero(as_tuple=True)
            mask[b_idx, :, :, t_idx] = MaskValue.MISSING.value
            d[MaskedOlmoEarthSample.get_masked_modality_name(name)] = mask
        frame_valid = torch.stack([mb.frame_valid for mb in mods]).any(0)
        return MaskedOlmoEarthSample(**d), frame_valid


def _timestamps(t_start: torch.Tensor) -> torch.Tensor:
    """(B, T) days -> (B, T, 3) [day, month (0-11), year]; padding reuses a real date."""
    B, T = t_start.shape
    out = torch.zeros(B, T, 3, dtype=torch.long)
    for b in range(B):
        fill = int(t_start[b][t_start[b] >= 0][0])
        for t in range(T):
            d = from_days(int(t_start[b, t]) if t_start[b, t] >= 0 else fill)
            out[b, t] = torch.tensor([d.day, d.month - 1, d.year])
    return out


class OlmoEarthEncoder(Encoder):
    TEMPORAL_MODES = ("series", "per_timestep")
    KEEPS_TIME = True

    def __init__(self, model_id: str, inputs: OlmoEarthInput, patch_size: int, tile_size: int,
                 name: str | None = None, amp: bool = True, attention: str = "flash",
                 pretrained: bool = True):
        """pretrained=False: the same architecture at its random initialization (the
        train-from-scratch reference; no OlmoEarth weights are loaded)."""
        super().__init__()
        self.model_id, self.inputs = model_id, inputs
        self.patch_size, self.tile_size, self.amp = patch_size, tile_size, amp
        self.key = name or model_id.lower()
        model = load_model_from_id(getattr(ModelID, model_id), load_weights=pretrained)
        self.encoder = model.encoder if hasattr(model, "encoder") else model
        if attention == "flash" and not amp:
            raise ValueError("FlashAttention runs in bf16 only; attention='flash' needs amp=True")
        self.attention = attention
        set_attention(self.encoder, attention)

    def requirements(self) -> Requirements:
        return Requirements(bands=self.inputs.bands(), aligned_time=True, max_frames=None,
                            size_multiple=self.tile_size)

    def check(self, dataset: MTMMDataset, temporal_mode: str) -> None:
        if temporal_mode not in self.TEMPORAL_MODES:
            raise ValueError(f"temporal_mode {temporal_mode!r} not in {self.TEMPORAL_MODES}")
        self.inputs.check(dataset)
        size = dataset.tiling.size if dataset.tiling else min(dataset.SPEC.native_shape)
        if size % self.tile_size or self.tile_size % self.patch_size:
            raise ValueError(f"sample {size}, tile {self.tile_size}, patch {self.patch_size} "
                             "must nest")

    def load_finetuned(self, ckpt_path: str) -> None:
        """Replace the pretrained weights with the backbone of an exp.pastis.finetune_olmoearth
        checkpoint (keys under "backbone."), strictly: a mismatch is an error."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not bb:
            raise ValueError(f"{ckpt_path} has no 'backbone.' keys "
                             f"(prefixes {sorted({k.split('.')[0] for k in sd})})")
        missing, unexpected = self.encoder.load_state_dict(bb, strict=False)
        if missing or unexpected:
            raise ValueError(f"{ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected "
                             f"keys (first {missing[:3]} / {unexpected[:3]})")

    # --- encoding -------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, batch: MTMMBatch, temporal_mode: str) -> FeatureSeries:
        sample, frame_valid = self.inputs.to_olmoearth(batch)
        tokens = self(sample, temporal_mode)                                    # (B,T,gH,gW,D)
        ref = batch.modalities[next(iter(self.inputs.modality_map))]
        gH, gW = tokens.shape[2:4]
        return FeatureSeries(
            ids=list(batch.ids), tokens=tokens, t_start=ref.t_start, t_end=ref.t_end,
            time_precision=ref.time_precision, frame_valid=frame_valid,
            geometry=[g.rescale((gH, gW)) for g in ref.geometry],
            provenance={"encoder": self.key, "model_id": self.model_id,
                        "patch_size": self.patch_size, "tile_size": self.tile_size,
                        "temporal_mode": temporal_mode, "attention": self.attention,
                        "modalities": {m: batch.modalities[m].history
                                       for m in self.inputs.modality_map},
                        "impute": self.inputs.impute})

    def forward(self, sample: MaskedOlmoEarthSample, temporal_mode: str) -> torch.Tensor:
        """A packed sample (OlmoEarthInput.to_olmoearth, e.g. in DataLoader workers) ->
        tokens (B, T, gH, gW, D), with gradients: the fine-tuning entry point. `encode` is
        the no-grad one that takes an MTMMBatch and returns a FeatureSeries."""
        if temporal_mode not in self.TEMPORAL_MODES:
            raise ValueError(f"temporal_mode {temporal_mode!r} not in {self.TEMPORAL_MODES}")
        device = next(self.encoder.parameters()).device
        sample = _to_device(sample, device, torch.bfloat16 if self.amp else torch.float32)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.amp):
            return self._encode_tiled(sample, temporal_mode)

    def _encode_tiled(self, sample: MaskedOlmoEarthSample, temporal_mode: str) -> torch.Tensor:
        B, Hs = sample.timestamps.shape[0], _spatial_size(sample)
        ts, p = self.tile_size, self.patch_size
        n, tg, grid = Hs // ts, ts // p, Hs // p
        tiles = [(i, j) for i in range(n) for j in range(n)]
        T = sample.timestamps.shape[1]
        fold = B * (T if temporal_mode == "per_timestep" else 1)
        per_call = max(1, min(len(tiles), 4096 // max(tg * tg, 1), MAX_FOLDED_ITEMS // max(fold, 1)))
        full = None
        for c0 in range(0, len(tiles), per_call):
            group = tiles[c0:c0 + per_call]
            parts = [_slice_space(sample, i * ts, (i + 1) * ts, j * ts, (j + 1) * ts) for i, j in group]
            block = self._encode_window(_cat(parts) if len(parts) > 1 else parts[0], temporal_mode)
            if full is None:
                full = block.new_zeros((B, grid, grid, block.shape[3], block.shape[4]))
            for g, (i, j) in enumerate(group):
                full[:, i * tg:(i + 1) * tg, j * tg:(j + 1) * tg] = block[g * B:(g + 1) * B]
        return full.permute(0, 3, 1, 2, 4).contiguous()                       # (B,T,gH,gW,D)

    def _encode_window(self, sample: MaskedOlmoEarthSample, temporal_mode: str) -> torch.Tensor:
        """One encoder call per window -> (B', tg, tg, T, D)."""
        T = sample.timestamps.shape[1]
        if temporal_mode == "per_timestep":
            folded = _cat([_slice_time(sample, t) for t in range(T)])
            # a step with no data in ANY modality (a YieldSAT month without an acquisition)
            # has no tokens, which OlmoEarth refuses: encode the others, zeros for it (every
            # head reads frame_valid and skips it)
            keep = _encoded_items(folded)
            if keep.all():
                pooled = pool_timestep(self._run(folded), 0)
            else:
                pooled = None
                if keep.any():
                    part = pool_timestep(self._run(_map(folded, lambda k, v: v[keep])), 0)
                    pooled = part.new_zeros((len(keep), *part.shape[1:]))
                    pooled[keep] = part
                else:
                    g = _spatial_size(folded) // self.patch_size
                    pooled = torch.zeros(len(keep), g, g, self.encoder.embedding_size,
                                         device=keep.device)
            per_t = list(pooled.chunk(T, dim=0))
        else:
            tam = self._run(sample)
            per_t = [pool_timestep(tam, t) for t in range(T)]
        return torch.stack(per_t, dim=-2)

    def _run(self, sample: MaskedOlmoEarthSample):
        if self.attention == "flash":
            # the varlen path packs valid tokens per sample: exact masking at any batch size
            fast_pass = False
        elif _has_missing(sample):
            if sample.timestamps.shape[0] > 1:
                raise ValueError("attention='sdpa' cannot mask missing tokens in a batch "
                                 "(padding leaks into attention); use attention='flash'")
            fast_pass = False   # one sample: no padding, so masking by token removal is exact
        else:
            fast_pass = True
        return self.encoder(sample, patch_size=self.patch_size, input_res=BASE_GSD,
                            fast_pass=fast_pass)["tokens_and_masks"]


ATTENTION = ("flash", "sdpa")


def set_attention(encoder: torch.nn.Module, attention: str) -> None:
    """Switch every OlmoEarth module to varlen FlashAttention ("flash") or SDPA ("sdpa")."""
    if attention not in ATTENTION:
        raise ValueError(f"attention {attention!r} not in {ATTENTION}")
    if attention == "flash":
        import flash_attn  # noqa: F401  (fail at setup, not mid-run, if it is not installed)
    for m in encoder.modules():
        if hasattr(m, "use_flash_attn"):
            m.use_flash_attn = attention == "flash"


def pool_timestep(tam, t: int) -> torch.Tensor:
    """Tokens of step t averaged over bandsets, then over the modalities present at each
    (sample, location) -> (B, gH, gW, D). With every token present this is exactly OlmoEarth's
    pool_spatially (mean over bandsets, then over modalities), bit for bit."""
    feats, weights = [], []
    for m in tam.modalities:
        x = getattr(tam, m)[:, :, :, t:t + 1]                     # (B, gH, gW, 1, S, D)
        mask = getattr(tam, tam.get_masked_modality_name(m))[:, :, :, t:t + 1]
        online = mask == MaskValue.ONLINE_ENCODER.value           # (B, gH, gW, 1, S)
        if online.all():
            feats.append(torch.mean(x, dim=(-2, -3)))
            weights.append(torch.ones(x.shape[:3], device=x.device, dtype=x.dtype))
        else:
            w = online.unsqueeze(-1).to(x.dtype)
            cnt = w.sum(dim=(-2, -3))                              # (B, gH, gW, 1)
            feats.append((x * w).sum(dim=(-2, -3)) / cnt.clamp(min=1))
            weights.append((cnt[..., 0] > 0).to(x.dtype))
    f, w = torch.stack(feats, -1), torch.stack(weights, -1)      # (B,gH,gW,D,M), (B,gH,gW,M)
    if (w == 1).all():
        return f.mean(dim=-1)
    return (f * w.unsqueeze(-2)).sum(-1) / w.sum(-1, keepdim=True).clamp(min=1)


def _has_missing(sample: MaskedOlmoEarthSample) -> bool:
    for name, v in sample.as_dict().items():
        if name.endswith("_mask") and v is not None and (v == MaskValue.MISSING.value).any():
            return True
    return False


def _encoded_items(sample: MaskedOlmoEarthSample) -> torch.Tensor:
    """(B,) bool: items with at least one token OlmoEarth will encode (any mask value that
    is not MISSING, in any modality)."""
    keep = None
    for name, v in sample.as_dict().items():
        if name.endswith("_mask") and v is not None:
            k = (v != MaskValue.MISSING.value).flatten(1).any(1)
            keep = k if keep is None else keep | k
    return keep


def _spatial_size(sample: MaskedOlmoEarthSample) -> int:
    for name, v in sample.as_dict().items():
        if name != "timestamps" and not name.endswith("_mask") and v is not None:
            return v.shape[1]
    raise ValueError("no spatial modality in sample")


def _map(sample: MaskedOlmoEarthSample, fn) -> MaskedOlmoEarthSample:
    return sample._replace(**{k: fn(k, v) for k, v in sample.as_dict().items() if v is not None})


def _to_device(sample, device, dtype):
    """Everything but timestamps in `dtype`. Under amp this is bf16 INPUTS, as OlmoEarth's own
    eval to_device does -- autocast alone would leave the pre-embedding ops in fp32."""
    return _map(sample, lambda k, v: v.to(device) if k == "timestamps" else v.to(device, dtype))


def _slice_space(sample, h0, h1, w0, w1):
    return _map(sample, lambda k, v: v if k == "timestamps" else v[:, h0:h1, w0:w1])


def _slice_time(sample, t):
    return _map(sample, lambda k, v: v[:, t:t + 1] if k == "timestamps" else v[:, :, :, t:t + 1])


def crop_patches(sample: MaskedOlmoEarthSample, b: torch.Tensor, t: torch.Tensor,
                 y: torch.Tensor, x: torch.Tensor, size: int,
                 pixels: bool = False) -> MaskedOlmoEarthSample:
    """N single-frame crops of a packed batch: crop n is the `size` x `size` pixel block at
    block-grid position (y[n], x[n]) of frame t[n] of sample b[n] -> a batch of N samples of
    size x size pixels and T = 1 (a high-res lookup at one location and date).
    pixels: (y, x) is the block's top-left PIXEL instead (a crop off the block grid)."""
    step = 1 if pixels else size
    r = y[:, None] * step + torch.arange(size, device=y.device)                 # (N, size)
    c = x[:, None] * step + torch.arange(size, device=x.device)

    def crop(k, v):
        if k == "timestamps":                                                   # (B, T, 3)
            return v[b, t][:, None]
        if v.ndim != 5:
            raise ValueError(f"crop_patches: {k} is not (B, H, W, T, C)")
        return v[b[:, None, None], r[:, :, None], c[:, None, :], t[:, None, None]][:, :, :, None]

    return _map(sample, crop)


def _cat(samples):
    """Stack along batch, group-major: [all B of sample0, all B of sample1, ...]."""
    first = samples[0].as_dict()
    return samples[0]._replace(**{k: torch.cat([getattr(s, k) for s in samples])
                                  for k in first})
