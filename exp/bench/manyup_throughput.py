"""Measure frozen mAnyUp upsampler throughput across (modality, LR grid, output grid).

Companion to olmo_throughput.py, which benches the ENCODER. This benches what runs on top
of it: the guidance-driven upsampler that turns a coarse (gh,gw) token map into a finer one.

Why it needs its own bench: mAnyUp's cost is driven by the CROSS-ATTENTION between output
queries and LR keys, so it scales with (out_h*out_w) x (gh*gw) -- not with the encoder's
knobs. A ps16->ps4 checkpoint asked for a 64x64 output does 16x the query work it was
trained for, and that is invisible in the encoder numbers.

Axes:
  modality  s2 (13-band guidance) | s1 (2) | s2s1 (15). Guidance band count is the only
            architectural difference between the arms -- it changes the image encoder's
            first conv, so the arms are genuinely different models, not just different data.
  ckpt      the "164" (ps16->ps4) and "84" (ps8->ps4) upsamplers, i.e. LR grids 4x4 and 8x8.
  out_size  the grid we ASK for, swept independently of what the ckpt was trained to emit.
            Includes each ckpt's native target (16x16) and the label resolution (64x64)
            that lp_cached_features.py requests by default.

DUMMY inputs throughout (shaped exactly like PASTIS: 64x64 px guidance, D=768 features), so
no dataset/IO is in the loop -- this measures the upsampler alone.

    source env_setup/env_olmo.sh
    python -u -m exp.bench.manyup_throughput --out results/bench/manyup_throughput.csv
"""
import argparse
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

CKPT_TMPL = ("checkpoints/manyup/oe_base_{m}_ps{lr}_tile64__to__oe_base_{m}_ps4_tile64/"
             "manyup_oe_base_{m}_ps{lr}_tile64_to_oe_base_{m}_ps4_tile64_ep31.pth")
# Guidance is at image resolution (64x64) regardless of the feature grid, mirroring how
# lp_cached_features.CachedManyUp.features() feeds it.
IMG = 64
EMBED = 768
CSV_COLUMNS = ["modality", "guidance_bands", "ckpt", "lr_patch", "lr_grid", "out_grid",
               "upsample_factor", "batch_size", "samples_per_s", "ms_per_sample",
               "step_s_median", "step_s_mean", "step_s_std", "peak_mem_gb", "gpu", "params_m"]


def load_upsampler(path: str, device):
    """Rebuild the checkpoint's architecture and load it, mirroring CachedManyUp.__init__."""
    sys.path.insert(0, "/scratch/timz/mAnyUp/third_party/anyup")
    from anyup.model import AnyUp
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "anyup")
    cls = AnyUp
    if arch == "manyup":
        from manyup.mAnyUp import mAnyUp as cls  # noqa: F811
    kw = dict(input_dim=ck["input_dim"], qk_dim=ck.get("qk_dim", 128),
              window_ratio=ck.get("window_ratio", 0.1))
    if arch == "manyup":
        kw["feat_dim"] = ck.get("feat_dim", EMBED)
        kw["transform_depth"] = ck.get("transform_depth", 1)
    model = cls(**kw)
    model.load_state_dict(ck["model"])
    proj = None
    if ck.get("proj_head") is not None:
        proj = torch.nn.Conv2d(EMBED, EMBED, 1)
        proj.load_state_dict(ck["proj_head"])
        proj = proj.to(device).eval()
    return model.to(device).eval(), proj, ck


@torch.no_grad()
def run_step(model, proj, guidance, feats, out):
    hr = model(guidance, feats, (out, out))
    return proj(hr) if proj is not None else hr


def fits(model, proj, bands, lr_grid, out, batch, device) -> bool:
    try:
        g = torch.randn(batch, bands, IMG, IMG, device=device)
        f = torch.randn(batch, EMBED, lr_grid, lr_grid, device=device)
        run_step(model, proj, g, f, out)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def find_max_batch(model, proj, bands, lr_grid, out, device, cap=256, safety=0.10) -> int:
    """Double until OOM, binary-search the gap, then back off by `safety` -- same strategy as
    olmo_throughput, so the reported rate is peak but not sitting on the OOM edge."""
    if not fits(model, proj, bands, lr_grid, out, 1, device):
        return 0
    lo, hi = 1, 1
    while hi < cap and fits(model, proj, bands, lr_grid, out, min(hi * 2, cap), device):
        lo, hi = hi, min(hi * 2, cap)
    if hi >= cap:
        lo = cap
    else:
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if fits(model, proj, bands, lr_grid, out, mid, device):
                lo = mid
            else:
                hi = mid
    return max(1, int(lo * (1 - safety)))


def bench(model, proj, bands, lr_grid, out, device, args) -> dict | None:
    batch = args.batch_size or find_max_batch(model, proj, bands, lr_grid, out, device)
    if batch == 0:
        print(f"    out{out}: OOM at batch 1 -- skipped", flush=True)
        return None
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    g = torch.randn(batch, bands, IMG, IMG, device=device)
    f = torch.randn(batch, EMBED, lr_grid, lr_grid, device=device)
    for _ in range(args.warmup):
        run_step(model, proj, g, f, out)
    torch.cuda.synchronize()
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_step(model, proj, g, f, out)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {
        "batch_size": batch,
        "samples_per_s": batch / med,
        "ms_per_sample": med / batch * 1000,
        "step_s_median": med,
        "step_s_mean": statistics.mean(times),
        "step_s_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/bench/manyup_throughput.csv")
    p.add_argument("--modalities", nargs="+", default=["s2", "s1", "s2s1"])
    p.add_argument("--lr_patches", nargs="+", type=int, default=[16, 8],
                   help="LR patch size of the checkpoint: 16 -> the '164' ckpt, 8 -> '84'")
    p.add_argument("--out_grids", nargs="+", type=int, default=[8, 16, 32, 64],
                   help="output token grids to request (16 = ckpt native, 64 = label res)")
    p.add_argument("--batch_size", type=int, default=0, help="0 = auto-tune per config")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows = []
    for mod in args.modalities:
        for lr_ps in args.lr_patches:
            ck_path = CKPT_TMPL.format(m=mod, lr=lr_ps)
            if not Path(ck_path).exists():
                print(f"skip {mod} ps{lr_ps}: no checkpoint at {ck_path}", flush=True)
                continue
            model, proj, ck = load_upsampler(ck_path, device)
            bands = ck["input_dim"]
            lr_grid = IMG // lr_ps                  # ps16 -> 4x4, ps8 -> 8x8
            n_par = sum(q.numel() for q in model.parameters()) / 1e6
            print(f"\n=== {mod} ps{lr_ps} (LR {lr_grid}x{lr_grid}, {bands}-band guidance, "
                  f"{n_par:.1f}M params) ===", flush=True)
            for out in args.out_grids:
                if out < lr_grid:
                    continue                        # upsampling only
                r = bench(model, proj, bands, lr_grid, out, device, args)
                if r is None:
                    continue
                r.update(modality=mod, guidance_bands=bands, ckpt=Path(ck_path).name,
                         lr_patch=lr_ps, lr_grid=lr_grid, out_grid=out,
                         upsample_factor=out // lr_grid, gpu=gpu, params_m=round(n_par, 2))
                rows.append(r)
                print(f"    {lr_grid}x{lr_grid} -> {out}x{out} ({out // lr_grid}x): "
                      f"{r['samples_per_s']:8.1f} samp/s | {r['ms_per_sample']:7.3f} ms/sample "
                      f"| bs={r['batch_size']:3d} | peak {r['peak_mem_gb']:.1f} GB", flush=True)
            del model, proj
            gc.collect()
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
