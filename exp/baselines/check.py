"""Each baseline against what it must reduce to, on one real batch (GPU, ~2 min):

    vid-TLDR    no merges = the fixed monthly checkpoint; the cluster map = upstream's source
    APT         every patch coarse = the fixed checkpoint; entropy = upstream's on an image
    upsamplers  Upsample Anything = upstream's UPA; bilinear / AnyUp / UPA run, weights free

    python -u -m exp.baselines.check [--task pastis] [--patch 8]
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.mixres.train import measure_compute, step  # noqa: E402
from manyup.mixres import MixedResModel, load_legacy  # noqa: E402
from manyup.mixres.baselines import build_model, build_selector_for  # noqa: E402
from manyup.mixres.baselines.apt import patch_entropy  # noqa: E402
from manyup.mixres.baselines.vid_tldr import merging  # noqa: E402
from manyup.mixres.config import apply_overrides, RunConfig  # noqa: E402
from manyup.mixres.data import loader  # noqa: E402
from manyup.mixres.model import ModelConfig  # noqa: E402
from manyup.upsamplers.jbu import JBU  # noqa: E402
from manyup.data.guidance import S2_RGB, GuidanceSpec  # noqa: E402


def upstream(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ok(name, good, detail=""):
    print(f"{'PASS' if good else 'FAIL'}  {name}  {detail}", flush=True)


def check_merge():
    up = upstream("references/vid-TLDR/multi_modality/utils/vid_TLDR.py", "up_tldr")
    torch.manual_seed(0)
    x = torch.randn(3, 41, 16)
    scores = torch.ones(3, 41, 1)
    # upstream's merge on the identity, the source trace (amax on the cluster's members)
    mg_up = up.merging(x, r_merge=7, score_obj=scores)
    source = mg_up(torch.eye(41)[None].expand(3, 41, 41), mode="amax")
    merge, remap = merging(x, 7, scores)
    mine = torch.zeros_like(source)
    mine.scatter_(1, remap[:, None], 1.0)
    ok("vid-TLDR cluster map = upstream source", bool((mine == source).all()))
    same = torch.allclose(merge(x), mg_up(x))
    ok("vid-TLDR merge = upstream merge", bool(same))
    a = torch.randn(2, 3, 41, 41).softmax(-1)
    from manyup.mixres.baselines.vid_tldr import get_objective_score
    ok("vid-TLDR saliency = upstream", bool(torch.allclose(get_objective_score(a),
                                                            up.get_objective_score(a))))


def check_entropy():
    torch.manual_seed(0)
    g = (torch.rand(4, 64, 64) * 255).floor()
    g[1] = g[1] / 40
    mine = patch_entropy(g, 16)
    ref = []
    for im in g:                         # upstream's compute_patch_entropy_vectorized, its loop
        ps = im.unfold(0, 16, 16).unfold(1, 16, 16).reshape(16, -1)
        h = torch.stack([torch.histc(p, bins=512, min=0, max=255) for p in ps]) / 256
        ref.append(-(h * torch.log2(h + 1e-10)).sum(1).view(4, 4))
    d = (mine - torch.stack(ref)).abs().max().item()
    ok("APT entropy = upstream", d < 1e-3, f"max diff {d:.5f}")


def check_upa():
    up = upstream("references/upsample-anything/src/upsample_anything.py", "up_upa")
    torch.manual_seed(0)
    # a smooth image with edges, one 4 x 4 feature map of 8 channels
    yy, xx = np.mgrid[0:64, 0:64]
    img = np.stack([(xx > 30) * 200 + 20, (yy > 20) * 150 + 40, (xx + yy) * 2], -1).astype(np.uint8)
    feat = torch.randn(1, 8, 4, 4).cuda()
    ref = up.UPA(img, feat).float()
    hr = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255
    jbu = JBU(GuidanceSpec({"S2": S2_RGB}, None, norm="percentile"), 50, "upa", range_sum=True)
    mine = jbu.upsample_frames(feat, hr, (64, 64))
    d = (mine - ref).abs().mean().item() / ref.abs().mean().item()
    ok("Upsample Anything = upstream", d < 0.02, f"relative mean diff {d:.4f}")


def model_of(task, over):
    cfg = apply_overrides(RunConfig(task=task.name), over)
    cfg.check(task.kind)
    return cfg, build_model(task, cfg).cuda().eval()


def fixed_logits(task, patch, ckpt, batch):
    m = MixedResModel(task, ModelConfig(lr_patch=patch)).cuda().eval()
    load_legacy(m, ckpt)
    B, T = batch.present.shape
    slots = torch.arange(T).expand(B, T).contiguous()
    return m.uniform(batch.ms, batch.ms, slots, batch.present).logits


@torch.no_grad()
def check_models(task, patch, ep):
    tail = "" if patch == 16 else f"_lrps{patch}"
    mon = f"inject_fft_{task.name}_tiny_none_native_set_lrall{tail}_lr0.001_{ep}"
    sea = f"inject_fft_{task.name}_tiny_none_native_set{tail}_lr0.001_{ep}"
    dl = loader(task, "val", 4, 2, False)
    batch = next(iter(dl))
    ref = fixed_logits(task, patch, mon, batch)
    com = [f"model.lr_patch={patch}", "train.freeze_model=1", "train.epochs=0"]

    def run(over):
        cfg, m = model_of(task, com + over)
        load_legacy(m, cfg.train.init_ckpt)
        sel = build_selector_for(task, cfg, m, None)
        return m, sel, step(m, sel, batch)[0]

    m, sel, out = run([f"train.init_ckpt={mon}", "baseline.method=tldr", "baseline.tokens=months",
                       "baseline.r=0"])
    d = (out - ref).abs().max().item()
    ok("vid-TLDR r=0 = fixed monthly", d < 5e-2 * max(ref.abs().max().item(), 1),
       f"max |diff| {d:.4f} of {ref.abs().max():.2f}")
    m, sel, out = run([f"train.init_ckpt={mon}", "baseline.method=tldr", "baseline.tokens=months"])
    print("  vid-TLDR default r:", out.shape, f"{measure_compute(m, sel, [batch], 1)}", flush=True)
    m, sel, out = run([f"train.init_ckpt={mon}", "baseline.method=apt", "baseline.tokens=months",
                       "baseline.keep=1", f"select.crop_px={patch // 2}"])
    d = (out - ref).abs().max().item()
    ok("APT all coarse = fixed monthly", d < 5e-2 * max(ref.abs().max().item(), 1),
       f"max |diff| {d:.4f} of {ref.abs().max():.2f}")
    m, sel, out = run([f"train.init_ckpt={mon}", "baseline.method=apt", "baseline.tokens=months",
                       "baseline.keep=0.5", f"select.crop_px={patch // 2}"])
    print("  APT keep .5:", out.shape, f"{measure_compute(m, sel, [batch], 1)}", flush=True)
    for up in ("bilinear", "anyup", "upa"):
        m, sel, out = run([f"train.init_ckpt={sea}", "baseline.method=upsample",
                           f"baseline.upsampler={up}"])
        print(f"  {up}:", out.shape, f"finite {bool(out.isfinite().all())}",
              f"{measure_compute(m, sel, [batch], 1)}", flush=True)
        del m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="pastis")
    ap.add_argument("--patch", type=int, default=8)
    a = ap.parse_args()
    check_merge()
    check_entropy()
    check_upa()
    check_models(TASKS[a.task], a.patch, "ep10" if a.task == "biomassters" else "ep50")


if __name__ == "__main__":
    main()
