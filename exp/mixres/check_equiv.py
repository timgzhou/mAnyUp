"""Step 4 of docs/restructure_mixed_res.md: manyup.mixres.MixedResModel vs the legacy InjectFT on
the same trained checkpoint and test batches, with the legacy model's own picks as the plan:
    uniform   MixedResModel.uniform vs InjectFT._crop_free (logits, season tokens)
    plan      MixedResModel(plan = legacy picks) vs InjectFT.forward (logits, crop count)
Prints the max abs difference per batch; exit 1 if any exceeds --tol.

    python -m exp.mixres.check_equiv --setup s16m8 --n 8
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers import finetune_inject as fi  # noqa: E402
from exp.upsamplers.learned_select import SETUPS, run_name  # noqa: E402
from manyup.mixres import MixedResModel, ModelConfig, Plan, load_legacy  # noqa: E402


def legacy(task, setup, n, ckpt):
    lp, cp, stride, _ = SETUPS[setup]
    m = fi.InjectFT(task, "tiny", 0, "oracle_bothalf", n / 256, "location",
                    readout="native_set", lr_patch=lp, crop_px=cp, pos="center").cuda()
    if stride:
        m.set_loc_stride(stride)
    sd = torch.load(f"checkpoints/{ckpt}.pt", map_location="cpu", weights_only=False)["model"]
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("task_loss_fn.")})
    return m.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pastis")
    ap.add_argument("--setup", default="s16m8", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    task = TASKS[a.task]
    ckpt = run_name(a.task, a.setup, a.n)
    old = legacy(task, a.setup, a.n, ckpt)
    new = MixedResModel(task, ModelConfig(lr_patch=SETUPS[a.setup][0])).cuda().eval()
    unused = load_legacy(new, ckpt)
    print(f"{ckpt}\n  legacy keys not in MixedResModel: {len(unused)} {unused[:4]}")
    dl = fi.loader(task, "test", 32, 8, False)
    worst = 0.0
    for i, (*x, _) in enumerate(dl):
        if i >= a.batches:
            break
        ms, ss, season, present = x[:4]
        dev = new.device
        # uniform pass vs the legacy crop-free pass
        u = new.uniform(ms, ss, season, present)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr = old.mix.head_blocks(old.mix.embed(fi._to_device(ss, dev, torch.bfloat16),
                                                   old.lr_ps))
            lr_F0, lg0 = old._crop_free(lr, season.to(dev), present.to(dev),
                                        ss.timestamps.shape[1])
        d_u = (u.logits - lg0).abs().max().item()
        d_uF = (u.lr_F - lr_F0).abs().max().item()
        # full pass with the legacy picks as the plan
        torch.manual_seed(0)
        old.last_pick = None
        lg1, _, nc = old(*x)
        mask = old.last_pick[0]
        plan = Plan.from_mask(mask, old.crop_px, old.loc_px)
        out = new(ms, ss, season, present, plan)
        d_p = (out.logits - lg1).abs().max().item()
        worst = max(worst, d_u, d_uF, d_p)
        print(f"  batch {i}: uniform logits {d_u:.2e} tokens {d_uF:.2e} | plan logits {d_p:.2e} "
              f"| crops legacy {nc} new {len(plan)}", flush=True)
    ok = worst <= a.tol
    print(f"{'PASS' if ok else 'FAIL'} (max diff {worst:.2e}, tol {a.tol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
