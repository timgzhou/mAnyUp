"""Where does a trained learned selector put its crops, vs random picks on the same samples?

For a finetune_inject checkpoint with --selector learned, on the val split:
  months     entropy of the crops over present months / log(present months) (1 = even) and the
             fraction of present months that get any crop
  cells      distinct (y, x) cells among the picks / picks (1 = no cell picked twice)
  neighbours mean distance (ps4 cells) from a pick to the nearest other pick of the same month
  gate       mean sigmoid(score) of the picks (the learned arms scale crops by it)
Random = the same number of picks drawn uniformly over the same present slots.

    python -u -m exp.upsamplers.analyze_inject_picks checkpoints/inject_fft_pastis_..._ep50.pt
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.upsamplers.finetune_inject import InjectFT, loader  # noqa: E402
from manyup.seasonal import select  # noqa: E402


def stats(mask: torch.Tensor, present: torch.Tensor) -> dict:
    """mask (B, T, H, W) bool picks -> per-batch sums of the statistics."""
    B, T, H, W = mask.shape
    per_t = mask.flatten(2).sum(2).float()                                    # (B, T)
    n = per_t.sum(1).clamp(min=1)
    p = per_t / n[:, None]
    ent = -(p * p.clamp_min(1e-12).log()).sum(1)
    n_present = present.sum(1).float()
    even = ent / n_present.clamp(min=2).log()
    covered = ((per_t > 0) & present).sum(1) / n_present.clamp(min=1)
    distinct = mask.any(1).flatten(1).sum(1) / n
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pos = torch.stack([yy, xx], -1).view(-1, 2).float().to(mask.device)       # (HW, 2)
    nn_sum, nn_cnt = 0.0, 0
    for b in range(B):
        for t in range(T):
            idx = mask[b, t].flatten().nonzero()[:, 0]
            if len(idx) < 2:
                continue
            d = torch.cdist(pos[idx], pos[idx])
            d.fill_diagonal_(math.inf)
            nn_sum += d.min(1).values.sum().item()
            nn_cnt += len(idx)
    return {"even_months": even.sum().item(), "months_covered": covered.sum().item(),
            "distinct_cells": distinct.sum().item(), "nn_sum": nn_sum, "nn_cnt": nn_cnt,
            "n": B}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    task = TASKS[args["task"]]
    model = InjectFT(task, args["model_size"], args["inject_layer"], args["selector"],
                     args["k_frac"], args["select_unit"], args["selector_hidden"],
                     args.get("pick", "topk"), args.get("head", "mean")).cuda().eval()
    model.load_state_dict(ck["model"])
    tot = {"learned": {}, "random": {}}
    gate, gate_n = 0.0, 0
    bs = task.batch_size[16]
    for i, (*x, _) in enumerate(loader(task, "val", bs, a.num_workers, False)):
        if i >= a.batches:
            break
        model(*x)
        mask, scores, present = model.last_pick
        rnd = select(torch.rand_like(scores.float()), present, int(mask[0].sum()))
        if model.unit == "location":                  # same unit as the learned picks
            loc = torch.rand_like(scores[:, 0].float())
            from manyup.timAnyUp import topk_mask
            rnd = topk_mask(loc[:, None], int(mask[0].any(0).sum())) & present[:, :, None, None]
        for k, m in (("learned", mask), ("random", rnd)):
            for s, v in stats(m, present).items():
                tot[k][s] = tot[k].get(s, 0) + v
        gate += torch.sigmoid(scores[mask].float()).sum().item()
        gate_n += int(mask.sum())
    out = {}
    for k, t in tot.items():
        out[k] = {"months evenness": t["even_months"] / t["n"],
                  "months covered": t["months_covered"] / t["n"],
                  "distinct cells / picks": t["distinct_cells"] / t["n"],
                  "nearest pick (cells)": t["nn_sum"] / max(t["nn_cnt"], 1)}
    out["learned"]["mean gate sigmoid(score)"] = gate / max(gate_n, 1)
    print(Path(a.ckpt).name)
    for s in out["random"]:
        print(f"  {s:26s} learned {out['learned'][s]:.3f} | random {out['random'][s]:.3f}")
    print(f"  {'mean gate sigmoid(score)':26s} learned {out['learned']['mean gate sigmoid(score)']:.3f}")
    dest = Path(f"results/{args['task']}/inject_picks") / f"{Path(a.ckpt).stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
