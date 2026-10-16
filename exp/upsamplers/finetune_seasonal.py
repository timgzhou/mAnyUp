"""Fine-tune seasonal timAnyUp end to end on the task loss (backbone included).

    seasonal lr   OlmoEarth ps16 tile64 on the season composites (GroupSteps(task.season),
                  mid-season timestamps)                                   -> (B, G, 4, 4, D)
    upsample      mAnyUp (init: a trained per-timestep checkpoint) upsamples every MONTH from
                  its season's tokens with the month's own guidance        -> (B, T, 16, 16, D)
    lookups       k locations (flat over month x 16 x 16) chosen by --selector; each is encoded
                  by the SAME OlmoEarth at ps4 from one 4x4 px crop of that month, and fused
                  into the upsample by a ResidualFuser (identity at init)
    head          the task's pa2px head (time mean over the upsampled months, or post - pre)

Only months with guidance of their own dates are upsampled (guidance_present); the head
averages over those. Gradients reach the backbone through both the seasonal pass and the crops:
ONE set of FlexiViT weights learns the coarse summary (ps16) and the localized lookup (ps4).

--selector (arm names, see arm_label):
    none          no lookups                                 "S4ps16->M12"
    random        k random (month, patch) slots              "... + k x ps4-1mo @ random"
    attn_kl       KL(A_month || season-mean A), untrained    "... @ attnKL"
    learned_attn  MLP on attention-divergence statistics     "... @ learned(attn stats)"
    learned_feat  MLP on the upsampled feature vector        "... @ learned(upsampled feature)"
(old names kl_mean / attn / content are aliases, kept so earlier runs resume under their names.)
A learned selector starts from an exp.upsamplers.train_seasonal checkpoint (--selector_ckpt) or
from scratch, and is either FROZEN or, with --train_selector, trained JOINTLY with everything
else on the task loss alone. Top-k has no gradient, so a jointly trained selector
    - gates every lookup by its own score: out = up + sigmoid(score) * (fuser(up, hr) - up),
      so the task loss moves a slot's score by whether its lookup helped;
    - selects by Gumbel-top-k while training (exploration: slots never picked get no
      gradient), plain top-k in evaluation.
Only the k selected slots are ever encoded, in every arm. The fuser starts at the identity.
The upsampler is FROZEN unless --train_upsampler: its full-resolution guidance encoder costs
more than the tiny backbone, and with frozen weights autograd skips it (gradients still reach
the backbone through the attention-weighted values). Guidance is built in the loader workers.
OlmoEarth's fine-tuning protocol, as exp.olmoearth.train fft:
backbone frozen for the first 20% of epochs, then everything at lr x 0.1; ReduceLROnPlateau on
the validation metric; the best validation epoch is tested. Resumable per epoch; --max_hours
stops cleanly, so a run goes as a chain of short jobs. One row in results/<task>/seasonal_fft.csv,
with the lookups actually spent per sample.

    python -u -m exp.upsamplers.finetune_seasonal --task pastis --epochs 50 --selector kl_mean \\
        --upsampler_ckpt checkpoints/upsamplers/manyup_t_pastis_tiny_ep20.pt
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from exp.common import olmo_bootstrap  # noqa: E402,F401  (OlmoEarth import shims; see module)
olmo_bootstrap.apply()

import argparse  # noqa: E402
import copy  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from exp.common.config import MODEL_SIZE_TO_ID  # noqa: E402
from exp.common.heads import Pa2PxHead  # noqa: E402
from exp.common.tasks import TASKS, Task  # noqa: E402
from exp.olmoearth.train import (FREEZE_EPOCH_FRACTION, SCHEDULER_COOLDOWN,  # noqa: E402
                                 SCHEDULER_FACTOR, SCHEDULER_PATIENCE, UNFREEZE_LR_FACTOR,
                                 _append, _brief)
from exp.upsamplers.task_data import OUT_GRID  # noqa: E402
from manyup.data import GroupSteps, mtmm_collate  # noqa: E402
from manyup.data.guidance import build_guidance_batch, guidance_present  # noqa: E402
from manyup.encoders.olmoearth import (OlmoEarthEncoder, OlmoEarthInput,  # noqa: E402
                                       _to_device, crop_patches)
from manyup.seasonal import (AttentionSelector, ContentSelector,  # noqa: E402
                             KLMeanSelector, RandomSelector, ResidualFuser,
                             SeasonalContext, season_index, season_mean, select,
                             upsample_with_attention)
from manyup.timAnyUp import topk_mask  # noqa: E402
from manyup.upsamplers import MAnyUp  # noqa: E402

SELECTORS = {"none": None, "random": RandomSelector, "attn_kl": KLMeanSelector,
             "learned_attn": AttentionSelector, "learned_feat": ContentSelector}
ALIASES = {"kl_mean": "attn_kl", "attn": "learned_attn", "content": "learned_feat"}
LEARNED = ("learned_attn", "learned_feat")
SELECTOR_LABEL = {"random": "random", "attn_kl": "attnKL", "learned_attn": "learned(attn stats)",
                  "learned_feat": "learned(upsampled feature)"}
CKPT_KEY = {"learned_attn": "attn", "learned_feat": "content"}   # train_seasonal's names


def arm_label(selector: str, k: int, joint: bool, G: int, T: int, unit: str = "slot",
              train_upsampler: bool = False, frozen_query: bool = False) -> str:
    """What an arm IS: G seasons at ps16 -> T months (M) or dates (D, GEOID's pre/post) on
    the ps4 grid, plus k lookups of one ps4 token of one month, placed by the named selector,
    and what was trained. E.g. "S4ps16->M12 + 512xps4-1mo @ attnKL".
      "frozen query"  OlmoEarth, mAnyUp, selector and fuser frozen; only the head trains
      "full FT"       OlmoEarth (both paths), mAnyUp, selector, fuser and head all train
      "joint"         selector trained with OlmoEarth + fuser + head; mAnyUp frozen
      "frozen"        learned selector frozen; OlmoEarth + fuser + head train"""
    sel = ALIASES.get(selector, selector)
    base = f"S{G}ps16->{'M' if T > 2 else 'D'}{T}"
    if sel == "none":
        return base + (", frozen query" if frozen_query else
                       ", FT incl. mAnyUp" if train_upsampler else "")
    how = ("frozen query" if frozen_query else "full FT" if joint and train_upsampler
           else "joint" if joint else "frozen" if sel in LEARNED else "")
    if train_upsampler and how != "full FT":
        how = (how + ", " if how else "") + "mAnyUp trained"
    d = "M" if T > 2 else "D"
    look = f"{k}xps4-1{d.lower()}" if unit == "slot" else f"{k // T}locx{T}{d}-ps4"
    return f"{base} + {look} @ {SELECTOR_LABEL[sel]}" + (f", {how}" if how else "")
LOOKUP_PS = 4                   # a lookup is one ps4 token: a 4 x 4 px crop
CROP_CHUNK = 16384              # crops per encoder call


class Paired(Dataset):
    """Each sample twice: as the task reads it (months) and with its steps grouped."""

    def __init__(self, ds, group):
        """group: steps per season (GroupSteps), or any TemporalPolicy (e.g. Pool months)."""
        self.ds, self.group = ds, GroupSteps(group) if isinstance(group, int) else group

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        spec = self.ds.SPEC.modalities
        return s, replace(s, modalities={m: self.group(md, spec[m])
                                         for m, md in s.modalities.items()})


class Collate:
    """In the workers: both batches packed for OlmoEarth, which months are upsampled (own-date
    guidance, a season with data), their guidance, and the task target."""

    def __init__(self, task: Task, spec):
        m, imp = task.olmoearth.modalities, task.olmoearth.impute
        self.task, self.spec, self.ref = task, spec, next(iter(m))
        self.month_in = OlmoEarthInput(m, imp)
        self.season_in = OlmoEarthInput(m, imp, timestamp="mid")

    def __call__(self, items):
        months = mtmm_collate([a for a, _ in items])
        seasons = mtmm_collate([b for _, b in items])
        ms, m_valid = self.month_in.to_olmoearth(months)
        ss, s_valid = self.season_in.to_olmoearth(seasons)
        ref, sref = months.modalities[self.ref], seasons.modalities[self.ref]
        t0, t1 = ref.t_start, ref.t_end
        season = season_index(sref.t_start, sref.t_end, t0, t1)                # (B, T)
        present = (m_valid & (season >= 0) & guidance_present(months, self.spec, t0, t1)
                   & s_valid.gather(1, season.clamp(min=0)))
        guide = build_guidance_batch(months, self.spec, match="nearest", f_start=t0, f_end=t1)
        return ms, ss, season, present, guide, self.task.target(months)


class SeasonalFT(nn.Module):
    def __init__(self, task: Task, model_size: str, upsampler_ckpt: str, selector: str,
                 k_frac: float, fuser_hidden: int, train_upsampler: bool = False,
                 selector_ckpt: str | None = None, train_selector: bool = False,
                 selector_hidden: int = 64, select_unit: str = "slot"):
        super().__init__()
        m, imp = task.olmoearth.modalities, task.olmoearth.impute
        self.lr_enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID[model_size], OlmoEarthInput(m, imp),
                                       16, 64, name=model_size)
        # the lookups run the SAME weights at ps4 on 4 x 4 px crops
        self.hr_enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID[model_size], OlmoEarthInput(m, imp),
                                       LOOKUP_PS, LOOKUP_PS, name=model_size)
        self.hr_enc.encoder = self.lr_enc.encoder
        up = MAnyUp.from_checkpoint(upsampler_ckpt, frozen=False)
        if up.time_matching == "pooled":
            raise SystemExit(f"{upsampler_ckpt} is time-pooled; need a per-timestep mAnyUp")
        self.spec, self.upsampler = up.guidance, up.model
        self.upsampler.requires_grad_(train_upsampler)
        D = self.lr_enc.encoder.embedding_size
        self.fuser = ResidualFuser(D, fuser_hidden)          # (a --fuser_ckpt reloads it)
        self.head = Pa2PxHead(task.reduction, D, task.num_outputs, 64 // OUT_GRID)
        selector = ALIASES.get(selector, selector)
        self.selector, self.train_selector = None, train_selector
        if select_unit not in ("slot", "location"):
            raise ValueError(f"select_unit {select_unit!r} not in ('slot', 'location')")
        self.unit = select_unit
        if selector in LEARNED:
            ck = (torch.load(selector_ckpt, map_location="cpu", weights_only=False)
                  if selector_ckpt else None)
            hid = ck["selector_hidden"] if ck else selector_hidden
            self.selector = (AttentionSelector(hid) if selector == "learned_attn"
                             else ContentSelector(D, hid))
            if ck:
                pre = f"{CKPT_KEY[selector]}."
                self.selector.load_state_dict({k[len(pre):]: v for k, v in
                                               ck["selectors"].items() if k.startswith(pre)})
            self.selector.requires_grad_(train_selector)
        elif train_selector:
            raise ValueError(f"--train_selector needs a learned selector, not {selector!r}")
        elif SELECTORS[selector]:
            self.selector = SELECTORS[selector]()
        self.k_frac, self.k, self.GT = k_frac, 0, (0, 0)

    def backbone(self):
        return self.lr_enc.encoder

    def forward(self, ms, ss, season, present, guide):
        """-> (logits, present (B, T), lookups spent)"""
        device = self.head.linear.weight.device
        lr = self.lr_enc(ss, "series").float()                                  # (B,G,h,w,D)
        B, T = present.shape
        G, D = lr.shape[1], lr.shape[-1]
        self.GT = (G, T)
        size = (OUT_GRID, OUT_GRID)
        guide = guide.to(device, non_blocking=True)
        lr = lr.permute(0, 1, 4, 2, 3)                                          # (B,G,D,h,w)
        h, w = lr.shape[-2:]
        feats = lr[torch.arange(B, device=device)[:, None], season.clamp(min=0).to(device)]
        sel = present.flatten().to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, a = upsample_with_attention(self.upsampler, feats.flatten(0, 1)[sel],
                                           guide.flatten(0, 1)[sel], size)
        up = torch.zeros(B * T, *size, D, device=device, dtype=o.dtype)
        up = up.index_put((sel.nonzero()[:, 0],), o.permute(0, 2, 3, 1))
        up = up.reshape(B, T, *size, D).float()                                # (B,T,H,W,D)
        present = present.to(device)
        k = round(self.k_frac * T * size[0] * size[1]) if self.selector is not None else 0
        n_look = 0
        if k:
            attn = torch.zeros(B * T, a.shape[1], h * w, device=device)
            attn[sel] = a.detach().float()
            attn = attn.reshape(B, T, -1, h * w)
            ctx = SeasonalContext(up.detach().permute(0, 1, 4, 2, 3), attn,
                                  season_mean(attn, season.to(device), present, G), present)
            self.k = k
            if self.train_selector:
                scores = self.selector(ctx).float()                            # with grad
            else:
                with torch.no_grad():
                    scores = self.selector(ctx).float()
            pick = scores.detach()
            explore = self.train_selector and self.training   # Gumbel-top-k: sample, not argmax
            gumbel = lambda x: x - torch.log(-torch.log(                      # noqa: E731
                torch.rand_like(x).clamp(1e-6, 1 - 1e-6)))
            where = present[:, :, None, None].expand_as(pick)
            if self.unit == "slot":                    # k (month, patch) slots
                mask = select(gumbel(pick) if explore else pick, present, k)   # (B,T,H,W)
            else:                                      # k // T patches, every present month
                loc = pick.masked_fill(~where, float("-inf")).amax(1)          # (B, H, W)
                loc = gumbel(loc) if explore else loc
                mask = topk_mask(loc[:, None], max(1, k // T)) & where
            b, t, y, x = mask.nonzero(as_tuple=True)
            n_look = len(b)
            if n_look:
                # in chunks: one call over every crop of a dense run (65k at k = 1 on GEOID)
                # exceeds a CUDA launch limit ("invalid configuration argument")
                msd = _to_device(ms, device, torch.bfloat16)
                hr = torch.cat([self.hr_enc(crop_patches(msd, b[i:i + CROP_CHUNK],
                                                         t[i:i + CROP_CHUNK], y[i:i + CROP_CHUNK],
                                                         x[i:i + CROP_CHUNK], LOOKUP_PS),
                                            "series")[:, 0, 0, 0].float()
                                for i in range(0, n_look, CROP_CHUNK)])        # (N, D)
                v = up[b, t, y, x]
                fused = self.fuser(v[:, None, :, None, None],
                                   hr[:, None, :, None, None])[:, 0, :, 0, 0]
                if self.train_selector:        # the task loss reaches the scores through this
                    fused = v + torch.sigmoid(scores[b, t, y, x])[:, None] * (fused - v)
                up = up.index_put((b, t, y, x), fused)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.head(up, present)
        return logits.float(), present, n_look


def loader(task, spec, split, batch_size, num_workers, shuffle):
    return DataLoader(Paired(task.dataset(split), task.season), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, collate_fn=Collate(task, spec),
                      persistent_workers=num_workers > 0, drop_last=shuffle, pin_memory=True)


def to_device(batch, device):
    ms, ss, season, present, guide, target = batch
    return ms, ss, season, present, guide, target.to(device)


@torch.no_grad()
def evaluate(model, task, dl, limit):
    model.eval()
    meter, looks, n = task.meter(), 0, 0
    for i, batch in enumerate(dl):
        if limit and i >= limit:
            break
        *x, target = to_device(batch, "cuda")
        logits, _, nl = model(*x)
        meter.update(logits, target)
        looks, n = looks + nl, n + target.shape[0]
    return meter.compute(), looks / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--upsampler_ckpt", required=True)
    ap.add_argument("--selector", required=True, choices=list(SELECTORS) + list(ALIASES))
    ap.add_argument("--selector_ckpt", default=None,
                    help="learned selectors: start from this exp.upsamplers.train_seasonal "
                         "checkpoint (else from scratch; a frozen one needs it)")
    ap.add_argument("--train_selector", action="store_true",
                    help="learned selectors: train jointly on the task loss (score gate + "
                         "Gumbel-top-k); default frozen")
    ap.add_argument("--selector_hidden", type=int, default=64, help="from-scratch selector")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--select_unit", default="slot", choices=("slot", "location"),
                    help="slot: k (month, patch) slots. location: k // T patches, each looked "
                         "up in EVERY present month -- what a diff (post - pre) or time-mean "
                         "head can use; a location's score is its max over months")
    ap.add_argument("--k_frac", type=float, default=1 / 6,
                    help="lookups per sample, as a fraction of months x 16 x 16")
    ap.add_argument("--fuser_hidden", type=int, default=0)
    ap.add_argument("--frozen_query", action="store_true",
                    help="the frozen baseline: OlmoEarth never unfrozen, mAnyUp and selector "
                         "frozen, fuser taken from --fuser_ckpt and frozen; only the head trains")
    ap.add_argument("--fuser_ckpt", default=None,
                    help="start the fuser from this exp.upsamplers.train_seasonal checkpoint")
    ap.add_argument("--train_upsampler", action="store_true",
                    help="fine-tune mAnyUp too (default: frozen)")
    ap.add_argument("--model_size", default="tiny", choices=list(MODEL_SIZE_TO_ID))
    ap.add_argument("--batch_size", type=int, default=None, help="default: the task's ps16")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=None)
    a = ap.parse_args()

    task = TASKS[a.task]
    torch.manual_seed(a.seed)
    bs = a.batch_size or task.batch_size[16]
    sel = ALIASES.get(a.selector, a.selector)
    k_tag = "" if sel == "none" else f"_k{a.k_frac:.3g}" + (
        "_loc" if a.select_unit == "location" else "")
    # the RAW --selector string names the run, so runs started under an alias still resume
    name = (f"seasonal_fft_{a.task}_{a.model_size}_{a.selector}{k_tag}"
            + ("" if sel not in LEARNED else "_joint" if a.train_selector else "")
            + ("" if sel not in LEARNED or a.selector_ckpt else "_scratch")
            + (f"_fh{a.fuser_hidden}" if a.fuser_hidden else "")
            + ("_tu" if a.train_upsampler else "") + ("_frozenq" if a.frozen_query else "")
            + f"_lr{a.lr:g}_ep{a.epochs}")
    smoke = "_smoke" if a.limit_batches else ""
    state_path = Path(f"checkpoints/{name}{smoke}_state.pt")
    if state_path.exists() and torch.load(state_path, weights_only=False).get("done"):
        print(f"{state_path} says this run is finished; nothing to do")
        return
    if a.selector_ckpt and sel not in LEARNED:
        raise SystemExit("--selector_ckpt is for a learned selector")
    if sel in LEARNED and not (a.selector_ckpt or a.train_selector):
        raise SystemExit("a frozen learned selector needs --selector_ckpt")
    model = SeasonalFT(task, a.model_size, a.upsampler_ckpt, sel, a.k_frac, a.fuser_hidden,
                       a.train_upsampler, a.selector_ckpt, a.train_selector,
                       a.selector_hidden, a.select_unit).cuda()
    dl = {s: loader(task, model.spec, s, bs, a.num_workers, s == "train")
          for s in ("train", "val", "test")}
    better = task.meter().higher_is_better
    # one optimizer over everything; the backbone has no gradients until it is unfrozen
    if a.frozen_query and (a.train_upsampler or a.train_selector or not a.fuser_ckpt):
        raise SystemExit("--frozen_query trains the head only: it needs --fuser_ckpt and "
                         "excludes --train_upsampler / --train_selector")
    if a.fuser_ckpt:
        fk = torch.load(a.fuser_ckpt, map_location="cpu", weights_only=False)
        if fk["fuser_hidden"] != a.fuser_hidden:
            raise SystemExit(f"{a.fuser_ckpt} has fuser_hidden {fk['fuser_hidden']}")
        model.fuser.load_state_dict(fk["fuser"])
    if a.frozen_query:
        model.fuser.requires_grad_(False)
    frozen = ("upsampler." if not a.train_upsampler else None,
              "lr_enc." if a.frozen_query else None, "fuser." if a.frozen_query else None)
    opt = torch.optim.AdamW([q for n, q in model.named_parameters()
                             if not any(f and n.startswith(f) for f in frozen)], lr=a.lr)
    sched = ReduceLROnPlateau(opt, mode="max" if better else "min", factor=SCHEDULER_FACTOR,
                              patience=SCHEDULER_PATIENCE, cooldown=SCHEDULER_COOLDOWN)
    freeze_epochs = math.ceil(FREEZE_EPOCH_FRACTION * a.epochs)
    loss_fn = task.loss().cuda()
    start, hours_before, epoch_s = 0, 0.0, None
    best, best_epoch, best_val, best_state = None, -1, None, None
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, hours_before, epoch_s = st["epoch"], st["hours"], st["epoch_s"]
        best, best_epoch, best_val, best_state = (st["best"], st["best_epoch"], st["best_val"],
                                                  st["best_state"])
        print(f"Resumed from {state_path} after epoch {start}")
        if tuple(st.get("val_excluded", ())) != task.val_excluded_events and best_state:
            # best-so-far was picked on another val: re-score IT on the current val so it stays a
            # candidate (the epochs it beat cannot be re-scored; their weights are gone)
            current = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            model.load_state_dict(best_state)
            best_val, _ = evaluate(model, task, dl["val"], a.limit_batches)
            best = best_val["primary"]
            model.load_state_dict(current)
            print(f"val definition changed ({st.get('val_excluded', ())} -> "
                  f"{task.val_excluded_events}): best-so-far (epoch {best_epoch}) re-scored "
                  f"on the new val: {best:.4f}")
    print(f"Run: {name} | batch {bs} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"params (backbone {sum(p.numel() for p in model.backbone().parameters()) / 1e6:.2f}M)")
    t_start = time.time()

    for epoch in range(start, a.epochs):
        if a.max_hours and epoch_s and (time.time() - t_start + 1.1 * epoch_s) / 3600 > a.max_hours:
            print(f"STOPPED before epoch {epoch + 1} (--max_hours); rerun to resume")
            return
        e0 = time.time()
        torch.manual_seed(a.seed + epoch)
        tuning = epoch >= freeze_epochs and not a.frozen_query
        model.backbone().requires_grad_(tuning)
        if epoch == freeze_epochs and not a.frozen_query:
            for g in opt.param_groups:
                g["lr"] = a.lr * UNFREEZE_LR_FACTOR
            print(f"epoch {epoch + 1}: backbone unfrozen, lr -> {a.lr * UNFREEZE_LR_FACTOR:g}")
        model.train()
        model.upsampler.train(a.train_upsampler)
        model.fuser.train(not a.frozen_query)
        if model.selector is not None:             # frozen / parameter-free stay in eval
            model.selector.train(a.train_selector)
        model.backbone().train(tuning)
        loss_sum, n, looks, samples = 0.0, 0, 0, 0
        for i, batch in enumerate(dl["train"]):
            if a.limit_batches and i >= a.limit_batches:
                break
            *x, target = to_device(batch, "cuda")
            logits, _, nl = model(*x)
            loss = loss_fn(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum, n = loss_sum + loss.item(), n + 1
            looks, samples = looks + nl, samples + target.shape[0]
            if i % 100 == 0:
                print(f"  step {i}/{len(dl['train'])} loss {loss.item():.4f}", flush=True)
        train_s = time.time() - e0
        val, _ = evaluate(model, task, dl["val"], a.limit_batches)
        sched.step(val["primary"])
        print(f"epoch {epoch + 1}/{a.epochs} | {samples / train_s:.1f} samples/s "
              f"({train_s / 60:.1f} min) | {looks / max(samples, 1):.0f} lookups/sample | "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB\n"
              f"   train_loss {loss_sum / max(n, 1):.4f} | val {_brief(val)}", flush=True)
        if best is None or (val["primary"] > best if better else val["primary"] < best):
            best, best_epoch, best_val = val["primary"], epoch + 1, val
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        epoch_s = time.time() - e0
        tmp = state_path.with_suffix(".tmp")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch + 1,
                    "hours": hours_before + (time.time() - t_start) / 3600, "epoch_s": epoch_s,
                    "best": best, "best_epoch": best_epoch, "best_val": best_val,
                    "best_state": best_state,
                    "val_excluded": list(task.val_excluded_events)}, tmp)
        tmp.replace(state_path)

    if best_state is None:          # no epoch was ever scored (e.g. resumed with nothing left)
        best_val, _ = evaluate(model, task, dl["val"], a.limit_batches)
        best, best_epoch = best_val["primary"], a.epochs
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    test, test_looks = evaluate(model, task, dl["test"], a.limit_batches)
    hours = hours_before + (time.time() - t_start) / 3600
    arm = arm_label(sel, model.k, a.train_selector, *model.GT, a.select_unit,
                    a.train_upsampler, a.frozen_query)
    print(f"TEST {arm} (best val epoch {best_epoch}, val {best:.4f}): {_brief(test)} | "
          f"{test_looks:.0f} lookups/sample")
    if not a.limit_batches:
        torch.save({"args": vars(a), "model": best_state}, f"checkpoints/{name}.pt")
    _append(Path(f"results/{a.task}/seasonal_fft{smoke}.csv"), {
        "job": os.environ.get("SLURM_JOB_ID", ""), "run": name, "arm": arm, "task": a.task,
        "model_size": a.model_size, "selector": sel, "train_selector": a.train_selector,
        "k_frac": a.k_frac if a.selector != "none" else 0, "select_unit": a.select_unit,
        "fuser_hidden": a.fuser_hidden,
        "train_upsampler": a.train_upsampler, "frozen_query": a.frozen_query,
        "fuser_ckpt": a.fuser_ckpt or "", "selector_ckpt": a.selector_ckpt or "",
        "upsampler_ckpt": a.upsampler_ckpt, "lr": a.lr, "epochs": a.epochs, "batch_size": bs,
        "best_epoch": best_epoch, "val_excluded": " ".join(task.val_excluded_events),
        "hours": round(hours, 2),
        "test_lookups_per_sample": round(test_looks, 1),
        **{f"val_{k}": round(v, 5) for k, v in best_val.items()},
        **{f"test_{k}": round(v, 5) for k, v in test.items()}})
    torch.save({"done": True}, state_path)


if __name__ == "__main__":
    main()
