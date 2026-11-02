"""The simplification flags (model.hr_type, time_step, months_prior, second_pass): run names apart,
each flag does what it says on a real PASTIS batch. python -m exp.mixres.check_simplify"""
import torch

from exp.common.tasks import TASKS
from manyup.mixres.config import RunConfig, apply_overrides
from manyup.mixres.data import loader
from manyup.mixres.model import MixedResModel, ModelConfig
from manyup.mixres.plan import Plan


def main():
    task, dev = TASKS["pastis"], "cuda"
    assert RunConfig().name() == apply_overrides(RunConfig(), ["model.hr_type=true"]).name()
    names = {apply_overrides(RunConfig(), o).name() for o in
             ([], ["model.hr_type=false"], ["model.time_step=month"],
              ["model.months_prior=false"], ["model.attn=cross_hr", "model.second_pass=true"])}
    assert len(names) == 5, names
    b = next(iter(loader(task, "val", 4, 2, False)))
    base = MixedResModel(task, ModelConfig(attn="cross_hr")).to(dev).eval()
    sd = base.state_dict()

    def build(**kw):
        m = MixedResModel(task, ModelConfig(attn="cross_hr", **kw)).to(dev).eval()
        m.load_state_dict(sd, strict=not kw.get("second_pass"))
        return m

    t = torch.tensor([6, 7, 8, 6, 7, 8], device=dev)
    i = torch.arange(6, device=dev)
    plan = Plan(b=i % 4, t=t, y=i * 2, x=i * 2, py=i * 8, px=i * 8, patch=8)

    def run(m):
        return m(b.ms, b.ss, b.season, b.present, plan).logits.float()

    with torch.no_grad(), torch.random.fork_rng():
        ref = run(base)
        assert torch.equal(run(base), ref), "the plain model must repeat"
        ss = base._inputs(b.ms, b.ss, b.season, b.present)[1]
        S, NS = ss.timestamps.shape[:2]
        times = torch.arange(NS, device=dev).float()[None].expand(S, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert torch.equal(base.mix.embed(ss, 16, times=times).x, base.mix.embed(ss, 16).x), \
                "times = the plain slots must change nothing"
        m = build(time_step="month")
        seen = {}
        emb = m.mix.embed
        m.mix.embed = lambda *a, **k: seen.update(k) or seen.setdefault("all", []).append(k) or emb(*a, **k)
        mo = run(m)
        m.mix.embed = emb
        season_t, crop_t = seen["all"][0]["times"], seen["all"][1]["slot"]
        print("seasons of sample 0:", b.season[0].tolist(), "-> times", season_t[0].tolist())
        print("crop slots", crop_t.tolist())
        assert crop_t.tolist() == t.tolist()
        print("mean |logit change| month steps", (mo - ref).abs().mean().item())
        assert torch.equal(run(build(hr_type=False)), ref), "zero at init: as the plain model"
        print("mean |logit change| no prior", (run(build(months_prior=False)) - ref).abs().mean().item())
        p2 = build(second_pass=True)
        p2.mix.sync_second_pass()
        assert torch.equal(run(p2), ref), "a synced second pass must be the plain model"
    n = lambda x: sum(p.numel() for p in x.parameters())                       # noqa: E731
    print(f"params {n(base) / 1e6:.2f}M -> {n(p2) / 1e6:.2f}M")
    frozen = build(hr_type=False)
    assert not frozen.mix.hr_type.requires_grad
    p2.train()
    p2(b.ms, b.ss, b.season, b.present, plan).logits.float().mean().backward()
    assert p2.mix.second_pass[-1].attn.q.weight.grad.abs().sum() > 0
    assert p2.mix.encoder.blocks[-1].attn.k.weight.grad is not None
    assert p2.mix.encoder.blocks[-1].attn.q.weight.grad.abs().sum() > 0
    # one_pass: picks and crops share one season pass, gradients reach the season blocks once
    op = MixedResModel(task, ModelConfig(attn="cross_hr", one_pass=True)).to(dev)
    op.load_state_dict(sd)
    op.train()
    assert op.reuses_pass
    o0 = op.uniform(b.ms, b.ss, b.season, b.present)
    assert o0.lr_F.requires_grad
    op(b.ms, b.ss, b.season, b.present, plan, None, None, o0.seasons, o0.lr_enc).logits.float().mean().backward()
    assert op.mix.encoder.blocks[0].attn.q.weight.grad.abs().sum() > 0
    op.eval()
    assert not op.uniform(b.ms, b.ss, b.season, b.present).lr_F.requires_grad
    # detach_lr + second_pass: the blended loss never reaches the encoder's blocks
    dl = MixedResModel(task, ModelConfig(attn="cross_hr", second_pass=True, detach_lr=True)).to(dev)
    dl.load_state_dict(sd, strict=False)
    dl.train()
    dl(b.ms, b.ss, b.season, b.present, plan).logits.float().mean().backward()
    assert all(p.grad is None or not p.grad.any() for p in dl.mix.encoder.blocks.parameters())
    assert dl.mix.second_pass[-1].attn.q.weight.grad.abs().sum() > 0
    print("PASS")


if __name__ == "__main__":
    main()
