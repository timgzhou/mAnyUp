"""SeasonResidualScorer at r = 0 equals the plain lowest-entropy time selector: identical picks and
all crop gates exactly 0, on the trained lowest-entropy one-season models (both setups).

    python -m exp.mixres.check_season_residual
"""
from exp.common import olmo_bootstrap
olmo_bootstrap.apply()
import torch
from exp.mixres.time_diag import ARMS, load, run_name
from manyup.mixres.data import loader
from manyup.mixres.residual import SeasonResidual, window_pixels
from manyup.mixres.selectors import (Context, Entropy, EveryLocation, LR_PS, TILE,
                                     SeasonResidualScorer, Selector)
for ps in (16, 8):
    task, model, sel = load(run_name(ps, ARMS["entropy"]), 32, 4)
    b0 = next(iter(loader(task, "val", 32, 4, False)))
    for arch in ("conv", "xattn_conv"):
        net = SeasonResidual(model.mix.encoder.embedding_size, model.head.linear.out_features // LR_PS ** 2,
                             arch=arch, C=window_pixels(b0.ms, 1).shape[-1], P=sel.geo.crop_px,
                             S=TILE // sel.geo.crop_px, gh=TILE // model.cfg.lr_patch,
                             NS=b0.ss.timestamps.shape[1]).cuda().eval()
        a = Selector(sel.geo, Entropy(), EveryLocation("one"))
        r = Selector(sel.geo, SeasonResidualScorer(net), EveryLocation("one"))
        same = gates = True
        with torch.no_grad():
            for i, b in enumerate(loader(task, "val", 32, 4, False)):
                if i >= 3: break
                sa, sr = a(Context.from_batch(model, b)), r(Context.from_batch(model, b))
                same &= bool((sa.mask == sr.mask).all()); gates &= bool((sr.logw == 0).all())
        print(f"ps{ps} {arch}: picks == lowest-entropy {same} | gates all 0 {gates}", flush=True)
