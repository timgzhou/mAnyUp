"""python -m exp.geoidchange.run name=x mode=pair ps=16 ... [split=val] [--bi]"""
import logging
import sys

from exp.geoidchange.harness import Cfg, eval_bi, log, report, train

logging.disable(logging.INFO)


def main(argv):
    kw = dict(a.split("=", 1) for a in argv if "=" in a)
    split = kw.pop("split", "val")
    typed = {k: type(Cfg.__dataclass_fields__[k].default)(v) for k, v in kw.items()}
    cfg = Cfg(**typed)
    net = train(cfg)
    if cfg.mode == "bi":
        res = {"val_iou": round(eval_bi(net, "val"), 4), "test_iou": round(eval_bi(net, "test"), 4)}
    else:
        res, _ = report(net, cfg, split)
    log(cfg, res)


if __name__ == "__main__":
    main(sys.argv[1:])
