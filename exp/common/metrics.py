"""Segmentation metrics, frozen at olmoearth-pretrain 0.1.0 semantics.

Every mIoU in results/ was computed by olmoearth_pretrain.evals.metrics as of 0.1.0. From
0.1.1 its segmentation_metrics REQUIRES per-pixel softmax scores and adds AUROC/PR-AUC over
every valid pixel (sklearn, ~8M pixels x 20 classes on PASTIS). The miou/acc/f1 math did not
change, but owning this copy keeps our numbers independent of the package version, and the
returned keys (miou, overall_acc, macro_acc, macro_f1) are exactly what the CSVs record.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE_LABEL = -1


@dataclass
class EvalResult:
    primary: float              # miou; used for model selection
    metrics: dict[str, float]   # miou, overall_acc, macro_acc, macro_f1


def _build_confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor,
                            num_classes: int, ignore_label: int = IGNORE_LABEL,
                            mask: torch.Tensor | None = None) -> torch.Tensor:
    """(num_classes, num_classes) counts; confusion[i, j] = true i predicted as j.
    predictions/labels: (N, H, W) integer class indices; ignore_label pixels are dropped.
    mask (N, H, W) bool, if given: pixels outside it are dropped too (see boundary_mask)."""
    if predictions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"predictions must be integer class indices, got {predictions.dtype}")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"labels must be integer class indices, got {labels.dtype}")
    labels = labels.to(predictions.device)
    valid = labels != ignore_label
    if mask is not None:
        valid = valid & mask.to(predictions.device)
    n = num_classes
    return torch.bincount(n * labels[valid] + predictions[valid], minlength=n**2).reshape(n, n)


def boundary_mask(labels: torch.Tensor, px: int, ignore_label: int = IGNORE_LABEL) -> torch.Tensor:
    """Pixels within `px` (chebyshev distance, native pixels) of a label boundary, per sample
    (Cheng et al. 2021, "Boundary IoU"). labels: (N, H, W) integer class indices.

    A pixel is interior (not a boundary pixel) iff every neighbour in its (2px+1)^2 window
    holds the same label; erosion is a one-hot min-pool so the cost is one pass per class
    actually present, not one per class in the task. Ignored pixels don't count as "same
    label" to either side, so the sample edge and ignored regions read as boundary too (both
    get dropped below with the usual ignore-label mask, as in the mIoU confusion matrix).
    """
    if px <= 0:
        return torch.ones_like(labels, dtype=torch.bool)
    k, pad = 2 * px + 1, px
    erode_min = lambda x: -torch.nn.functional.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)

    interior = torch.ones_like(labels, dtype=torch.bool)
    for c in labels[labels != ignore_label].unique():
        is_c = (labels == c).float().unsqueeze(1)
        eroded_c = erode_min(is_c).squeeze(1) > 0.5          # 1 only where the whole window is c
        interior &= (labels != c) | eroded_c
    return ~interior & (labels != ignore_label)


def segmentation_metrics(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int,
                         ignore_label: int = IGNORE_LABEL) -> EvalResult:
    """mIoU (over classes present in labels or predictions), overall accuracy, macro
    accuracy and macro F1 (both over classes present in labels)."""
    confusion = _build_confusion_matrix(predictions, labels, num_classes, ignore_label)
    tp = confusion.diagonal().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp

    union = tp + fp + fn
    miou = (tp / (union + 1e-8))[union > 0].mean().item()
    overall_acc = (tp.sum() / (confusion.sum() + 1e-8)).item()

    class_totals = tp + fn
    present = class_totals > 0
    macro_acc = (tp / (class_totals + 1e-8))[present].mean().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1 = f1[present].mean().item()

    return EvalResult(primary=miou, metrics={"miou": miou, "overall_acc": overall_acc,
                                             "macro_acc": macro_acc, "macro_f1": macro_f1})


# ---- streaming meters (exp.olmoearth.train, exp.utae.train) --------------------------------
class SegMeter:
    """Confusion matrix over batches of logits (B, C, H, W) and targets (B, H, W).
    primary = mIoU over `headline` classes (all classes if None), as segmentation_metrics
    counts them (classes absent from both labels and predictions are skipped).

    boundary_px: also accumulate a second confusion matrix restricted to each sample's
    boundary_mask(target, boundary_px) band (Cheng et al. 2021, "Boundary IoU" at a fixed
    pixel radius) -> biou / biou_headline / biou_<class>, read the same way as miou. A fixed
    pixel radius (not a token- or patch_size-relative one) is what makes it comparable across
    OlmoEarth arms of different patch_size: the head always predicts at native pixel
    resolution (exp/common/heads.py Pa2PxHead), only the token grid coarseness differs."""
    higher_is_better = True

    def __init__(self, num_classes: int, class_names: tuple[str, ...],
                 headline: tuple[int, ...] | None = None, ignore_label: int = IGNORE_LABEL,
                 boundary_px: int = 2):
        self.n, self.names, self.headline, self.ignore = num_classes, class_names, headline, ignore_label
        self.boundary_px = boundary_px
        self.conf = self.bconf = None

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(1)
        c = _build_confusion_matrix(pred, target, self.n, self.ignore)
        self.conf = c if self.conf is None else self.conf + c
        if self.boundary_px:
            bc = _build_confusion_matrix(pred, target, self.n, self.ignore,
                                         mask=boundary_mask(target, self.boundary_px, self.ignore))
            self.bconf = bc if self.bconf is None else self.bconf + bc

    def merge(self, other: "SegMeter") -> "SegMeter":
        """Add another meter's counts (e.g. from a shard of the same split)."""
        self.conf = other.conf if self.conf is None else self.conf + other.conf.to(self.conf)
        if other.bconf is not None:
            self.bconf = other.bconf if self.bconf is None else self.bconf + other.bconf.to(self.bconf)
        return self

    @staticmethod
    def _iou(conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tp = conf.diagonal()
        fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
        union = tp + fp + fn
        return tp / (union + 1e-8), union > 0

    def compute(self) -> dict[str, float]:
        conf = self.conf.double()
        tp = conf.diagonal()
        fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
        iou, keep = self._iou(conf)
        present = (tp + fn) > 0
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        out = {"miou": iou[keep].mean().item(),
               "overall_acc": (tp.sum() / conf.sum().clamp(min=1)).item(),
               "macro_f1": f1[present].mean().item()}
        if self.headline is not None:
            h = [c for c in self.headline if keep[c]]
            out["miou_headline"] = iou[h].mean().item() if h else float("nan")
        for c, name in enumerate(self.names):
            out[f"iou_{name}"] = iou[c].item() if keep[c] else float("nan")
        if self.bconf is not None:
            biou, bkeep = self._iou(self.bconf.double())
            out["biou"] = biou[bkeep].mean().item() if bkeep.any() else float("nan")
            if self.headline is not None:
                bh = [c for c in self.headline if bkeep[c]]
                out["biou_headline"] = biou[bh].mean().item() if bh else float("nan")
            for c, name in enumerate(self.names):
                out[f"biou_{name}"] = biou[c].item() if bkeep[c] else float("nan")
        out["primary"] = out["miou_headline" if self.headline is not None else "miou"]
        return out


class RegMeter:
    """RMSE / MAE / R^2 of per-pixel regression. Predictions and targets arrive standardized
    ((y - mean) / scale); errors are reported in the target's own units (x scale)."""
    higher_is_better = False

    def __init__(self, scale: float):
        self.scale = scale
        self.sse = self.sae = self.sy = self.syy = 0.0
        self.n = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = pred[:, 0].float(), target.float()
        err = pred - target
        self.sse += (err * err).sum().item()
        self.sae += err.abs().sum().item()
        self.sy += target.sum().item()
        self.syy += (target * target).sum().item()
        self.n += target.numel()

    def merge(self, other: "RegMeter") -> "RegMeter":
        for k in ("sse", "sae", "sy", "syy", "n"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self

    def compute(self) -> dict[str, float]:
        var = self.syy / self.n - (self.sy / self.n) ** 2
        rmse = (self.sse / self.n) ** 0.5 * self.scale
        return {"rmse": rmse, "mae": self.sae / self.n * self.scale,
                "r2": 1 - (self.sse / self.n) / max(var, 1e-12), "primary": rmse}


class YieldMeter:
    """Per-pixel yield regression with unlabelled pixels, reported pooled and per group
    (YieldSAT: country x crop), at pixel level and at FIELD level (a field's prediction and
    target are the means over its labelled pixels, as the YieldSAT paper reports).

    pred (B, 1, H, W) and target[:, 0] arrive standardized ((y - mean) / std); target[:, 1] is
    the pixel's field index into `field_group` (one group name per field). NaN in target[:, 0]
    = unlabelled, never scored. Everything is accumulated per field, so a field cut into
    several tiles, batches or shards is scored once. Errors are in the target's units."""
    higher_is_better = False
    SUMS = ("n", "sp", "st", "stt", "sse", "sae")

    def __init__(self, mean: float, std: float, field_group: tuple[str, ...]):
        self.mean, self.std = mean, std
        self.field_group = field_group
        self.groups = tuple(sorted(set(field_group)))
        self.acc = torch.zeros(len(self.SUMS), len(field_group), dtype=torch.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        y = target[:, 0].double()
        keep = torch.isfinite(y)
        p = pred[:, 0].double()[keep] * self.std + self.mean
        t = y[keep] * self.std + self.mean
        f = target[:, 1][keep].long().cpu()
        e = p - t
        vals = torch.stack([torch.ones_like(t), p, t, t * t, e * e, e.abs()]).cpu()
        self.acc.index_add_(1, f, vals)

    def merge(self, other: "YieldMeter") -> "YieldMeter":
        self.acc += other.acc
        return self

    @staticmethod
    def _scores(a: torch.Tensor) -> dict[str, float]:
        """Pixel and field scores of the fields in accumulator columns a (6, F)."""
        n, sp, st, stt, sse, sae = a
        seen = n > 0
        N = n.sum().item()
        if N == 0:
            return {}
        r2 = lambda mse, var: 1 - mse / var if var > 1e-9 else float("nan")   # noqa: E731
        var = stt.sum().item() / N - (st.sum().item() / N) ** 2
        out = {"rmse": (sse.sum().item() / N) ** 0.5, "mae": sae.sum().item() / N,
               "r2": r2(sse.sum().item() / N, var)}
        fp, ft = (sp / n.clamp(min=1))[seen], (st / n.clamp(min=1))[seen]
        fe = ((fp - ft) ** 2).mean().item()
        out.update(field_rmse=fe ** 0.5, fields=int(seen.sum()),         # R^2 needs >= 2 fields
                   field_r2=r2(fe, ft.var(unbiased=False).item()) if len(ft) > 1 else float("nan"))
        return out

    def compute(self) -> dict[str, float]:
        out = self._scores(self.acc)
        by = []
        for g in self.groups:
            cols = torch.tensor([i for i, fg in enumerate(self.field_group) if fg == g])
            s = self._scores(self.acc[:, cols])
            out.update({f"{g}_{k}": v for k, v in s.items() if k != "mae"})
            if s:
                by.append(s)
        # the paper's view: every country x crop subset counts once, whatever its size
        for k in ("r2", "field_r2"):
            v = [s[k] for s in by if s[k] == s[k]]
            out[f"group_mean_{k}"] = sum(v) / len(v) if v else float("nan")
        out["primary"] = out.get("rmse", float("nan"))
        return out
