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
                            num_classes: int, ignore_label: int = IGNORE_LABEL) -> torch.Tensor:
    """(num_classes, num_classes) counts; confusion[i, j] = true i predicted as j.
    predictions/labels: (N, H, W) integer class indices; ignore_label pixels are dropped."""
    if predictions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"predictions must be integer class indices, got {predictions.dtype}")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"labels must be integer class indices, got {labels.dtype}")
    labels = labels.to(predictions.device)
    valid = labels != ignore_label
    n = num_classes
    return torch.bincount(n * labels[valid] + predictions[valid], minlength=n**2).reshape(n, n)


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
    counts them (classes absent from both labels and predictions are skipped)."""
    higher_is_better = True

    def __init__(self, num_classes: int, class_names: tuple[str, ...],
                 headline: tuple[int, ...] | None = None, ignore_label: int = IGNORE_LABEL):
        self.n, self.names, self.headline, self.ignore = num_classes, class_names, headline, ignore_label
        self.conf = None

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        c = _build_confusion_matrix(logits.argmax(1), target, self.n, self.ignore)
        self.conf = c if self.conf is None else self.conf + c

    def compute(self) -> dict[str, float]:
        conf = self.conf.double()
        tp = conf.diagonal()
        fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
        union = tp + fp + fn
        iou = tp / (union + 1e-8)
        keep = union > 0
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

    def compute(self) -> dict[str, float]:
        var = self.syy / self.n - (self.sy / self.n) ** 2
        rmse = (self.sse / self.n) ** 0.5 * self.scale
        return {"rmse": rmse, "mae": self.sae / self.n * self.scale,
                "r2": 1 - (self.sse / self.n) / max(var, 1e-12), "primary": rmse}
