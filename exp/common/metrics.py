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
