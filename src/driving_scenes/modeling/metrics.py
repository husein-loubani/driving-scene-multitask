"""
One definition of every score, used by training, evaluation and the notebook.

**Road:** every labeled pixel is a true or false positive or negative for the
road class. Intersection over union (IoU) is the share of the pixels called
road or truly road that are both; pixel accuracy is the share of all labeled
pixels called correctly, which a network can push up by calling everything
"not road", so IoU is the score that matters. The counts come from
torchmetrics and are kept per frame, so that any group of frames can be scored
later without predicting again.

**Signs:** a detection is a box with a confidence. Average precision (AP)
ranks all detections by confidence, walks down the list and averages the
precision over the recall reached, counting a detection as correct when it
overlaps a sign that no better-ranked detection has taken by at least a given
IoU. AP50 uses an IoU of 0.5; COCO's mAP averages AP over ten IoU levels from
0.5 to 0.95 and so also rewards tight boxes. IoU measures one box against one
sign; AP measures the whole ranked list. Both come from torchmetrics, which
runs the COCO reference implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.functional.classification import binary_stat_scores

from driving_scenes.config import IGNORE_LABEL, ROAD_THRESHOLD

COUNT_NAMES = ("tp", "fp", "tn", "fn")
SIGN_SCORES = {
    "ap50": "map_50",
    "map": "map",
    "ap75": "map_75",
    "ap_small": "map_small",
    "ap_medium": "map_medium",
    "ap_large": "map_large",
    "ar100": "mar_100",
}


def road_counts(
    probability: torch.Tensor, road: torch.Tensor, threshold: float = ROAD_THRESHOLD
) -> torch.Tensor:
    """
    True positives, false positives, true negatives and false negatives per frame.

    `probability` and `road` have shape (frames, height, width); unlabeled
    pixels are left out of every count.
    """
    # One call per frame: the library's per-sample mode drops the frame axis
    # when a batch holds a single frame.
    counts = [
        binary_stat_scores(frame, truth, threshold=threshold, ignore_index=IGNORE_LABEL)
        for frame, truth in zip(probability.cpu(), road.cpu().long(), strict=True)
    ]
    return torch.stack(counts)[:, : len(COUNT_NAMES)]


def road_scores(counts) -> dict[str, float]:
    """
    Road scores from counts summed over any set of frames.

    `counts` holds tp, fp, tn, fn, or rows of them. The mean IoU averages the
    road's IoU with the IoU of everything else, the two-class mIoU of the
    segmentation literature.
    """
    tp, fp, tn, fn = (float(value) for value in np.asarray(counts).reshape(-1, 4).sum(axis=0))
    road_iou = tp / max(tp + fp + fn, 1)
    other_iou = tn / max(tn + fp + fn, 1)
    return {
        "road_iou": road_iou,
        "miou": (road_iou + other_iou) / 2,
        "pixel_accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "road_precision": tp / max(tp + fp, 1),
        "road_recall": tp / max(tp + fn, 1),
    }


def sign_metric() -> MeanAveragePrecision:
    """COCO average precision for boxes given as (x_min, y_min, x_max, y_max)."""
    return MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")


def to_cpu(items: list[Mapping[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Detections or targets moved off the GPU, where the COCO evaluation runs."""
    return [{name: value.detach().cpu() for name, value in item.items()} for item in items]


def sign_scores(metric: MeanAveragePrecision) -> dict[str, float]:
    """The scores the project reports, by its own names; a size class without signs gives NaN."""
    result = metric.compute()
    scores = {name: float(result[key]) for name, key in SIGN_SCORES.items()}
    return {name: np.nan if value < 0 else value for name, value in scores.items()}
