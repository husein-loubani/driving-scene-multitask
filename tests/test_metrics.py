"""Tests for the scores, pinned against hand counts and against torchmetrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex

from driving_scenes.config import IGNORE_LABEL
from driving_scenes.modeling.metrics import (
    road_counts,
    road_scores,
    sign_metric,
    sign_scores,
    to_cpu,
)


def test_road_counts_per_frame_leave_unlabeled_pixels_out():
    probability = torch.tensor([[[0.9, 0.8, 0.2, 0.1, 0.7]], [[0.6, 0.4, 0.4, 0.4, 0.4]]])
    road = torch.tensor([[[1, 0, 1, 0, IGNORE_LABEL]], [[1, 1, 0, 0, 0]]], dtype=torch.uint8)
    # Columns: tp, fp, tn, fn.
    assert road_counts(probability, road).tolist() == [[1, 1, 1, 1], [1, 0, 3, 1]]


def test_a_batch_of_one_frame_keeps_its_frame_axis():
    counts = road_counts(torch.tensor([[[0.9, 0.2]]]), torch.tensor([[[1, 1]]], dtype=torch.uint8))
    assert counts.tolist() == [[1, 0, 0, 1]]


def test_road_scores_match_a_hand_count():
    scores = road_scores([6, 2, 10, 2])
    assert scores["road_iou"] == pytest.approx(6 / 10)
    assert scores["miou"] == pytest.approx((6 / 10 + 10 / 14) / 2)
    assert scores["pixel_accuracy"] == pytest.approx(16 / 20)
    assert scores["road_precision"] == pytest.approx(6 / 8)
    assert scores["road_recall"] == pytest.approx(6 / 8)


def test_road_scores_sum_rows_of_counts_before_dividing():
    rows = np.array([[1, 0, 0, 9], [9, 0, 0, 1]])
    assert road_scores(rows)["road_iou"] == pytest.approx(10 / 20)


def test_road_scores_agree_with_torchmetrics():
    generator = torch.Generator().manual_seed(0)
    probability = torch.rand(3, 20, 30, generator=generator)
    road = (torch.rand(3, 20, 30, generator=generator) > 0.6).to(torch.uint8)
    road[0, :3] = IGNORE_LABEL
    scores = road_scores(road_counts(probability, road))
    jaccard = BinaryJaccardIndex(ignore_index=IGNORE_LABEL)(probability, road.long())
    accuracy = BinaryAccuracy(ignore_index=IGNORE_LABEL)(probability, road.long())
    assert scores["road_iou"] == pytest.approx(jaccard.item(), abs=1e-6)
    assert scores["pixel_accuracy"] == pytest.approx(accuracy.item(), abs=1e-6)


def test_perfect_detections_score_one_and_an_empty_size_class_is_missing():
    truth = [{"boxes": torch.tensor([[10.0, 10, 20, 30]]), "labels": torch.tensor([1])}]
    found = [truth[0] | {"scores": torch.tensor([0.9])}]
    metric = sign_metric()
    metric.update(to_cpu(found), to_cpu(truth))
    scores = sign_scores(metric)
    assert scores["ap50"] == pytest.approx(1.0) and scores["map"] == pytest.approx(1.0)
    assert scores["ap_small"] == pytest.approx(1.0), "a 10 x 20 box is small"
    assert np.isnan(scores["ap_large"]), "there is no large sign to score"


def test_a_loose_box_counts_at_iou_50_but_not_at_75():
    truth = [{"boxes": torch.tensor([[0.0, 0, 10, 40]]), "labels": torch.tensor([1])}]
    loose = [
        {
            "boxes": torch.tensor([[0.0, 0, 10, 26]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }
    ]
    metric = sign_metric()
    metric.update(loose, truth)
    scores = sign_scores(metric)
    assert scores["ap50"] == pytest.approx(1.0) and scores["ap75"] == pytest.approx(0.0)
