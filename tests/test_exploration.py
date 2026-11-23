"""Tests for the exploration's measurements, pinned against hand counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

from driving_scenes.config import ANCHOR_STRIDES
from driving_scenes.exploration import (
    anchor_coverage,
    anchor_shapes,
    best_anchor_iou,
    box_gap,
    box_heatmap,
    class_frequency,
    class_shares,
    coarse_mask_ceilings,
    describe_signs,
    frame_repetition,
    has_close_neighbor,
    mask_iou,
    matched_share,
    rendering_agreement,
)
from tests.conftest import HEIGHT, ROAD_TOP, WIDTH


def test_class_shares_sum_to_one_per_world(indexed):
    frames, _, _ = indexed
    shares = class_shares(frames)
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    assert shares.loc["0001", "road"] == pytest.approx((HEIGHT - ROAD_TOP) / HEIGHT)
    assert shares.loc["0001", "unlabeled pixels"] == pytest.approx(6 / (HEIGHT * WIDTH))


def test_signs_are_described_by_shape_size_and_position(indexed):
    frames, signs, _ = indexed
    table = describe_signs(signs, frames)
    tall = table[(table["frame_id"] == "0001_clone_00001") & (table["x_min"] == 12)].iloc[0]
    assert (tall["width"], tall["height"], tall["aspect"]) == (6, 20, pytest.approx(20 / 6))
    assert tall["fill"] == 1.0 and tall["size"] == "small" and not tall["at_border"]
    assert tall["world"] == "0001" and tall["variation"] == "clone" and tall["step"] == 1


def test_size_classes_follow_the_coco_edges():
    frames = pd.DataFrame(
        {"frame_id": ["a"], "world": ["w"], "variation": ["clone"], "step": [0], "width": [500]}
    ).assign(height=500)
    signs = pd.DataFrame(
        {
            "frame_id": ["a"] * 4,
            "x_min": [10, 10, 10, 0],
            "y_min": [10, 10, 10, 10],
            "x_max": [41, 42, 106, 20],
            "y_max": [41, 42, 106, 20],
            "pixels": [100] * 4,
        }
    )
    table = describe_signs(signs, frames)
    assert table["size"].tolist() == ["small", "medium", "large", "small"]
    assert table["at_border"].tolist() == [False, False, False, True]


def test_box_gap_is_the_empty_space_between_boxes():
    first = np.array([[0, 0, 10, 10]])
    others = np.array([[12, 0, 20, 10], [5, 5, 8, 8], [10, 14, 20, 20]])
    assert box_gap(first, others).tolist() == [[2, 0, 4]]


def test_regions_that_almost_touch_are_flagged_as_neighbors():
    signs = pd.DataFrame(
        {
            "frame_id": ["a", "a", "a", "b"],
            "x_min": [0, 12, 60, 0],
            "y_min": [0, 0, 0, 0],
            "x_max": [10, 20, 70, 10],
            "y_max": [10, 10, 10, 10],
        }
    )
    assert has_close_neighbor(signs, gap=3).tolist() == [True, True, False, False]


def test_box_heatmap_counts_the_boxes_over_every_pixel():
    heat = box_heatmap(np.array([[1, 1, 4, 3], [2, 0, 6, 2]]), height=4, width=6)
    expected = np.array(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 2, 2, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(heat, expected)


def test_class_frequency_is_the_share_of_frames_per_pixel(indexed):
    frames, _, label_dir = indexed
    frequency = class_frequency(frames, label_dir=label_dir)
    assert frequency.shape == (HEIGHT, WIDTH)
    assert (frequency[ROAD_TOP:] == 1).all() and (frequency[:ROAD_TOP] == 0).all()


def test_mask_iou_and_matched_share_match_hand_counts():
    first = np.array([[1, 1, 0, 0]], dtype=bool)
    second = np.array([[0, 1, 1, 0]], dtype=bool)
    assert mask_iou(first, second) == pytest.approx(1 / 3)
    assert mask_iou(np.zeros(4, bool), np.zeros(4, bool)) == 1.0
    boxes = np.array([[0.0, 0, 10, 10], [50, 50, 60, 60]])
    others = np.array([[0.0, 0, 10, 8]])
    assert matched_share(boxes, others, threshold=0.5) == 0.5
    assert matched_share(boxes, np.zeros((0, 4)), threshold=0.5) == 0.0
    assert np.isnan(matched_share(np.zeros((0, 4)), others, threshold=0.5))


def test_frame_repetition_reads_the_labels_of_later_frames(indexed):
    frames, signs, label_dir = indexed
    table = frame_repetition(frames, signs, (1, 2), "clone", 0.5, label_dir).set_index(
        ["world", "lag"]
    )
    assert (table["road_iou"] == 1.0).all(), "the synthetic road never moves"
    # Step 0 to 1: the sign moved two of six pixels (IoU 0.5); step 1 to 2: both signs are gone.
    assert table.loc[("0001", 1), "signs_still_matched"] == pytest.approx((1.0 + 0.0) / 2)
    assert table.loc[("0001", 2), "signs_still_matched"] == 0.0


def test_renderings_of_one_moment_share_their_labels(indexed):
    frames, signs, label_dir = indexed
    table = rendering_agreement(frames, signs, "clone", 0.5, 3, 0, label_dir)
    assert table.index.tolist() == ["fog"]
    assert table.loc["fog", "road_iou"] == 1.0 and table.loc["fog", "signs_matched"] == 1.0


def test_anchor_shapes_follow_the_detector():
    sizes, ratios = ((32,), (64,), (128,), (256,), (512,)), (0.5, 1.0, 2.0)
    generator = AnchorGenerator(sizes, ratios)
    expected = [
        (stride, float(cell[2] - cell[0]), float(cell[3] - cell[1]))
        for stride, level in zip(ANCHOR_STRIDES, generator.cell_anchors, strict=True)
        for cell in level
    ]
    assert anchor_shapes(sizes, ratios) == expected


@pytest.mark.parametrize("ratios", [(0.5, 1.0, 2.0), (1.0, 2.5, 6.0)])
def test_best_anchor_iou_matches_a_search_over_every_anchor(ratios):
    sizes = ((16,), (32,), (64,), (128,), (256,))
    height, width = 384, 1248
    generator = AnchorGenerator(sizes, (ratios,) * len(sizes))
    maps = [torch.zeros(1, 1, height // s, -(-width // s)) for s in ANCHOR_STRIDES]
    anchors = generator(ImageList(torch.zeros(1, 3, height, width), [(height, width)]), maps)[0]
    rng = np.random.default_rng(0)
    corner = rng.uniform((0, 0), (1100, 250), size=(300, 2))
    extent = rng.uniform((3, 4), (140, 120), size=(300, 2))
    boxes = np.concatenate([corner, corner + extent], axis=1)
    brute = box_iou(torch.as_tensor(boxes, dtype=torch.float32), anchors).max(dim=1).values
    np.testing.assert_allclose(best_anchor_iou(boxes, sizes, ratios), brute.numpy(), atol=1e-4)


def test_anchor_coverage_ranks_a_matching_shape_above_a_mismatched_one():
    tall = np.array([[100.0, 100, 110, 160]] * 5)
    table = anchor_coverage(
        tall,
        {"square": (((32,),) * 5, (1.0,)), "tall": (((24,),) * 5, (6.0,))},
        thresholds=(0.5,),
    )
    assert table.loc["tall", "median best IoU"] > table.loc["square", "median best IoU"]
    assert (
        table.loc["tall", "best IoU >= 0.5"] == 1.0
        and table.loc["square", "best IoU >= 0.5"] == 0.0
    )


def test_a_coarse_box_mask_loses_more_of_the_road_than_a_coarse_grid(tmp_path):
    from PIL import Image

    from driving_scenes.dataset import CLASS_INDEX, label_path

    # A thin diagonal road: easy on a fine grid, hard for a 4 x 4 mask stretched over its box.
    labels = np.full((64, 128), CLASS_INDEX["sky"], dtype=np.uint8)
    for row in range(64):
        labels[row, 2 * row : 2 * row + 6] = CLASS_INDEX["road"]
    Image.fromarray(labels).save(label_path("frame", tmp_path))
    frames = pd.DataFrame({"frame_id": ["frame"]})
    scores = coarse_mask_ceilings(frames, tmp_path, mask_size=4, stride=2).iloc[0]
    assert scores["semantic head"] > 0.8 > scores["mask head"]
