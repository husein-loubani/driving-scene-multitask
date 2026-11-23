"""Tests for decoding the archive's annotations and serving frames to the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from driving_scenes.config import IGNORE_LABEL
from driving_scenes.dataset import (
    BOX_COLUMNS,
    CLASS_INDEX,
    FrameDataset,
    assign_partitions,
    audit_summary,
    build_index,
    collate,
    decode_bitmap,
    frame_table,
    load_labels,
    mask_box,
    one_rendering_per_step,
    read_annotation,
    road_target,
)
from tests.conftest import HEIGHT, ROAD_TOP, SIGNS, UNLABELED, WIDTH, encode_bitmap


def test_frame_names_are_parsed_into_world_rendering_and_step(raw_dir):
    frames = frame_table(raw_dir)
    assert len(frames) == 12
    first = frames.iloc[0]
    assert (first["frame_id"], first["world"], first["variation"], first["step"]) == (
        "0001_clone_00000",
        "0001",
        "clone",
        0,
    )


def test_an_unknown_rendering_is_refused(raw_dir):
    (raw_dir / "img" / "rgb_0001_snow_00000.png").write_bytes(b"")
    with pytest.raises(ValueError, match="unexpected renderings"):
        frame_table(raw_dir)


def test_a_bitmap_survives_the_archive_encoding():
    mask = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
    decoded, left, top = decode_bitmap(encode_bitmap(mask, left=7, top=3))
    np.testing.assert_array_equal(decoded, mask)
    assert (left, top) == (7, 3)


def test_mask_box_is_tight_and_offset_by_the_origin():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 3:4] = True
    assert mask_box(mask) == (3, 2, 4, 5)
    assert mask_box(mask, left=10, top=20) == (13, 22, 14, 25)


def test_an_annotation_becomes_a_label_map_and_sign_boxes(raw_dir):
    labels, signs, checks = read_annotation(raw_dir / "ann" / "rgb_0001_clone_00001.png.json")
    assert labels.shape == (HEIGHT, WIDTH)
    assert (labels[ROAD_TOP:] == CLASS_INDEX["road"]).all()
    assert (labels[UNLABELED] == IGNORE_LABEL).all()
    assert (labels == IGNORE_LABEL).sum() == 6, "only the unlabeled block is left blank"
    boxes = [tuple(sign[column] for column in BOX_COLUMNS) for sign in signs]
    assert boxes == SIGNS[1]
    assert [sign["pixels"] for sign in signs] == [6 * 20, 20 * 8]
    assert checks == {"overlapping_pixels": 0, "tagged_world": "0001", "tagged_variation": "clone"}


def test_the_tracking_rectangle_of_a_car_leaves_no_trace(raw_dir):
    labels, _, _ = read_annotation(raw_dir / "ann" / "rgb_0001_clone_00002.png.json")
    assert (labels != CLASS_INDEX["car"]).all()


def test_the_index_measures_every_frame_and_writes_its_label_map(indexed):
    frames, signs, label_dir = indexed
    assert len(frames) == 12 and len(signs) == 4 * 3
    assert frames["signs"].tolist() == [1, 2, 0] * 4
    row = frames.iloc[1]
    assert row["pixels_road"] == (HEIGHT - ROAD_TOP) * WIDTH
    assert row["pixels_traffic sign"] == 6 * 20 + 20 * 8
    assert row["unlabeled_pixels"] == 6
    np.testing.assert_array_equal(
        load_labels(row["frame_id"], label_dir)[ROAD_TOP:], CLASS_INDEX["road"]
    )
    fog, clone = frames[frames["variation"] == "fog"], frames[frames["variation"] == "clone"]
    assert fog["blue"].mean() < clone["blue"].mean(), "the gray rendering is less blue"


def test_the_audit_reports_the_gaps_the_archive_leaves(indexed):
    frames, signs, _ = indexed
    audit = audit_summary(frames, signs)
    assert audit["frames"] == 12 and audit["worlds"] == 2
    assert audit["frame sizes (width x height)"] == f"{WIDTH} x {HEIGHT}"
    assert audit["frames with every pixel labeled"] == 0
    assert audit["percent of pixels without a label"] == pytest.approx(600 / (HEIGHT * WIDTH))
    assert audit["pixels claimed by two regions"] == 0
    assert audit["frames whose tags disagree with their name"] == 0
    assert audit["labeled pixels accounted for"]


def test_the_index_is_reused_until_an_annotation_changes(raw_dir, tmp_path, monkeypatch):
    from driving_scenes import dataset

    paths = (tmp_path / "labels", tmp_path / "frames.csv", tmp_path / "signs.csv")
    frames = frame_table(raw_dir)
    first, _ = build_index(frames, raw_dir, *paths, n_jobs=1)
    calls = []
    real = dataset.index_frame
    monkeypatch.setattr(dataset, "index_frame", lambda *args: calls.append(1) or real(*args))
    again, signs = build_index(frames, raw_dir, *paths, n_jobs=1)
    assert not calls, "nothing is decoded a second time"
    pd.testing.assert_frame_equal(again, first)
    assert again["world"].tolist() == first["world"].tolist() == ["0001"] * 6 + ["0002"] * 6
    target = raw_dir / "ann" / "rgb_0001_clone_00000.png.json"
    target.write_text(target.read_text() + " ")
    build_index(frames, raw_dir, *paths, n_jobs=1)
    assert len(calls) == 12, "a changed annotation rebuilds the index"


def test_road_target_keeps_unlabeled_pixels_apart():
    labels = np.array([[CLASS_INDEX["road"], CLASS_INDEX["sky"], IGNORE_LABEL]], dtype=np.uint8)
    assert road_target(labels).tolist() == [[1, 0, IGNORE_LABEL]]
    assert road_target(torch.from_numpy(labels)).tolist() == [[1, 0, IGNORE_LABEL]]


def test_a_world_never_straddles_two_partitions(indexed):
    frames, _, _ = indexed
    partition = assign_partitions(frames, {"train": ["0001"], "test": ["0002"]})
    assert frames.groupby("world")[[]].size().index.tolist() == ["0001", "0002"]
    assert partition[frames["world"] == "0001"].unique().tolist() == ["train"]
    assert partition[frames["world"] == "0002"].unique().tolist() == ["test"]
    with pytest.raises(ValueError, match="two partitions"):
        assign_partitions(frames, {"train": ["0001", "0002"], "test": ["0002"]})
    with pytest.raises(ValueError, match="without a partition"):
        assign_partitions(frames, {"train": ["0001"]})


def test_one_rendering_per_step_covers_every_frame_once_over_the_shifts():
    renderings = ("clone", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right")
    renderings += ("fog", "morning", "overcast", "rain", "sunset")
    frames = pd.DataFrame(
        [
            {"frame_id": f"{name}_{step}", "variation": name, "step": step}
            for name in renderings
            for step in range(25)
        ]
    )
    epochs = [one_rendering_per_step(frames, shift) for shift in range(10)]
    assert all(len(epoch) == 25 and epoch["step"].is_unique for epoch in epochs)
    assert all(epoch["variation"].nunique() == 10 for epoch in epochs)
    seen = pd.concat(epochs)["frame_id"]
    assert seen.is_unique and len(seen) == len(frames)


def test_a_dataset_item_has_the_detector_format(indexed, raw_dir):
    frames, signs, label_dir = indexed
    data = FrameDataset(frames, signs, raw_dir, label_dir)
    image, target, road, row = data[1]
    assert image.shape == (3, HEIGHT, WIDTH) and image.dtype == torch.float32
    assert image.min() >= 0 and image.max() <= 1
    assert target["boxes"].tolist() == [list(map(float, box)) for box in SIGNS[1]]
    assert target["labels"].tolist() == [1, 1] and target["labels"].dtype == torch.int64
    assert road.dtype == torch.uint8 and road[ROAD_TOP:].eq(1).all() and road[5, 50] == 0
    assert road[UNLABELED].eq(IGNORE_LABEL).all() and row == 1
    _, empty, _, _ = data[2]
    assert empty["boxes"].shape == (0, 4)


def test_a_flip_mirrors_the_frame_its_mask_and_its_boxes(indexed, raw_dir):
    frames, signs, label_dir = indexed
    plain = FrameDataset(frames, signs, raw_dir, label_dir)[1]
    flipped = FrameDataset(frames, signs, raw_dir, label_dir, flip=1.0)[1]
    torch.testing.assert_close(flipped[0], plain[0].flip(-1))
    torch.testing.assert_close(flipped[2], plain[2].flip(-1))
    mirrored = [
        [WIDTH - x_max, y_min, WIDTH - x_min, y_max] for x_min, y_min, x_max, y_max in SIGNS[1]
    ]
    assert flipped[1]["boxes"].tolist() == [list(map(float, box)) for box in mirrored]
    red = flipped[0][0] > 0.8
    x_min, y_min, x_max, y_max = (int(v) for v in flipped[1]["boxes"][0])
    assert red[y_min:y_max, x_min:x_max].all(), "the mirrored box still frames the sign"


def test_collate_keeps_lists_for_the_detector_and_stacks_the_masks(indexed, raw_dir):
    frames, signs, label_dir = indexed
    data = FrameDataset(frames, signs, raw_dir, label_dir)
    images, targets, roads, rows = collate([data[0], data[2]])
    assert isinstance(images, list) and len(images) == len(targets) == 2
    assert roads.shape == (2, HEIGHT, WIDTH) and rows.tolist() == [0, 2]
