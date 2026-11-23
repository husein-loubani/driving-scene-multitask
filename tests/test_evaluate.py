"""Tests for stored predictions and the scores read from them, against hand counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from driving_scenes.modeling import evaluate as evaluate_module
from driving_scenes.modeling.evaluate import (
    Predictions,
    cached_predictions,
    cached_road_calls,
    compare_runs,
    detection_reliability,
    extreme_frames,
    fixed_mask_counts,
    frame_scores,
    frame_titles,
    match_detections,
    recall_by,
    road_calls_by_class,
    score_predictions,
    scores_by_group,
    size_scores,
    threshold_sweep,
    time_per_frame,
    timing_images,
)

SIGNS = pd.DataFrame(
    {
        "frame_id": ["a", "a", "b"],
        "x_min": [0.0, 50.0, 10.0],
        "y_min": [0.0, 0.0, 10.0],
        "x_max": [10.0, 60.0, 20.0],
        "y_max": [30.0, 30.0, 40.0],
    }
)


def make_predictions(counts=True) -> Predictions:
    """Frame a: a sign found, one missed, a false alarm. Frame b: its sign found. c: empty."""
    return Predictions(
        tasks=("signs", "road") if counts else ("signs",),
        frame_ids=np.array(["a", "b", "c"]),
        counts=np.array([[6, 2, 10, 2], [5, 0, 10, 0], [0, 1, 19, 0]]) if counts else None,
        boxes=np.array(
            [[0, 0, 10, 30], [100, 0, 110, 30], [10, 10, 20, 40], [11, 10, 21, 40]],
            dtype=np.float32,
        ),
        scores=np.array([0.9, 0.6, 0.8, 0.3], dtype=np.float32),
        box_frame=np.array([0, 0, 1, 1]),
    )


def test_predictions_survive_a_round_trip_to_disk(tmp_path):
    original = make_predictions()
    original.save(tmp_path / "p.npz")
    loaded = Predictions.load(tmp_path / "p.npz")
    assert loaded.tasks == ("signs", "road")
    for field in ("frame_ids", "counts", "boxes", "scores", "box_frame"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(original, field))
    signs_only = make_predictions(counts=False)
    signs_only.save(tmp_path / "s.npz")
    assert Predictions.load(tmp_path / "s.npz").counts is None


def test_a_subset_keeps_each_detection_with_its_frame():
    part = make_predictions().subset(np.array([False, True, True]))
    assert part.frame_ids.tolist() == ["b", "c"]
    assert part.counts.tolist() == [[5, 0, 10, 0], [0, 1, 19, 0]]
    assert part.box_frame.tolist() == [0, 0] and part.scores.tolist() == pytest.approx([0.8, 0.3])


def test_detections_are_grouped_per_frame_in_the_detector_format():
    per_frame = make_predictions().detections()
    assert [len(item["boxes"]) for item in per_frame] == [2, 2, 0]
    assert per_frame[0]["labels"].tolist() == [1, 1] and per_frame[2]["boxes"].shape == (0, 4)


def test_matching_is_greedy_from_the_most_confident_detection():
    truth = np.array([[0.0, 0, 10, 10]])
    boxes = np.array([[1.0, 0, 11, 10], [0.0, 0, 10, 10]])
    # The more confident box takes the sign even though the other one fits it better.
    assert match_detections(boxes, np.array([0.9, 0.5]), truth).tolist() == [True, False]
    assert match_detections(boxes, np.array([0.5, 0.9]), truth).tolist() == [False, True]
    assert match_detections(boxes, np.array([0.9, 0.5]), np.zeros((0, 4))).tolist() == [False] * 2


def test_frame_scores_count_found_missed_and_false_alarms():
    table = frame_scores(make_predictions(), SIGNS, score_threshold=0.5).set_index("frame_id")
    assert table.loc["a", ["signs", "found", "missed", "false_alarms"]].tolist() == [2, 1, 1, 1]
    assert table.loc["a", "sign_f1"] == pytest.approx(2 / (2 + 1 + 1))
    assert table.loc["b", ["found", "false_alarms"]].tolist() == [1, 0]
    assert table.loc["a", "road_iou"] == pytest.approx(6 / 10)
    assert table.loc["c", "road_iou"] == 0.0, "road called where there is none"
    assert np.isnan(table.loc["c", "sign_f1"]), "a frame without signs or alarms has no F1"
    assert table["errors"].tolist() == [2, 0, 0]
    assert table.loc["a", "combined"] == pytest.approx((6 / 10 + 2 / 4) / 2)
    assert np.isnan(table.loc["c", "combined"])


def test_extreme_frames_rank_by_score_and_break_ties_by_errors():
    table = pd.DataFrame(
        {
            "frame_id": ["a", "b", "c", "d", "e"],
            "score": [0.9, 0.2, 0.2, np.nan, 0.5],
            "errors": [0, 1, 4, 0, 2],
            "step": [0, 100, 200, 300, 400],
        }
    )
    chosen = extreme_frames(table, "score", count=2)
    assert chosen["frame_id"].tolist() == ["a", "e", "c", "b"]
    assert chosen["rank"].tolist() == ["best", "best", "worst", "worst"]


def test_extreme_frames_show_different_moments():
    table = pd.DataFrame(
        {
            "frame_id": ["a", "b", "c", "d"],
            "score": [0.9, 0.8, 0.7, 0.1],
            "errors": [0, 0, 0, 0],
            "step": [10, 12, 60, 11],
        }
    )
    chosen = extreme_frames(table, "score", count=2, gap=25)
    # b is two steps from a, so the second best is c; d is the worst, then c.
    assert chosen["frame_id"].tolist() == ["a", "c", "d", "c"]


def test_frame_titles_say_how_the_frame_went():
    chosen = pd.DataFrame(
        {"variation": ["fog", "rain"], "step": [12, 5], "road_iou": [0.456, 0.9]}
    ).assign(found=[1, 0], signs=[3, 0], false_alarms=[2, 1])
    assert frame_titles(chosen) == [
        "fog, step 12: road IoU 0.46, 1 of 3 signs found, 2 false alarms",
        "rain, step 5: road IoU 0.90, 0 of 0 signs found, 1 false alarm",
    ]


def test_the_threshold_sweep_trades_recall_for_precision():
    sweep = threshold_sweep(make_predictions(), SIGNS, thresholds=(0.2, 0.7)).set_index("threshold")
    # At 0.2: three hits (the 0.3 duplicate on frame b is a false alarm), two false alarms.
    assert sweep.loc[0.2, "recall"] == pytest.approx(2 / 3)
    assert sweep.loc[0.2, "precision"] == pytest.approx(2 / 4)
    assert sweep.loc[0.7, "recall"] == pytest.approx(2 / 3)
    assert sweep.loc[0.7, "precision"] == pytest.approx(1.0)


def test_reliability_compares_confidence_with_the_hit_rate():
    table, error = detection_reliability(make_predictions(), SIGNS, bins=2)
    # Below 0.5: the 0.3 duplicate, a miss. From 0.5: hits at 0.9 and 0.8, a false alarm at 0.6.
    assert table["detections"].tolist() == [1, 3]
    assert table["hit rate"].tolist() == pytest.approx([0.0, 2 / 3])
    assert table["mean confidence"].tolist() == pytest.approx([0.3, (0.9 + 0.6 + 0.8) / 3])
    gaps = [abs(0.3 - 0.0) * 1, abs((0.9 + 0.6 + 0.8) / 3 - 2 / 3) * 3]
    assert error == pytest.approx(sum(gaps) / 4)


def test_recall_by_size_counts_found_signs_per_class():
    described = pd.DataFrame(
        {"size": pd.Categorical(["small", "large", "small"], ["small", "medium", "large"])},
        index=[10, 11, 12],
    )
    found = pd.Series([True, False, False], index=[10, 11, 12])
    table = recall_by(described, found, "size")
    assert table.index.tolist() == ["small", "medium", "large"]
    assert table.loc["small", ["signs", "found"]].tolist() == [2, 1]
    assert table.loc["small", "recall"] == pytest.approx(0.5)
    assert table.loc["medium", "signs"] == 0 and np.isnan(table.loc["medium", "recall"])
    scores = {"sign_ap_small": 0.2, "sign_ap_medium": np.nan, "sign_ap_large": 0.7}
    assert size_scores(described, found, scores)["ap"].tolist() == pytest.approx(
        [0.2, np.nan, 0.7], nan_ok=True
    )


def test_scores_cover_the_tasks_the_network_has():
    both = score_predictions(make_predictions(), SIGNS)
    assert {"sign_ap50", "sign_map", "road_iou", "miou", "pixel_accuracy"} <= set(both)
    assert both["road_iou"] == pytest.approx(11 / (11 + 3 + 2))
    signs_only = score_predictions(make_predictions(counts=False), SIGNS)
    assert "road_iou" not in signs_only and "sign_ap50" in signs_only


def test_scores_by_group_score_each_group_on_its_own_frames():
    groups = pd.Series({"a": "clone", "b": "fog", "c": "fog"})
    table = scores_by_group(make_predictions(), SIGNS, groups)
    assert table.index.tolist() == ["clone", "fog"]
    assert table.loc["fog", "sign_ap50"] == pytest.approx(1.0)
    assert table.loc["clone", "road_iou"] == pytest.approx(6 / 10)


@pytest.fixture
def trained(indexed, raw_dir, tmp_path, monkeypatch):
    """A network trained for one epoch on the synthetic frames, on the CPU."""
    from driving_scenes import dataset as dataset_module
    from driving_scenes.modeling import model as model_module
    from driving_scenes.modeling.train import fit
    from tests.conftest import HEIGHT, WIDTH

    monkeypatch.setattr(model_module, "MIN_SIZE", HEIGHT)
    monkeypatch.setattr(model_module, "MAX_SIZE", WIDTH)
    monkeypatch.setattr(dataset_module, "VARIATIONS", ("clone", "fog"))
    monkeypatch.setattr(evaluate_module, "device", lambda: torch.device("cpu"))
    frames, signs, label_dir = indexed
    run = fit(
        frames[frames["world"] == "0001"], frames[frames["world"] == "0002"], signs, "run",
        stages=0, epochs=1, lr=0.01, run_dir=tmp_path / "runs", raw_dir=raw_dir,
        label_dir=label_dir, workers=0, accelerator="cpu",
    )  # fmt: skip
    return run, frames, signs, label_dir


def test_predictions_are_computed_once_and_reused(trained, raw_dir, tmp_path, monkeypatch):
    run, frames, signs, label_dir = trained
    path = tmp_path / "predictions" / "run_test.npz"
    first = cached_predictions(run, frames, signs, path, raw_dir, label_dir, workers=0)
    assert first.frame_ids.tolist() == frames["frame_id"].tolist()
    assert first.counts.shape == (len(frames), 4) and first.tasks == ("signs", "road")
    assert (first.counts.sum(axis=1) == frames["width"] * frames["height"] - 6).all()
    calls = []
    monkeypatch.setattr(evaluate_module, "predict_frames", lambda *args: calls.append(1))
    again = cached_predictions(run, frames, signs, path, raw_dir, label_dir, workers=0)
    assert not calls
    np.testing.assert_array_equal(again.counts, first.counts)


class RoadBelow:
    """A stand-in network that calls every pixel from one row down road."""

    tasks = ("road",)

    def __init__(self, row: int):
        self.row = row

    def to(self, _):
        return self

    def eval(self):
        return self

    def predict(self, images):
        probability = torch.zeros(len(images), *images[0].shape[-2:])
        probability[:, self.row :] = 1.0
        return [], probability


def test_road_calls_are_counted_by_what_the_pixels_really_are(indexed, raw_dir, monkeypatch):
    from tests.conftest import HEIGHT, ROAD_TOP, WIDTH

    monkeypatch.setattr(evaluate_module, "device", lambda: torch.device("cpu"))
    frames, signs, label_dir = indexed
    world = frames[frames["world"] == "0001"]
    # One row of sky above the road is called road as well; no sign reaches that row.
    network = RoadBelow(ROAD_TOP - 1)
    table = road_calls_by_class(network, world, signs, raw_dir, label_dir, workers=0)
    road = len(world) * (HEIGHT - ROAD_TOP) * WIDTH
    assert table.loc["road", ["pixels", "called road"]].tolist() == [road, road]
    assert table.loc["sky", "called road"] == len(world) * WIDTH
    assert table.loc["sky", "share of false road"] == 1.0
    assert table["called road"].drop(["road", "sky"]).sum() == 0
    assert np.isnan(table.loc["road", "share of false road"])


def test_road_calls_of_a_run_are_computed_once(trained, raw_dir, tmp_path, monkeypatch):
    run, frames, signs, label_dir = trained
    path = tmp_path / "calls.csv"
    first = cached_road_calls(run, frames, signs, path, raw_dir, label_dir, workers=0)
    labeled = (frames["width"] * frames["height"]).sum() - 6 * len(frames)
    assert first["pixels"].sum() == labeled
    calls = []
    monkeypatch.setattr(evaluate_module, "road_calls_by_class", lambda *args: calls.append(1))
    again = cached_road_calls(run, frames, signs, path, raw_dir, label_dir, workers=0)
    assert not calls
    pd.testing.assert_frame_equal(again, first)


def test_latency_is_timed_one_frame_at_a_time(trained, raw_dir):
    from driving_scenes.modeling.train import load_model

    run, frames, _, _ = trained
    images = timing_images(frames, count=4, raw_dir=raw_dir)
    assert len(images) == 4 and images[0].dtype == torch.float32
    timing = time_per_frame(load_model(run), images, warmup=1)
    assert timing["frames"] == 3 and 0 < timing["median_ms"] <= timing["p95_ms"]


def test_assignment_names_the_sign_each_detection_took():
    from driving_scenes.modeling.evaluate import assign_detections

    truth = np.array([[0.0, 0, 10, 10], [20.0, 0, 30, 10]])
    boxes = np.array([[21.0, 0, 31, 10], [0.0, 0, 10, 10], [1.0, 0, 11, 10]])
    assigned = assign_detections(boxes, np.array([0.9, 0.8, 0.7]), truth)
    assert assigned.tolist() == [1, 0, -1], "the third box finds its sign already taken"


def test_sign_outcomes_mark_each_true_sign_found_or_missed():
    from driving_scenes.modeling.evaluate import sign_outcomes

    found = sign_outcomes(make_predictions(), SIGNS, score_threshold=0.5)
    assert found.tolist() == [True, False, True]
    only_b = sign_outcomes(make_predictions().subset(np.array([False, True, False])), SIGNS, 0.5)
    assert only_b.index.tolist() == [2] and only_b.tolist() == [True]


def test_box_kinds_draw_hits_misses_and_false_alarms():
    from driving_scenes.modeling.evaluate import box_kinds

    truth = np.array([[0.0, 0, 10, 30], [50.0, 0, 60, 30]])
    boxes = np.array([[0.0, 0, 10, 30], [100.0, 0, 110, 30], [50.0, 0, 60, 30]], dtype=np.float32)
    kinds = box_kinds(boxes, np.array([0.9, 0.6, 0.2]), truth, score_threshold=0.5)
    assert [kind for _, kind in kinds] == ["hit", "false", "miss"]
    np.testing.assert_array_equal(kinds[2][0], truth[1])


def test_runs_are_compared_per_frame_set_with_the_seed_range():
    validation = pd.DataFrame(
        {"sign_ap50": [0.9, 0.8, 0.7], "road_iou": [0.95, 0.97, 0.9], "score": [1, 1, 1]},
        index=["seed42", "seed43", "sum"],
    )
    test = validation[["sign_ap50", "road_iou"]] - 0.2
    table, band = compare_runs({"validation": validation, "test": test}, ["seed42", "seed43"])
    assert table.columns.tolist() == [
        ("validation", "sign_ap50"),
        ("validation", "road_iou"),
        ("test", "sign_ap50"),
        ("test", "road_iou"),
    ]
    assert band[("validation", "sign_ap50")].tolist() == pytest.approx([0.8, 0.9])
    assert band[("test", "road_iou")].tolist() == pytest.approx([0.75, 0.77])


def test_a_fixed_mask_is_scored_against_every_frame_s_road(indexed):
    from tests.conftest import HEIGHT, ROAD_TOP, WIDTH

    frames, _, label_dir = indexed
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[ROAD_TOP - 2 :] = True
    counts = fixed_mask_counts(mask, frames.head(2), label_dir)
    road = (HEIGHT - ROAD_TOP) * WIDTH
    # Two rows of sky above the road are called road too; six pixels are unlabeled.
    assert counts.tolist() == [[road, 2 * WIDTH, HEIGHT * WIDTH - road - 2 * WIDTH - 6, 0]] * 2
