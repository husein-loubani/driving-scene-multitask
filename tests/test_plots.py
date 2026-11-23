"""Smoke tests for the figures: each builds, labels its axes and saves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from driving_scenes import plots
from driving_scenes.config import IGNORE_LABEL, ROAD_COLORS


def labeled(figure) -> bool:
    """Every axis that draws data has both axis labels; a color scale needs only its own."""
    for axis in figure.axes:
        if axis.get_label() == "<colorbar>":
            if not axis.get_ylabel():
                return False
            continue
        if not axis.has_data() or not axis.axison:
            continue
        if not (axis.get_xlabel() and axis.get_ylabel()):
            return False
    return True


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_the_road_overlay_colors_each_kind_of_call():
    image = np.full((1, 4, 3), 100, dtype=np.uint8)
    truth = np.array([[1, 0, 1, IGNORE_LABEL]], dtype=np.uint8)
    predicted = np.array([[True, True, False, True]])
    shown = plots.road_overlay(image, predicted, truth).astype(int)
    expected = [
        (1 - plots.ROAD_ALPHA) * 100 + plots.ROAD_ALPHA * np.array(ROAD_COLORS[name])
        for name in ("hit", "false", "miss")
    ]
    for column, color in enumerate(expected):
        np.testing.assert_allclose(shown[0, column], color, atol=1)
    assert shown[0, 3].tolist() == [100, 100, 100], "unlabeled pixels stay untouched"
    truth_only = plots.road_overlay(image, truth=truth).astype(int)
    assert truth_only[0, 1].tolist() == [100, 100, 100] and truth_only[0, 0, 0] != 100


def test_frame_panels_draw_boxes_and_a_legend_of_their_kinds():
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    panels = [
        {
            "image": image,
            "title": "a",
            "boxes": [((1, 1, 10, 30), "hit"), ((20, 1, 30, 30), "miss")],
        },
        {"image": image, "title": "b"},
    ]
    figure = plots.frame_panels(panels, columns=1)
    assert len(figure.axes[0].patches) == 2
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == ["sign found", "sign missed"]


def test_every_chart_labels_its_axes(tmp_path):
    shares = pd.DataFrame(
        {"road": [0.3, 0.2], "sky": [0.7, 0.8]}, index=pd.Index(["a", "b"], name="world")
    )
    signs = pd.DataFrame(
        {
            "width": [5, 10, 40, 100],
            "height": [20, 30, 80, 120],
            "aspect": [4.0, 3.0, 2.0, 1.2],
            "world": ["a", "a", "b", "b"],
            "size": pd.Categorical(["small", "small", "medium", "large"]),
        }
    )
    frames = pd.DataFrame(
        {
            "variation": ["clone", "fog"] * 3,
            "brightness": np.arange(6.0),
            "contrast": np.arange(6.0),
        }
    )
    repetition = pd.DataFrame(
        {
            "world": ["a", "a"],
            "lag": [1, 5],
            "road_iou": [0.9, 0.7],
            "signs_still_matched": [0.8, 0.5],
        }
    )
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss_classifier": [0.5, 0.4],
            "train_loss_road": [0.2, 0.1],
            "val_loss_classifier": [0.5, 0.4],
            "val_loss_road": [0.2, 0.1],
            "val_sign_ap50": [0.6, 0.7],
            "val_road_iou": [0.8, 0.9],
        }
    )
    pulls = pd.DataFrame(
        {
            "step": [0, 10],
            "pull_signs": [0.01, 0.02],
            "pull_road": [0.001, 0.002],
            "weight_signs": [1.0, 0.6],
            "weight_road": [1.0, 3.0],
        }
    )
    runs = pd.DataFrame(
        {("validation", "sign_ap50"): [0.9, 0.8], ("test", "road_iou"): [0.8, 0.7]},
        index=["recipe", "sum"],
    )
    groups = pd.DataFrame({"sign_ap50": [0.6, 0.5], "road_iou": [0.8, 0.7]}, index=["clone", "fog"])
    sweep = pd.DataFrame(
        {"threshold": [0.1, 0.5], "precision": [0.5, 0.8], "recall": [0.9, 0.6], "f1": [0.64, 0.69]}
    )
    sizes = pd.DataFrame({"recall": [0.4, 0.8], "ap": [0.2, 0.5]}, index=["small", "large"])
    figures = {
        "shares": plots.plot_class_shares(shares),
        "shapes": plots.plot_sign_shapes(signs),
        "light": plots.plot_rendering_light(frames),
        "repetition": plots.plot_repetition(repetition),
        "anchors": plots.plot_anchor_coverage(
            {"a": np.array([0.2, 0.6]), "b": np.array([0.5, 0.9])}, (0.5,)
        ),
        "ceiling": plots.plot_mask_ceiling(
            pd.DataFrame({"semantic head": [0.95, 0.9], "mask head": [0.8, 0.7]})
        ),
        "history": plots.plot_history({"one": history, "two": history}),
        "pulls": plots.plot_pulls(pulls),
        "runs": plots.plot_runs(runs),
        "groups": plots.plot_group_scores(groups, "Rendering"),
        "sweep": plots.plot_threshold_sweep(sweep, 0.5),
        "sizes": plots.plot_size_scores(sizes),
        "heat": plots.plot_heatmaps(
            {"map": (np.random.default_rng(0).random((4, 8)), "Share of frames")}
        ),
    }
    for name, figure in figures.items():
        assert labeled(figure), name
        assert plots.save_figure(figure, name, "test", root=tmp_path).exists()


def test_truth_panels_tint_the_road_and_frame_the_signs(indexed, raw_dir):
    from driving_scenes.dataset import boxes_by_frame

    frames, signs, label_dir = indexed
    chosen = frames.iloc[[1]]
    panels = plots.truth_panels(chosen, boxes_by_frame(signs), ["frame"], raw_dir, label_dir)
    assert len(panels[0]["boxes"]) == 2 and all(kind == "truth" for _, kind in panels[0]["boxes"])
    assert panels[0]["image"].shape == (48, 96, 3)
