"""Tests for the training stages: the schedule, the task balance and the cached runs."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from driving_scenes import cache
from driving_scenes import dataset as dataset_module
from driving_scenes.modeling import model as model_module
from driving_scenes.modeling import train as train_module
from driving_scenes.modeling.train import (
    MONITOR,
    TaskBalance,
    best_epoch,
    fit,
    load_model,
    pull_on,
    read_history,
    read_pulls,
    read_settings,
    scale_gradient,
    steps_per_epoch,
    warmup_cosine,
)
from tests.conftest import HEIGHT, WIDTH


def test_the_learning_rate_climbs_then_follows_a_cosine_to_zero():
    rate = [warmup_cosine(step, total=1000, warmup=100, start=0.001) for step in range(1001)]
    assert rate[0] == pytest.approx(0.001) and rate[100] == pytest.approx(1.0)
    assert rate[50] == pytest.approx(0.001 + 0.999 * 0.5)
    assert rate[550] == pytest.approx(0.5) and rate[1000] == pytest.approx(0.0, abs=1e-12)
    assert all(a <= b for a, b in zip(rate[:100], rate[1:101], strict=True))
    assert all(a >= b for a, b in zip(rate[100:-1], rate[101:], strict=True))


def test_a_short_stage_never_spends_more_than_half_its_steps_warming_up():
    assert warmup_cosine(10, total=20, warmup=100) == pytest.approx(1.0)


def test_gradient_scaling_changes_the_gradient_and_not_the_value():
    tensor = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    scaled = scale_gradient(tensor, 0.25)
    torch.testing.assert_close(scaled, tensor)
    scaled.sum().backward()
    assert tensor.grad.tolist() == [0.25, 0.25, 0.25]


def test_a_task_s_pull_is_the_norm_of_its_gradient_at_the_pyramid():
    features = {level: torch.ones(2, requires_grad=True) for level in "0123"}
    loss = sum(3.0 * features[level].sum() for level in "0123")
    # Eight entries, each with gradient 3.
    assert pull_on(features, loss) == pytest.approx(3.0 * math.sqrt(8))


def test_the_sum_mode_never_reweights():
    balance = TaskBalance(("signs", "road"), "sum")
    balance.update({"signs": 8.0, "road": 0.5})
    assert balance.weights() == {"signs": 1.0, "road": 1.0}


def test_the_equal_mode_makes_the_weighted_pulls_equal_and_keeps_their_sum():
    balance = TaskBalance(("signs", "road"), "equal")
    assert balance.weights() == {"signs": 1.0, "road": 1.0}, "nothing is measured yet"
    balance.update({"signs": 8.0, "road": 2.0})
    weights = balance.weights()
    assert weights["signs"] * 8.0 == pytest.approx(weights["road"] * 2.0)
    assert weights["signs"] * 8.0 + weights["road"] * 2.0 == pytest.approx(10.0)


def test_pulls_are_smoothed_across_measurements():
    balance = TaskBalance(("signs", "road"), "equal", momentum=0.5)
    balance.update({"signs": 4.0, "road": 1.0})
    balance.update({"signs": 8.0, "road": 1.0})
    assert balance.pull == {"signs": 6.0, "road": 1.0}


def test_an_unknown_balance_mode_is_refused():
    with pytest.raises(ValueError, match="balance must be one of"):
        TaskBalance(("signs", "road"), "loudest")


@pytest.fixture
def tiny(indexed, raw_dir, tmp_path, monkeypatch):
    """Everything `fit` needs to train on the synthetic frames in a few seconds on the CPU."""
    monkeypatch.setattr(model_module, "MIN_SIZE", HEIGHT)
    monkeypatch.setattr(model_module, "MAX_SIZE", WIDTH)
    monkeypatch.setattr(train_module, "PULL_EVERY", 1)
    # The synthetic worlds have two renderings, so the cycle over renderings has length two.
    monkeypatch.setattr(dataset_module, "VARIATIONS", ("clone", "fog"))
    frames, signs, label_dir = indexed
    train, val = frames[frames["world"] == "0001"], frames[frames["world"] == "0002"]
    common = {
        "run_dir": tmp_path / "runs",
        "raw_dir": raw_dir,
        "label_dir": label_dir,
        "workers": 0,
        "accelerator": "cpu",
    }
    return train, val, signs, common


def test_two_stages_train_reload_and_log(tiny):
    train, val, signs, common = tiny
    assert steps_per_epoch(train) == 1, "three moments, one rendering each, in one batch of four"
    heads = fit(train, val, signs, "heads", stages=0, epochs=1, lr=0.01, **common)
    assert cache.is_current(heads, cache.read_stamp(heads))
    assert read_settings(heads)["start"] is None
    history = read_history(heads)
    assert history["epoch"].tolist() == [1]
    assert {"train_loss", "train_loss_road", "val_loss_classifier", "val_road_iou"} <= set(history)
    assert {"val_sign_ap50", "val_sign_map", MONITOR} <= set(history)
    assert read_pulls(heads).empty, "a frozen backbone is not shared, so nothing is measured"

    tuned = fit(
        train, val, signs, "tuned", stages=1, epochs=2, lr=0.01, balance="equal",
        start=heads, shift=1, **common,
    )  # fmt: skip
    assert read_settings(tuned)["start"] == cache.read_stamp(heads)
    pulls = read_pulls(tuned)
    assert len(pulls) == 2 and (pulls[["pull_signs", "pull_road"]] > 0).all().all()
    first, second = pulls.iloc[0], pulls.iloc[1]
    assert (first["weight_signs"], first["weight_road"]) == (1.0, 1.0)
    assert second["weight_signs"] * first["pull_signs"] == pytest.approx(
        second["weight_road"] * first["pull_road"]
    )

    network = load_model(tuned)
    assert not network.training and network.cache_key == cache.read_stamp(tuned)
    kept = torch.load(tuned / "best.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    torch.testing.assert_close(
        network.road_head.predictor.weight, kept["model.road_head.predictor.weight"]
    )


def test_a_finished_run_is_reused_until_a_setting_changes(tiny, monkeypatch):
    train, val, signs, common = tiny
    first = fit(train, val, signs, "heads", stages=0, epochs=1, lr=0.01, **common)
    stamp = cache.read_stamp(first)
    calls = []
    real = train_module.Trainer.fit
    monkeypatch.setattr(
        train_module.Trainer, "fit", lambda self, *args: calls.append(1) or real(self, *args)
    )
    assert fit(train, val, signs, "heads", stages=0, epochs=1, lr=0.01, **common) == first
    assert not calls, "the same request trains nothing"
    fit(train, val, signs, "heads", stages=0, epochs=1, lr=0.02, **common)
    assert calls == [1] and cache.read_stamp(first) != stamp, "another learning rate retrains"


def test_a_single_task_run_trains_and_scores_only_its_task(tiny):
    train, val, signs, common = tiny
    run = fit(train, val, signs, "road", stages=1, epochs=1, lr=0.01, tasks=("road",), **common)
    history = read_history(run)
    assert "val_road_iou" in history and "val_sign_ap50" not in history
    assert history[MONITOR].iloc[0] == pytest.approx(history["val_road_iou"].iloc[0])
    assert read_pulls(run).empty, "one task has nobody to balance against"
    assert load_model(run).tasks == ("road",)


def test_the_kept_epoch_is_the_first_with_the_best_validation_score(tmp_path):
    log = pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1, 2, 2],
            "step": [1, 1, 3, 3, 5, 5],
            "train_loss": [None, 2.0, None, 1.0, None, 0.5],
            "val_loss_road": [0.4, None, 0.3, None, 0.2, None],
            "val_road_iou": [0.5, None, 0.8, None, 0.8, None],
            MONITOR: [0.5, None, 0.8, None, 0.8, None],
        }
    )
    log.to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "settings.json").write_text('{"minutes": 12.5}')
    kept = best_epoch(tmp_path)
    assert kept.to_dict() == {"epoch": 2, "minutes": 12.5, "val_road_iou": 0.8, MONITOR: 0.8}


def test_history_has_one_row_per_epoch(tmp_path):
    log = pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1],
            "step": [1, 1, 3, 3],
            "train_loss": [None, 2.0, None, 1.0],
            "val_score": [0.5, None, 0.7, None],
        }
    )
    log.to_csv(tmp_path / "metrics.csv", index=False)
    history = read_history(tmp_path)
    assert history["epoch"].tolist() == [1, 2]
    assert history["train_loss"].tolist() == [2.0, 1.0] and "step" not in history
