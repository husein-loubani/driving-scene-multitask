"""Tests for the training plan: which runs exist and what each one changes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from driving_scenes.config import (
    BALANCE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FINETUNE_STAGES,
    HEAD_EPOCHS,
    HEAD_LR,
    HELD_OUT,
    PARTITIONS,
    TASKS,
    VARIANTS,
)
from driving_scenes.modeling import experiments
from driving_scenes.modeling.experiments import (
    carried_forward,
    held_out_predictions,
    split_frames,
    train_held_out,
    train_recipe,
    train_seeds,
    train_variant,
    variant_verdicts,
)


@pytest.fixture
def calls(monkeypatch):
    """Record every call to `fit` instead of training."""
    recorded = []

    def fake_fit(train, val, signs, name, **settings):
        recorded.append({"name": name, "train": train, "val": val} | settings)
        return Path(name)

    monkeypatch.setattr(experiments, "fit", fake_fit)
    return recorded


def frames_of(worlds) -> pd.DataFrame:
    return pd.DataFrame({"frame_id": [f"{w}_x" for w in worlds], "world": list(worlds)})


def test_the_split_keeps_every_world_whole():
    frames = frames_of(["0001", "0002", "0006", "0018", "0020", "0002"])
    parts = split_frames(frames)
    assert set(parts) == {"train", "validation", "test"}
    for name, worlds in PARTITIONS.items():
        assert set(parts[name]["world"]) == set(worlds)
    assert sum(len(part) for part in parts.values()) == len(frames)


def test_the_recipe_trains_heads_then_fine_tunes_from_them(calls):
    heads, tuned = train_recipe(frames_of(["0002"]), frames_of(["0018"]), None, seed=43)
    first, second = calls
    assert (first["name"], second["name"]) == ("heads_seed43", "recipe_seed43")
    assert (first["stages"], first["epochs"], first["lr"]) == (0, HEAD_EPOCHS, HEAD_LR)
    assert (second["stages"], second["epochs"], second["lr"]) == (
        FINETUNE_STAGES,
        FINETUNE_EPOCHS,
        FINETUNE_LR,
    )
    assert second["start"] == heads and second["shift"] == HEAD_EPOCHS
    assert second["balance"] == BALANCE and first["seed"] == second["seed"] == 43
    assert tuned == Path("recipe_seed43")


def test_each_variant_changes_exactly_one_setting_of_stage_two(calls):
    recipe = {
        "stages": FINETUNE_STAGES,
        "epochs": FINETUNE_EPOCHS,
        "lr": FINETUNE_LR,
        "balance": BALANCE,
        "tasks": TASKS,
    }
    for name, change in VARIANTS.items():
        train_variant(name, frames_of(["0002"]), frames_of(["0018"]), None, Path("heads"))
        settings = {key: calls[-1][key] for key in recipe}
        differs = {key for key in recipe if settings[key] != recipe[key]}
        assert differs == set(change), name
        assert calls[-1]["start"] == Path("heads") and calls[-1]["shift"] == HEAD_EPOCHS


def test_a_variant_with_another_seed_starts_from_that_seed_s_heads(calls):
    train_variant("lr_0.01", frames_of(["0002"]), frames_of(["0018"]), None, Path("h43"), seed=43)
    assert calls[-1]["name"] == "lr_0.01_seed43" and calls[-1]["seed"] == 43
    assert calls[-1]["start"] == Path("h43") and calls[-1]["lr"] == VARIANTS["lr_0.01"]["lr"]
    train_variant("lr_0.01", frames_of(["0002"]), frames_of(["0018"]), None, Path("h42"))
    assert calls[-1]["name"] == "lr_0.01" and calls[-1]["seed"] == 42


def test_every_seed_of_a_configuration_is_trained_from_its_own_heads(calls):
    recipes = {42: (Path("h42"), Path("recipe_seed42")), 43: (Path("h43"), Path("recipe_seed43"))}
    train, val = frames_of(["0002"]), frames_of(["0018"])
    assert train_seeds("recipe_seed42", train, val, None, recipes) == {
        "recipe_seed42": Path("recipe_seed42"),
        "recipe_seed43": Path("recipe_seed43"),
    }
    assert not calls, "the recipe's seeds are already trained"
    runs = train_seeds("lr_0.01", train, val, None, recipes)
    assert list(runs) == ["lr_0.01", "lr_0.01_seed43"]
    assert [call["start"] for call in calls] == [Path("h42"), Path("h43")]


def test_the_best_replacing_candidate_is_carried_forward():
    verdicts = pd.DataFrame(
        {"val_score": [0.95, 0.97, 0.96], "replaces the recipe": [False, True, True]},
        index=["sum", "lr_0.01", "lr_0.02"],
    )
    assert carried_forward(verdicts, "recipe_seed42") == "lr_0.01"
    verdicts["replaces the recipe"] = False
    assert carried_forward(verdicts, "recipe_seed42") == "recipe_seed42"


def test_a_variant_replaces_the_recipe_only_beyond_the_seed_range():
    scores = pd.Series(
        {"seed42": 0.90, "seed43": 0.92, "seed44": 0.91, "close": 0.925, "far": 0.94, "low": 0.85}
    )
    table = variant_verdicts(scores, ["seed42", "seed43", "seed44"], ["close", "far", "low"])
    # Seed mean 0.91 and range 0.02: only a score above 0.93 replaces the recipe.
    assert table["points above the seed mean"].tolist() == pytest.approx([1.5, 3.0, -6.0])
    assert table["seed range in points"].tolist() == pytest.approx([2.0] * 3)
    assert table["replaces the recipe"].tolist() == [False, True, False]


def test_a_held_out_run_trains_on_its_worlds_and_returns_the_others(calls):
    frames = frames_of(["0001", "0002", "0006", "0018", "0020"])
    for name, plan in HELD_OUT.items():
        tuned, test = train_held_out(name, frames, None)
        assert set(calls[-1]["train"]["world"]) == set(plan["train"])
        assert set(calls[-1]["val"]["world"]) == set(PARTITIONS["validation"])
        assert set(test["world"]) == set(plan["test"])
        assert tuned == Path(f"{name}_recipe_seed42")
        assert not set(plan["train"]) & set(plan["test"])
        assert not set(plan["train"]) & set(PARTITIONS["validation"])


def test_held_out_predictions_score_each_run_on_the_worlds_it_never_saw(monkeypatch, tmp_path):
    worlds = ["0001", "0002", "0006", "0018", "0020"]
    frames = frames_of(worlds).assign(variation="clone", step=0)
    predicted = []

    def fake_predictions(run, chosen, signs, path):
        predicted.append((run, set(chosen["world"]), path))
        return run

    monkeypatch.setattr(
        experiments,
        "train_held_out",
        lambda name, all_frames, signs: (
            Path(f"{name}_recipe_seed42"),
            all_frames[all_frames["world"].isin(HELD_OUT[name]["test"])],
        ),
    )
    monkeypatch.setattr(experiments, "cached_predictions", fake_predictions)
    result = held_out_predictions(frames, None, tmp_path)
    assert list(result) == list(HELD_OUT)
    for (run, worlds, path), (name, plan) in zip(predicted, HELD_OUT.items(), strict=True):
        assert worlds == set(plan["test"])
        assert path == tmp_path / f"{run.name}_test_moments.npz"
        assert run == Path(f"{name}_recipe_seed42")
