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
    split_frames,
    train_held_out,
    train_recipe,
    train_variant,
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
