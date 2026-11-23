"""
Every training run of the project, defined once.

The notebook and any background job call the same functions with the same
arguments, so a run trained in one is found finished by the other.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from driving_scenes.config import (
    BALANCE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FINETUNE_STAGES,
    HEAD_EPOCHS,
    HEAD_LR,
    HELD_OUT,
    PARTITIONS,
    RANDOM_SEED,
    TASKS,
    VARIANTS,
)
from driving_scenes.dataset import assign_partitions
from driving_scenes.modeling.train import fit


def split_frames(frames: pd.DataFrame, worlds=PARTITIONS) -> dict[str, pd.DataFrame]:
    """The frames of each partition, worlds kept whole."""
    partition = assign_partitions(frames, worlds)
    return {name: frames[partition == name] for name in worlds}


def train_recipe(
    train: pd.DataFrame,
    val: pd.DataFrame,
    signs: pd.DataFrame,
    seed: int = RANDOM_SEED,
    anchors: str = "adapted",
    prefix: str = "",
) -> tuple[Path, Path]:
    """
    The two stages of the final recipe: heads on the frozen backbone, then fine-tuning.

    Stage two continues the rendering cycle where stage one left it, so the
    eight epochs together show each moment under eight different renderings.
    """
    heads = fit(
        train, val, signs, f"{prefix}heads_seed{seed}",
        stages=0, epochs=HEAD_EPOCHS, lr=HEAD_LR, anchors=anchors, seed=seed,
    )  # fmt: skip
    tuned = fit(
        train, val, signs, f"{prefix}recipe_seed{seed}",
        stages=FINETUNE_STAGES, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, balance=BALANCE,
        anchors=anchors, seed=seed, start=heads, shift=HEAD_EPOCHS,
    )  # fmt: skip
    return heads, tuned


def train_variant(
    name: str, train: pd.DataFrame, val: pd.DataFrame, signs: pd.DataFrame, heads: Path
) -> Path:
    """Stage two of the recipe with the one change `VARIANTS[name]` makes, from the same heads."""
    settings = {
        "stages": FINETUNE_STAGES,
        "epochs": FINETUNE_EPOCHS,
        "lr": FINETUNE_LR,
        "balance": BALANCE,
        "tasks": TASKS,
    } | VARIANTS[name]
    return fit(
        train, val, signs, name, start=heads, shift=HEAD_EPOCHS, seed=RANDOM_SEED, **settings
    )


def train_held_out(
    name: str, frames: pd.DataFrame, signs: pd.DataFrame
) -> tuple[Path, pd.DataFrame]:
    """
    The recipe trained with the worlds of `HELD_OUT[name]`, and the frames it is tested on.

    Validation is the main split's validation world.
    """
    plan = HELD_OUT[name]
    train = frames[frames["world"].isin(plan["train"])]
    val = frames[frames["world"].isin(PARTITIONS["validation"])]
    _, tuned = train_recipe(train, val, signs, prefix=f"{name}_")
    return tuned, frames[frames["world"].isin(plan["test"])]
