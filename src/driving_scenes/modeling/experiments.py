"""
Every training run of the project, defined once.

The notebook and any background job call the same functions with the same
arguments, so a run trained in one is found finished by the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    PREDICTION_DIR,
    RANDOM_SEED,
    TASKS,
    VARIANTS,
)
from driving_scenes.dataset import assign_partitions, one_rendering_per_step
from driving_scenes.modeling.evaluate import Predictions, cached_predictions
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
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    signs: pd.DataFrame,
    heads: Path,
    seed: int = RANDOM_SEED,
) -> Path:
    """
    Stage two of the recipe with the one change `VARIANTS[name]` makes, from the given heads.

    With another seed than the default, `heads` should be the stage one of that
    seed, and the run is named after the seed as well.
    """
    settings = {
        "stages": FINETUNE_STAGES,
        "epochs": FINETUNE_EPOCHS,
        "lr": FINETUNE_LR,
        "balance": BALANCE,
        "tasks": TASKS,
    } | VARIANTS[name]
    run = name if seed == RANDOM_SEED else f"{name}_seed{seed}"
    return fit(train, val, signs, run, start=heads, shift=HEAD_EPOCHS, seed=seed, **settings)


def variant_verdicts(
    scores: pd.Series, seeds: Sequence[str], candidates: Sequence[str]
) -> pd.DataFrame:
    """
    Whether each candidate replaces the recipe, by the rule fixed before the runs.

    `scores` holds the validation score of every run by name. A candidate
    replaces the recipe only when it beats the mean of the recipe's seed runs
    by more than their range, the run-to-run noise of the recipe itself. The
    margin and the range are given in points (hundredths of the score).
    """
    recipe = scores[list(seeds)]
    mean, noise = recipe.mean(), recipe.max() - recipe.min()
    table = pd.DataFrame({"val_score": scores[list(candidates)]})
    table["points above the seed mean"] = 100 * (table["val_score"] - mean)
    table["seed range in points"] = 100 * noise
    table["replaces the recipe"] = table["val_score"] - mean > noise
    return table


def carried_forward(verdicts: pd.DataFrame, recipe: str) -> str:
    """The run carried forward: the best candidate that replaces the recipe, or the recipe."""
    winners = verdicts[verdicts["replaces the recipe"]]
    return recipe if winners.empty else str(winners["val_score"].astype(float).idxmax())


def train_seeds(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    signs: pd.DataFrame,
    recipes: Mapping[int, tuple[Path, Path]],
) -> dict[str, Path]:
    """
    Every seed of one configuration, by run name: the recipe's own runs, or a variant's.

    `recipes` maps each seed to the two stages `train_recipe` returned for it; a
    variant is trained from each seed's own stage one.
    """
    if name not in VARIANTS:
        return {tuned.name: tuned for _, tuned in recipes.values()}
    runs = [
        train_variant(name, train, val, signs, heads, seed) for seed, (heads, _) in recipes.items()
    ]
    return {run.name: run for run in runs}


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


def held_out_predictions(
    frames: pd.DataFrame, signs: pd.DataFrame, prediction_dir: Path = PREDICTION_DIR
) -> dict[str, Predictions]:
    """
    Each held-out run's predictions on one rendering per moment of the worlds it never saw.

    The prediction files are named as the training queue names them, so the
    predictions it computed are found and reused.
    """
    result = {}
    for name in HELD_OUT:
        run, held = train_held_out(name, frames, signs)
        path = prediction_dir / f"{run.name}_test_moments.npz"
        result[name] = cached_predictions(run, one_rendering_per_step(held), signs, path)
    return result
