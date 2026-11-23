"""Tests that the settings in `config` agree with each other."""

from __future__ import annotations

from driving_scenes import config


def test_the_partitions_use_every_world_once():
    worlds = [world for group in config.PARTITIONS.values() for world in group]
    assert len(worlds) == len(set(worlds)) == 5
    assert set(config.PARTITIONS) == {"train", "validation", "test"}


def test_held_out_runs_never_train_on_their_test_worlds_or_on_validation():
    validation = set(config.PARTITIONS["validation"])
    for plan in config.HELD_OUT.values():
        assert not set(plan["train"]) & set(plan["test"])
        assert not set(plan["train"]) & validation and not set(plan["test"]) & validation


def test_every_anchor_set_places_the_same_number_of_anchors_on_each_level():
    for sizes, ratios in config.ANCHOR_SETS.values():
        assert len(sizes) == len(config.ANCHOR_STRIDES)
        assert len({len(level) for level in sizes}) == 1
        assert all(ratio > 0 for ratio in ratios)


def test_variants_change_only_settings_that_a_run_has():
    allowed = {"stages", "epochs", "lr", "balance", "tasks"}
    for change in config.VARIANTS.values():
        assert set(change) <= allowed and len(change) == 1
    assert config.BALANCE in config.BALANCE_MODES


def test_thresholds_and_classes_are_consistent():
    assert list(config.SCORE_THRESHOLDS) == sorted(config.SCORE_THRESHOLDS)
    assert min(config.SCORE_THRESHOLDS) > 0 and max(config.SCORE_THRESHOLDS) < 1
    assert config.SIGN_CLASS in config.CLASSES and config.ROAD_CLASS in config.CLASSES
    assert len(config.CLASSES) <= config.IGNORE_LABEL
    assert (config.MIN_SIZE, config.MAX_SIZE) == (config.FRAME_HEIGHT, config.FRAME_WIDTH)
    assert len(config.SEEDS) >= 3 and config.SEEDS[0] == config.RANDOM_SEED
