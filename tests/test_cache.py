"""Tests for the fingerprints that decide when a cached result can be reused."""

from __future__ import annotations

import numpy as np
import torch

from driving_scenes.cache import fingerprint, is_current, read_stamp, record, stamp_path


def test_the_same_inputs_give_the_same_fingerprint():
    parts = (
        np.arange(6, dtype=np.float32).reshape(2, 3),
        ["a.jpg", "b.jpg"],
        {"lr": 3e-5, "epochs": 4},
        42,
    )
    assert fingerprint(*parts) == fingerprint(*parts)


def test_any_change_in_data_settings_or_order_changes_the_fingerprint():
    base = fingerprint(np.array([1.0, 2.0]), {"lr": 3e-5}, 42)
    assert fingerprint(np.array([1.0, 2.5]), {"lr": 3e-5}, 42) != base
    assert fingerprint(np.array([1.0, 2.0]), {"lr": 1e-4}, 42) != base
    assert fingerprint(np.array([1.0, 2.0]), {"lr": 3e-5}, 43) != base
    assert fingerprint(np.array([2.0, 1.0]), {"lr": 3e-5}, 42) != base


def test_dictionary_order_does_not_matter_but_dtype_does():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint(np.array([1, 2], dtype=np.int64)) != fingerprint(
        np.array([1, 2], dtype=np.int32)
    )


def test_tensors_hash_like_their_arrays():
    values = np.arange(4, dtype=np.float32)
    assert fingerprint(torch.from_numpy(values)) == fingerprint(values)


def test_a_target_is_current_only_with_a_matching_stamp(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert not is_current(run, "abc"), "no stamp yet"
    record(run, "abc")
    assert stamp_path(run) == run / "fingerprint.txt"
    assert is_current(run, "abc") and not is_current(run, "xyz")
    assert read_stamp(run) == "abc"


def test_a_cached_file_keeps_its_stamp_beside_it(tmp_path):
    path = tmp_path / "scores.npy"
    record(path, "abc")
    assert stamp_path(path) == tmp_path / "scores.npy.fingerprint"
    assert not is_current(path, "abc"), "the stamp alone is not a cached result"
    np.save(path, np.zeros(2))
    assert is_current(path, "abc")


def test_a_run_folder_with_a_dot_in_its_name_keeps_its_stamp_inside(tmp_path):
    run = tmp_path / "lr_0.0001"
    run.mkdir()
    record(run, "abc")
    assert (run / "fingerprint.txt").exists() and is_current(run, "abc")
