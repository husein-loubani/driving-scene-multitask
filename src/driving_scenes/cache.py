"""
Fingerprints for every result cached on disk.

A cached result is reused only when the fingerprint stored beside it matches
the fingerprint of the inputs and settings that would produce it now. Change
the training frames, a label, a hyperparameter or a seed, and the old
result is recomputed instead of being reloaded because its name still matches.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

STAMP_NAME = "fingerprint.txt"
DIGEST_LENGTH = 16


def fingerprint(*parts) -> str:
    """A short SHA-256 over arrays, tensors, mappings, sequences, strings and numbers."""
    digest = hashlib.sha256()
    for part in parts:
        _feed(digest, part)
    return digest.hexdigest()[:DIGEST_LENGTH]


def _feed(digest, part) -> None:
    if isinstance(part, torch.Tensor):
        part = part.detach().cpu().numpy()
    if isinstance(part, np.ndarray):
        array = part.astype(str) if part.dtype == object else np.ascontiguousarray(part)
        digest.update(f"array{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    elif isinstance(part, Mapping):
        digest.update(f"map{len(part)}".encode())
        for key in sorted(part, key=str):
            _feed(digest, str(key))
            _feed(digest, part[key])
    elif isinstance(part, list | tuple):
        digest.update(f"seq{len(part)}".encode())
        for item in part:
            _feed(digest, item)
    else:
        digest.update(f"{type(part).__name__}:{part!r}".encode())


def stamp_path(target: Path) -> Path:
    """
    Where a target's fingerprint lives: inside a run folder, or beside a cached file.

    A folder is recognized on disk rather than by its name, since run names such
    as `lr_0.0001` contain dots. A target that does not exist yet is never
    current, so its stamp location only matters once it has been written.
    """
    target = Path(target)
    if target.is_dir():
        return target / STAMP_NAME
    return target.with_name(target.name + ".fingerprint")


def is_current(target: Path, key: str) -> bool:
    """True when the target exists and was produced from exactly these inputs."""
    stamp = stamp_path(target)
    return target.exists() and stamp.exists() and stamp.read_text().strip() == key


def record(target: Path, key: str) -> None:
    stamp = stamp_path(target)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(key + "\n")


def read_stamp(target: Path) -> str:
    """The fingerprint a finished run or cached file was recorded with."""
    return stamp_path(target).read_text().strip()
