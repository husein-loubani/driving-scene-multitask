"""
Fetch the RGB frames and their annotations from the archive the brief links.

The archive is a single 29.5 GB tar that carries three renderings of every
frame (RGB, depth, optical flow), each with a copy of the same annotation file.
The project needs the RGB third of it, about 17 GB. The tar is therefore never
stored: it is read as a stream straight from the HTTP response, the wanted
members are written to disk and everything else is skipped as it passes.

A tar has no index, but every member starts on a block boundary and the
position of the next member is known once the current one has been read. That
position is recorded after each member, so a dropped connection resumes with an
HTTP Range request at the last member boundary instead of starting over.
"""

from __future__ import annotations

import shutil
import tarfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import requests
from urllib3.exceptions import HTTPError as BrokenStream

from driving_scenes.config import (
    ANNOTATION_FOLDER,
    ARCHIVE_URL,
    CONNECT_TIMEOUT,
    EXPECTED_FRAMES,
    IMAGE_FOLDER,
    IMAGE_KIND,
    META_NAME,
    RAW_DIR,
    READ_TIMEOUT,
    RETRY_WAIT,
    STALLED_ATTEMPTS,
    STREAM_BLOCK,
)

STATE_NAME = "download_offset.txt"
DONE = "done"
# Errors that mean the connection broke, not that the archive or the disk is wrong.
STREAM_ERRORS = (
    requests.RequestException,
    BrokenStream,
    tarfile.ReadError,
    ConnectionError,
    TimeoutError,
)


def wanted_member(name: str) -> Path | None:
    """
    Where an archive member belongs under the raw folder, or None when it is not needed.

    Only the file name and its parent folder are taken from the archive, so a
    member can never be written outside the raw folder.
    """
    path = PurePosixPath(name)
    if path.name == META_NAME:
        return Path(META_NAME)
    if path.parent.name in (ANNOTATION_FOLDER, IMAGE_FOLDER) and path.name.startswith(
        f"{IMAGE_KIND}_"
    ):
        return Path(path.parent.name) / path.name
    return None


def read_offset(state: Path) -> int | None:
    """The archive position to resume from, or None when the extraction has finished."""
    if not state.exists():
        return 0
    text = state.read_text().strip()
    return None if text == DONE else int(text)


def extract_stream(
    stream: BinaryIO,
    start: int,
    destination: Path,
    keep: Callable[[str], Path | None],
    state: Path,
) -> None:
    """
    Read tar members from `stream`, which begins at archive position `start`.

    Wanted members are written through a temporary name, so a member cut off by
    a broken connection never leaves a truncated file behind. After each member
    the position of the next one is saved to `state`.
    """
    with tarfile.open(fileobj=stream, mode="r|", bufsize=STREAM_BLOCK) as archive:
        for member in archive:
            target = keep(member.name) if member.isfile() else None
            if target is not None:
                path = destination / target
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".part")
                with open(partial, "wb") as output:
                    shutil.copyfileobj(archive.extractfile(member), output, STREAM_BLOCK)
                partial.replace(path)
            state.write_text(str(start + archive.offset))


def extract_archive(
    open_at: Callable[[int], tuple[BinaryIO, int]],
    destination: Path,
    keep: Callable[[str], Path | None] = wanted_member,
    stalled_attempts: int = STALLED_ATTEMPTS,
    wait: float = RETRY_WAIT,
) -> None:
    """
    Extract the wanted members of a remote tar, resuming after every broken stream.

    `open_at(offset)` returns a readable stream that starts at byte `offset`
    of the archive, and the archive's total size. The extraction is finished
    when a pass ends within the tar's end-of-archive padding.
    """
    destination.mkdir(parents=True, exist_ok=True)
    state = destination / STATE_NAME
    failures = 0
    while (offset := read_offset(state)) is not None:
        try:
            stream, total = open_at(offset)
            with stream:
                extract_stream(stream, offset, destination, keep, state)
            reached = read_offset(state)
            # A tar ends with two empty blocks and is padded to a full record.
            if total - reached <= tarfile.RECORDSIZE:
                state.write_text(DONE)
        except STREAM_ERRORS:
            reached = read_offset(state)
        failures = 0 if reached is None or reached > offset else failures + 1
        if failures >= stalled_attempts:
            raise ConnectionError(f"the archive stream stalled at byte {offset:,}")
        if failures:
            time.sleep(wait)


def open_remote(url: str) -> Callable[[int], tuple[BinaryIO, int]]:
    """A function that opens `url` from a byte offset with an HTTP Range request."""

    def open_at(offset: int) -> tuple[BinaryIO, int]:
        response = requests.get(
            url,
            headers={"Range": f"bytes={offset}-"},
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        total = int(response.headers["Content-Range"].rpartition("/")[2])
        return response.raw, total

    return open_at


def count_frames(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """How many RGB frames and annotation files the raw folder holds."""
    return {
        folder: sum(1 for _ in (raw_dir / folder).glob(f"{IMAGE_KIND}_*"))
        for folder in (IMAGE_FOLDER, ANNOTATION_FOLDER)
    }


def download_virtual_kitti(
    raw_dir: Path = RAW_DIR, url: str = ARCHIVE_URL, expected: int = EXPECTED_FRAMES
) -> dict[str, int]:
    """
    Make sure every RGB frame and its annotation is on disk, and return their counts.

    The first call streams the archive (about 20 minutes on a fast line). Later
    calls only count the files.
    """
    extract_archive(open_remote(url), raw_dir)
    counts = count_frames(raw_dir)
    if set(counts.values()) != {expected}:
        raise RuntimeError(f"expected {expected:,} frames and annotations, found {counts}")
    return counts
