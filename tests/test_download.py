"""Tests for the streamed, resumable extraction of the dataset archive."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from driving_scenes.download import (
    DONE,
    STATE_NAME,
    count_frames,
    extract_archive,
    read_offset,
    wanted_member,
)

FILES = {
    "meta.json": b'{"classes": []}',
    "ds/ann/depthgt_0001_clone_00000.png.json": b"d" * 3000,
    "ds/ann/rgb_0001_clone_00000.png.json": b"a" * 2500,
    "ds/ann/rgb_0001_clone_00001.png.json": b"b" * 700,
    "ds/img/flowgt_0001_clone_00000.png": b"f" * 5000,
    "ds/img/rgb_0001_clone_00000.png": bytes(range(256)) * 20,
    "ds/img/rgb_0001_clone_00001.png": b"i" * 1300,
}
KEPT = {
    "meta.json",
    "ann/rgb_0001_clone_00000.png.json",
    "ann/rgb_0001_clone_00001.png.json",
    "img/rgb_0001_clone_00000.png",
    "img/rgb_0001_clone_00001.png",
}


def make_archive(files: dict[str, bytes] = FILES) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        folder = tarfile.TarInfo("ds")
        folder.type = tarfile.DIRTYPE
        archive.addfile(folder)
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class Connection(io.BytesIO):
    """A stream that hands out small pieces and breaks after `limit` bytes."""

    def __init__(self, data: bytes, limit: int | None):
        super().__init__(data)
        self.limit = limit

    def read(self, size: int = -1) -> bytes:
        if self.limit is not None and self.tell() >= self.limit:
            raise ConnectionError("connection dropped")
        return super().read(700)


class Remote:
    """Serves an archive from any offset; each entry of `breaks` cuts one connection."""

    def __init__(self, data: bytes, breaks: tuple[int, ...] = (), served: int | None = None):
        self.data = data
        self.breaks = list(breaks)
        self.served = served
        self.offsets: list[int] = []

    def __call__(self, offset: int) -> tuple[Connection, int]:
        self.offsets.append(offset)
        limit = self.breaks.pop(0) if self.breaks else None
        end = self.served if self.served is not None and not self.offsets[1:] else None
        return Connection(self.data[offset:end], limit), len(self.data)


def extracted(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file() and path.name != STATE_NAME
    }


def expected_files() -> dict[str, bytes]:
    return {name: FILES[f"ds/{name}" if "/" in name else name] for name in KEPT}


def test_only_rgb_frames_their_annotations_and_the_meta_file_are_wanted():
    assert wanted_member("ds/img/rgb_0001_clone_00000.png") == Path("img/rgb_0001_clone_00000.png")
    assert wanted_member("/ds/ann/rgb_0020_fog_00012.png.json") == Path(
        "ann/rgb_0020_fog_00012.png.json"
    )
    assert wanted_member("meta.json") == Path("meta.json")
    assert wanted_member("ds/img/depthgt_0001_clone_00000.png") is None
    assert wanted_member("ds/ann/flowgt_0001_clone_00000.png.json") is None
    assert wanted_member("README.md") is None


def test_a_member_can_never_land_outside_the_raw_folder():
    assert wanted_member("../../etc/img/rgb_x.png") == Path("img/rgb_x.png")


def test_the_wanted_members_are_extracted_byte_for_byte(tmp_path):
    remote = Remote(make_archive())
    extract_archive(remote, tmp_path, wait=0)
    assert extracted(tmp_path) == expected_files()
    assert read_offset(tmp_path / STATE_NAME) is None
    assert (tmp_path / STATE_NAME).read_text() == DONE
    assert remote.offsets == [0]


def test_a_finished_extraction_never_opens_the_archive_again(tmp_path):
    extract_archive(Remote(make_archive()), tmp_path, wait=0)

    def unreachable(offset: int):
        raise AssertionError("the archive was opened again")

    extract_archive(unreachable, tmp_path, wait=0)


@pytest.mark.parametrize("breaks", [(1500,), (4200,), (6000, 2000, 3000), (9000, 100, 100)])
def test_a_dropped_connection_resumes_at_a_member_boundary(tmp_path, breaks):
    remote = Remote(make_archive(), breaks)
    extract_archive(remote, tmp_path, wait=0)
    assert extracted(tmp_path) == expected_files(), "every file is whole after the resumes"
    assert len(remote.offsets) > 1 and remote.offsets[0] == 0
    assert all(offset % tarfile.BLOCKSIZE == 0 for offset in remote.offsets)
    assert not list(tmp_path.rglob("*.part")), "no half-written file is left behind"


def test_a_stream_that_ends_early_without_an_error_is_resumed(tmp_path):
    data = make_archive()
    # The first connection ends silently after the fourth member, as a closed socket would.
    boundary = 512 + sum(512 + -(-len(c) // 512) * 512 for c in list(FILES.values())[:3])
    remote = Remote(data, served=boundary)
    extract_archive(remote, tmp_path, wait=0)
    assert extracted(tmp_path) == expected_files()
    assert remote.offsets == [0, boundary]


def test_a_stream_that_never_delivers_gives_up(tmp_path):
    remote = Remote(make_archive(), breaks=(0,) * 50)
    with pytest.raises(ConnectionError, match="stalled at byte 0"):
        extract_archive(remote, tmp_path, stalled_attempts=3, wait=0)
    assert remote.offsets == [0, 0, 0]


def test_frames_are_counted_per_folder(tmp_path):
    extract_archive(Remote(make_archive()), tmp_path, wait=0)
    assert count_frames(tmp_path) == {"img": 2, "ann": 2}
