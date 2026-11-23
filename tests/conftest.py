"""
Shared fixtures: a tiny synthetic copy of the dataset in the archive's own format.

The suite never touches Virtual KITTI. A synthetic frame is a flat sky over a
band of road with one or two "signs" painted on it, annotated the way the
archive annotates a frame: one bitmap per connected region, plus a tracking
rectangle that the project must ignore.
"""

from __future__ import annotations

import base64
import io
import json
import zlib

import matplotlib
import numpy as np
import pytest
from PIL import Image

from driving_scenes.config import ANNOTATION_FOLDER, IMAGE_FOLDER

matplotlib.use("Agg")

HEIGHT, WIDTH = 48, 96
ROAD_TOP = 30
# Sign boxes (x_min, y_min, x_max, y_max) per step; step 2 has no sign.
SIGNS = {0: [(10, 8, 16, 28)], 1: [(12, 8, 18, 28), (70, 4, 90, 12)], 2: []}
WORLDS = ("0001", "0002")
RENDERINGS = ("clone", "fog")
# One pixel block per frame is left without a label, as in the archive.
UNLABELED = (slice(0, 2), slice(0, 3))


def encode_bitmap(mask: np.ndarray, left: int, top: int) -> dict:
    """A mask in the archive's bitmap format: palette PNG, zlib, base64."""
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette([0, 0, 0, 255, 255, 255])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(zlib.compress(buffer.getvalue())).decode()
    return {"data": data, "origin": [left, top]}


def region(class_title: str, mask: np.ndarray) -> dict:
    """One connected region as an annotation object, cropped to its extent."""
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    crop = mask[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    return {
        "classTitle": class_title,
        "geometryType": "bitmap",
        "bitmap": encode_bitmap(crop, int(columns[0]), int(rows[0])),
        "tags": [],
    }


def make_frame(world: str, rendering: str, step: int) -> tuple[Image.Image, dict]:
    """A synthetic frame and its annotation."""
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[:ROAD_TOP] = (120, 170, 230) if rendering == "clone" else (150, 150, 150)
    pixels[ROAD_TOP:] = (90, 90, 90)
    sky = np.zeros((HEIGHT, WIDTH), dtype=bool)
    sky[:ROAD_TOP] = True
    road = ~sky
    objects = []
    for x_min, y_min, x_max, y_max in SIGNS[step]:
        sign = np.zeros((HEIGHT, WIDTH), dtype=bool)
        sign[y_min:y_max, x_min:x_max] = True
        pixels[sign] = (220, 40, 40)
        sky &= ~sign
        objects.append(region("traffic sign", sign))
    sky[UNLABELED] = False
    objects += [region("sky", sky), region("road", road)]
    objects.append(
        {
            "classTitle": "car",
            "geometryType": "rectangle",
            "points": {"exterior": [[1, 1], [9, 9]], "interior": []},
            "tags": [],
        }
    )
    annotation = {
        "size": {"height": HEIGHT, "width": WIDTH},
        "tags": [
            {"name": "im_id", "value": f"{world}_{rendering}_{step:05d}"},
            {"name": "sequence", "value": world},
            {"name": "rotation", "value": rendering},
        ],
        "objects": objects,
    }
    return Image.fromarray(pixels), annotation


@pytest.fixture
def raw_dir(tmp_path):
    """A raw folder with two worlds, two renderings and three steps each."""
    root = tmp_path / "raw"
    (root / IMAGE_FOLDER).mkdir(parents=True)
    (root / ANNOTATION_FOLDER).mkdir()
    for world in WORLDS:
        for rendering in RENDERINGS:
            for step in SIGNS:
                name = f"rgb_{world}_{rendering}_{step:05d}.png"
                image, annotation = make_frame(world, rendering, step)
                image.save(root / IMAGE_FOLDER / name)
                (root / ANNOTATION_FOLDER / f"{name}.json").write_text(json.dumps(annotation))
    return root


@pytest.fixture
def indexed(raw_dir, tmp_path):
    """The synthetic dataset after indexing: frame table, sign table and label folder."""
    from driving_scenes.dataset import build_index, frame_table

    label_dir = tmp_path / "labels"
    frames, signs = build_index(
        frame_table(raw_dir),
        raw_dir,
        label_dir,
        tmp_path / "frames.csv",
        tmp_path / "signs.csv",
        n_jobs=1,
    )
    return frames, signs, label_dir
