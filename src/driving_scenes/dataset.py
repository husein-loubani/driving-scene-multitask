"""
Frames, their labels, and the tables built from them.

The archive describes each frame with a JSON file of objects. An object is one
connected region of one class, stored as a bitmap: a PNG cropped to the region,
compressed and written as text, plus the position of its top-left corner. The
brief's two targets come out of those bitmaps:

- the road mask is the union of the road regions;
- a traffic sign box is the extent of one traffic-sign region, since the
  dataset draws boxes only around vehicles.

Decoding 150 bitmaps per frame on every training step would dominate the
loader, so each annotation is decoded once into a label map: a small PNG whose
pixels hold the class index, and 255 where the archive gives no label. The same
pass decodes the frame itself in full, which doubles as the integrity check of
the download, and measures what the exploration needs.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from PIL import Image
from torchvision.transforms.v2 import functional as transforms

from driving_scenes import cache
from driving_scenes.config import (
    ANNOTATION_FOLDER,
    CLASSES,
    FRAME_TABLE,
    IGNORE_LABEL,
    IMAGE_FOLDER,
    IMAGE_KIND,
    LABEL_DIR,
    N_JOBS,
    PROCESSED_DIR,
    RAW_DIR,
    ROAD_CLASS,
    SIGN_CLASS,
    SIGN_TABLE,
    VARIATIONS,
)

FRAME_NAME = re.compile(
    rf"{IMAGE_KIND}_(?P<world>\d{{4}})_(?P<variation>[a-z0-9-]+)_(?P<step>\d{{5}})\.png$"
)
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
ROAD_INDEX = CLASS_INDEX[ROAD_CLASS]
SIGN_INDEX = CLASS_INDEX[SIGN_CLASS]
BOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
# ITU-R BT.601 weights: how bright a color looks.
LUMA = np.array([0.299, 0.587, 0.114])


def frame_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    One row per RGB frame on disk: its world, rendering and position in the video.

    A world is one of the five KITTI videos rebuilt in the game engine, a
    variation is one of its ten renderings, and `step` counts the frames of the
    video, so a (world, step) pair names one moment shown ten times.
    """
    rows = []
    for path in sorted((raw_dir / IMAGE_FOLDER).glob(f"{IMAGE_KIND}_*.png")):
        match = FRAME_NAME.match(path.name)
        if match is None:
            raise ValueError(f"unexpected frame name: {path.name}")
        world, variation, step = match["world"], match["variation"], int(match["step"])
        rows.append(
            {
                "frame_id": f"{world}_{variation}_{match['step']}",
                "world": world,
                "variation": variation,
                "step": step,
                "file": path.name,
            }
        )
    frames = pd.DataFrame(rows, columns=["frame_id", "world", "variation", "step", "file"])
    unknown = set(frames["variation"]) - set(VARIATIONS)
    if unknown:
        raise ValueError(f"unexpected renderings: {sorted(unknown)}")
    return frames


def decode_bitmap(bitmap: Mapping) -> tuple[np.ndarray, int, int]:
    """
    A Supervisely bitmap as a boolean mask with the column and row of its top-left corner.

    The mask travels as a palette PNG, compressed with zlib and written in base64.
    """
    png = zlib.decompress(base64.b64decode(bitmap["data"]))
    mask = np.asarray(Image.open(io.BytesIO(png))) > 0
    left, top = bitmap["origin"]
    return mask, left, top


def mask_box(mask: np.ndarray, left: int = 0, top: int = 0) -> tuple[int, int, int, int]:
    """The tight box (x_min, y_min, x_max, y_max) of a mask, the maxima exclusive."""
    columns = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    return (
        left + int(columns[0]),
        top + int(rows[0]),
        left + int(columns[-1]) + 1,
        top + int(rows[-1]) + 1,
    )


def read_annotation(path: Path) -> tuple[np.ndarray, list[dict], dict]:
    """
    Decode one annotation file into a label map, its traffic-sign regions and a few checks.

    Only bitmaps are read: the file's rectangles are the tracking boxes of
    cars and vans, which this project does not use.
    """
    annotation = json.loads(path.read_bytes())
    size = annotation["size"]
    labels = np.full((size["height"], size["width"]), IGNORE_LABEL, dtype=np.uint8)
    signs, overlapping = [], 0
    for item in annotation["objects"]:
        if item["geometryType"] != "bitmap":
            continue
        mask, left, top = decode_bitmap(item["bitmap"])
        window = labels[top : top + mask.shape[0], left : left + mask.shape[1]]
        overlapping += int((window[mask] != IGNORE_LABEL).sum())
        window[mask] = CLASS_INDEX[item["classTitle"]]
        if item["classTitle"] == SIGN_CLASS:
            box = mask_box(mask, left, top)
            signs.append(dict(zip(BOX_COLUMNS, box, strict=True)) | {"pixels": int(mask.sum())})
    tags = {tag["name"]: tag["value"] for tag in annotation["tags"]}
    checks = {
        "overlapping_pixels": overlapping,
        "tagged_world": tags.get("sequence"),
        "tagged_variation": tags.get("rotation"),
    }
    return labels, signs, checks


def label_path(frame_id: str, label_dir: Path = LABEL_DIR) -> Path:
    return label_dir / f"{frame_id}.png"


def index_frame(file: str, frame_id: str, raw_dir: Path, label_dir: Path) -> tuple[dict, list]:
    """Decode one frame and its annotation, write the label map, and return what was measured."""
    pixels = np.asarray(Image.open(raw_dir / IMAGE_FOLDER / file).convert("RGB"))
    labels, signs, checks = read_annotation(raw_dir / ANNOTATION_FOLDER / f"{file}.json")
    Image.fromarray(labels).save(label_path(frame_id, label_dir))
    counts = np.bincount(labels.ravel(), minlength=IGNORE_LABEL + 1)
    luma = pixels @ LUMA
    row = {
        "frame_id": frame_id,
        "width": pixels.shape[1],
        "height": pixels.shape[0],
        "labels_match_frame": labels.shape == pixels.shape[:2],
        "brightness": float(luma.mean()),
        "contrast": float(luma.std()),
        "red": float(pixels[..., 0].mean()),
        "green": float(pixels[..., 1].mean()),
        "blue": float(pixels[..., 2].mean()),
        "unlabeled_pixels": int(counts[IGNORE_LABEL]),
        "signs": len(signs),
        **checks,
        **{f"pixels_{name}": int(counts[index]) for name, index in CLASS_INDEX.items()},
    }
    return row, [{"frame_id": frame_id} | sign for sign in signs]


def build_index(
    frames: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
    frame_table_path: Path = FRAME_TABLE,
    sign_table_path: Path = SIGN_TABLE,
    n_jobs: int = N_JOBS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decode every frame and annotation once, in parallel, and cache the result.

    Returns the frame table with one row of measurements per frame, and the
    sign table with one row per traffic-sign region. The cache is keyed on the
    frame list and the size of every annotation file, so a changed download
    rebuilds it.
    """
    annotation_sizes = np.array(
        [(raw_dir / ANNOTATION_FOLDER / f"{file}.json").stat().st_size for file in frames["file"]]
    )
    key = cache.fingerprint("index", frames["frame_id"].to_numpy(), annotation_sizes, CLASSES)
    if cache.is_current(frame_table_path, key) and cache.is_current(sign_table_path, key):
        return (
            pd.read_csv(frame_table_path, dtype={"world": str, "tagged_world": str}),
            pd.read_csv(sign_table_path),
        )
    label_dir.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(index_frame)(file, frame_id, raw_dir, label_dir)
        for file, frame_id in zip(frames["file"], frames["frame_id"], strict=True)
    )
    measured = pd.DataFrame([row for row, _ in results])
    table = frames.merge(measured, on="frame_id", validate="one_to_one")
    signs = pd.DataFrame(
        [sign for _, found in results for sign in found],
        columns=["frame_id", *BOX_COLUMNS, "pixels"],
    )
    frame_table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(frame_table_path, index=False)
    signs.to_csv(sign_table_path, index=False)
    cache.record(frame_table_path, key)
    cache.record(sign_table_path, key)
    return table, signs


def audit_summary(frames: pd.DataFrame, signs: pd.DataFrame) -> pd.Series:
    """What the full decode found, as counts a reader can check at a glance."""
    pixels = frames["width"] * frames["height"]
    labeled = [f"pixels_{name}" for name in CLASSES]
    return pd.Series(
        {
            "frames": len(frames),
            "worlds": frames["world"].nunique(),
            "renderings per world": frames.groupby("world")["variation"].nunique().min(),
            "frame sizes (width x height)": ", ".join(
                sorted(
                    {f"{w} x {h}" for w, h in zip(frames["width"], frames["height"], strict=True)}
                )
            ),
            "frames whose label map has another size": int((~frames["labels_match_frame"]).sum()),
            "frames whose tags disagree with their name": int(
                (
                    (frames["tagged_world"] != frames["world"])
                    | (frames["tagged_variation"] != frames["variation"])
                ).sum()
            ),
            "pixels claimed by two regions": int(frames["overlapping_pixels"].sum()),
            "frames with every pixel labeled": int((frames["unlabeled_pixels"] == 0).sum()),
            "percent of pixels without a label": float(
                100 * frames["unlabeled_pixels"].sum() / pixels.sum()
            ),
            "labeled pixels accounted for": bool(
                (frames[labeled].sum(axis=1) + frames["unlabeled_pixels"] == pixels).all()
            ),
            "traffic-sign regions": len(signs),
            "smallest region kept by the archive (pixels)": int(signs["pixels"].min()),
        }
    )


def load_image(file: str, raw_dir: Path = RAW_DIR) -> np.ndarray:
    """A frame as an RGB array of shape (height, width, 3)."""
    return np.asarray(Image.open(raw_dir / IMAGE_FOLDER / file).convert("RGB"))


def load_labels(frame_id: str, label_dir: Path = LABEL_DIR) -> np.ndarray:
    """A frame's label map: the class index of every pixel, 255 where unlabeled."""
    return np.asarray(Image.open(label_path(frame_id, label_dir)))


def road_target(labels: np.ndarray | torch.Tensor):
    """Road as 1, every other class as 0, and unlabeled pixels left at the ignore value."""
    road = labels == ROAD_INDEX
    target = road.to(torch.uint8) if isinstance(labels, torch.Tensor) else road.astype(np.uint8)
    target[labels == IGNORE_LABEL] = IGNORE_LABEL
    return target


def assign_partitions(frames: pd.DataFrame, worlds: Mapping[str, Sequence[str]]) -> pd.Series:
    """
    The partition of every frame, decided by its world alone.

    `worlds` maps a partition name to the worlds it holds. All ten renderings
    of a world show the same drive, so a world never straddles two partitions.
    """
    partition_of = {world: name for name, group in worlds.items() for world in group}
    if len(partition_of) != sum(len(group) for group in worlds.values()):
        raise ValueError("a world is assigned to two partitions")
    missing = set(frames["world"]) - set(partition_of)
    if missing:
        raise ValueError(f"worlds without a partition: {sorted(missing)}")
    return frames["world"].map(partition_of).rename("partition")


def one_rendering_per_step(frames: pd.DataFrame, shift: int = 0) -> pd.DataFrame:
    """
    Every moment of every video once, each under one of its ten renderings.

    The rendering cycles with the step, and `shift` moves the cycle by one, so
    consecutive shifts show the same moments under different weather and
    camera angles, and ten shifts cover every frame exactly once.
    """
    rank = frames["variation"].map({name: i for i, name in enumerate(VARIATIONS)})
    return frames[(frames["step"] + shift) % len(VARIATIONS) == rank]


def frame_tensor(file: str, raw_dir: Path = RAW_DIR) -> torch.Tensor:
    """A frame as a float tensor of shape (3, height, width) with values in [0, 1]."""
    frame = Image.open(raw_dir / IMAGE_FOLDER / file).convert("RGB")
    return transforms.to_dtype(transforms.pil_to_tensor(frame), torch.float32, scale=True)


def boxes_by_frame(signs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each frame's sign boxes as a float array of shape (signs, 4)."""
    return {
        frame_id: group[BOX_COLUMNS].to_numpy(dtype=np.float32)
        for frame_id, group in signs.groupby("frame_id", sort=False)
    }


class FrameDataset(torch.utils.data.Dataset):
    """
    Frames with their sign boxes and road mask, ready for the network.

    An item is the image as floats in [0, 1], a target with `boxes` and
    `labels` in the detector's format, the road mask (1 road, 0 other, 255
    unlabeled) and the row number of the frame. With `flip` above zero the
    frame, its mask and its boxes are mirrored left to right at that rate.
    """

    def __init__(
        self,
        frames: pd.DataFrame,
        signs: pd.DataFrame,
        raw_dir: Path = RAW_DIR,
        label_dir: Path = LABEL_DIR,
        flip: float = 0.0,
    ):
        self.files = frames["file"].tolist()
        self.frame_ids = frames["frame_id"].tolist()
        boxes = boxes_by_frame(signs[signs["frame_id"].isin(set(self.frame_ids))])
        self.boxes = [
            boxes.get(frame_id, np.zeros((0, 4), np.float32)) for frame_id in self.frame_ids
        ]
        self.raw_dir, self.label_dir, self.flip = raw_dir, label_dir, flip

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, row: int):
        image = frame_tensor(self.files[row], self.raw_dir)
        labels = Image.open(label_path(self.frame_ids[row], self.label_dir))
        road = road_target(transforms.pil_to_tensor(labels)[0])
        boxes = torch.tensor(self.boxes[row])
        if self.flip and torch.rand(()) < self.flip:
            image, road = image.flip(-1), road.flip(-1)
            width = image.shape[-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return image, target, road, row


def collate(batch: Sequence) -> tuple[list, list, torch.Tensor, torch.Tensor]:
    """Images and targets stay lists, as the detector expects; the masks share one size."""
    images, targets, roads, rows = zip(*batch, strict=True)
    return list(images), list(targets), torch.stack(roads), torch.tensor(rows)


def write_processed(tables: Mapping[str, pd.DataFrame], root: Path = PROCESSED_DIR) -> list[str]:
    """Write every results table the notebook reports to `data/processed`, one CSV each."""
    root.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(root / f"{name}.csv", index=False)
    return sorted(f"{name}.csv" for name in tables)
