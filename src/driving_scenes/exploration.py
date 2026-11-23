"""
Measurements for the exploration: what the frames, the road and the signs look like.

Everything here reads the tables and label maps built by `dataset` and returns
a table or an array; the figures live in `plots`. The questions are chosen for
what the model will need: how small and how elongated the signs are (which
decides the anchor boxes), how much consecutive frames and the ten renderings
of a world repeat each other (which decides the split and the sampling), and
how coarse a mask can be before it stops describing the road (which decides
the kind of segmentation head).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torchvision.ops import box_iou

from driving_scenes.config import (
    ANCHOR_STRIDES,
    BORDER_MARGIN,
    CLASSES,
    COCO_SIZE_EDGES,
    IGNORE_LABEL,
    LABEL_DIR,
    MASK_HEAD_SIZE,
    NEIGHBOR_GAP,
    ROAD_HEAD_STRIDE,
    TALL_ASPECT,
    VARIATIONS,
)
from driving_scenes.dataset import BOX_COLUMNS, ROAD_INDEX, load_labels

SIZE_NAMES = ("small", "medium", "large")


def group_summary(frames: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Per group of frames, such as a world or a partition: its size and what it shows.

    A moment is one step of one world's video, shown in up to ten renderings.
    The road share counts every pixel of the frames, unlabeled ones included.
    """
    table = frames.assign(
        moment=frames["world"] + "_" + frames["step"].astype(str),
        pixels=frames["width"] * frames["height"],
        has_sign=frames["signs"] > 0,
    )
    grouped = table.groupby(by)
    summary = pd.DataFrame(
        {
            "moments": grouped["moment"].nunique(),
            "frames": grouped.size(),
            "renderings": grouped["variation"].nunique(),
            "signs per frame": grouped["signs"].mean(),
            "frames with a sign": grouped["has_sign"].mean(),
            "road share": grouped["pixels_road"].sum() / grouped["pixels"].sum(),
        }
    )
    summary.insert(1, "share of moments", summary["moments"] / summary["moments"].sum())
    if by != "world":
        summary.insert(0, "worlds", grouped["world"].unique().map(lambda w: ", ".join(sorted(w))))
    return summary


def class_shares(frames: pd.DataFrame, by: str = "world") -> pd.DataFrame:
    """Share of all pixels each class takes, per group, with the unlabeled share last."""
    columns = [f"pixels_{name}" for name in CLASSES] + ["unlabeled_pixels"]
    totals = frames.groupby(by)[columns].sum()
    shares = totals.div(totals.sum(axis=1), axis=0)
    return shares.rename(columns=lambda name: name.removeprefix("pixels_").replace("_", " "))


def describe_signs(signs: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """
    Every traffic-sign region with its world, its shape and what makes it hard.

    `size` follows the COCO convention on box area (small under 32 x 32
    pixels, large from 96 x 96), `aspect` is height over width, `fill` is the
    share of the box the region covers, and `at_border` marks a box cut off by
    the frame's edge.
    """
    frame_columns = ["frame_id", "world", "variation", "step", "width", "height"]
    table = signs.merge(
        frames[frame_columns].rename(columns={"width": "frame_width", "height": "frame_height"}),
        on="frame_id",
        validate="many_to_one",
    )
    table["width"] = table["x_max"] - table["x_min"]
    table["height"] = table["y_max"] - table["y_min"]
    table["box_area"] = table["width"] * table["height"]
    table["aspect"] = table["height"] / table["width"]
    table["fill"] = table["pixels"] / table["box_area"]
    table["size"] = pd.cut(
        table["box_area"], [0, *COCO_SIZE_EDGES, np.inf], labels=SIZE_NAMES, right=False
    )
    table["at_border"] = (
        (table["x_min"] <= BORDER_MARGIN)
        | (table["y_min"] <= BORDER_MARGIN)
        | (table["x_max"] >= table["frame_width"] - BORDER_MARGIN)
        | (table["y_max"] >= table["frame_height"] - BORDER_MARGIN)
    )
    return table.drop(columns=["frame_width", "frame_height"])


def sign_summary(described: pd.DataFrame) -> pd.Series:
    """
    The sign regions in a few numbers: size, elongation, fill, frame edges and fragments.

    `described` is the output of `describe_signs` with a `fragment` column from
    `has_close_neighbor`.
    """
    sizes = described["size"].value_counts(normalize=True)
    return pd.Series(
        {
            "regions": len(described),
            "median width (px)": described["width"].median(),
            "median height (px)": described["height"].median(),
            "median height over width": described["aspect"].median(),
            f"share more than {TALL_ASPECT} times taller than wide": (
                described["aspect"] > TALL_ASPECT
            ).mean(),
            "median share of the box the region fills": described["fill"].median(),
            **{f"share {size} (COCO)": sizes.get(size, 0.0) for size in SIZE_NAMES},
            "share cut off by the frame edge": described["at_border"].mean(),
            f"share with another region within {NEIGHBOR_GAP} px": described["fragment"].mean(),
            "smallest region (pixels)": described["pixels"].min(),
        }
    )


def sign_examples(described: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    One frame for each kind of sign region the cleaning has to decide about.

    Returns the frame ids with the reason each was chosen: a sign split into
    fragments, a sign cut by the frame edge, the smallest region, and the most
    elongated one.
    """
    picks = {
        "a sign split into touching regions": described[described["fragment"]].sample(
            1, random_state=seed
        ),
        "a sign cut off by the frame edge": described[described["at_border"]].sample(
            1, random_state=seed
        ),
        "the smallest region": described.nsmallest(1, "pixels"),
        "the most elongated region": described.nlargest(1, "aspect"),
    }
    return pd.DataFrame(
        {"frame_id": [pick["frame_id"].iloc[0] for pick in picks.values()], "reason": list(picks)}
    )


def world_examples(frames: pd.DataFrame, variation: str) -> pd.DataFrame:
    """One frame per world, from the middle of its video, in the given rendering."""
    video = frames[frames["variation"] == variation].sort_values(["world", "step"])
    position = video.groupby("world").cumcount()
    length = video.groupby("world")["step"].transform("size")
    return video[position == length // 2].reset_index(drop=True)


def rendering_examples(frames: pd.DataFrame, world: str, step: int) -> pd.DataFrame:
    """The renderings of one moment, in the order of `VARIATIONS`."""
    moment = frames[(frames["world"] == world) & (frames["step"] == step)]
    rank = {name: index for index, name in enumerate(VARIATIONS)}
    return moment.sort_values("variation", key=lambda names: names.map(rank)).reset_index(drop=True)


def box_gap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pixels of empty space between every box of `first` and every box of `second`."""
    dx = np.maximum(first[:, None, 0] - second[None, :, 2], second[None, :, 0] - first[:, None, 2])
    dy = np.maximum(first[:, None, 1] - second[None, :, 3], second[None, :, 1] - first[:, None, 3])
    return np.maximum(np.maximum(dx, dy), 0)


def has_close_neighbor(signs: pd.DataFrame, gap: int = NEIGHBOR_GAP) -> pd.Series:
    """
    Whether another sign region of the same frame lies within `gap` pixels of each box.

    A sign partly hidden by a pole or a branch is stored as several regions, so
    regions that almost touch are usually pieces of one sign.
    """
    close = pd.Series(False, index=signs.index)
    for _, group in signs.groupby("frame_id", sort=False):
        if len(group) > 1:
            boxes = group[BOX_COLUMNS].to_numpy()
            gaps = box_gap(boxes, boxes).astype(float)
            np.fill_diagonal(gaps, np.inf)
            close.loc[group.index] = gaps.min(axis=1) <= gap
    return close


def box_heatmap(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    How many boxes cover each pixel.

    Each box adds one at its top-left corner of a difference image and takes
    it back past its right and bottom edges; two running sums then fill every
    box at once, so the cost does not grow with the boxes' areas.
    """
    difference = np.zeros((height + 1, width + 1), dtype=np.int64)
    x_min, y_min, x_max, y_max = np.asarray(boxes, dtype=np.int64).T
    np.add.at(difference, (y_min, x_min), 1)
    np.add.at(difference, (y_min, x_max), -1)
    np.add.at(difference, (y_max, x_min), -1)
    np.add.at(difference, (y_max, x_max), 1)
    return difference.cumsum(axis=0).cumsum(axis=1)[:height, :width]


def class_frequency(
    frames: pd.DataFrame, class_index: int = ROAD_INDEX, label_dir: Path = LABEL_DIR
) -> np.ndarray:
    """For every pixel, the share of the given frames in which it belongs to the class."""
    total = 0
    for frame_id in frames["frame_id"]:
        total = total + (load_labels(frame_id, label_dir) == class_index)
    return total / len(frames)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two boolean masks; two empty masks agree fully."""
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def matched_share(boxes: np.ndarray, others: np.ndarray, threshold: float) -> float:
    """Share of `boxes` that overlap some box of `others` by at least `threshold` IoU."""
    if not len(boxes):
        return np.nan
    if not len(others):
        return 0.0
    overlap = box_iou(torch.tensor(boxes), torch.tensor(others))
    return float((overlap.max(dim=1).values >= threshold).float().mean())


def frame_repetition(
    frames: pd.DataFrame,
    signs: pd.DataFrame,
    lags: Sequence[int],
    variation: str,
    box_threshold: float,
    label_dir: Path = LABEL_DIR,
) -> pd.DataFrame:
    """
    How much of a frame's labels are still there a few frames later.

    For every world and lag, in one rendering: the mean IoU between the road
    masks of frames `lag` steps apart, and the share of sign boxes that still
    overlap a box of the later frame.
    """
    boxes = {
        frame_id: group[BOX_COLUMNS].to_numpy() for frame_id, group in signs.groupby("frame_id")
    }
    empty = np.zeros((0, 4))
    rows = []
    video = frames[frames["variation"] == variation]
    for world, group in video.groupby("world"):
        ids = group.sort_values("step")["frame_id"].tolist()
        roads = [load_labels(frame_id, label_dir) == ROAD_INDEX for frame_id in ids]
        for lag in lags:
            pairs = list(zip(range(len(ids) - lag), range(lag, len(ids)), strict=True))
            kept = [
                matched_share(boxes.get(ids[a], empty), boxes.get(ids[b], empty), box_threshold)
                for a, b in pairs
            ]
            rows.append(
                {
                    "world": world,
                    "lag": lag,
                    "road_iou": float(np.mean([mask_iou(roads[a], roads[b]) for a, b in pairs])),
                    "signs_still_matched": float(pd.Series(kept).mean()),
                }
            )
    return pd.DataFrame(rows)


def rendering_agreement(
    frames: pd.DataFrame,
    signs: pd.DataFrame,
    reference: str,
    box_threshold: float,
    steps_per_world: int,
    seed: int,
    label_dir: Path = LABEL_DIR,
) -> pd.DataFrame:
    """
    How closely each rendering's labels match the reference rendering's at the same moment.

    A value of 1 means the two renderings show the very same road mask or sign
    boxes: the same drive under another sky.
    """
    boxes = {
        frame_id: group[BOX_COLUMNS].to_numpy() for frame_id, group in signs.groupby("frame_id")
    }
    empty = np.zeros((0, 4))
    rng = np.random.default_rng(seed)
    by_key = frames.set_index(["world", "variation", "step"])["frame_id"]
    present = set(frames["variation"])
    others = [name for name in VARIATIONS if name in present and name != reference]
    rows = []
    for world, group in frames.groupby("world"):
        steps = rng.choice(np.sort(group["step"].unique()), steps_per_world, replace=False)
        for step in steps:
            anchor_id = by_key[(world, reference, step)]
            road = load_labels(anchor_id, label_dir) == ROAD_INDEX
            for variation in others:
                other_id = by_key[(world, variation, step)]
                rows.append(
                    {
                        "world": world,
                        "variation": variation,
                        "road_iou": mask_iou(road, load_labels(other_id, label_dir) == ROAD_INDEX),
                        "signs_matched": matched_share(
                            boxes.get(anchor_id, empty), boxes.get(other_id, empty), box_threshold
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    return table.groupby("variation")[["road_iou", "signs_matched"]].mean().loc[others]


def anchor_shapes(
    sizes: Sequence[Sequence[int]], ratios: Sequence[float]
) -> list[tuple[int, float, float]]:
    """
    Every anchor as (stride, width, height), built the way the detector builds them.

    An anchor of size s and ratio r (height over width) is s / sqrt(r) wide and
    s * sqrt(r) high, with its half-extents rounded to whole pixels.
    """
    shapes = []
    for stride, level_sizes in zip(ANCHOR_STRIDES, sizes, strict=True):
        for size in level_sizes:
            for ratio in ratios:
                half_width = round(size / np.sqrt(ratio) / 2)
                half_height = round(size * np.sqrt(ratio) / 2)
                shapes.append((stride, 2.0 * half_width, 2.0 * half_height))
    return shapes


def best_anchor_iou(
    boxes: np.ndarray, sizes: Sequence[Sequence[int]], ratios: Sequence[float]
) -> np.ndarray:
    """
    The highest IoU any anchor reaches with each box.

    Anchors of one shape sit on a grid with the level's stride. The overlap
    along each axis is largest for the grid point nearest the box's center, so
    the best anchor of every shape is found directly, without listing the
    120,000 anchors of a frame.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    best = np.zeros(len(boxes))
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for stride, width, height in anchor_shapes(sizes, ratios):
        overlap = np.ones(len(boxes))
        for low, high, extent in (
            (boxes[:, 0], boxes[:, 2], width),
            (boxes[:, 1], boxes[:, 3], height),
        ):
            center = np.round((low + high) / 2 / stride) * stride
            overlap *= np.clip(
                np.minimum(high, center + extent / 2) - np.maximum(low, center - extent / 2),
                0,
                None,
            )
        best = np.maximum(best, overlap / (area + width * height - overlap))
    return best


def anchor_coverage(
    boxes: np.ndarray,
    candidates: dict[str, tuple[Sequence[Sequence[int]], Sequence[float]]],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Per candidate anchor set: the median best IoU and the share of boxes above each level."""
    rows = {}
    for name, (sizes, ratios) in candidates.items():
        best = best_anchor_iou(boxes, sizes, ratios)
        rows[name] = {"median best IoU": float(np.median(best))} | {
            f"best IoU >= {level:g}": float((best >= level).mean()) for level in thresholds
        }
    return pd.DataFrame(rows).T


def resample(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """A mask of shape (height, width) resized bilinearly, as float."""
    return functional.interpolate(
        mask[None, None].float(), size=size, mode="bilinear", align_corners=False, antialias=True
    )[0, 0]


def coarse_mask_ceilings(
    frames: pd.DataFrame,
    label_dir: Path = LABEL_DIR,
    mask_size: int = MASK_HEAD_SIZE,
    stride: int = ROAD_HEAD_STRIDE,
) -> pd.DataFrame:
    """
    The best road IoU two kinds of head could reach if they predicted perfectly.

    A Mask R-CNN mask head draws a `mask_size` x `mask_size` mask inside the
    object's box; a semantic head predicts on a grid `stride` times coarser
    than the frame. Each true road mask is squeezed through both bottlenecks
    and compared with itself.
    """
    rows = []
    for frame_id in frames["frame_id"]:
        labels = load_labels(frame_id, label_dir)
        road = torch.from_numpy(labels == ROAD_INDEX)
        labeled = torch.from_numpy(labels != IGNORE_LABEL)
        if not road.any():
            continue
        height, width = road.shape
        coarse = resample(road, (height // stride, width // stride))
        semantic = resample(coarse, (height, width)) > 0.5
        rows_with_road = torch.where(road.any(dim=1))[0]
        columns_with_road = torch.where(road.any(dim=0))[0]
        top, bottom = int(rows_with_road[0]), int(rows_with_road[-1]) + 1
        left, right = int(columns_with_road[0]), int(columns_with_road[-1]) + 1
        small = resample(road[top:bottom, left:right], (mask_size, mask_size))
        instance = torch.zeros_like(road)
        instance[top:bottom, left:right] = resample(small, (bottom - top, right - left)) > 0.5
        scores = {}
        for name, rebuilt in (("semantic head", semantic), ("mask head", instance)):
            union = ((rebuilt | road) & labeled).sum()
            scores[name] = float(((rebuilt & road) & labeled).sum() / union)
        rows.append({"frame_id": frame_id} | scores)
    return pd.DataFrame(rows)
