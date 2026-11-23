"""
Predictions of a finished run on a set of frames, kept on disk, and the scores read from them.

Predicting the 4,470 test frames takes about a quarter of an hour, so a run's
predictions are computed once and stored with a fingerprint of the weights and
the frames. What is stored is what every later analysis needs and no more: the
road counts of every frame (true and false positives and negatives) and every
detection with its box, its confidence and its frame. Any group of frames, a
rendering, a size class or a single frame, can then be scored without
predicting again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_calibration_error
from torchvision.ops import box_iou

from driving_scenes import cache
from driving_scenes.config import (
    BATCH_SIZE,
    CALIBRATION_BINS,
    CLASSES,
    EXTREME_STEP_GAP,
    IGNORE_LABEL,
    LABEL_DIR,
    LOADER_WORKERS,
    MATCH_IOU,
    RAW_DIR,
    ROAD_CLASS,
    ROAD_THRESHOLD,
    SCORE_FLOOR,
    SCORE_THRESHOLDS,
    TIMING_FRAMES,
    TIMING_WARMUP,
)
from driving_scenes.dataset import (
    BOX_COLUMNS,
    FrameDataset,
    boxes_by_frame,
    collate,
    frame_tensor,
    load_image,
    load_labels,
    road_target,
)
from driving_scenes.modeling.metrics import road_counts, road_scores, sign_metric, sign_scores
from driving_scenes.modeling.model import MultiTaskRCNN
from driving_scenes.modeling.train import load_model


def device() -> torch.device:
    """The accelerator PyTorch finds (CUDA, Apple GPU or other), or the CPU."""
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def synchronize(target: torch.device) -> None:
    """Wait for the accelerator to finish its queued work, so a timer measures the work."""
    if target.type != "cpu":
        torch.accelerator.synchronize()


@dataclass
class Predictions:
    """
    Everything a run predicted on a set of frames.

    `tasks` are the heads the network computed. `counts` holds tp, fp, tn, fn
    per frame, or is None for a network without a road head; the detections
    are flat arrays, with `box_frame` giving the row of each detection's frame
    in `frame_ids`.
    """

    tasks: tuple[str, ...]
    frame_ids: np.ndarray
    counts: np.ndarray | None
    boxes: np.ndarray
    scores: np.ndarray
    box_frame: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "tasks": np.array(self.tasks),
            "frame_ids": self.frame_ids.astype(str),
            "boxes": self.boxes,
            "scores": self.scores,
            "box_frame": self.box_frame,
        }
        if self.counts is not None:
            arrays["counts"] = self.counts
        with open(path, "wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: Path) -> Predictions:
        with np.load(path) as stored:
            return cls(
                tasks=tuple(str(task) for task in stored["tasks"]),
                frame_ids=stored["frame_ids"],
                counts=stored.get("counts"),
                boxes=stored["boxes"],
                scores=stored["scores"],
                box_frame=stored["box_frame"],
            )

    def subset(self, keep: np.ndarray) -> Predictions:
        """The predictions of the frames where `keep` is true, in their order."""
        rows = np.flatnonzero(keep)
        new_row = np.full(len(self.frame_ids), -1)
        new_row[rows] = np.arange(len(rows))
        chosen = np.isin(self.box_frame, rows)
        return Predictions(
            tasks=self.tasks,
            frame_ids=self.frame_ids[rows],
            counts=None if self.counts is None else self.counts[rows],
            boxes=self.boxes[chosen],
            scores=self.scores[chosen],
            box_frame=new_row[self.box_frame[chosen]],
        )

    def detections(self) -> list[dict[str, torch.Tensor]]:
        """Per frame, the detections in the detector's format."""
        order = np.argsort(self.box_frame, kind="stable")
        bounds = np.searchsorted(self.box_frame[order], np.arange(len(self.frame_ids) + 1))
        result = []
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            rows = order[start:end]
            result.append(
                {
                    "boxes": torch.from_numpy(self.boxes[rows].astype(np.float32)),
                    "scores": torch.from_numpy(self.scores[rows].astype(np.float32)),
                    "labels": torch.ones(len(rows), dtype=torch.int64),
                }
            )
        return result


@torch.no_grad()
def predict_frames(
    model: MultiTaskRCNN,
    frames: pd.DataFrame,
    signs: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
    workers: int = LOADER_WORKERS,
    batch_size: int = BATCH_SIZE,
) -> Predictions:
    """Run the network over the frames in batches and keep its road counts and detections."""
    target = device()
    model = model.to(target).eval()
    data = FrameDataset(frames, signs, raw_dir, label_dir)
    loader = DataLoader(data, batch_size=batch_size, num_workers=workers, collate_fn=collate)
    counts, boxes, scores, box_frame = [], [], [], []
    for images, _, road, rows in loader:
        detections, probability = model.predict([image.to(target) for image in images])
        if probability is not None:
            counts.append(road_counts(probability, road))
        for row, found in zip(rows.tolist(), detections, strict=True):
            boxes.append(found["boxes"].cpu().numpy())
            scores.append(found["scores"].cpu().numpy())
            box_frame.append(np.full(len(found["scores"]), row))
    return Predictions(
        tasks=model.tasks,
        frame_ids=frames["frame_id"].to_numpy(),
        counts=torch.cat(counts).numpy() if counts else None,
        boxes=np.concatenate(boxes).astype(np.float32),
        scores=np.concatenate(scores).astype(np.float32),
        box_frame=np.concatenate(box_frame).astype(np.int64),
    )


def cached_predictions(
    run: Path,
    frames: pd.DataFrame,
    signs: pd.DataFrame,
    path: Path,
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
    workers: int = LOADER_WORKERS,
) -> Predictions:
    """
    A run's predictions on the frames, computed once and reloaded while nothing has changed.

    The fingerprint covers the run's own fingerprint (so retrained weights are
    predicted again), the frames, and the thresholds applied while predicting.
    """
    key = cache.fingerprint(
        "predictions",
        cache.read_stamp(run),
        frames["frame_id"].to_numpy(),
        ROAD_THRESHOLD,
        SCORE_FLOOR,
    )
    if cache.is_current(path, key):
        return Predictions.load(path)
    model = load_model(run)
    predictions = predict_frames(model, frames, signs, raw_dir, label_dir, workers)
    predictions.save(path)
    cache.record(path, key)
    del model
    return predictions


def truth_by_frame(frame_ids: np.ndarray, signs: pd.DataFrame) -> list[dict[str, torch.Tensor]]:
    """The true sign boxes of each frame, in the detector's format."""
    boxes = boxes_by_frame(signs[signs["frame_id"].isin(set(frame_ids))])
    empty = np.zeros((0, 4), np.float32)
    return [
        {
            "boxes": torch.tensor(boxes.get(frame_id, empty)),
            "labels": torch.ones(len(boxes.get(frame_id, empty)), dtype=torch.int64),
        }
        for frame_id in frame_ids
    ]


def score_predictions(predictions: Predictions, signs: pd.DataFrame) -> dict[str, float]:
    """Every sign and road score of a set of predictions; a task the network lacks is left out."""
    scores: dict[str, float] = {}
    if "signs" in predictions.tasks:
        metric = sign_metric()
        metric.update(predictions.detections(), truth_by_frame(predictions.frame_ids, signs))
        scores |= {f"sign_{name}": value for name, value in sign_scores(metric).items()}
    if "road" in predictions.tasks:
        scores |= road_scores(predictions.counts)
    return scores


def scores_by_group(
    predictions: Predictions, signs: pd.DataFrame, groups: pd.Series
) -> pd.DataFrame:
    """
    Scores per group of frames, one row per group.

    `groups` maps each frame id to its group, such as the rendering.
    """
    labels = groups.reindex(predictions.frame_ids).to_numpy()
    rows = {
        group: score_predictions(predictions.subset(labels == group), signs)
        for group in pd.unique(labels)
    }
    return pd.DataFrame(rows).T


def compare_runs(
    scores: dict[str, pd.DataFrame], seeds: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The AP50 and road IoU of every run on each frame set, and their range over the recipe's seeds.

    `scores` maps a frame set's name, such as "validation" or "test", to a
    table with one row per run and `sign_ap50` and `road_iou` columns. Both
    results carry a two-level column index (frame set, score), as `plots.plot_runs`
    reads it; the range has a `min` and a `max` row.
    """
    columns = ["sign_ap50", "road_iou"]
    table = pd.concat({name: frame[columns] for name, frame in scores.items()}, axis=1)
    return table, table.loc[seeds].agg(["min", "max"])


def assign_detections(
    boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, iou: float = MATCH_IOU
) -> np.ndarray:
    """
    For each detection, the index of the sign it hits, or -1 when it hits none.

    Detections are matched greedily from the most confident down: a detection
    hits the sign it overlaps most among those no more confident detection has
    taken, if that overlap reaches `iou`. This is the rule COCO uses.
    """
    assigned = np.full(len(boxes), -1)
    if not len(boxes) or not len(truth):
        return assigned
    overlap = box_iou(
        torch.tensor(boxes, dtype=torch.float32), torch.tensor(truth, dtype=torch.float32)
    ).numpy()
    taken = np.zeros(len(truth), dtype=bool)
    for row in np.argsort(-scores, kind="stable"):
        candidates = np.where(taken, -1.0, overlap[row])
        best = int(np.argmax(candidates))
        if candidates[best] >= iou:
            assigned[row] = best
            taken[best] = True
    return assigned


def match_detections(
    boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, iou: float = MATCH_IOU
) -> np.ndarray:
    """Which detections hit a sign, one boolean per detection in the order given."""
    return assign_detections(boxes, scores, truth, iou) >= 0


def sign_outcomes(
    predictions: Predictions,
    sign_table: pd.DataFrame,
    score_threshold: float,
    iou: float = MATCH_IOU,
) -> pd.Series:
    """Whether each true sign of the predicted frames was found, indexed like `sign_table`."""
    row_of = {frame_id: row for row, frame_id in enumerate(predictions.frame_ids)}
    kept = predictions.scores >= score_threshold
    found = pd.Series(False, index=sign_table.index, name="found")
    for frame_id, group in sign_table[sign_table["frame_id"].isin(row_of)].groupby("frame_id"):
        mine = kept & (predictions.box_frame == row_of[frame_id])
        assigned = assign_detections(
            predictions.boxes[mine],
            predictions.scores[mine],
            group[BOX_COLUMNS].to_numpy(dtype=np.float32),
            iou,
        )
        found.loc[group.index[assigned[assigned >= 0]]] = True
    return found[sign_table["frame_id"].isin(row_of)]


def frame_scores(
    predictions: Predictions, signs: pd.DataFrame, score_threshold: float, iou: float = MATCH_IOU
) -> pd.DataFrame:
    """
    One row per frame: its road IoU and how its signs fared at a confidence threshold.

    A sign is found when a detection at or above the threshold hits it; a
    detection at or above the threshold that hits nothing is a false alarm.
    """
    truth = truth_by_frame(predictions.frame_ids, signs)
    kept = predictions.scores >= score_threshold
    rows = []
    for row, frame_id in enumerate(predictions.frame_ids):
        mine = kept & (predictions.box_frame == row)
        true_boxes = truth[row]["boxes"].numpy()
        hits = match_detections(predictions.boxes[mine], predictions.scores[mine], true_boxes, iou)
        found, alarms = int(hits.sum()), int((~hits).sum())
        rows.append(
            {
                "frame_id": frame_id,
                "signs": len(true_boxes),
                "found": found,
                "missed": len(true_boxes) - found,
                "false_alarms": alarms,
            }
        )
    table = pd.DataFrame(rows)
    table["errors"] = table["missed"] + table["false_alarms"]
    table["sign_f1"] = 2 * table["found"] / (2 * table["found"] + table["errors"])
    if predictions.counts is not None:
        tp, fp, _, fn = predictions.counts.T.astype(float)
        union = tp + fp + fn
        table["road_iou"] = np.divide(tp, union, out=np.ones_like(tp), where=union > 0)
        # Frames with no sign and no false alarm have no F1, so they get no combined score.
        table["combined"] = (table["road_iou"] + table["sign_f1"]) / 2
    return table


def distinct_moments(ranked: pd.DataFrame, count: int, gap: int) -> pd.DataFrame:
    """The first `count` rows whose steps lie at least `gap` apart, in the order given."""
    chosen: list[int] = []
    for index, step in zip(ranked.index, ranked["step"], strict=True):
        if all(abs(step - ranked.at[other, "step"]) >= gap for other in chosen):
            chosen.append(index)
            if len(chosen) == count:
                break
    return ranked.loc[chosen]


def extreme_frames(
    table: pd.DataFrame, column: str, count: int, gap: int = EXTREME_STEP_GAP
) -> pd.DataFrame:
    """
    The `count` best frames by one score, best first, then the `count` worst, worst first.

    Frames without the score are left out. Among equal scores, fewer sign
    errors rank higher, so the worst of a tie is the frame with the most errors.
    Neighboring frames and the renderings of one moment are near-copies, so the
    frames chosen at each end lie at least `gap` steps apart.
    """
    ranked = table.dropna(subset=[column]).sort_values(
        [column, "errors", "frame_id"], ascending=[False, True, True]
    )
    best = distinct_moments(ranked, count, gap).assign(rank="best")
    worst = distinct_moments(ranked.iloc[::-1], count, gap).assign(rank="worst")
    return pd.concat([best, worst])


def frame_titles(chosen: pd.DataFrame) -> list[str]:
    """A title per frame: its rendering and step, its road IoU and how its signs fared."""
    return [
        f"{row.variation}, step {row.step}: road IoU {row.road_iou:.2f}, "
        f"{row.found} of {row.signs} signs found, "
        f"{row.false_alarms} false alarm{'' if row.false_alarms == 1 else 's'}"
        for row in chosen.itertuples()
    ]


def detection_hits(
    predictions: Predictions, signs: pd.DataFrame, iou: float = MATCH_IOU
) -> np.ndarray:
    """
    Whether each stored detection hits a sign, with every stored detection matched.

    A detection's hit does not depend on less confident detections, so the
    hits found here hold at any higher confidence threshold as well.
    """
    truth = truth_by_frame(predictions.frame_ids, signs)
    hits = np.zeros(len(predictions.scores), dtype=bool)
    for row in range(len(predictions.frame_ids)):
        mine = np.flatnonzero(predictions.box_frame == row)
        hits[mine] = match_detections(
            predictions.boxes[mine], predictions.scores[mine], truth[row]["boxes"].numpy(), iou
        )
    return hits


def threshold_sweep(
    predictions: Predictions,
    signs: pd.DataFrame,
    thresholds=SCORE_THRESHOLDS,
    iou: float = MATCH_IOU,
) -> pd.DataFrame:
    """Precision, recall and F1 over all frames at each confidence threshold."""
    total_signs = int(signs["frame_id"].isin(set(predictions.frame_ids)).sum())
    hits = detection_hits(predictions, signs, iou)
    rows = []
    for threshold in thresholds:
        kept = predictions.scores >= threshold
        found = int((hits & kept).sum())
        alarms = int((~hits & kept).sum())
        precision = found / max(found + alarms, 1)
        recall = found / max(total_signs, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def detection_reliability(
    predictions: Predictions,
    signs: pd.DataFrame,
    bins: int = CALIBRATION_BINS,
    iou: float = MATCH_IOU,
) -> tuple[pd.DataFrame, float]:
    """
    How far a detection's confidence can be read as the chance that it is a sign.

    Returns, per confidence bin, the detections, their mean confidence and the
    share of them that hit a sign, plus the expected calibration error over
    all detections (torchmetrics, the gap between confidence and hit rate
    averaged over the bins, weighted by their detections).
    """
    hits = detection_hits(predictions, signs, iou)
    table = pd.DataFrame(
        {
            "bin": pd.cut(predictions.scores, np.linspace(0, 1, bins + 1), include_lowest=True),
            "confidence": predictions.scores,
            "hit": hits,
        }
    )
    grouped = table.groupby("bin", observed=True)
    reliability = pd.DataFrame(
        {
            "detections": grouped.size(),
            "mean confidence": grouped["confidence"].mean(),
            "hit rate": grouped["hit"].mean(),
        }
    )
    error = binary_calibration_error(
        torch.from_numpy(predictions.scores.astype(np.float32)),
        torch.from_numpy(hits.astype(np.int64)),
        n_bins=bins,
        norm="l1",
    )
    return reliability, float(error)


def recall_by(described: pd.DataFrame, found: pd.Series, column: str) -> pd.DataFrame:
    """
    The signs, the signs found and the share found, per value of one column of `describe_signs`.

    `found` is indexed like the described signs, as `sign_outcomes` returns it.
    """
    table = described.loc[found.index].assign(found=found)
    grouped = table.groupby(column, observed=False)["found"]
    return pd.DataFrame({"signs": grouped.size(), "found": grouped.sum(), "recall": grouped.mean()})


def size_scores(
    described: pd.DataFrame, found: pd.Series, scores: dict[str, float]
) -> pd.DataFrame:
    """Per COCO size class: the signs, the share found at a threshold, and the class's COCO AP."""
    table = recall_by(described, found, "size")
    table["ap"] = [scores[f"sign_ap_{size}"] for size in table.index]
    return table


def fixed_mask_counts(
    mask: np.ndarray, frames: pd.DataFrame, label_dir: Path = LABEL_DIR
) -> np.ndarray:
    """
    Road counts per frame (tp, fp, tn, fn) for one fixed mask called road in every frame.

    The baseline a road head has to beat: a mask that knows where road usually
    is and looks at nothing.
    """
    called = torch.from_numpy(mask.astype(np.float32))[None]
    counts = [
        road_counts(called, torch.from_numpy(road_target(load_labels(frame_id, label_dir)))[None])
        for frame_id in frames["frame_id"]
    ]
    return torch.cat(counts).numpy()


@torch.no_grad()
def road_calls_by_class(
    model: MultiTaskRCNN,
    frames: pd.DataFrame,
    signs: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
    workers: int = LOADER_WORKERS,
) -> pd.DataFrame:
    """
    Every labeled class with its pixels in the frames and how many of them the network calls road.

    The road row counts the road found. Every other row counts false road by
    what the pixels really are, and its `share of false road` says how much of
    all false road that class makes up.
    """
    target = device()
    model = model.to(target).eval()
    loader = DataLoader(
        FrameDataset(frames, signs, raw_dir, label_dir),
        batch_size=BATCH_SIZE,
        num_workers=workers,
        collate_fn=collate,
    )
    frame_ids = frames["frame_id"].tolist()
    totals = torch.zeros(IGNORE_LABEL + 1, dtype=torch.long)
    called = torch.zeros(IGNORE_LABEL + 1, dtype=torch.long)
    for images, _, _, rows in loader:
        _, probability = model.predict([image.to(target) for image in images])
        for road, row in zip((probability >= ROAD_THRESHOLD).cpu(), rows.tolist(), strict=True):
            labels = torch.from_numpy(load_labels(frame_ids[row], label_dir).astype(np.int64))
            totals += torch.bincount(labels.ravel(), minlength=IGNORE_LABEL + 1)
            called += torch.bincount(labels[road], minlength=IGNORE_LABEL + 1)
    table = pd.DataFrame(
        {"pixels": totals[: len(CLASSES)].numpy(), "called road": called[: len(CLASSES)].numpy()},
        index=pd.Index(CLASSES, name="true class"),
    )
    table["share called road"] = table["called road"] / table["pixels"].where(table["pixels"] > 0)
    false_road = table["called road"].drop(ROAD_CLASS)
    table["share of false road"] = false_road / max(false_road.sum(), 1)
    return table


def cached_road_calls(
    run: Path,
    frames: pd.DataFrame,
    signs: pd.DataFrame,
    path: Path,
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
    workers: int = LOADER_WORKERS,
) -> pd.DataFrame:
    """`road_calls_by_class` for a finished run, computed once and reused while nothing changed."""
    key = cache.fingerprint(
        "road calls", cache.read_stamp(run), frames["frame_id"].to_numpy(), ROAD_THRESHOLD
    )
    if cache.is_current(path, key):
        return pd.read_csv(path, index_col="true class")
    model = load_model(run)
    table = road_calls_by_class(model, frames, signs, raw_dir, label_dir, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    cache.record(path, key)
    del model
    return table


@torch.no_grad()
def time_per_frame(
    model: MultiTaskRCNN,
    images: list[torch.Tensor],
    warmup: int = TIMING_WARMUP,
) -> pd.Series:
    """
    Milliseconds per frame for one frame at a time, from tensor to boxes and road mask.

    The first `warmup` frames are run untimed, so the accelerator's start-up
    costs stay out of the median.
    """
    target = device()
    model = model.to(target).eval()
    timings = []
    for index, image in enumerate(images):
        image = image.to(target)
        synchronize(target)
        start = time.perf_counter()
        model.predict([image])
        synchronize(target)
        if index >= warmup:
            timings.append(1000 * (time.perf_counter() - start))
    return pd.Series(
        {
            "frames": len(timings),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
        }
    )


def timing_images(
    frames: pd.DataFrame, count: int = TIMING_FRAMES, raw_dir: Path = RAW_DIR
) -> list[torch.Tensor]:
    """The first `count` frames as tensors, loaded before any timer starts."""
    return [frame_tensor(file, raw_dir) for file in frames["file"].head(count)]


def box_kinds(
    boxes: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    score_threshold: float,
    iou: float = MATCH_IOU,
) -> list[tuple[np.ndarray, str]]:
    """
    Every box of a frame with what it was: a sign found, a sign missed or a false alarm.

    Detections below the threshold are not shown; a found sign is drawn with
    the detection's box, a missed one with its label box.
    """
    kept = scores >= score_threshold
    assigned = assign_detections(boxes[kept], scores[kept], truth, iou)
    drawn = [
        (box, "hit" if sign >= 0 else "false")
        for box, sign in zip(boxes[kept], assigned, strict=True)
    ]
    found = np.zeros(len(truth), dtype=bool)
    found[assigned[assigned >= 0]] = True
    return drawn + [(box, "miss") for box in truth[~found]]


@torch.no_grad()
def predict_some(
    model: MultiTaskRCNN, files: list[str], raw_dir: Path = RAW_DIR
) -> list[tuple[dict[str, np.ndarray], np.ndarray | None]]:
    """Detections and road probabilities for a handful of frames, to draw them."""
    target = device()
    model = model.to(target).eval()
    results = []
    for file in files:
        detections, road = model.predict([frame_tensor(file, raw_dir).to(target)])
        found = {name: value.cpu().numpy() for name, value in detections[0].items()}
        results.append((found, None if road is None else road[0].cpu().numpy()))
    return results


def prediction_panels(
    model: MultiTaskRCNN,
    chosen: pd.DataFrame,
    signs: pd.DataFrame,
    score_threshold: float,
    titles: list[str],
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
) -> list[dict]:
    """
    Panels for `plots.frame_panels`: each frame with its road calls and its boxes.

    Road pixels are colored by how the call went, and boxes by whether they
    found a sign, missed one or raised a false alarm.
    """
    from driving_scenes.plots import road_overlay

    truth = boxes_by_frame(signs[signs["frame_id"].isin(set(chosen["frame_id"]))])
    empty = np.zeros((0, 4), np.float32)
    panels = []
    outputs = predict_some(model, chosen["file"].tolist(), raw_dir)
    for frame_id, file, (found, probability), title in zip(
        chosen["frame_id"], chosen["file"], outputs, titles, strict=True
    ):
        image = load_image(file, raw_dir)
        road = road_target(load_labels(frame_id, label_dir))
        predicted = None if probability is None else probability >= ROAD_THRESHOLD
        kinds = box_kinds(
            found["boxes"], found["scores"], truth.get(frame_id, empty), score_threshold
        )
        panels.append(
            {"image": road_overlay(image, predicted, road), "title": title, "boxes": kinds}
        )
    return panels
