"""
Every figure in the notebook.

Each function returns a matplotlib Figure and never shows it, so the notebook
decides when to display and `save_figure` decides where it lands. Colors come
from the palette in `config`, every axis is labeled, and bars start at zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from driving_scenes.config import (
    CLASSES,
    FIGURE_DIR,
    FIGURE_DPI,
    IGNORE_LABEL,
    LABEL_DIR,
    PALETTE,
    RAW_DIR,
    ROAD_ALPHA,
    ROAD_COLORS,
    VARIATIONS,
)
from driving_scenes.dataset import load_image, load_labels, road_target

CLASS_COLORS = dict(zip(CLASSES, plt.get_cmap("tab20").colors, strict=False))
BOX_STYLES = {
    "truth": {"color": PALETTE["truth"], "linestyle": "-"},
    "hit": {"color": PALETTE["hit"], "linestyle": "-"},
    "miss": {"color": PALETTE["miss"], "linestyle": "--"},
    "false": {"color": PALETTE["false"], "linestyle": "-"},
}
BOX_LABELS = {
    "truth": "sign (label)",
    "hit": "sign found",
    "miss": "sign missed",
    "false": "false alarm",
}


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )


def save_figure(figure: Figure, name: str, section: str, root: Path = FIGURE_DIR) -> Path:
    path = root / section / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    return path


def road_overlay(
    image: np.ndarray, predicted: np.ndarray | None = None, truth: np.ndarray | None = None
) -> np.ndarray:
    """
    The frame with the road tinted.

    With only `truth` (1 road, 0 other, 255 unlabeled), the true road is tinted.
    With `predicted` (a boolean mask) as well, each pixel shows how the call
    went: road called road, other called road, road missed. Unlabeled pixels
    are left as they are.
    """
    shown = image.astype(float).copy()
    if predicted is None:
        regions = {"hit": truth == 1}
    else:
        labeled = truth != IGNORE_LABEL
        regions = {
            "hit": predicted & (truth == 1),
            "false": predicted & (truth == 0),
            "miss": ~predicted & (truth == 1) & labeled,
        }
    for name, region in regions.items():
        shown[region] = (1 - ROAD_ALPHA) * shown[region] + ROAD_ALPHA * np.array(ROAD_COLORS[name])
    return shown.astype(np.uint8)


def frame_panels(panels: Sequence[Mapping], columns: int = 1, width: float = 11.0) -> Figure:
    """
    Frames in a grid, each with a title and optional boxes.

    A panel is a mapping with `image`, `title` and optionally `boxes`: a list
    of (box, kind) pairs, the box as (x_min, y_min, x_max, y_max) and the kind
    one of the `BOX_STYLES`. A legend lists the kinds that appear.
    """
    rows = int(np.ceil(len(panels) / columns))
    height, frame_width = np.asarray(panels[0]["image"]).shape[:2]
    panel_height = width / columns * height / frame_width + 0.45
    figure, axes = plt.subplots(rows, columns, figsize=(width, rows * panel_height), squeeze=False)
    kinds = set()
    for axis in axes.ravel():
        axis.axis("off")
    for axis, panel in zip(axes.ravel(), panels, strict=False):
        axis.imshow(panel["image"])
        axis.set_title(panel["title"], fontsize=8)
        for (x_min, y_min, x_max, y_max), kind in panel.get("boxes", []):
            style = BOX_STYLES[kind]
            axis.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    linewidth=1.4,
                    **style,
                )
            )
            kinds.add(kind)
    handles = [
        Patch(facecolor="none", edgecolor=BOX_STYLES[kind]["color"], label=BOX_LABELS[kind])
        for kind in BOX_STYLES
        if kind in kinds
    ]
    if handles:
        figure.legend(handles=handles, loc="lower center", ncols=len(handles), frameon=False)
    figure.tight_layout(rect=(0, 0.03 if handles else 0, 1, 1))
    return figure


def plot_class_shares(shares: pd.DataFrame) -> Figure:
    """Share of the pixels each class covers, one stacked bar per group."""
    figure, axis = plt.subplots(figsize=(10, 0.45 * len(shares) + 1.8))
    left = np.zeros(len(shares))
    labels = [str(group) for group in shares.index]
    for name in shares.columns:
        color = CLASS_COLORS.get(name, (0.85, 0.85, 0.85))
        axis.barh(labels, shares[name], left=left, color=color, label=name)
        left += shares[name].to_numpy()
    axis.set(xlim=(0, 1), xlabel="Share of pixels", ylabel=shares.index.name or "Group")
    axis.invert_yaxis()
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=5, frameon=False)
    axis.set_title("What the frames show: pixel share of each class")
    figure.tight_layout()
    return figure


def plot_sign_shapes(signs: pd.DataFrame) -> Figure:
    """Box width against height, the height-to-width ratio, and the COCO size classes per world."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sample = signs.sample(min(len(signs), 8000), random_state=0)
    axes[0].scatter(sample["width"], sample["height"], s=3, alpha=0.25, color=PALETTE["signs"])
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Box width (pixels)",
        ylabel="Box height (pixels)",
        title="Sign boxes are narrow and tall",
    )
    reference = np.array([3, 400])
    for ratio in (1, 4):
        axes[0].plot(reference, reference * ratio, color="0.4", linewidth=0.8, linestyle=":")
        axes[0].annotate(
            f"height = {ratio} x width", (reference[0] * 1.5, reference[0] * 1.5 * ratio)
        )
    bins = np.logspace(np.log10(0.1), np.log10(30), 50)
    axes[1].hist(signs["aspect"], bins=bins, color=PALETTE["signs"])
    axes[1].set(
        xscale="log",
        xlabel="Height over width (log scale)",
        ylabel="Sign regions",
        title="Height-to-width ratio",
    )
    shares = pd.crosstab(signs["world"], signs["size"], normalize="index")
    bottom = np.zeros(len(shares))
    for name, color in zip(shares.columns, ("#a6d96a", "#1b9e77", "#00441b"), strict=True):
        axes[2].bar(shares.index.astype(str), shares[name], bottom=bottom, color=color, label=name)
        bottom += shares[name].to_numpy()
    axes[2].set(xlabel="World", ylabel="Share of sign regions", title="COCO size classes")
    axes[2].legend(title="size", loc="lower right")
    figure.tight_layout()
    return figure


def plot_heatmaps(maps: Mapping[str, tuple[np.ndarray, str]]) -> Figure:
    """
    One image per entry: how often something covers each pixel of the frame.

    `maps` maps a title to the values and the label of their color scale.
    """
    figure, axes = plt.subplots(len(maps), 1, figsize=(10, 2.3 * len(maps)), squeeze=False)
    for axis, (title, (values, label)) in zip(axes[:, 0], maps.items(), strict=True):
        shown = axis.imshow(values, cmap="magma", aspect="equal")
        axis.set(title=title, xlabel="Column (pixels)", ylabel="Row (pixels)")
        axis.grid(False)
        figure.colorbar(shown, ax=axis, fraction=0.015, pad=0.01).set_label(label)
    figure.tight_layout()
    return figure


def plot_rendering_light(frames: pd.DataFrame) -> Figure:
    """Brightness and contrast of the frames under each rendering."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    order = [name for name in VARIATIONS if name in set(frames["variation"])]
    for axis, column, label in (
        (axes[0], "brightness", "Mean brightness (0-255)"),
        (axes[1], "contrast", "Brightness standard deviation"),
    ):
        groups = [frames.loc[frames["variation"] == name, column] for name in order]
        axis.boxplot(groups, tick_labels=order, showfliers=False)
        axis.set(ylabel=label, xlabel="Rendering")
        axis.tick_params(axis="x", rotation=40)
    axes[0].set_title("How bright each rendering is")
    axes[1].set_title("How much contrast it keeps")
    figure.tight_layout()
    return figure


def plot_repetition(table: pd.DataFrame) -> Figure:
    """How much of a frame's labels survive a given number of frames later, per world."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for world, group in table.groupby("world"):
        axes[0].plot(group["lag"], group["road_iou"], marker="o", label=world)
        axes[1].plot(group["lag"], group["signs_still_matched"], marker="o", label=world)
    axes[0].set(
        xscale="log",
        xlabel="Frames apart",
        ylabel="Road mask IoU",
        title="Road masks of nearby frames",
        ylim=(0, 1),
    )
    axes[1].set(
        xscale="log",
        xlabel="Frames apart",
        ylabel="Share of sign boxes still matched",
        title="Signs of nearby frames (IoU >= 0.5)",
        ylim=(0, 1),
    )
    axes[1].legend(title="World")
    figure.tight_layout()
    return figure


def plot_anchor_coverage(best: Mapping[str, np.ndarray], levels: Sequence[float]) -> Figure:
    """The share of sign boxes whose best anchor reaches each IoU, per anchor set."""
    figure, axis = plt.subplots(figsize=(9, 4.2))
    grid = np.linspace(0, 1, 201)
    for name, values in best.items():
        axis.plot(grid, [(values >= level).mean() for level in grid], label=name)
    for level in levels:
        axis.axvline(level, color="0.5", linewidth=0.8, linestyle=":")
    axis.set(
        xlabel="IoU of the best anchor with the sign box",
        ylabel="Share of training sign boxes at or above",
        title="How well each anchor set fits the training signs",
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower left")
    figure.tight_layout()
    return figure


def plot_mask_ceiling(table: pd.DataFrame) -> Figure:
    """The road IoU each head could reach at best, frame by frame."""
    figure, axis = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(min(table.min(numeric_only=True).min(), 0.5), 1, 41)
    for column, color in zip(
        ("semantic head", "mask head"), (PALETTE["road"], PALETTE["single"]), strict=True
    ):
        axis.hist(table[column], bins=bins, alpha=0.7, color=color, label=column)
    axis.set(
        xlabel="Best possible road IoU of the frame",
        ylabel="Frames",
        title="How much of the road each kind of head can draw",
    )
    axis.legend()
    figure.tight_layout()
    return figure


def plot_history(histories: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Losses and validation scores per epoch across the training stages.

    `histories` maps a stage name to its history; the stages are drawn one
    after the other, with a line where each new stage begins.
    """
    figure, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    offset = 0
    for index, history in enumerate(histories.values()):
        epochs = history["epoch"] + offset
        style = {"marker": "o", "linestyle": "-" if index else "--"}
        signs = history[[c for c in history if c.startswith("train_loss_") and "road" not in c]]
        axes[0].plot(epochs, signs.sum(axis=1), color=PALETTE["signs"], **style)
        axes[0].plot(epochs, history["train_loss_road"], color=PALETTE["road"], **style)
        val_signs = history[[c for c in history if c.startswith("val_loss_") and "road" not in c]]
        axes[1].plot(epochs, val_signs.sum(axis=1), color=PALETTE["signs"], **style)
        axes[1].plot(epochs, history["val_loss_road"], color=PALETTE["road"], **style)
        axes[2].plot(epochs, history["val_sign_ap50"], color=PALETTE["signs"], **style)
        axes[2].plot(epochs, history["val_road_iou"], color=PALETTE["road"], **style)
        if index:
            for axis in axes:
                axis.axvline(offset + 0.5, color="0.5", linewidth=0.8, linestyle=":")
        offset += len(history)
    handles = [
        plt.Line2D([], [], color=PALETTE["signs"], label="signs"),
        plt.Line2D([], [], color=PALETTE["road"], label="road"),
        plt.Line2D([], [], color="0.3", linestyle="--", label=list(histories)[0]),
        plt.Line2D([], [], color="0.3", linestyle="-", label=list(histories)[-1]),
    ]
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss per task")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Validation loss per task")
    axes[2].set(xlabel="Epoch", ylabel="Score", title="Validation AP50 and road IoU", ylim=(0, 1))
    axes[2].legend(handles=handles, loc="lower right")
    figure.tight_layout()
    return figure


def plot_pulls(pulls: pd.DataFrame) -> Figure:
    """Each task's measured pull on the shared features, and the weight it was given."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharex=True)
    for task in ("signs", "road"):
        color = PALETTE[task]
        smooth = pulls[f"pull_{task}"].rolling(10, min_periods=1).median()
        axes[0].plot(pulls["step"], pulls[f"pull_{task}"], color=color, alpha=0.25)
        axes[0].plot(pulls["step"], smooth, color=color, label=task)
        axes[1].plot(pulls["step"], pulls[f"weight_{task}"], color=color, label=task)
    axes[0].set(
        yscale="log",
        xlabel="Training step of stage two",
        ylabel="Gradient norm at the pyramid",
        title="Each task's pull on the shared features",
    )
    axes[1].axhline(1, color="0.5", linewidth=0.8, linestyle=":")
    axes[1].set(
        xlabel="Training step of stage two",
        ylabel="Weight on the gradient",
        title="Weight that makes the pulls equal",
    )
    axes[0].legend()
    figure.tight_layout()
    return figure


def plot_runs(table: pd.DataFrame, band: pd.DataFrame | None = None) -> Figure:
    """
    Validation and test scores of each run, signs and road side by side.

    `table` has one row per run with `sign_ap50` and `road_iou` columns for
    `validation` and `test` (a two-level column index). `band`, when given,
    shades each score's range over the recipe's seeds.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(table) + 1.6), sharey=True)
    positions = np.arange(len(table))
    for axis, (score, title) in zip(
        axes, (("sign_ap50", "Signs: AP50"), ("road_iou", "Road: IoU")), strict=True
    ):
        for partition, marker in (("validation", "o"), ("test", "s")):
            if (partition, score) in table:
                axis.scatter(
                    table[(partition, score)], positions, marker=marker, s=30, label=partition
                )
            if band is not None and (partition, score) in band:
                low, high = band[(partition, score)]
                axis.axvspan(low, high, alpha=0.12, color="0.4" if partition == "test" else "C0")
        axis.set(title=title, xlabel="Score")
    axes[0].set_yticks(positions, table.index)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.set_ylabel("Run")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, frameon=False)
    figure.tight_layout()
    return figure


def plot_group_scores(table: pd.DataFrame, group_name: str) -> Figure:
    """AP50 and road IoU for each group of test frames, such as each rendering."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharex=True)
    labels = [str(index) for index in table.index]
    for axis, (score, task, title) in zip(
        axes,
        (("sign_ap50", "signs", "Signs: AP50"), ("road_iou", "road", "Road: IoU")),
        strict=True,
    ):
        axis.bar(labels, table[score], color=PALETTE[task])
        axis.axhline(table[score].mean(), color="0.3", linewidth=0.8, linestyle=":")
        axis.set(title=title, xlabel=group_name, ylabel="Score", ylim=(0, 1))
        axis.tick_params(axis="x", rotation=40)
    figure.tight_layout()
    return figure


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float) -> Figure:
    """Precision, recall and F1 of the sign detections at each confidence threshold."""
    figure, axis = plt.subplots(figsize=(8, 3.8))
    for column in ("precision", "recall", "f1"):
        axis.plot(sweep["threshold"], sweep[column], marker="o", markersize=3, label=column)
    axis.axvline(chosen, color="0.4", linestyle=":", label=f"chosen ({chosen:.2f})")
    axis.set(
        xlabel="Confidence threshold",
        ylabel="Score",
        title="Validation: choosing where a detection counts as a sign",
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    return figure


def plot_reliability(table: pd.DataFrame, error: float) -> Figure:
    """Hit rate against mean confidence per confidence bin, beside the diagonal of calibration."""
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, linestyle=":", label="calibrated")
    axis.plot(
        table["mean confidence"],
        table["hit rate"],
        marker="o",
        color=PALETTE["signs"],
        label="detections",
    )
    for _, row in table.iterrows():
        axis.annotate(
            f"{row['detections']:,.0f}",
            (row["mean confidence"], row["hit rate"]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=7,
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence in the bin",
        ylabel="Share of detections that hit a sign",
        title=f"Reliability of the detection confidence (ECE {error:.3f})",
    )
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_road_calls(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """
    What the false road really is, for each set of frames.

    `tables` maps a frame set's name to its `road_calls_by_class` table. The
    left panel shows how much of each class is called road, the right panel
    how much of all false road each class makes up.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    first = next(iter(tables.values()))
    order = first.drop(index="road").sort_values("share called road").index
    positions = np.arange(len(order))
    height = 0.8 / len(tables)
    for index, (name, table) in enumerate(tables.items()):
        shown = table.loc[order]
        offset = positions + (index - (len(tables) - 1) / 2) * height
        for axis, column in zip(axes, ("share called road", "share of false road"), strict=True):
            axis.barh(offset, shown[column].fillna(0), height=height, label=name)
    axes[0].set(
        xlabel="Share of the class's pixels called road",
        ylabel="True class",
        title="How much of each class is called road",
    )
    axes[1].set(
        xlabel="Share of all false road",
        ylabel="True class",
        title="What the false road is made of",
    )
    axes[0].set_yticks(positions, order)
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    return figure


def plot_size_scores(table: pd.DataFrame) -> Figure:
    """Recall at the chosen threshold, and AP, for each COCO size class of sign."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = [str(index) for index in table.index]
    axes[0].bar(labels, table["recall"], color=PALETTE["signs"])
    axes[0].set(xlabel="Sign size (COCO)", ylabel="Share of signs found", ylim=(0, 1))
    axes[0].set_title("Signs found at the chosen threshold")
    axes[1].bar(labels, table["ap"], color=PALETTE["signs"])
    axes[1].set(
        xlabel="Sign size (COCO)", ylabel="Average precision (IoU 0.5 to 0.95)", ylim=(0, 1)
    )
    axes[1].set_title("COCO average precision")
    figure.tight_layout()
    return figure


def truth_panels(
    chosen: pd.DataFrame,
    boxes: Mapping[str, np.ndarray],
    titles: Sequence[str],
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
) -> list[dict]:
    """Panels for `frame_panels`: each frame with its labeled road tinted and its sign boxes."""
    panels = []
    for frame_id, file, title in zip(chosen["frame_id"], chosen["file"], titles, strict=True):
        road = road_target(load_labels(frame_id, label_dir))
        panels.append(
            {
                "image": road_overlay(load_image(file, raw_dir), truth=road),
                "title": title,
                "boxes": [(box, "truth") for box in boxes.get(frame_id, [])],
            }
        )
    return panels
