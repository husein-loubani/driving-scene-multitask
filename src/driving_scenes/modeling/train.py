"""
Training in two stages, and keeping the two tasks even on the backbone they share.

**The optimizer** is stochastic gradient descent with momentum, the choice of
the Faster R-CNN, Mask R-CNN and Panoptic FPN papers. Each step moves every
weight against a running average of its recent gradients: the average keeps
90% of its previous value and adds the new gradient, so directions the batches
agree on build up speed and directions they disagree on cancel. Weight decay
pulls every weight slightly toward zero at each step. The learning rate warms
up linearly, because the new heads start random and their first gradients are
large, then follows a cosine down to zero.

**Stage one** trains the two heads on the frozen backbone. A randomly
initialized head sends noise into the features it reads, so the backbone stays
fixed until the heads have learned something. With nothing shared being
trained, the two tasks cannot interfere here.

**Stage two** also trains the feature pyramid and the last ResNet stages. Now
both losses send gradients into the same weights, and the one with the larger
gradient decides where the backbone goes. Adding the two losses with equal
weights does not make their gradients equal. The trainer therefore measures
each task's pull, the norm of its loss gradient at the pyramid's outputs, and
in the "equal" mode rescales the two gradients on their way into the backbone
so that both arrive with the same norm. The heads themselves always learn from
their own unscaled loss.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import shutil
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from driving_scenes import cache
from driving_scenes.config import (
    ANCHOR_SETS,
    BALANCE_MODES,
    BATCH_SIZE,
    DETECTIONS_PER_IMAGE,
    FLIP_RATE,
    LABEL_DIR,
    LOADER_WORKERS,
    MAX_SIZE,
    MIN_SIZE,
    MOMENTUM,
    PULL_EVERY,
    PULL_MOMENTUM,
    PYRAMID_LEVELS,
    RANDOM_SEED,
    RAW_DIR,
    ROAD_HEAD_CHANNELS,
    RUN_DIR,
    SCORE_FLOOR,
    TASKS,
    WARMUP_START,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from driving_scenes.dataset import BOX_COLUMNS, FrameDataset, collate, one_rendering_per_step
from driving_scenes.modeling.metrics import (
    road_counts,
    road_scores,
    sign_metric,
    sign_scores,
    to_cpu,
)
from driving_scenes.modeling.model import (
    ROAD_LOSS,
    MultiTaskRCNN,
    build_model,
    set_backbone_trainable,
)

CHECKPOINT = "best"
SETTINGS_NAME = "settings.json"
PULLS_NAME = "pulls.csv"
MONITOR = "val_score"
# Lightning's notices that are expected here and say nothing about the result.
EXPECTED_WARNINGS = (
    # The loaders are rebuilt every epoch to show another rendering of each moment.
    r".*does not have many workers.*",
    r".*persistent_workers.*",
    # A run folder is cleared before it is trained again.
    r".*Checkpoint directory .* exists and is not empty.*",
    # Scores are logged once per epoch, so the step interval never matters.
    r".*is smaller than the logging interval.*",
    # The caller may choose the CPU, as the tests do.
    r".*GPU available but not used.*",
    # Lightning's own use of a PyTorch helper that newer PyTorch versions deprecate.
    r".*isinstance\(treespec, LeafSpec\).*",
)


@contextmanager
def expected_warnings_held() -> Iterator[None]:
    """Hold back the expected Lightning warnings and per-run INFO lines; anything else surfaces."""
    for logger_name in ("lightning.pytorch", "lightning.fabric"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    with warnings.catch_warnings():
        for message in EXPECTED_WARNINGS:
            warnings.filterwarnings("ignore", message=message)
        yield


def release_memory() -> None:
    """Hand the memory of finished networks back, so the next stage starts with all of it."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def warmup_cosine(
    step: int, total: int, warmup: int = WARMUP_STEPS, start: float = WARMUP_START
) -> float:
    """The learning rate's multiplier at a step: a linear climb, then a cosine down to zero."""
    warmup = min(warmup, total // 2)
    if step < warmup:
        return start + (1 - start) * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def scale_gradient(tensor: torch.Tensor, weight: float) -> torch.Tensor:
    """`tensor` unchanged going forward, its gradient multiplied by `weight` coming back."""
    return tensor * weight + tensor.detach() * (1 - weight)


def pull_on(features: Mapping[str, torch.Tensor], loss: torch.Tensor) -> float:
    """The norm of a loss's gradient at the pyramid levels both heads read."""
    gradients = torch.autograd.grad(
        loss, [features[level] for level in PYRAMID_LEVELS], retain_graph=True
    )
    return float(torch.sqrt(sum((gradient**2).sum() for gradient in gradients)))


class TaskBalance:
    """
    The weight each task's gradient carries into the shared backbone.

    In the "sum" mode every weight is one. In the "equal" mode a task's weight
    is the mean pull over its own smoothed pull, so the weighted pulls are
    equal and their sum stays what it was.
    """

    def __init__(self, tasks: Sequence[str], mode: str, momentum: float = PULL_MOMENTUM):
        if mode not in BALANCE_MODES:
            raise ValueError(f"balance must be one of {BALANCE_MODES}, got {mode!r}")
        self.mode, self.momentum = mode, momentum
        self.pull: dict[str, float | None] = dict.fromkeys(tasks)

    def update(self, measured: Mapping[str, float]) -> None:
        for task, value in measured.items():
            old = self.pull[task]
            self.pull[task] = (
                value if old is None else self.momentum * old + (1 - self.momentum) * value
            )

    def weights(self) -> dict[str, float]:
        pulls = list(self.pull.values())
        if self.mode == "sum" or len(pulls) < 2 or any(not pull for pull in pulls):
            return dict.fromkeys(self.pull, 1.0)
        mean = sum(pulls) / len(pulls)
        return {task: mean / pull for task, pull in self.pull.items()}


class TwoHeadTraining(LightningModule):
    """One training stage of the multi-task network."""

    def __init__(self, model: MultiTaskRCNN, lr: float, total_steps: int, balance: str = "sum"):
        super().__init__()
        self.model = model
        self.lr, self.total_steps = lr, total_steps
        self.balance = TaskBalance(model.tasks, balance)
        self.pulls: list[dict[str, float]] = []
        # Kept outside the module tree: the COCO evaluation runs on the CPU.
        self.scorers = {"signs": sign_metric(), "road": torch.zeros(4, dtype=torch.long)}

    def shares_backbone(self) -> bool:
        """True when both tasks send gradients into a backbone that is being trained."""
        trained = any(p.requires_grad for p in self.model.detector.backbone.parameters())
        return trained and len(self.model.tasks) > 1

    def training_step(self, batch, batch_index: int) -> torch.Tensor:
        images, targets, road, _ = batch
        sizes = [image.shape[-2:] for image in images]
        image_list, features, targets = self.model.shared_features(images, targets)
        weights = self.balance.weights()
        views = {
            task: features
            if weight == 1.0
            else {level: scale_gradient(value, weight) for level, value in features.items()}
            for task, weight in weights.items()
        }
        terms: dict[str, torch.Tensor] = {}
        task_loss: dict[str, torch.Tensor] = {}
        if "signs" in views:
            sign_terms = self.model.detection_losses(image_list, views["signs"], targets)
            terms |= sign_terms
            task_loss["signs"] = sum(sign_terms.values())
        if "road" in views:
            task_loss["road"] = self.model.segmentation_loss(image_list, views["road"], road, sizes)
            terms[ROAD_LOSS] = task_loss["road"]
        if self.shares_backbone() and self.global_step % PULL_EVERY == 0:
            # The gradient at the unscaled features carries the weight; dividing
            # it out leaves the task's own pull.
            measured = {
                task: pull_on(features, loss) / weights[task] for task, loss in task_loss.items()
            }
            self.balance.update(measured)
            self.pulls.append(
                {"step": self.global_step, "epoch": self.current_epoch}
                | {f"pull_{task}": value for task, value in measured.items()}
                | {f"weight_{task}": weights[task] for task in measured}
            )
        total = sum(task_loss.values())
        logged = {f"train_{name}": value for name, value in terms.items()} | {"train_loss": total}
        self.log_dict(logged, on_step=False, on_epoch=True, batch_size=len(images))
        return total

    def validation_step(self, batch, batch_index: int) -> None:
        images, targets, road, _ = batch
        detections, probability, terms = self.model.predict_with_losses(images, targets, road)
        if "signs" in self.model.tasks:
            self.scorers["signs"].update(to_cpu(detections), to_cpu(targets))
        if probability is not None:
            self.scorers["road"] += road_counts(probability, road).sum(dim=0)
        logged = {f"val_{name}": value for name, value in terms.items()}
        self.log_dict(logged, batch_size=len(images))

    def on_validation_epoch_end(self) -> None:
        scores: dict[str, float] = {}
        monitored = []
        if "signs" in self.model.tasks:
            signs = sign_scores(self.scorers["signs"])
            scores |= {"val_sign_ap50": signs["ap50"], "val_sign_map": signs["map"]}
            monitored.append(signs["ap50"])
            self.scorers["signs"].reset()
        if "road" in self.model.tasks:
            road = road_scores(self.scorers["road"])
            scores |= {
                "val_road_iou": road["road_iou"],
                "val_road_miou": road["miou"],
                "val_pixel_accuracy": road["pixel_accuracy"],
            }
            monitored.append(road["road_iou"])
            self.scorers["road"] = torch.zeros(4, dtype=torch.long)
        # One network serves both tasks, so the epoch kept is the one with the
        # best average of the two primary scores.
        scores[MONITOR] = float(np.mean(monitored))
        self.log_dict(scores)

    def configure_optimizers(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            parameters, lr=self.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, partial(warmup_cosine, total=self.total_steps)
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": schedule, "interval": "step"}}


class FrameData(LightningDataModule):
    """
    Training and validation loaders.

    Every training epoch shows each moment of each training video once, under
    the next of its ten renderings, so the weather and the camera angle change
    from epoch to epoch at no extra cost. Validation always uses the same
    rendering of each moment.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        signs: pd.DataFrame,
        raw_dir: Path,
        label_dir: Path,
        shift: int,
        workers: int,
    ):
        super().__init__()
        self.train, self.val, self.signs = train, val, signs
        self.raw_dir, self.label_dir = raw_dir, label_dir
        self.shift, self.workers = shift, workers

    def loader(self, frames: pd.DataFrame, training: bool) -> DataLoader:
        data = FrameDataset(
            frames, self.signs, self.raw_dir, self.label_dir, flip=FLIP_RATE if training else 0.0
        )
        return DataLoader(
            data,
            batch_size=BATCH_SIZE,
            shuffle=training,
            num_workers=self.workers,
            collate_fn=collate,
        )

    def train_dataloader(self) -> DataLoader:
        shift = self.shift + self.trainer.current_epoch
        return self.loader(one_rendering_per_step(self.train, shift), training=True)

    def val_dataloader(self) -> DataLoader:
        return self.loader(one_rendering_per_step(self.val), training=False)


def steps_per_epoch(train: pd.DataFrame) -> int:
    return math.ceil(len(one_rendering_per_step(train)) / BATCH_SIZE)


def label_digest(frames: pd.DataFrame, signs: pd.DataFrame) -> list:
    """What identifies a set of frames and their labels in a fingerprint."""
    boxes = signs[signs["frame_id"].isin(set(frames["frame_id"]))]
    return [
        frames["frame_id"].to_numpy(),
        frames[["pixels_road", "unlabeled_pixels"]].to_numpy(),
        boxes[BOX_COLUMNS].to_numpy(),
    ]


def read_settings(run: Path) -> dict:
    return json.loads((run / SETTINGS_NAME).read_text())


def load_weights(model: MultiTaskRCNN, run: Path) -> MultiTaskRCNN:
    """Load a run's kept epoch into `model`, the weights that were scored on validation."""
    state = torch.load(run / f"{CHECKPOINT}.ckpt", map_location="cpu", weights_only=True)
    prefix = "model."
    model.load_state_dict(
        {name.removeprefix(prefix): value for name, value in state["state_dict"].items()}
    )
    return model


def load_model(run: Path) -> MultiTaskRCNN:
    """
    The network of a finished run, in evaluation mode.

    It carries the run's fingerprint as `cache_key`, so that anything computed
    from it can be cached against the exact weights.
    """
    settings = read_settings(run)
    network = build_model(settings["tasks"], pretrained=False, anchors=settings["anchors"])
    model = load_weights(network, run).eval()
    model.cache_key = cache.read_stamp(run)
    return model


def read_history(run: Path) -> pd.DataFrame:
    """One row per epoch from a run's log, training losses and validation scores side by side."""
    log = pd.read_csv(run / "metrics.csv")
    history = log.groupby("epoch").first().reset_index()
    return history.drop(columns="step").assign(epoch=history["epoch"] + 1)


def read_pulls(run: Path) -> pd.DataFrame:
    """Each task's measured pull on the shared features, and the weight it carried."""
    return pd.read_csv(run / PULLS_NAME)


def fit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    signs: pd.DataFrame,
    name: str,
    *,
    stages: int,
    epochs: int,
    lr: float,
    tasks: Sequence[str] = TASKS,
    balance: str = "sum",
    anchors: str = "adapted",
    seed: int = RANDOM_SEED,
    start: Path | None = None,
    shift: int = 0,
    run_dir: Path = RUN_DIR,
    raw_dir: Path = RAW_DIR,
    label_dir: Path = LABEL_DIR,
    workers: int = LOADER_WORKERS,
    accelerator: str = "auto",
) -> Path:
    """
    Train one stage and return its run folder, or reuse the finished run.

    `stages` is how many ResNet stages are fine-tuned (zero freezes the whole
    backbone), `start` is the run whose kept weights this stage begins from
    (COCO weights when None), and `shift` is the rendering cycle of the first
    epoch. A stage takes up to an hour on the Apple-silicon GPU, so a finished
    run is reused, but only when its fingerprint matches: the same frames and
    labels, the same starting weights, the same settings and seed.
    """
    run = run_dir / name
    settings = {
        "tasks": list(tasks),
        "stages": stages,
        "epochs": epochs,
        "lr": lr,
        "balance": balance,
        "anchors": anchors,
        "seed": seed,
        "shift": shift,
        "start": None if start is None else cache.read_stamp(start),
    }
    recipe = {
        "batch": BATCH_SIZE,
        "flip": FLIP_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "warmup": (WARMUP_START, WARMUP_STEPS),
        "pull": (PULL_EVERY, PULL_MOMENTUM),
        "anchors": ANCHOR_SETS[anchors],
        "input": (MIN_SIZE, MAX_SIZE),
        "road_head": ROAD_HEAD_CHANNELS,
        "detections": (SCORE_FLOOR, DETECTIONS_PER_IMAGE),
    }
    key = cache.fingerprint(
        "fit", label_digest(train, signs), label_digest(val, signs), settings, recipe
    )
    if cache.is_current(run, key):
        return run
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    release_memory()
    started = time.perf_counter()
    with expected_warnings_held():
        seed_everything(seed, workers=True)
        model = build_model(tasks, pretrained=start is None, anchors=anchors)
        if start is not None:
            load_weights(model, start)
        set_backbone_trainable(model, stages)
        module = TwoHeadTraining(model, lr, epochs * steps_per_epoch(train), balance)
        trainer = Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            logger=CSVLogger(save_dir=run, name="", version=""),
            callbacks=[
                ModelCheckpoint(
                    dirpath=run,
                    filename=CHECKPOINT,
                    monitor=MONITOR,
                    mode="max",
                    save_weights_only=True,
                    enable_version_counter=False,
                )
            ],
            reload_dataloaders_every_n_epochs=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, FrameData(train, val, signs, raw_dir, label_dir, shift, workers))
    del trainer
    release_memory()
    pulls = pd.DataFrame(module.pulls or {"step": [], "epoch": []})
    pulls.to_csv(run / PULLS_NAME, index=False)
    minutes = round((time.perf_counter() - started) / 60, 1)
    (run / SETTINGS_NAME).write_text(json.dumps(settings | {"minutes": minutes}, indent=1))
    cache.record(run, key)
    return run
