"""
One network with two heads: traffic sign detection and road segmentation.

The shared part is the backbone of the Mask R-CNN family, a ResNet-50 whose
last four stages feed a feature pyramid network (FPN). The pyramid turns one
image into four feature maps at strides 4, 8, 16 and 32, each with 256
channels, so small and large objects are both described at a suitable scale.

Two heads read those maps in the same forward pass:

- The sign head is Faster R-CNN's (Ren et al. 2015). A region proposal network
  slides over every pyramid level and, for a fixed set of anchor boxes at each
  position, predicts whether the anchor covers an object and how to shift it.
  The best proposals are cropped from the pyramid with RoIAlign, and a small
  network classifies each crop as sign or background and refines its box.
- The road head is the semantic branch of Panoptic FPN (Kirillov et al. 2019).
  A road is "stuff": one amorphous region with no instances to tell apart, so
  it gets a per-pixel classifier, not a box and a mask per object. Each
  pyramid level is brought to stride 4 by 3x3 convolutions and bilinear
  upsampling, the four results are summed, and a 1x1 convolution gives one
  road logit per position, upsampled to the input size.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import torch
from torch import nn
from torch.nn import functional
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import FrozenBatchNorm2d

from driving_scenes.config import (
    ANCHOR_SETS,
    BACKBONE_STAGES,
    DETECTIONS_PER_IMAGE,
    IGNORE_LABEL,
    MAX_SIZE,
    MIN_SIZE,
    PYRAMID_LEVELS,
    ROAD_HEAD_CHANNELS,
    ROAD_HEAD_GROUPS,
    SCORE_FLOOR,
    SIGN_CLASSES,
    TASKS,
)

SIGN_LOSSES = ("loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg")
ROAD_LOSS = "loss_road"


class RoadHead(nn.Module):
    """Panoptic FPN's semantic branch with a single output channel, the road logit."""

    def __init__(
        self,
        in_channels: int,
        channels: int = ROAD_HEAD_CHANNELS,
        levels: Sequence[str] = PYRAMID_LEVELS,
        groups: int = ROAD_HEAD_GROUPS,
    ):
        super().__init__()
        self.scale_heads = nn.ModuleDict()
        # Level i has stride 4 * 2**i, so it needs i doublings to reach stride 4.
        for doublings, level in enumerate(levels):
            layers: list[nn.Module] = []
            for step in range(max(doublings, 1)):
                layers += [
                    nn.Conv2d(in_channels if step == 0 else channels, channels, 3, padding=1),
                    nn.GroupNorm(groups, channels),
                    nn.ReLU(inplace=True),
                ]
                if doublings:
                    layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.scale_heads[level] = nn.Sequential(*layers)
        self.predictor = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features: dict[str, torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        """Road logits of shape (batch, height, width) for an input of `size`."""
        merged = sum(head(features[level]) for level, head in self.scale_heads.items())
        logits = functional.interpolate(
            self.predictor(merged), size=size, mode="bilinear", align_corners=False
        )
        return logits[:, 0]


def road_loss(logits: torch.Tensor, road: torch.Tensor, ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """
    Binary cross-entropy per pixel, averaged over the labeled pixels.

    `road` holds 1 for road, 0 for everything else and `ignore` where the
    dataset gives no label; those pixels count neither for nor against.
    """
    labeled = (road != ignore).to(logits.dtype)
    total = functional.binary_cross_entropy_with_logits(
        logits, (road == 1).to(logits.dtype), weight=labeled, reduction="sum"
    )
    return total / labeled.sum().clamp(min=1)


class MultiTaskRCNN(nn.Module):
    """
    A Faster R-CNN detector and a road head on one shared backbone.

    `tasks` names the heads that are computed. With both, one forward pass
    yields the sign boxes and the road mask; with one, the same network serves
    as the single-task baseline for that task.
    """

    def __init__(self, detector: FasterRCNN, road_head: RoadHead, tasks: Sequence[str] = TASKS):
        super().__init__()
        unknown = set(tasks) - set(TASKS)
        if unknown or not tasks:
            raise ValueError(f"tasks must be drawn from {TASKS}, got {tuple(tasks)}")
        self.detector = detector
        self.road_head = road_head
        self.tasks = tuple(tasks)

    def shared_features(self, images: Sequence[torch.Tensor], targets=None):
        """Normalize and batch the images, then run the shared backbone once."""
        batch, targets = self.detector.transform(list(images), targets)
        return batch, self.detector.backbone(batch.tensors), targets

    def road_logits(self, features, batch, sizes) -> torch.Tensor:
        """Road logits in each image's own pixel grid, stacked (the frames share one size)."""
        padded = self.road_head(features, batch.tensors.shape[-2:])
        logits = []
        for row, resized, original in zip(padded, batch.image_sizes, sizes, strict=True):
            row = row[: resized[0], : resized[1]]
            if tuple(resized) != tuple(original):
                row = functional.interpolate(
                    row[None, None], size=original, mode="bilinear", align_corners=False
                )[0, 0]
            logits.append(row)
        return torch.stack(logits)

    def detection_losses(
        self, batch, features: dict[str, torch.Tensor], targets: Sequence[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """
        The four Faster R-CNN loss terms. The two detection heads must be in training mode.

        The proposal network contributes a binary cross-entropy on "does this
        anchor cover a sign" and a smooth L1 loss on the anchor's shift; the
        box head contributes a cross-entropy over sign and background and a
        smooth L1 loss on the final box.
        """
        proposals, proposal_losses = self.detector.rpn(batch, features, targets)
        _, box_losses = self.detector.roi_heads(features, proposals, batch.image_sizes, targets)
        return proposal_losses | box_losses

    def segmentation_loss(
        self,
        batch,
        features: dict[str, torch.Tensor],
        road: torch.Tensor,
        sizes: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        return road_loss(self.road_logits(features, batch, sizes), road)

    def losses(
        self,
        images: Sequence[torch.Tensor],
        sign_targets: Sequence[dict[str, torch.Tensor]],
        road: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Every loss term of the active tasks. The network must be in training mode."""
        sizes = [image.shape[-2:] for image in images]
        batch, features, targets = self.shared_features(images, list(sign_targets))
        terms: dict[str, torch.Tensor] = {}
        if "signs" in self.tasks:
            terms |= self.detection_losses(batch, features, targets)
        if "road" in self.tasks:
            terms[ROAD_LOSS] = self.segmentation_loss(batch, features, road, sizes)
        return terms

    def _outputs(self, batch, features, sizes):
        """Detections and road probabilities from the shared features, heads in evaluation mode."""
        if "signs" in self.tasks:
            proposals, _ = self.detector.rpn(batch, features)
            detections, _ = self.detector.roi_heads(features, proposals, batch.image_sizes)
            detections = self.detector.transform.postprocess(detections, batch.image_sizes, sizes)
        else:
            empty = batch.tensors.new_zeros
            detections = [
                {"boxes": empty((0, 4)), "scores": empty(0), "labels": empty(0, dtype=torch.long)}
                for _ in sizes
            ]
        road = None
        if "road" in self.tasks:
            road = torch.sigmoid(self.road_logits(features, batch, sizes))
        return detections, road

    @torch.no_grad()
    def predict(
        self, images: Sequence[torch.Tensor]
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor | None]:
        """
        Sign detections and road probabilities from one forward pass.

        The network must be in evaluation mode. A task that is not active
        returns no boxes, or None for the road.
        """
        sizes = [image.shape[-2:] for image in images]
        batch, features, _ = self.shared_features(images)
        return self._outputs(batch, features, sizes)

    @torch.no_grad()
    def predict_with_losses(
        self,
        images: Sequence[torch.Tensor],
        sign_targets: Sequence[dict[str, torch.Tensor]],
        road: torch.Tensor,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor | None, dict[str, torch.Tensor]]:
        """
        Predictions and loss terms for a validation batch from one pass through the backbone.

        The detection heads report losses only in training mode, so they are
        switched to it for the loss and back for the predictions. They hold no
        dropout or batch statistics, so the switch changes nothing else.
        """
        sizes = [image.shape[-2:] for image in images]
        batch, features, targets = self.shared_features(images, list(sign_targets))
        terms: dict[str, torch.Tensor] = {}
        if "signs" in self.tasks:
            heads = (self.detector.rpn, self.detector.roi_heads)
            for head in heads:
                head.train()
            terms |= self.detection_losses(batch, features, targets)
            for head in heads:
                head.eval()
        if "road" in self.tasks:
            terms[ROAD_LOSS] = self.segmentation_loss(batch, features, road, sizes)
        return *self._outputs(batch, features, sizes), terms

    def forward(self, images, sign_targets=None, road=None):
        if self.training:
            return self.losses(images, sign_targets, road)
        return self.predict(images)


def build_detector(pretrained: bool = True, anchors: str = "adapted") -> FasterRCNN:
    """
    Faster R-CNN with a ResNet-50 FPN backbone, COCO weights and a new sign classifier.

    The frames keep their own resolution: the detector's built-in resize is set
    to the frame size, so no sign loses pixels to interpolation. The COCO
    weights were trained with frozen batch-norm statistics and a zero epsilon,
    which the backbone reproduces.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=partial(FrozenBatchNorm2d, eps=0.0),
        trainable_layers=len(BACKBONE_STAGES),
    )
    detector = FasterRCNN(
        backbone,
        num_classes=len(weights.meta["categories"]),
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        box_score_thresh=SCORE_FLOOR,
        box_detections_per_img=DETECTIONS_PER_IMAGE,
    )
    if pretrained:
        detector.load_state_dict(weights.get_state_dict(progress=False, check_hash=True))
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(SIGN_CLASSES))
    anchor_sizes, anchor_ratios = ANCHOR_SETS[anchors]
    default = detector.rpn.anchor_generator
    ratios = (tuple(anchor_ratios),) * len(anchor_sizes)
    sizes = tuple(tuple(level) for level in anchor_sizes)
    if (sizes, ratios) != (default.sizes, default.aspect_ratios):
        generator = AnchorGenerator(sizes, ratios)
        # A different number of anchors per position changes the shape of the
        # proposal network's two output layers, so those start from scratch;
        # the convolution in front of them keeps its COCO weights.
        if generator.num_anchors_per_location() != default.num_anchors_per_location():
            head = RPNHead(backbone.out_channels, generator.num_anchors_per_location()[0])
            head.conv.load_state_dict(detector.rpn.head.conv.state_dict())
            detector.rpn.head = head
        detector.rpn.anchor_generator = generator
    return detector


def build_model(
    tasks: Sequence[str] = TASKS, pretrained: bool = True, anchors: str = "adapted"
) -> MultiTaskRCNN:
    detector = build_detector(pretrained, anchors)
    return MultiTaskRCNN(detector, RoadHead(detector.backbone.out_channels), tasks)


def shared_parameters(model: MultiTaskRCNN) -> list[nn.Parameter]:
    """The parameters both tasks read: the ResNet stages and the feature pyramid."""
    return list(model.detector.backbone.parameters())


def set_backbone_trainable(model: MultiTaskRCNN, stages: int) -> None:
    """
    Train the last `stages` ResNet stages and the pyramid, or freeze the whole backbone.

    With `stages` at zero only the two heads learn. Otherwise the pyramid and
    the named stages (counted from the output: layer4, layer3, ...) are
    fine-tuned and the earlier ones stay frozen. Batch-norm statistics are
    frozen throughout, as in the pretrained network.
    """
    trained = BACKBONE_STAGES[:stages]
    for name, parameter in model.detector.backbone.body.named_parameters():
        parameter.requires_grad_(name.split(".")[0] in trained)
    for parameter in model.detector.backbone.fpn.parameters():
        parameter.requires_grad_(stages > 0)


def trainable_share(model: nn.Module) -> float:
    """Share of the network's weights that receive gradients."""
    counts = [(p.numel(), p.requires_grad) for p in model.parameters()]
    return sum(n for n, trained in counts if trained) / sum(n for n, _ in counts)
