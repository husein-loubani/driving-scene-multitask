"""Tests for the two-headed network: shapes, losses and what is trained."""

from __future__ import annotations

import math

import pytest
import torch

from driving_scenes.config import IGNORE_LABEL
from driving_scenes.modeling import model as model_module
from driving_scenes.modeling.model import (
    ROAD_LOSS,
    SIGN_LOSSES,
    RoadHead,
    build_model,
    road_loss,
    set_backbone_trainable,
    shared_parameters,
    trainable_share,
)

SIZE = (64, 96)


@pytest.fixture
def small_frames(monkeypatch):
    """Make the detector take small frames as they are, so a forward pass costs milliseconds."""
    monkeypatch.setattr(model_module, "MIN_SIZE", SIZE[0])
    monkeypatch.setattr(model_module, "MAX_SIZE", SIZE[1])


@pytest.fixture
def batch():
    torch.manual_seed(0)
    images = [torch.rand(3, *SIZE) for _ in range(2)]
    targets = [
        {"boxes": torch.tensor([[10.0, 8.0, 22.0, 40.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)},
    ]
    road = torch.zeros(2, *SIZE, dtype=torch.uint8)
    road[:, 40:] = 1
    road[0, :2] = IGNORE_LABEL
    return images, targets, road


def test_the_road_head_returns_one_logit_per_pixel():
    features = {
        level: torch.rand(2, 16, 64 // stride, 96 // stride)
        for level, stride in zip("0123", (4, 8, 16, 32), strict=True)
    }
    head = RoadHead(in_channels=16, channels=8, groups=4)
    assert head(features, (64, 96)).shape == (2, 64, 96)


def test_the_road_loss_is_the_mean_over_labeled_pixels_only():
    logits = torch.tensor([[[0.0, 0.0, 9.0]]])
    road = torch.tensor([[[1, 0, IGNORE_LABEL]]], dtype=torch.uint8)
    assert road_loss(logits, road).item() == pytest.approx(math.log(2))
    confident = torch.tensor([[[3.0, -3.0, -3.0]]])
    assert road_loss(confident, road).item() == pytest.approx(math.log1p(math.exp(-3)), rel=1e-4)


def test_a_frame_without_labels_gives_a_zero_loss_not_an_error():
    road = torch.full((1, 2, 2), IGNORE_LABEL, dtype=torch.uint8)
    assert road_loss(torch.zeros(1, 2, 2), road).item() == 0.0


def test_training_returns_the_loss_terms_of_both_tasks(small_frames, batch):
    network = build_model(pretrained=False).train()
    terms = network(*batch)
    assert set(terms) == {*SIGN_LOSSES, ROAD_LOSS}
    assert all(torch.isfinite(value) and value.ndim == 0 for value in terms.values())


def test_one_forward_pass_yields_boxes_and_a_road_map(small_frames, batch):
    network = build_model(pretrained=False).eval()
    detections, road = network(batch[0])
    assert len(detections) == 2 and set(detections[0]) == {"boxes", "labels", "scores"}
    assert road.shape == (2, *SIZE) and road.min() >= 0 and road.max() <= 1
    assert all((item["labels"] == 1).all() for item in detections), "the only class is the sign"


def test_a_single_task_network_computes_only_its_own_head(small_frames, batch):
    signs_only = build_model(("signs",), pretrained=False)
    assert set(signs_only.train()(*batch)) == set(SIGN_LOSSES)
    assert signs_only.eval()(batch[0])[1] is None
    road_only = build_model(("road",), pretrained=False)
    assert set(road_only.train()(*batch)) == {ROAD_LOSS}
    detections, road = road_only.eval()(batch[0])
    assert all(len(item["boxes"]) == 0 for item in detections) and road.shape == (2, *SIZE)


def test_unknown_or_missing_tasks_are_refused():
    with pytest.raises(ValueError, match="tasks must be drawn from"):
        build_model(("lanes",), pretrained=False)
    with pytest.raises(ValueError, match="tasks must be drawn from"):
        build_model((), pretrained=False)


def test_validation_gets_losses_and_predictions_from_one_backbone_pass(small_frames, batch):
    network = build_model(pretrained=False).eval()
    calls = []
    network.detector.backbone.register_forward_hook(lambda *_: calls.append(1))
    detections, road, terms = network.predict_with_losses(*batch)
    assert len(calls) == 1
    assert set(terms) == {*SIGN_LOSSES, ROAD_LOSS}
    assert not network.detector.rpn.training and not network.detector.roi_heads.training
    plain_detections, plain_road = network.predict(batch[0])
    torch.testing.assert_close(road, plain_road)
    torch.testing.assert_close(detections[0]["boxes"], plain_detections[0]["boxes"])


def test_predictions_come_back_in_the_frame_s_own_pixels(batch):
    # Without the small-frame patch the detector enlarges the frame; the outputs must not.
    network = build_model(pretrained=False).eval()
    detections, road = network(batch[0][:1])
    assert road.shape == (1, *SIZE)
    assert (detections[0]["boxes"][:, [0, 2]] <= SIZE[1]).all()
    assert (detections[0]["boxes"][:, [1, 3]] <= SIZE[0]).all()


def test_the_backbone_is_frozen_or_fine_tuned_by_stage():
    network = build_model(pretrained=False)
    body = network.detector.backbone.body
    set_backbone_trainable(network, 0)
    assert not any(p.requires_grad for p in shared_parameters(network))
    assert all(p.requires_grad for p in network.road_head.parameters())
    assert all(p.requires_grad for p in network.detector.roi_heads.parameters())
    frozen_share = trainable_share(network)
    set_backbone_trainable(network, 3)
    trained = {name.split(".")[0] for name, p in body.named_parameters() if p.requires_grad}
    assert trained == {"layer2", "layer3", "layer4"}
    assert all(p.requires_grad for p in network.detector.backbone.fpn.parameters())
    assert frozen_share < trainable_share(network) < 1.0
    set_backbone_trainable(network, 5)
    assert trainable_share(network) == 1.0


def test_the_adapted_anchors_replace_the_coco_ones_and_resize_the_proposal_outputs():
    from driving_scenes.config import ANCHOR_SETS

    coco = build_model(pretrained=False, anchors="coco").detector.rpn
    adapted = build_model(pretrained=False, anchors="adapted").detector.rpn
    sizes, ratios = ANCHOR_SETS["adapted"]
    assert adapted.anchor_generator.sizes == sizes
    assert adapted.anchor_generator.aspect_ratios[0] == ratios
    assert coco.head.cls_logits.out_channels == 3
    assert adapted.head.cls_logits.out_channels == len(sizes[0]) * len(ratios)
    assert adapted.head.bbox_pred.out_channels == 4 * len(sizes[0]) * len(ratios)


def test_the_proposal_convolution_keeps_its_coco_weights_when_the_outputs_change():
    coco = build_model(anchors="coco").detector.rpn.head.conv
    adapted = build_model(anchors="adapted").detector.rpn.head.conv
    for kept, original in zip(adapted.parameters(), coco.parameters(), strict=True):
        torch.testing.assert_close(kept, original)
