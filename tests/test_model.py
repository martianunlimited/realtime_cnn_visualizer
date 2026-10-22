"""Tests for MultiOutput CNN architectures."""

import torch

from src.core.model import (
    MultiOutputMobileNetV2,
    MultiOutputResNet18,
    MultiOutputResNet50,
    create_multi_output_model,
)


def test_resnet18_forward_training():
    model = MultiOutputResNet18(num_classes=256, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    logits = model(x, return_features=False)
    assert logits.shape == (2, 256)


def test_resnet18_forward_features():
    model = MultiOutputResNet18(num_classes=256, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    logits, act1, act2, act3, act4, act5 = model(x, return_features=True)

    assert logits.shape == (2, 256)
    assert act1.shape == (2, 64, 112, 112)
    assert act2.shape == (2, 64, 56, 56)
    assert act3.shape == (2, 128, 28, 28)
    assert act4.shape == (2, 256, 14, 14)
    assert act5.shape == (2, 512, 7, 7)


def test_mobilenet_v2_forward_features():
    model = MultiOutputMobileNetV2(num_classes=256, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    logits, act1, act2, act3, act4, act5 = model(x, return_features=True)

    assert logits.shape == (2, 256)
    assert act1.shape == (2, 16, 112, 112)
    assert act2.shape == (2, 24, 56, 56)
    assert act3.shape == (2, 32, 28, 28)
    assert act4.shape == (2, 96, 14, 14)
    assert act5.shape == (2, 1280, 7, 7)


def test_resnet50_forward_features():
    model = MultiOutputResNet50(num_classes=256, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    logits, act1, act2, act3, act4, act5 = model(x, return_features=True)

    assert logits.shape == (2, 256)
    assert act1.shape == (2, 64, 112, 112)
    assert act2.shape == (2, 256, 56, 56)
    assert act3.shape == (2, 512, 28, 28)
    assert act4.shape == (2, 1024, 14, 14)
    assert act5.shape == (2, 2048, 7, 7)


def test_freeze_unfreeze():
    for model in [
        MultiOutputResNet18(num_classes=256, pretrained=False),
        MultiOutputMobileNetV2(num_classes=256, pretrained=False),
        MultiOutputResNet50(num_classes=256, pretrained=False),
    ]:
        model.freeze_backbone()
        groups = model.get_parameter_groups(lr_backbone=1e-5, lr_head=1e-3, weight_decay=1e-4)
        assert len(groups) == 2
        # Head parameters should require grad
        assert any(p.requires_grad for p in groups[1]["params"])
        # Backbone parameters should NOT require grad
        assert not any(p.requires_grad for p in groups[0]["params"])

        model.unfreeze_backbone()
        groups_unfrozen = model.get_parameter_groups(
            lr_backbone=1e-5, lr_head=1e-3, weight_decay=1e-4
        )
        assert all(p.requires_grad for p in groups_unfrozen[0]["params"])


def test_create_multi_output_model_factory():
    m1 = create_multi_output_model("resnet18", pretrained=False)
    assert isinstance(m1, MultiOutputResNet18)

    m2 = create_multi_output_model("mobilenet_v2", pretrained=False)
    assert isinstance(m2, MultiOutputMobileNetV2)

    m3 = create_multi_output_model("resnet50", pretrained=False)
    assert isinstance(m3, MultiOutputResNet50)
