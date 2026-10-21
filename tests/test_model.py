"""Tests for MultiOutputResNet18 architecture."""

import torch

from src.core.model import MultiOutputResNet18


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


def test_freeze_backbone():
    model = MultiOutputResNet18(num_classes=256, pretrained=False)
    model.freeze_backbone()

    # Backbone parameters should not require gradients
    assert not model.conv1.weight.requires_grad
    assert not model.layer1[0].conv1.weight.requires_grad
    assert not model.layer4[0].conv1.weight.requires_grad

    # Head parameters should require gradients
    assert model.fc.weight.requires_grad
    assert model.fc.bias.requires_grad

    model.unfreeze_backbone()
    assert model.conv1.weight.requires_grad
    assert model.layer4[0].conv1.weight.requires_grad
