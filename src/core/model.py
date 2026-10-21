"""ResNet18 architecture with multi-layer activation taps for visualization."""

import torch
from torch import nn
from torchvision import models


class MultiOutputResNet18(nn.Module):
    """ResNet18 model with multi-layer activation taps.

    Taps:
        act_layer1: Output after conv1 + bn1 + relu (64 channels, 112x112)
        act_layer2: Output after layer1 (64 channels, 56x56)
        act_layer3: Output after layer2 (128 channels, 28x28)
        act_layer4: Output after layer3 (256 channels, 14x14)
        act_layer5: Output after layer4 (512 channels, 7x7)
        logits: Output after global average pooling and linear classification head (num_classes)
    """

    def __init__(self, num_classes: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

        # Initialize linear head with normal distribution
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # Layer 1: conv1 + bn1 + relu -> [B, 64, 112, 112]
        act1 = self.relu(self.bn1(self.conv1(x)))
        x_pool = self.maxpool(act1)

        # Layer 2: Residual Block 1 -> [B, 64, 56, 56]
        act2 = self.layer1(x_pool)

        # Layer 3: Residual Block 2 -> [B, 128, 28, 28]
        act3 = self.layer2(act2)

        # Layer 4: Residual Block 3 -> [B, 256, 14, 14]
        act4 = self.layer3(act3)

        # Layer 5: Residual Block 4 -> [B, 512, 7, 7]
        act5 = self.layer4(act4)

        # Global average pool & classification head -> [B, num_classes]
        pooled = self.avgpool(act5)
        flattened = torch.flatten(pooled, 1)
        logits = self.fc(flattened)

        if return_features:
            return logits, act1, act2, act3, act4, act5
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final classification head."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> list[dict[str, object]]:
        """Return parameter groups for differential learning rate training."""
        backbone_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("fc.") and p.requires_grad
        ]
        head_params = [
            p for name, p in self.named_parameters() if name.startswith("fc.") and p.requires_grad
        ]
        return [
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]
