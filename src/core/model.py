"""CNN architectures with multi-layer activation taps for visualization."""

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
        act1 = self.relu(self.bn1(self.conv1(x)))
        x_pool = self.maxpool(act1)
        act2 = self.layer1(x_pool)
        act3 = self.layer2(act2)
        act4 = self.layer3(act3)
        act5 = self.layer4(act4)

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


class MultiOutputMobileNetV2(nn.Module):
    """MobileNetV2 with multi-layer activation taps for lightweight WebGPU inference.

    Taps:
        act_layer1: Output after features[0:2] (16 channels, 112x112)
        act_layer2: Output after features[2:4] (24 channels, 56x56)
        act_layer3: Output after features[4:7] (32 channels, 28x28)
        act_layer4: Output after features[7:14] (96 channels, 14x14)
        act_layer5: Output after features[14:19] (1280 channels, 7x7)
        logits: Output after global pool & dropout + linear head (num_classes)
    """

    def __init__(self, num_classes: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)

        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(base.last_channel, num_classes),
        )

        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier[1].bias, 0.0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        act1 = self.features[0:2](x)
        act2 = self.features[2:4](act1)
        act3 = self.features[4:7](act2)
        act4 = self.features[7:14](act3)
        act5 = self.features[14:19](act4)

        pooled = act5.mean([2, 3])
        logits = self.classifier(pooled)

        if return_features:
            return logits, act1, act2, act3, act4, act5
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all layers in feature backbone."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> list[dict[str, object]]:
        """Return parameter groups for differential learning rate training."""
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]
        return [
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]


class MultiOutputResNet50(nn.Module):
    """ResNet50 with multi-layer activation taps for deep residual representations.

    Taps:
        act_layer1: Output after conv1 + bn1 + relu (64 channels, 112x112)
        act_layer2: Output after layer1 (256 channels, 56x56)
        act_layer3: Output after layer2 (512 channels, 28x28)
        act_layer4: Output after layer3 (1024 channels, 14x14)
        act_layer5: Output after layer4 (2048 channels, 7x7)
        logits: Output after avgpool + linear head (num_classes)
    """

    def __init__(self, num_classes: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(2048, num_classes)

        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        act1 = self.relu(self.bn1(self.conv1(x)))
        x_pool = self.maxpool(act1)
        act2 = self.layer1(x_pool)
        act3 = self.layer2(act2)
        act4 = self.layer3(act3)
        act5 = self.layer4(act4)

        pooled = self.avgpool(act5)
        flattened = torch.flatten(pooled, 1)
        logits = self.fc(flattened)

        if return_features:
            return logits, act1, act2, act3, act4, act5
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all layers except the classification head."""
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


def create_multi_output_model(
    model_name: str, num_classes: int = 256, pretrained: bool = True
) -> nn.Module:
    """Factory creating multi-output CNN model matching requested architecture."""
    clean_name = model_name.replace("-", "").replace("_", "").lower()
    if "mobilenet" in clean_name:
        return MultiOutputMobileNetV2(num_classes=num_classes, pretrained=pretrained)
    elif "resnet50" in clean_name:
        return MultiOutputResNet50(num_classes=num_classes, pretrained=pretrained)
    elif "resnet18" in clean_name:
        return MultiOutputResNet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}. Supported: 'resnet18', 'mobilenet_v2', 'resnet50'"
        )
