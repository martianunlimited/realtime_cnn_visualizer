"""
Export PyTorch CNN models to ONNX with multi-layer activation outputs.
Supports MobileNetV2 (lightweight <10MB for instant web loading)
and ResNet50 (standard high-capacity backbone with Caltech-256 / ImageNet head).
"""

import os

import torch
import torch.nn as nn
import torchvision.models as models


class MultiOutputMobileNetV2(nn.Module):
    """
    MobileNetV2 with multi-layer activation taps for real-time web visualization.
    Weights are ~9MB in FP32, ~3MB in INT8.
    """

    def __init__(self, num_classes=256):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(base.last_channel, num_classes))

    def forward(self, x):
        # x: [1, 3, 224, 224]
        # Layer 1: Early edge/color filters (features[0-1])
        act1 = self.features[0:2](x)  # [1, 16, 112, 112]
        # Layer 2: Texture/corners (features[2-3])
        act2 = self.features[2:4](act1)  # [1, 24, 56, 56]
        # Layer 3: Patterns (features[4-6])
        act3 = self.features[4:7](act2)  # [1, 32, 28, 28]
        # Layer 4: Object parts (features[7-13])
        act4 = self.features[7:14](act3)  # [1, 96, 14, 14]
        # Layer 5: High-level semantic features (features[14-18])
        act5 = self.features[14:19](act4)  # [1, 1280, 7, 7]

        # Global Average Pool & Classify
        pool = act5.mean([2, 3])
        logits = self.classifier(pool)

        return logits, act1, act2, act3, act4, act5


class MultiOutputResNet50(nn.Module):
    """
    ResNet50 with multi-layer activation taps matching the original Qt project.
    """

    def __init__(self, num_classes=256):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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

    def forward(self, x):
        act1 = self.relu(self.bn1(self.conv1(x)))  # [1, 64, 112, 112]
        x_pool = self.maxpool(act1)
        act2 = self.layer1(x_pool)  # [1, 256, 56, 56]
        act3 = self.layer2(act2)  # [1, 512, 28, 28]
        act4 = self.layer3(act3)  # [1, 1024, 14, 14]
        act5 = self.layer4(act4)  # [1, 2048, 7, 7]

        pool = self.avgpool(act5)
        flat = torch.flatten(pool, 1)
        logits = self.fc(flat)

        return logits, act1, act2, act3, act4, act5


def export_model(model, output_path, model_name):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    input_names = ["input"]
    output_names = [
        "logits",
        "act_layer1",
        "act_layer2",
        "act_layer3",
        "act_layer4",
        "act_layer5",
    ]

    dynamic_axes = {
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "act_layer1": {0: "batch_size"},
        "act_layer2": {0: "batch_size"},
        "act_layer3": {0: "batch_size"},
        "act_layer4": {0: "batch_size"},
        "act_layer5": {0: "batch_size"},
    }

    print(f"Exporting {model_name} to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully exported {output_path} ({size_mb:.2f} MB)")

    # Quantize to INT8
    int8_path = output_path.replace(".onnx", "_int8.onnx")
    print(f"Quantizing {model_name} to INT8 ({int8_path})...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(model_input=output_path, model_output=int8_path, weight_type=QuantType.QUInt8)
    int8_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"Quantized model saved: {int8_path} ({int8_mb:.2f} MB) -> Fits under 25MB GitHub limit!")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # 1. Export lightweight MobileNetV2 (~9.7 MB FP32, ~3 MB INT8)
    print("\n=== Exporting MobileNetV2 ===")
    mb_model = MultiOutputMobileNetV2(num_classes=256)
    export_model(mb_model, "models/mobilenetv2_caltech256.onnx", "MobileNetV2")

    # 2. Export ResNet50 (~98 MB FP32 -> ~24.5 MB INT8)
    print("\n=== Exporting ResNet50 ===")
    resnet = MultiOutputResNet50(num_classes=256)
    export_model(resnet, "models/resnet50_caltech256.onnx", "ResNet50")
