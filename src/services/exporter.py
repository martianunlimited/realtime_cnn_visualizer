"""Export and Quantization service for multi-output CNNs."""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from src.config.settings import TrainingSettings


class MultiOutputOnnxWrapper(nn.Module):
    """Wrapper ensuring return_features=True is fixed for ONNX graph tracing."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        logits, act1, act2, act3, act4, act5 = self.model(x, return_features=True)
        return logits, act1, act2, act3, act4, act5


# Backward-compatibility alias
ResNet18OnnxWrapper = MultiOutputOnnxWrapper


def export_model_to_onnx(
    model: nn.Module,
    settings: TrainingSettings | None = None,
) -> tuple[Path, Path]:
    """Export multi-output CNN model to ONNX and quantize to INT8.

    Returns:
        tuple of (fp32_onnx_path, int8_onnx_path)
    """
    cfg = settings or TrainingSettings()
    cfg.onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.onnx_int8_output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = MultiOutputOnnxWrapper(model).cpu()
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

    print(f"\n[ONNX] Exporting {cfg.model_name} to {cfg.onnx_output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(cfg.onnx_output_path),
        export_params=True,
        opset_version=cfg.opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    size_fp32_mb = os.path.getsize(cfg.onnx_output_path) / (1024 * 1024)
    print(f"[ONNX] Exported FP32 model: {cfg.onnx_output_path} ({size_fp32_mb:.2f} MB)")

    print(f"[ONNX] Quantizing to INT8: {cfg.onnx_int8_output_path}...")
    quantize_dynamic(
        model_input=str(cfg.onnx_output_path),
        model_output=str(cfg.onnx_int8_output_path),
        weight_type=QuantType.QUInt8,
    )
    size_int8_mb = os.path.getsize(cfg.onnx_int8_output_path) / (1024 * 1024)
    print(f"[ONNX] Quantized INT8 model: {cfg.onnx_int8_output_path} ({size_int8_mb:.2f} MB)")

    # Verify exported INT8 model with ONNX Runtime
    print("[ONNX] Verifying INT8 model with ONNX Runtime...")
    session = ort.InferenceSession(
        str(cfg.onnx_int8_output_path), providers=["CPUExecutionProvider"]
    )
    ort_inputs = {session.get_inputs()[0].name: np.random.randn(1, 3, 224, 224).astype(np.float32)}
    ort_outputs = session.run(None, ort_inputs)

    assert len(ort_outputs) == 6, f"Expected 6 outputs, got {len(ort_outputs)}"
    print(f"[ONNX] Verification successful! Logits shape: {ort_outputs[0].shape}")

    return cfg.onnx_output_path, cfg.onnx_int8_output_path


# Backward-compatibility alias
export_resnet18_to_onnx = export_model_to_onnx
