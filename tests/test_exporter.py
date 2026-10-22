"""Tests for ONNX export and quantization service."""

from pathlib import Path

from src.config.settings import TrainingSettings
from src.core.model import MultiOutputMobileNetV2, MultiOutputResNet18
from src.services.exporter import export_model_to_onnx, export_resnet18_to_onnx


def test_export_and_quantize_resnet18(tmp_path: Path):
    settings = TrainingSettings(
        model_name="resnet18",
        onnx_output_path=tmp_path / "test_model.onnx",
        onnx_int8_output_path=tmp_path / "test_model_int8.onnx",
    )

    model = MultiOutputResNet18(num_classes=256, pretrained=False)
    fp32_path, int8_path = export_resnet18_to_onnx(model, settings)

    assert fp32_path.exists()
    assert int8_path.exists()
    assert int8_path.stat().st_size < fp32_path.stat().st_size


def test_export_and_quantize_mobilenet(tmp_path: Path):
    settings = TrainingSettings(
        model_name="mobilenet_v2",
        onnx_output_path=tmp_path / "test_mb.onnx",
        onnx_int8_output_path=tmp_path / "test_mb_int8.onnx",
    )

    model = MultiOutputMobileNetV2(num_classes=256, pretrained=False)
    fp32_path, int8_path = export_model_to_onnx(model, settings)

    assert fp32_path.exists()
    assert int8_path.exists()
    assert int8_path.stat().st_size < fp32_path.stat().st_size
