"""Tests for configuration settings."""

from pathlib import Path

from src.config.settings import TrainingSettings


def test_default_settings():
    settings = TrainingSettings()
    assert settings.num_classes == 256
    assert settings.batch_size == 64
    assert settings.epochs_warmup == 2
    assert settings.epochs_finetune == 8
    assert settings.onnx_output_path == Path("models/resnet18_caltech256.onnx")
    assert settings.onnx_int8_output_path == Path(
        "models/resnet18_caltech256_int8.onnx"
    )


def test_custom_settings():
    settings = TrainingSettings(
        batch_size=32,
        num_classes=256,
        epochs_finetune=5,
        device="cpu",
    )
    assert settings.batch_size == 32
    assert settings.epochs_finetune == 5
    assert settings.device == "cpu"
