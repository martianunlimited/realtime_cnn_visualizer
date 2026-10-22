"""Tests for configuration settings."""

from pathlib import Path

from src.config.settings import TrainingSettings


def test_default_settings():
    settings = TrainingSettings()
    assert settings.model_name == "resnet18"
    assert settings.num_classes == 256
    assert settings.batch_size == 64
    assert settings.epochs_warmup == 2
    assert settings.epochs_finetune == 10
    assert settings.lr_backbone == 2e-5
    assert settings.lr_head == 3e-4
    assert settings.onnx_output_path == Path("models/resnet18_caltech256.onnx")
    assert settings.onnx_int8_output_path == Path("models/resnet18_caltech256_int8.onnx")


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


def test_mobilenet_path_resolution():
    settings = TrainingSettings(model_name="mobilenet_v2")
    assert settings.checkpoint_name == "mobilenetv2_caltech256_best.pth"
    assert settings.tensorboard_dir == Path("runs/mobilenetv2_caltech256")
    assert settings.onnx_output_path == Path("models/mobilenetv2_caltech256.onnx")
    assert settings.onnx_int8_output_path == Path("models/mobilenetv2_caltech256_int8.onnx")


def test_resnet50_path_resolution():
    settings = TrainingSettings(model_name="resnet50")
    assert settings.checkpoint_name == "resnet50_caltech256_best.pth"
    assert settings.tensorboard_dir == Path("runs/resnet50_caltech256")
    assert settings.onnx_output_path == Path("models/resnet50_caltech256.onnx")
    assert settings.onnx_int8_output_path == Path("models/resnet50_caltech256_int8.onnx")
