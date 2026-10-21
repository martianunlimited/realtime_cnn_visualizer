"""Training and Export Configuration for ResNet18 on Caltech-256."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    """Configuration settings for fine-tuning ResNet18 on Caltech-256."""

    model_config = SettingsConfigDict(
        env_prefix="CALTECH_",
        env_file=".env",
        extra="ignore",
    )

    # Data paths & parameters
    data_dir: Path = Field(
        default=Path("data/caltech256/256_ObjectCategories"),
        description="Path to extracted 256_ObjectCategories directory",
    )
    labels_csv_path: Path = Field(
        default=Path("caltech.csv"),
        description="Path to caltech.csv label mapping",
    )
    num_classes: int = Field(default=256, ge=2, le=1000)
    val_split: float = Field(default=0.15, gt=0.0, lt=0.5)
    seed: int = Field(default=42)

    # Training hyperparameters (Low learning rate schedule for gentle convergence)
    batch_size: int = Field(default=64, ge=1, le=512)
    num_workers: int = Field(default=4, ge=0)
    epochs_warmup: int = Field(
        default=2,
        ge=0,
        description="Epochs to train only the linear classification head",
    )
    epochs_finetune: int = Field(
        default=10, ge=1, description="Epochs to fine-tune backbone and head"
    )
    lr_head: float = Field(default=3e-4, gt=0.0, description="Warmup/head learning rate")
    lr_backbone: float = Field(
        default=2e-5, gt=0.0, description="Backbone fine-tuning learning rate"
    )
    weight_decay: float = Field(default=1e-4, ge=0.0)
    label_smoothing: float = Field(
        default=0.1, ge=0.0, le=0.3, description="Label smoothing for cross-entropy"
    )

    # Device & checkpointing
    device: str = Field(default="cuda", description="Computation device ('cuda' or 'cpu')")
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    checkpoint_name: str = Field(default="resnet18_caltech256_best.pth")
    tensorboard_dir: Path = Field(default=Path("runs/resnet18_caltech256"))
    plot_path: Path = Field(default=Path("runs/resnet18_caltech256/training_curves.png"))
    history_path: Path = Field(default=Path("runs/resnet18_caltech256/training_history.json"))

    # Export & Quantization
    onnx_output_path: Path = Field(default=Path("models/resnet18_caltech256.onnx"))
    onnx_int8_output_path: Path = Field(default=Path("models/resnet18_caltech256_int8.onnx"))
    opset_version: int = Field(default=17)
