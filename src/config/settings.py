"""Training and Export Configuration for CNNs on Caltech-256."""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    """Configuration settings for fine-tuning CNN backbones on Caltech-256."""

    model_config = SettingsConfigDict(
        env_prefix="CALTECH_",
        env_file=".env",
        extra="ignore",
    )

    # Model architecture
    model_name: str = Field(
        default="resnet18",
        description="Backbone architecture ('resnet18', 'mobilenet_v2', 'resnet50')",
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

    # Training hyperparameters
    batch_size: int = Field(default=64, ge=1, le=512)
    num_workers: int = Field(default=4, ge=0)
    epochs_warmup: int = Field(
        default=2,
        ge=0,
        description="Epochs to train only the classification head",
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

    @model_validator(mode="after")
    def resolve_model_paths(self) -> "TrainingSettings":
        """Auto-configure default paths if model_name is set and paths were not explicitly overridden."""
        clean_name = self.model_name.replace("-", "").replace("_", "").lower()
        # Canonical names: resnet18, mobilenetv2, resnet50
        if "mobilenet" in clean_name:
            tag = "mobilenetv2"
        elif "resnet50" in clean_name:
            tag = "resnet50"
        else:
            tag = "resnet18"

        # Update defaults if they still reference resnet18 and model_name is different
        if tag != "resnet18":
            if self.checkpoint_name == "resnet18_caltech256_best.pth":
                self.checkpoint_name = f"{tag}_caltech256_best.pth"
            if self.tensorboard_dir == Path("runs/resnet18_caltech256"):
                self.tensorboard_dir = Path(f"runs/{tag}_caltech256")
            if self.plot_path == Path("runs/resnet18_caltech256/training_curves.png"):
                self.plot_path = Path(f"runs/{tag}_caltech256/training_curves.png")
            if self.history_path == Path("runs/resnet18_caltech256/training_history.json"):
                self.history_path = Path(f"runs/{tag}_caltech256/training_history.json")
            if self.onnx_output_path == Path("models/resnet18_caltech256.onnx"):
                self.onnx_output_path = Path(f"models/{tag}_caltech256.onnx")
            if self.onnx_int8_output_path == Path("models/resnet18_caltech256_int8.onnx"):
                self.onnx_int8_output_path = Path(f"models/{tag}_caltech256_int8.onnx")

        return self
