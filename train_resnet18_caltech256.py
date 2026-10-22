"""CLI Delivery Layer for Caltech-256 Training, Evaluation, and ONNX Export.

Supports ResNet18, MobileNetV2, and ResNet50.
"""

from pathlib import Path
from typing import Annotated

import torch
import typer

from src.config.settings import TrainingSettings
from src.core.model import create_multi_output_model
from src.services.exporter import export_model_to_onnx
from src.services.trainer import CNNTrainer

app = typer.Typer(
    name="cnn-caltech256",
    help="Train, fine-tune, evaluate, and export CNN models on the Caltech-256 dataset.",
    add_completion=False,
)


@app.command()
def train(
    model_name: Annotated[
        str,
        typer.Option(help="Backbone architecture ('resnet18', 'mobilenet_v2', 'resnet50')"),
    ] = "resnet18",
    batch_size: Annotated[int, typer.Option(help="Batch size for training and validation")] = 64,
    epochs_warmup: Annotated[int, typer.Option(help="Warmup epochs (head only)")] = 2,
    epochs_finetune: Annotated[int, typer.Option(help="End-to-end fine-tuning epochs")] = 8,
    lr_head: Annotated[float, typer.Option(help="Learning rate for classification head")] = 3e-4,
    lr_backbone: Annotated[float, typer.Option(help="Learning rate for backbone")] = 2e-5,
    weight_decay: Annotated[float, typer.Option(help="Weight decay for AdamW")] = 1e-4,
    label_smoothing: Annotated[
        float, typer.Option(help="Label smoothing factor for CrossEntropyLoss")
    ] = 0.1,
    device: Annotated[str, typer.Option(help="Computation device ('cuda' or 'cpu')")] = "cuda",
    export_onnx: Annotated[
        bool, typer.Option(help="Automatically export best model to ONNX & INT8")
    ] = True,
) -> None:
    """Train and fine-tune a CNN architecture on Caltech-256 dataset."""
    settings = TrainingSettings(
        model_name=model_name,
        batch_size=batch_size,
        epochs_warmup=epochs_warmup,
        epochs_finetune=epochs_finetune,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        device=device,
    )

    typer.echo(f"Initializing {model_name} Trainer...")
    trainer = CNNTrainer(settings)
    best_checkpoint_path = trainer.train()

    if export_onnx:
        typer.echo("\n--- Exporting Best Model to ONNX (Multi-Tap + INT8) ---")
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
        export_model = create_multi_output_model(
            model_name=settings.model_name,
            num_classes=settings.num_classes,
            pretrained=False,
        )
        export_model.load_state_dict(checkpoint["model_state_dict"])

        fp32_path, int8_path = export_model_to_onnx(export_model, settings)
        typer.echo("\n Finished! Ready for Web Visualizer:")
        typer.echo(f"   - FP32 ONNX: {fp32_path}")
        typer.echo(f"   - INT8 ONNX: {int8_path}")


@app.command()
def export(
    checkpoint_path: Annotated[Path, typer.Argument(help="Path to .pth checkpoint file")],
    model_name: Annotated[
        str,
        typer.Option(help="Backbone architecture ('resnet18', 'mobilenet_v2', 'resnet50')"),
    ] = "resnet18",
    num_classes: Annotated[int, typer.Option(help="Number of classes")] = 256,
) -> None:
    """Export an existing checkpoint to multi-output ONNX and INT8."""
    if not checkpoint_path.exists():
        typer.secho(f"Error: Checkpoint file not found: {checkpoint_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    settings = TrainingSettings(
        model_name=model_name,
        num_classes=num_classes,
    )

    typer.echo(f"Loading {model_name} checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = create_multi_output_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    fp32, int8 = export_model_to_onnx(model, settings)
    typer.echo(f"Export completed: {fp32} and {int8}")


@app.command()
def evaluate(
    checkpoint_path: Annotated[Path, typer.Argument(help="Path to .pth checkpoint file")],
    model_name: Annotated[str, typer.Option(help="Backbone architecture")] = "resnet18",
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    device: Annotated[str, typer.Option(help="Device")] = "cuda",
) -> None:
    """Evaluate accuracy of a saved checkpoint on the validation set."""
    if not checkpoint_path.exists():
        typer.secho(f"Error: Checkpoint file not found: {checkpoint_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    settings = TrainingSettings(model_name=model_name, batch_size=batch_size, device=device)
    trainer = CNNTrainer(settings)

    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, top1, top5 = trainer.evaluate()
    typer.echo(f"Validation Loss: {val_loss:.4f} | Top-1 Acc: {top1:.2f}% | Top-5 Acc: {top5:.2f}%")


if __name__ == "__main__":
    app()
