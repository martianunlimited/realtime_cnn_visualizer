"""Tests for curve plotting and history tracking."""

from pathlib import Path

from src.services.trainer import plot_training_curves


def test_plot_training_curves(tmp_path: Path):
    history = {
        "epoch": [1, 2, 3],
        "train_loss": [1.5, 0.9, 0.4],
        "val_loss": [1.3, 0.8, 0.5],
        "train_top1": [60.0, 75.0, 85.0],
        "val_top1": [62.0, 78.0, 86.5],
        "val_top5": [80.0, 90.0, 95.0],
    }
    plot_path = tmp_path / "test_curves.png"
    result = plot_training_curves(history, plot_path)

    assert result.exists()
    assert result.stat().st_size > 1000  # Ensure image was rendered
