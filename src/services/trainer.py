"""Trainer service for fine-tuning ResNet18 on Caltech-256 with curve tracking."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.config.settings import TrainingSettings
from src.core.dataset import build_dataloaders
from src.core.model import MultiOutputResNet18


def compute_topk_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> list[float]:
    """Compute top-k accuracy percentages."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: list[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.item() * 100.0) / batch_size)
        return res


def plot_training_curves(history: dict[str, list], output_path: Path) -> Path:
    """Generate and save publication-grade train/validation loss and accuracy curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", [])
    if not epochs:
        return output_path

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    for ax in (ax_loss, ax_acc):
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.35, color="#64748b")
        ax.tick_params(colors="#cbd5e1", which="both", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Loss curves
    ax_loss.plot(
        epochs,
        history["train_loss"],
        marker="o",
        color="#38bdf8",
        linewidth=2,
        label="Train Loss",
    )
    ax_loss.plot(
        epochs,
        history["val_loss"],
        marker="s",
        color="#f43f5e",
        linewidth=2,
        label="Validation Loss",
    )
    ax_loss.set_title(
        "Cross-Entropy Loss (Lower is Better)",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_loss.set_xlabel("Epoch", color="#94a3b8", fontsize=11)
    ax_loss.set_ylabel("Loss", color="#94a3b8", fontsize=11)
    ax_loss.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    # Panel 2: Accuracy curves
    ax_acc.plot(
        epochs,
        history["train_top1"],
        marker="o",
        color="#38bdf8",
        linewidth=2,
        label="Train Top-1 Acc (%)",
    )
    ax_acc.plot(
        epochs,
        history["val_top1"],
        marker="s",
        color="#10b981",
        linewidth=2.5,
        label="Val Top-1 Acc (%)",
    )
    ax_acc.plot(
        epochs,
        history["val_top5"],
        marker="^",
        color="#a855f7",
        linestyle="--",
        linewidth=1.8,
        label="Val Top-5 Acc (%)",
    )

    # Highlight best validation point
    if history["val_top1"]:
        best_val = max(history["val_top1"])
        best_idx = history["val_top1"].index(best_val)
        best_ep = epochs[best_idx]
        ax_acc.annotate(
            f"Peak: {best_val:.2f}%",
            xy=(best_ep, best_val),
            xytext=(best_ep, best_val - 4),
            arrowprops={
                "facecolor": "#f59e0b",
                "edgecolor": "#f59e0b",
                "shrink": 0.08,
                "width": 1.5,
                "headwidth": 6,
            },
            color="#fef08a",
            fontweight="bold",
            fontsize=10,
            ha="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#0f172a",
                "edgecolor": "#f59e0b",
                "alpha": 0.9,
            },
        )

    ax_acc.set_title(
        "Top-1 & Top-5 Accuracy (Higher is Better)",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_acc.set_xlabel("Epoch", color="#94a3b8", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=11)
    ax_acc.legend(
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#f8fafc",
        loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


class ResNet18Trainer:
    """Manages the 2-phase warmup and fine-tuning lifecycle for ResNet18."""

    def __init__(self, settings: TrainingSettings | None = None):
        self.settings = settings or TrainingSettings()
        self.device = torch.device(
            self.settings.device if torch.cuda.is_available() else "cpu"
        )

        self.model = MultiOutputResNet18(
            num_classes=self.settings.num_classes,
            pretrained=True,
        ).to(self.device)

        # Label smoothing helps avoid overconfident probability distributions
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.settings.label_smoothing
        )
        self.train_loader, self.val_loader, self.classes = build_dataloaders(
            self.settings
        )

        self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.settings.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.settings.tensorboard_dir))

        self.best_val_acc: float = 0.0

        # Full metric history
        self.history: dict[str, list] = {
            "epoch": [],
            "phase": [],
            "train_loss": [],
            "val_loss": [],
            "train_top1": [],
            "val_top1": [],
            "val_top5": [],
            "lr_backbone": [],
            "lr_head": [],
        }

    def _train_one_epoch(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        phase_name: str,
    ) -> tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        running_top1 = 0.0
        running_top5 = 0.0
        total_batches = len(self.train_loader)

        start_time = time.time()
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images, return_features=False)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            top1, top5 = compute_topk_accuracy(outputs, targets, topk=(1, 5))
            running_loss += loss.item()
            running_top1 += top1
            running_top5 += top5

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                print(
                    f"[{phase_name} Ep {epoch}] Step {batch_idx + 1}/{total_batches} "
                    f"| Loss: {running_loss / (batch_idx + 1):.4f} "
                    f"| Top1: {running_top1 / (batch_idx + 1):.2f}% "
                    f"| ({elapsed:.1f}s)",
                    flush=True,
                )

        avg_loss = running_loss / total_batches
        avg_top1 = running_top1 / total_batches
        avg_top5 = running_top5 / total_batches
        return avg_loss, avg_top1, avg_top5

    def evaluate(self) -> tuple[float, float, float]:
        """Evaluate current model weights on the validation dataset."""
        self.model.eval()
        running_loss = 0.0
        running_top1 = 0.0
        running_top5 = 0.0
        total_batches = len(self.val_loader)

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(images, return_features=False)
                loss = self.criterion(outputs, targets)

                top1, top5 = compute_topk_accuracy(outputs, targets, topk=(1, 5))
                running_loss += loss.item()
                running_top1 += top1
                running_top5 += top5

        avg_loss = running_loss / max(1, total_batches)
        avg_top1 = running_top1 / max(1, total_batches)
        avg_top5 = running_top5 / max(1, total_batches)
        return avg_loss, avg_top1, avg_top5

    def save_checkpoint(self, path: Path, val_acc: float, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "val_acc": val_acc,
            "classes": self.classes,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, path)
        print(
            f" Saved checkpoint with Top-1 Val Acc {val_acc:.2f}% to {path}", flush=True
        )

    def _record_epoch_metrics(
        self,
        epoch: int,
        phase: str,
        train_loss: float,
        train_top1: float,
        val_loss: float,
        val_top1: float,
        val_top5: float,
        lr_head: float,
        lr_backbone: float,
    ) -> None:
        """Record epoch metrics into history, JSON, and plot curves."""
        self.history["epoch"].append(epoch)
        self.history["phase"].append(phase)
        self.history["train_loss"].append(round(train_loss, 4))
        self.history["val_loss"].append(round(val_loss, 4))
        self.history["train_top1"].append(round(train_top1, 2))
        self.history["val_top1"].append(round(val_top1, 2))
        self.history["val_top5"].append(round(val_top5, 2))
        self.history["lr_head"].append(lr_head)
        self.history["lr_backbone"].append(lr_backbone)

        # Write to JSON
        self.settings.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Update curves plot image
        plot_training_curves(self.history, self.settings.plot_path)

        # TensorBoard
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Accuracy/train_top1", train_top1, epoch)
        self.writer.add_scalar("Accuracy/val_top1", val_top1, epoch)
        self.writer.add_scalar("Accuracy/val_top5", val_top5, epoch)
        self.writer.add_scalar("LR/head", lr_head, epoch)
        self.writer.add_scalar("LR/backbone", lr_backbone, epoch)

    def train(self) -> Path:
        """Execute complete warmup and fine-tuning workflow with curve tracking."""
        print("\n==========================================", flush=True)
        print(" Starting ResNet18 Caltech-256 Training", flush=True)
        print(f" Device: {self.device}", flush=True)
        print(
            f" Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}",
            flush=True,
        )
        print(f" Classes: {len(self.classes)}", flush=True)
        print(f" Warmup Epochs: {self.settings.epochs_warmup}", flush=True)
        print(f" Fine-tune Epochs: {self.settings.epochs_finetune}", flush=True)
        print(
            f" Head LR: {self.settings.lr_head} | Backbone LR: {self.settings.lr_backbone}",
            flush=True,
        )
        print(f" Label Smoothing: {self.settings.label_smoothing}", flush=True)
        print("==========================================\n", flush=True)

        global_epoch = 0
        best_checkpoint_path = (
            self.settings.checkpoint_dir / self.settings.checkpoint_name
        )

        # Phase 1: Warmup - Freeze backbone, train only classification head
        if self.settings.epochs_warmup > 0:
            print("--- Phase 1: Classification Head Warmup ---", flush=True)
            self.model.freeze_backbone()
            optimizer_warmup = AdamW(
                self.model.fc.parameters(),
                lr=self.settings.lr_head,
                weight_decay=self.settings.weight_decay,
            )

            for ep in range(1, self.settings.epochs_warmup + 1):
                global_epoch += 1
                train_loss, train_top1, _ = self._train_one_epoch(
                    global_epoch, optimizer_warmup, "Warmup"
                )
                val_loss, val_top1, val_top5 = self.evaluate()

                print(
                    f"--> [Warmup Ep {ep}/{self.settings.epochs_warmup}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%\n",
                    flush=True,
                )

                self._record_epoch_metrics(
                    epoch=global_epoch,
                    phase="Warmup",
                    train_loss=train_loss,
                    train_top1=train_top1,
                    val_loss=val_loss,
                    val_top1=val_top1,
                    val_top5=val_top5,
                    lr_head=self.settings.lr_head,
                    lr_backbone=0.0,
                )

                if val_top1 > self.best_val_acc:
                    self.best_val_acc = val_top1
                    self.save_checkpoint(best_checkpoint_path, val_top1, global_epoch)

        # Phase 2: Full Fine-Tuning - Unfreeze backbone with lower differential learning rates
        print(
            "\n--- Phase 2: End-to-End Fine-Tuning (Low Learning Rates) ---", flush=True
        )
        self.model.unfreeze_backbone()
        param_groups = self.model.get_parameter_groups(
            lr_backbone=self.settings.lr_backbone,
            lr_head=self.settings.lr_head * 0.5,
            weight_decay=self.settings.weight_decay,
        )
        optimizer_ft = AdamW(param_groups)
        scheduler = CosineAnnealingLR(
            optimizer_ft, T_max=self.settings.epochs_finetune, eta_min=1e-7
        )

        for ep in range(1, self.settings.epochs_finetune + 1):
            global_epoch += 1
            train_loss, train_top1, _ = self._train_one_epoch(
                global_epoch, optimizer_ft, "FineTune"
            )
            val_loss, val_top1, val_top5 = self.evaluate()

            current_lr_backbone = optimizer_ft.param_groups[0]["lr"]
            current_lr_head = optimizer_ft.param_groups[1]["lr"]
            scheduler.step()

            print(
                f"--> [FineTune Ep {ep}/{self.settings.epochs_finetune}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | "
                f"LR: bb={current_lr_backbone:.2e}, head={current_lr_head:.2e}\n",
                flush=True,
            )

            self._record_epoch_metrics(
                epoch=global_epoch,
                phase="FineTune",
                train_loss=train_loss,
                train_top1=train_top1,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                lr_head=current_lr_head,
                lr_backbone=current_lr_backbone,
            )

            if val_top1 > self.best_val_acc:
                self.best_val_acc = val_top1
                self.save_checkpoint(best_checkpoint_path, val_top1, global_epoch)

        self.writer.close()
        print(
            f"\n Training finished! Best Val Top-1 Accuracy: {self.best_val_acc:.2f}%",
            flush=True,
        )
        print(f" Training curves saved to: {self.settings.plot_path}", flush=True)
        print(f" Metric history saved to: {self.settings.history_path}", flush=True)
        return best_checkpoint_path
