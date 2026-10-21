"""Trainer service for fine-tuning ResNet18 on Caltech-256."""

import time
from pathlib import Path

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

        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.val_loader, self.classes = build_dataloaders(
            self.settings
        )

        self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.settings.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.settings.tensorboard_dir))

        self.best_val_acc: float = 0.0

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
                    f"| ({elapsed:.1f}s)"
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
        print(f" Saved checkpoint with Top-1 Val Acc {val_acc:.2f}% to {path}")

    def train(self) -> Path:
        """Execute complete warmup and fine-tuning workflow."""
        print("\n==========================================")
        print(" Starting ResNet18 Caltech-256 Training")
        print(f" Device: {self.device}")
        print(
            f" Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )
        print(f" Classes: {len(self.classes)}")
        print(f" Warmup Epochs: {self.settings.epochs_warmup}")
        print(f" Fine-tune Epochs: {self.settings.epochs_finetune}")
        print("==========================================\n")

        global_epoch = 0

        # Phase 1: Warmup - Freeze backbone, train only classification head
        if self.settings.epochs_warmup > 0:
            print("--- Phase 1: Classification Head Warmup ---")
            self.model.freeze_backbone()
            optimizer_warmup = AdamW(
                self.model.fc.parameters(),
                lr=self.settings.lr_head,
                weight_decay=self.settings.weight_decay,
            )

            for ep in range(1, self.settings.epochs_warmup + 1):
                global_epoch += 1
                train_loss, train_top1, _ = self._train_one_epoch(
                    ep, optimizer_warmup, "Warmup"
                )
                val_loss, val_top1, val_top5 = self.evaluate()

                print(
                    f"--> [Warmup Ep {ep}/{self.settings.epochs_warmup}] "
                    f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%\n"
                )

                self.writer.add_scalar("Loss/train", train_loss, global_epoch)
                self.writer.add_scalar("Loss/val", val_loss, global_epoch)
                self.writer.add_scalar("Accuracy/train_top1", train_top1, global_epoch)
                self.writer.add_scalar("Accuracy/val_top1", val_top1, global_epoch)
                self.writer.add_scalar("Accuracy/val_top5", val_top5, global_epoch)

                if val_top1 > self.best_val_acc:
                    self.best_val_acc = val_top1
                    best_path = (
                        self.settings.checkpoint_dir / self.settings.checkpoint_name
                    )
                    self.save_checkpoint(best_path, val_top1, global_epoch)

        # Phase 2: Full Fine-Tuning - Unfreeze backbone with differential learning rate
        print("\n--- Phase 2: End-to-End Fine-Tuning ---")
        self.model.unfreeze_backbone()
        param_groups = self.model.get_parameter_groups(
            lr_backbone=self.settings.lr_backbone,
            lr_head=self.settings.lr_head * 0.5,
            weight_decay=self.settings.weight_decay,
        )
        optimizer_ft = AdamW(param_groups)
        scheduler = CosineAnnealingLR(
            optimizer_ft, T_max=self.settings.epochs_finetune, eta_min=1e-6
        )

        best_checkpoint_path = (
            self.settings.checkpoint_dir / self.settings.checkpoint_name
        )

        for ep in range(1, self.settings.epochs_finetune + 1):
            global_epoch += 1
            train_loss, train_top1, _ = self._train_one_epoch(
                ep, optimizer_ft, "FineTune"
            )
            val_loss, val_top1, val_top5 = self.evaluate()
            scheduler.step()

            print(
                f"--> [FineTune Ep {ep}/{self.settings.epochs_finetune}] "
                f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%\n"
            )

            self.writer.add_scalar("Loss/train", train_loss, global_epoch)
            self.writer.add_scalar("Loss/val", val_loss, global_epoch)
            self.writer.add_scalar("Accuracy/train_top1", train_top1, global_epoch)
            self.writer.add_scalar("Accuracy/val_top1", val_top1, global_epoch)
            self.writer.add_scalar("Accuracy/val_top5", val_top5, global_epoch)

            if val_top1 > self.best_val_acc:
                self.best_val_acc = val_top1
                self.save_checkpoint(best_checkpoint_path, val_top1, global_epoch)

        self.writer.close()
        print(
            f"\n Training finished! Best Val Top-1 Accuracy: {self.best_val_acc:.2f}%"
        )
        return best_checkpoint_path
