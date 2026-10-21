"""Caltech-256 Dataset loader and transform pipelines."""

import random
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.config.settings import TrainingSettings


def parse_caltech256_categories(data_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Parse category folders from Caltech-256 directory.

    Caltech-256 consists of folders '001.ak47' through '256.toad', plus '257.clutter'.
    Standard 256-class classification excludes '257.clutter', yielding classes 0 to 255.

    Returns:
        tuple of (class_names, folder_to_index)
    """
    if not data_dir.exists():
        return [], {}

    categories = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    class_names: list[str] = [""] * 256
    folder_to_index: dict[str, int] = {}

    for cat in categories:
        parts = cat.split(".", 1)
        if len(parts) != 2:
            continue
        try:
            folder_num = int(parts[0])
        except ValueError:
            continue

        # Exclude 257.clutter or any invalid folder numbers
        if folder_num < 1 or folder_num > 256:
            continue

        class_idx = folder_num - 1
        clean_name = parts[1].replace("-", " ").strip()
        class_names[class_idx] = clean_name
        folder_to_index[cat] = class_idx

    # Filter out empty slots if any folders were missing
    valid_classes = [name for name in class_names if name]
    return valid_classes, folder_to_index


class Caltech256Dataset(Dataset):
    """PyTorch Dataset for Caltech-256."""

    def __init__(
        self,
        data_dir: Path,
        transform: Callable | None = None,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        self.classes, self.folder_to_index = parse_caltech256_categories(self.data_dir)
        self.samples: list[tuple[Path, int]] = []

        if self.folder_to_index:
            for folder, class_idx in self.folder_to_index.items():
                folder_path = self.data_dir / folder
                for img_path in folder_path.glob("*.jpg"):
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            default_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            img_tensor = default_transform(img)

        return img_tensor, target


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return train and validation transform pipelines."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def build_dataloaders(
    settings: TrainingSettings,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train and validation DataLoaders with stratified-like reproducible split."""
    train_transform, val_transform = get_transforms()

    # Load base dataset without transform to inspect samples
    base_dataset = Caltech256Dataset(data_dir=settings.data_dir)
    num_samples = len(base_dataset)
    if num_samples == 0:
        raise RuntimeError(f"No samples found in dataset directory: {settings.data_dir}")

    # Reproducible train / validation index split
    indices = list(range(num_samples))
    rng = random.Random(settings.seed)
    rng.shuffle(indices)

    val_size = int(num_samples * settings.val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Create distinct instances with appropriate transforms
    full_train_dataset = Caltech256Dataset(
        data_dir=settings.data_dir, transform=train_transform, split="train"
    )
    full_val_dataset = Caltech256Dataset(
        data_dir=settings.data_dir, transform=val_transform, split="val"
    )

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    pin_memory = torch.cuda.is_available() and "cuda" in settings.device

    train_loader = DataLoader(
        train_subset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, base_dataset.classes
