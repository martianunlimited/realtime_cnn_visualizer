"""Tests for Caltech-256 Dataset Loading and Transforms."""

from pathlib import Path

import pytest

from src.core.dataset import Caltech256Dataset, parse_caltech256_categories


def test_parse_categories_mock(tmp_path: Path):
    # Setup mock categories
    (tmp_path / "001.ak47").mkdir()
    (tmp_path / "002.american-flag").mkdir()
    (tmp_path / "256.toad").mkdir()
    (tmp_path / "257.clutter").mkdir()  # Should be excluded

    classes, class_to_idx = parse_caltech256_categories(tmp_path)
    assert len(classes) == 3
    assert "ak47" in classes
    assert "american flag" in classes
    assert "toad" in classes
    assert "clutter" not in classes
    assert class_to_idx["001.ak47"] == 0
    assert class_to_idx["002.american-flag"] == 1
    assert class_to_idx["256.toad"] == 255


def test_real_dataset_integrity():
    data_dir = Path("data/caltech256/256_ObjectCategories")
    if not data_dir.exists():
        pytest.skip("Caltech-256 dataset not found on disk")

    classes, class_to_idx = parse_caltech256_categories(data_dir)
    assert len(classes) == 256
    assert class_to_idx["001.ak47"] == 0
    assert class_to_idx["256.toad"] == 255

    dataset = Caltech256Dataset(data_dir=data_dir, split="val")
    assert len(dataset) > 0
    img, target = dataset[0]
    assert 0 <= target < 256
    assert img.shape == (3, 224, 224)
