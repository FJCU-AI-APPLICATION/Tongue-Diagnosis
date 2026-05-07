"""Tests for TongueDataset (extracted from Amanda's train_*.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai.training.dataset import TongueDataset


def _make_label_studio_record(image_filename: str, label: str) -> dict:
    return {
        "data": {"image": f"/data/upload/{image_filename}"},
        "annotations": [
            {"result": [{"value": {"choices": [label]}}]}
        ],
    }


def test_dataset_yields_image_and_label_idx(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img_dir / "a.png")
    data = [_make_label_studio_record("a.png", "淡紅")]
    labels_map = {"淡紅": 0}

    ds = TongueDataset(data, str(img_dir), labels_map, transform=None)
    image, label_idx = ds[0]
    assert label_idx == 0
    assert image.size == (16, 16)


def test_dataset_returns_blank_image_on_missing_file(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    data = [_make_label_studio_record("missing.png", "淡")]
    labels_map = {"淡": 0}
    ds = TongueDataset(data, str(img_dir), labels_map, transform=None)
    image, _ = ds[0]
    assert image.size == (224, 224)  # blank fallback per Amanda's behaviour


def test_dataset_len_matches_data_list_length(tmp_path: Path):
    data = [_make_label_studio_record("x.png", "a")] * 5
    ds = TongueDataset(data, str(tmp_path), {"a": 0}, transform=None)
    assert len(ds) == 5
