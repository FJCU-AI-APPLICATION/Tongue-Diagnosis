"""Shared TongueDataset for the train_front and train_sublingual entry points."""
from __future__ import annotations

import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class TongueDataset(Dataset):
    """Label Studio JSON record → (PIL.Image, label_idx).

    Records expected to look like:
      {"data": {"image": ".../filename.png"},
       "annotations": [{"result": [{"value": {"choices": ["label"]}}]}]}
    """

    def __init__(
        self,
        data_list: list[dict[str, Any]],
        img_dir: str,
        labels_map: dict[str, int],
        transform=None,
    ):
        self.data_list = data_list
        self.img_dir = img_dir
        self.labels_map = labels_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        img_path_raw = item["data"]["image"]
        img_filename = os.path.basename(img_path_raw)
        img_path = os.path.join(self.img_dir, img_filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224))

        label_str = item["annotations"][0]["result"][0]["value"]["choices"][0]
        label_idx = self.labels_map[label_str]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx
