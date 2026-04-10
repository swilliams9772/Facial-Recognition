"""
Dataset and data-loading utilities for face recognition training.

Expects a directory laid out as:

    data/
      person_1/
        img_001.jpg
        img_002.jpg
      person_2/
        img_001.jpg
        ...

The TripletFaceDataset yields (anchor, positive, negative) tuples
where anchor and positive belong to the same identity and negative
belongs to a different one.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class TripletFaceDataset(Dataset):
    """Generates (anchor, positive, negative) triplets on-the-fly.

    Parameters
    ----------
    root_dir : str | Path
        Root directory containing one sub-folder per identity.
    transform : torchvision.transforms.Compose, optional
        Image transforms applied to every sample.
    """

    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_train_transforms()

        # Build a mapping: label -> [image_path, ...]
        self.label_to_paths: dict[str, list[Path]] = defaultdict(list)
        for person_dir in sorted(self.root_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            for img_file in sorted(person_dir.iterdir()):
                if img_file.suffix.lower() in self.SUPPORTED_EXT:
                    self.label_to_paths[person_dir.name].append(img_file)

        # Only keep identities with at least 2 images (need a positive pair)
        self.label_to_paths = {
            k: v for k, v in self.label_to_paths.items() if len(v) >= 2
        }
        self.labels = sorted(self.label_to_paths.keys())

        if len(self.labels) < 2:
            raise ValueError(
                f"Need at least 2 identities with >= 2 images each. "
                f"Found {len(self.labels)} in {self.root_dir}"
            )

        # Flat list for __len__ — one entry per image
        self.samples: list[tuple[Path, str]] = []
        for label, paths in self.label_to_paths.items():
            for p in paths:
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, anchor_label = self.samples[idx]

        # Positive: different image, same identity
        pos_candidates = [
            p for p in self.label_to_paths[anchor_label] if p != anchor_path
        ]
        positive_path = random.choice(pos_candidates)

        # Negative: different identity
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        negative_path = random.choice(self.label_to_paths[neg_label])

        return self._load(anchor_path), self._load(positive_path), self._load(negative_path)
