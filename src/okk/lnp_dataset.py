from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class LNPItem:
    path: str
    label: int
    group: str = "unknown"
    split: str = "train"


def read_lnp_manifest(path: str | Path, split: Optional[str] = None) -> List[LNPItem]:
    items: List[LNPItem] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = row.get("split", "train") or "train"
            if split is not None and row_split != split:
                continue
            items.append(LNPItem(
                path=row["path"],
                label=int(row["label"]),
                group=row.get("group", "unknown") or "unknown",
                split=row_split,
            ))
    if not items:
        raise ValueError(f"LNP manifest 娌℃湁鏍锋湰: {path}, split={split}")
    return items


class LNPDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: Optional[str] = None, image_size: int = 224, train: bool = False):
        self.items = read_lnp_manifest(manifest_path, split=split)
        ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
        if train:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(item.path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(item.label, dtype=torch.long),
            "path": item.path,
            "group": item.group,
            "split": item.split,
        }

