from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from okk.transforms import build_image_transform


@dataclass
class ManifestItem:
    path: str
    label: int
    group: str = "unknown"
    pair_id: str = ""
    mask_path: str = ""
    operation: str = "unknown"
    generator: str = "unknown"
    split: str = "train"


REQUIRED_COLUMNS = ["path", "label"]
OPTIONAL_COLUMNS = ["group", "pair_id", "mask_path", "operation", "generator", "split"]


def read_manifest(path: str | Path, split: Optional[str] = None) -> List[ManifestItem]:
    path = Path(path)
    items: List[ManifestItem] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest is missing required columns: {missing}")
        for row in reader:
            row_split = row.get("split", "train") or "train"
            if split is not None and row_split != split:
                continue
            item = ManifestItem(
                path=row["path"],
                label=int(row["label"]),
                group=row.get("group", "unknown") or "unknown",
                pair_id=row.get("pair_id", "") or "",
                mask_path=row.get("mask_path", "") or "",
                operation=row.get("operation", "unknown") or "unknown",
                generator=row.get("generator", "unknown") or "unknown",
                split=row_split,
            )
            items.append(item)
    if not items:
        raise ValueError(f"manifest has no samples: {path}, split={split}")
    return items


class ImageManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: Optional[str] = None, image_size: int = 224, train: bool = False):
        self.items = read_manifest(manifest_path, split=split)
        self.transform = build_image_transform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        image = Image.open(item.path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(item.label, dtype=torch.long),
            "path": item.path,
            "group": item.group,
            "pair_id": item.pair_id,
            "mask_path": item.mask_path,
            "operation": item.operation,
            "generator": item.generator,
            "split": item.split,
        }


class PairedManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: Optional[str] = None, image_size: int = 224, train: bool = True):
        items = read_manifest(manifest_path, split=split)
        pairs: Dict[str, Dict[str, ManifestItem]] = {}
        for item in items:
            if not item.pair_id:
                continue
            pair = pairs.setdefault(item.pair_id, {})
            if item.label == 0:
                pair["real"] = item
            else:
                pair["fake"] = item
        self.pairs = [(pid, pair["real"], pair["fake"]) for pid, pair in pairs.items() if "real" in pair and "fake" in pair]
        if not self.pairs:
            raise ValueError("paired manifest has no complete real/fake pairs; check pair_id and label")
        self.transform = build_image_transform(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict:
        pair_id, real_item, fake_item = self.pairs[index]
        real = Image.open(real_item.path).convert("RGB")
        fake = Image.open(fake_item.path).convert("RGB")
        real = self.transform(real)
        fake = self.transform(fake)
        return {
            "real_image": real,
            "fake_image": fake,
            "pair_id": pair_id,
            "real_path": real_item.path,
            "fake_path": fake_item.path,
            "operation": fake_item.operation,
            "generator": fake_item.generator,
            "group": fake_item.group,
        }

