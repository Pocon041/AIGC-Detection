from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from okk.beyond_lowlevel import BeyondLowLevelExtractor
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.metrics import compute_binary_metrics, format_binary_metrics
from okk.transforms import IMAGENET_MEAN, IMAGENET_STD
from okk.utils import configure_torch, get_device, save_json, set_seed


class PrecomputedDiffusionPretextDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str | None, image_size: int):
        self.samples = []
        with Path(manifest).open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "original" not in reader.fieldnames:
                raise ValueError(f"precomputed manifest 缺少 original 列: {manifest}")
            self.variant_cols = [c for c in reader.fieldnames if c not in {"id", "split", "group", "original"}]
            for row in reader:
                row_split = row.get("split", "train") or "train"
                if split is not None and row_split != split:
                    continue
                entries = [(row["original"], 0)]
                for idx, col in enumerate(self.variant_cols, start=1):
                    if row.get(col, ""):
                        entries.append((row[col], idx))
                self.samples.append(entries)
        if not self.samples:
            raise ValueError(f"precomputed manifest 没有样本: {manifest}, split={split}")
        self.num_classes = 1 + len(self.variant_cols)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        images = []
        labels = []
        mae_targets = []
        original_path = self.samples[index][0][0]
        original = self.image_transform(Image.open(original_path).convert("RGB"))
        for path, label in self.samples[index]:
            image = self.image_transform(Image.open(path).convert("RGB"))
            images.append(self.normalize(image))
            labels.append(label)
            mae_targets.append(torch.mean(torch.abs(image - original)) * 255.0)
        return {
            "images": torch.stack(images, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mae_targets": torch.stack(mae_targets, dim=0).float(),
        }


def collate_precomputed(batch):
    images = torch.cat([b["images"] for b in batch], dim=0)
    labels = torch.cat([b["labels"] for b in batch], dim=0)
    mae_targets = torch.cat([b["mae_targets"] for b in batch], dim=0)
    return {"images": images, "labels": labels, "mae_targets": mae_targets}


def run_epoch(model, loader, optimizer, device, train: bool, lambda_reg: float):
    model.train(train)
    labels_all = []
    scores_all = []
    losses = []
    for batch in tqdm(loader, desc="train_precomputed" if train else "eval_precomputed"):
        x = batch["images"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        mae_targets = batch["mae_targets"].to(device, non_blocking=True)
        out = model(x)
        image_logits = out["image_logits"]
        regression = out["regression"]
        loss_cls = F.cross_entropy(image_logits, y)
        loss_reg = F.l1_loss(regression, mae_targets)
        loss = loss_cls + lambda_reg * loss_reg
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        suspicious = 1.0 - torch.softmax(image_logits, dim=1)[:, 0]
        binary = (y != 0).long()
        labels_all.append(binary.detach().cpu().numpy())
        scores_all.append(suspicious.detach().cpu().numpy())
        losses.append(float(loss.detach().cpu()))
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    metrics = compute_binary_metrics(labels, scores).to_dict()
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, default="")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--arch", type=str, default="vit_srm", choices=["vit_srm"])
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--conv-depth", type=int, default=3)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", type=str, default="checkpoints/beyond_lowlevel_precomputed_best.pth")
    args = parser.parse_args()

    cfg = ExperimentConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    train_set = PrecomputedDiffusionPretextDataset(args.manifest, args.train_split, cfg.image_size)
    val_manifest = args.val_manifest if args.val_manifest else args.manifest
    val_set = PrecomputedDiffusionPretextDataset(val_manifest, args.val_split, cfg.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_precomputed)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_precomputed)

    model = BeyondLowLevelExtractor(
        arch=args.arch,
        image_size=cfg.image_size,
        patch_size=args.patch_size,
        feature_dim=args.feature_dim,
        num_pretext_classes=train_set.num_classes,
        conv_channels=args.conv_channels,
        conv_depth=args.conv_depth,
        transformer_depth=args.transformer_depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_highpass=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    best = -1.0
    out_path = cfg.project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True, lambda_reg=args.lambda_reg)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False, lambda_reg=args.lambda_reg)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(f"epoch {epoch}")
        print(format_binary_metrics("train", train_metrics))
        print(format_binary_metrics("val", val_metrics))
        print(f"loss train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f}")
        if val_metrics["auroc"] > best:
            best = val_metrics["auroc"]
            torch.save({
                "model": model.state_dict(),
                "arch": args.arch,
                "feature_dim": args.feature_dim,
                "model_config": model.model_config(),
                "pretext_modes": ["original"] + train_set.variant_cols,
                "lambda_reg": args.lambda_reg,
                "best_epoch": epoch,
                "best_val": val_metrics,
            }, out_path)
            save_json({"history": history, "best": row}, out_path.with_suffix(".json"))
    print(f"最佳 val AUROC: {best:.6f}, checkpoint: {out_path}")


if __name__ == "__main__":
    main()


