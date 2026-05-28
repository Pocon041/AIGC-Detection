from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from okk.config import ExperimentConfig, ensure_project_dirs
from okk.dataset import ImageManifestDataset
from okk.features_dino_response import DINOResponseExtractor, ResponseProbe
from okk.metrics import compute_binary_metrics, format_binary_metrics
from okk.utils import configure_torch, get_device, save_json, set_seed


def run_epoch(model, extractor, loader, optimizer, device, train: bool):
    model.train(train)
    labels_all = []
    scores_all = []
    losses = []
    for batch in tqdm(loader, desc="train" if train else "eval"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].float().to(device, non_blocking=True)
        with torch.no_grad():
            feat = extractor(x)["image_features"]
        with torch.set_grad_enabled(train):
            logits = model(feat)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels_all.append(y.detach().cpu().numpy())
        scores_all.append(torch.sigmoid(logits).detach().cpu().numpy())
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    metrics = compute_binary_metrics(labels, scores).to_dict()
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="timm/vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--out", type=str, default="checkpoints/probe_best.pth")
    args = parser.parse_args()

    cfg = ExperimentConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, backbone_name=args.backbone)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    train_set = ImageManifestDataset(args.manifest, split=args.train_split, image_size=cfg.image_size, train=True)
    val_set = ImageManifestDataset(args.manifest, split=args.val_split, image_size=cfg.image_size, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    extractor = DINOResponseExtractor(args.backbone, pretrained=True).to(device)
    model = ResponseProbe(extractor.response_dim, hidden_dim=cfg.probe_hidden_dim, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    best = -1.0
    out_path = cfg.project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, extractor, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, extractor, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(f"epoch {epoch}")
        print(format_binary_metrics("train", train_metrics))
        print(format_binary_metrics("val", val_metrics))
        print(f"loss train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f}")
        score = val_metrics["auroc"]
        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(),
                "backbone": args.backbone,
                "response_dim": extractor.response_dim,
                "cfg": cfg.__dict__,
                "best_epoch": epoch,
                "best_val": val_metrics,
            }, out_path)
            save_json({"history": history, "best": row}, out_path.with_suffix(".json"))
    print(f"best val AUROC: {best:.6f}, checkpoint: {out_path}")


if __name__ == "__main__":
    main()


