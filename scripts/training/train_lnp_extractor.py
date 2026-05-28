from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from okk.config import ExperimentConfig, ensure_project_dirs
from okk.lnp_dataset import LNPDataset
from okk.lnp_model import LNPResNetExtractor
from okk.metrics import compute_binary_metrics
from okk.utils import configure_torch, get_device, save_json, set_seed


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    labels_all = []
    scores_all = []
    losses = []
    for batch in tqdm(loader, desc="train_lnp" if train else "eval_lnp"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(x)
            logits = out["logits"]
            loss = F.cross_entropy(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob_fake = torch.softmax(logits, dim=1)[:, 1]
        labels_all.append(y.detach().cpu().numpy())
        scores_all.append(prob_fake.detach().cpu().numpy())
        losses.append(float(loss.detach().cpu()))
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
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints/lnp_extractor_best.pth")
    args = parser.parse_args()

    cfg = ExperimentConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    train_set = LNPDataset(args.manifest, split=args.train_split, image_size=cfg.image_size, train=True)
    val_set = LNPDataset(args.manifest, split=args.val_split, image_size=cfg.image_size, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LNPResNetExtractor(arch=args.arch, feature_dim=args.feature_dim, num_classes=2, use_srm=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    best = -1.0
    out_path = cfg.project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(row)
        if val_metrics["auroc"] > best:
            best = val_metrics["auroc"]
            torch.save({
                "model": model.state_dict(),
                "arch": args.arch,
                "feature_dim": args.feature_dim,
                "best_epoch": epoch,
                "best_val": val_metrics,
            }, out_path)
            save_json({"history": history, "best": row}, out_path.with_suffix(".json"))
    print(f"best val AUROC: {best:.6f}, checkpoint: {out_path}")


if __name__ == "__main__":
    main()


