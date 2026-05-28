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
from torch.utils.data import DataLoader
from tqdm import tqdm

from okk.condition_encoder import HybridConditionEncoder, LowFrequencyConditionEncoder
from okk.conditional_detector import ConditionalGaussianDetector, ConditionalResidualDetector, PatchScoreAggregator
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.dataset import PairedManifestDataset
from okk.losses import pair_ranking_loss, patch_bce_loss
from okk.lowlevel_encoder import BeyondPretextLowLevelEncoder, OnlineLNPLowLevelEncoder
from okk.metrics import compute_binary_metrics, format_binary_metrics
from okk.utils import configure_torch, get_device, save_json, set_seed


def build_condition_encoder(kind: str, cfg: ExperimentConfig, out_dim: int):
    if kind == "lowfreq":
        return LowFrequencyConditionEncoder(out_dim=out_dim), out_dim
    if kind == "hybrid":
        enc = HybridConditionEncoder(cfg.backbone_name, pretrained=True, dino_dim=out_dim, low_dim=out_dim // 2)
        return enc, enc.out_dim
    raise ValueError(f"?? condition encoder: {kind}")


def build_residual_encoder(kind: str, cfg: ExperimentConfig, out_dim: int, lnp_checkpoint: str = "", beyond_checkpoint: str = ""):
    if kind == "lnp":
        if not lnp_checkpoint:
            raise ValueError("--residual lnp ???? --lnp-checkpoint")
        return OnlineLNPLowLevelEncoder(lnp_checkpoint, feature_dim=out_dim, out_dim=out_dim), out_dim
    if kind == "beyond":
        if not beyond_checkpoint:
            raise ValueError("--residual beyond ???? --beyond-checkpoint")
        return BeyondPretextLowLevelEncoder(beyond_checkpoint, feature_dim=out_dim, out_dim=out_dim), out_dim
    raise ValueError(f"?? residual encoder: {kind}")


def build_detector(kind: str, c_dim: int, r_dim: int, cfg: ExperimentConfig):
    if kind == "mlp":
        return ConditionalResidualDetector(c_dim, r_dim, hidden_dim=cfg.detector_hidden_dim, depth=cfg.detector_depth, dropout=cfg.dropout)
    if kind == "gaussian":
        return ConditionalGaussianDetector(c_dim, r_dim, hidden_dim=cfg.detector_hidden_dim, depth=cfg.detector_depth, dropout=cfg.dropout)
    raise ValueError(f"unknown detector: {kind}")


def forward_scores(c_encoder, r_encoder, detector, aggregator, image):
    residual = r_encoder(image)
    condition = c_encoder(image, target_tokens=residual.shape[1])
    patch_scores = detector(condition, residual)
    image_scores = aggregator(patch_scores)
    return patch_scores, image_scores


def run_epoch(c_encoder, r_encoder, detector, aggregator, loader, optimizer, device, cfg, train: bool, lambda_nll: float, lambda_rank: float, lambda_patch: float):
    c_encoder.train(train)
    r_encoder.train(train)
    detector.train(train)
    labels_all = []
    scores_all = []
    losses = []
    rank_losses = []
    patch_losses = []
    nll_losses = []
    desc = "train_cond" if train else "eval_cond"
    for batch in tqdm(loader, desc=desc):
        real = batch["real_image"].to(device, non_blocking=True)
        fake = batch["fake_image"].to(device, non_blocking=True)
        real_residual = r_encoder(real)
        fake_residual = r_encoder(fake)
        real_condition = c_encoder(real, target_tokens=real_residual.shape[1])
        fake_condition = c_encoder(fake, target_tokens=fake_residual.shape[1])
        real_patch = detector(real_condition, real_residual)
        fake_self_patch = detector(fake_condition, fake_residual)
        fake_anchor_patch = detector(real_condition, fake_residual)
        real_img = aggregator(real_patch)
        fake_img = aggregator(fake_self_patch)
        loss_rank = (
            pair_ranking_loss(real_patch, fake_anchor_patch, margin=cfg.margin)
            + pair_ranking_loss(real_patch, fake_self_patch, margin=cfg.margin)
        )
        loss_patch = patch_bce_loss(real_patch, fake_self_patch) + patch_bce_loss(real_patch, fake_anchor_patch)
        if hasattr(detector, "nll"):
            loss_nll = -real_patch.mean()
        else:
            loss_nll = torch.zeros((), device=device)
        loss = lambda_rank * loss_rank + lambda_patch * loss_patch + lambda_nll * loss_nll
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        rank_losses.append(float(loss_rank.detach().cpu()))
        patch_losses.append(float(loss_patch.detach().cpu()))
        nll_losses.append(float(loss_nll.detach().cpu()))
        real_suspicious = (-real_img).detach().cpu().numpy()
        fake_suspicious = (-fake_img).detach().cpu().numpy()
        labels_all.append(np.concatenate([np.zeros_like(real_suspicious), np.ones_like(fake_suspicious)]))
        scores_all.append(np.concatenate([real_suspicious, fake_suspicious]))
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    metrics = compute_binary_metrics(labels, scores).to_dict()
    metrics["loss"] = float(np.mean(losses))
    metrics["loss_rank"] = float(np.mean(rank_losses))
    metrics["loss_patch"] = float(np.mean(patch_losses))
    metrics["loss_nll"] = float(np.mean(nll_losses))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--condition", type=str, default="hybrid", choices=["lowfreq", "hybrid"])
    parser.add_argument("--residual", type=str, default="beyond", choices=["lnp", "beyond"])
    parser.add_argument("--lnp-checkpoint", type=str, default="")
    parser.add_argument("--beyond-checkpoint", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="timm/vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--residual-dim", type=int, default=256)
    parser.add_argument("--detector", type=str, default="gaussian", choices=["gaussian", "mlp"])
    parser.add_argument("--lambda-nll", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-patch", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="checkpoints/conditional_best.pth")
    args = parser.parse_args()

    cfg = ExperimentConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, backbone_name=args.backbone)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    train_set = PairedManifestDataset(args.manifest, split=args.train_split, image_size=cfg.image_size, train=True)
    val_set = PairedManifestDataset(args.manifest, split=args.val_split, image_size=cfg.image_size, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    c_encoder, c_dim = build_condition_encoder(args.condition, cfg, args.condition_dim)
    r_encoder, r_dim = build_residual_encoder(args.residual, cfg, args.residual_dim, args.lnp_checkpoint, args.beyond_checkpoint)
    detector = build_detector(args.detector, c_dim, r_dim, cfg)
    aggregator = PatchScoreAggregator(cfg.lower_tail_ratio, cfg.agg_alpha, cfg.agg_beta)
    c_encoder = c_encoder.to(device)
    r_encoder = r_encoder.to(device)
    detector = detector.to(device)
    aggregator = aggregator.to(device)

    params = list(c_encoder.parameters()) + list(r_encoder.parameters()) + list(detector.parameters())
    params = [p for p in params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=cfg.weight_decay)

    best = -1.0
    out_path = cfg.project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(c_encoder, r_encoder, detector, aggregator, train_loader, optimizer, device, cfg, train=True, lambda_nll=args.lambda_nll, lambda_rank=args.lambda_rank, lambda_patch=args.lambda_patch)
        val_metrics = run_epoch(c_encoder, r_encoder, detector, aggregator, val_loader, optimizer, device, cfg, train=False, lambda_nll=args.lambda_nll, lambda_rank=args.lambda_rank, lambda_patch=args.lambda_patch)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(f"epoch {epoch}")
        print(format_binary_metrics("train", train_metrics))
        print(format_binary_metrics("val", val_metrics))
        print(
            f"loss train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} | "
            f"rank={val_metrics['loss_rank']:.4f} patch={val_metrics['loss_patch']:.4f} nll={val_metrics['loss_nll']:.4f}"
        )
        score = val_metrics["auroc"]
        if score > best:
            best = score
            torch.save({
                "condition_encoder": c_encoder.state_dict(),
                "residual_encoder": r_encoder.state_dict(),
                "detector": detector.state_dict(),
                "detector_kind": args.detector,
                "lambda_nll": args.lambda_nll,
                "lambda_rank": args.lambda_rank,
                "lambda_patch": args.lambda_patch,
                "condition_kind": args.condition,
                "residual_kind": args.residual,
                "lnp_checkpoint": args.lnp_checkpoint,
                "beyond_checkpoint": args.beyond_checkpoint,
                "condition_dim_arg": args.condition_dim,
                "residual_dim_arg": args.residual_dim,
                "condition_dim": c_dim,
                "residual_dim": r_dim,
                "backbone": args.backbone,
                "cfg": cfg.__dict__,
                "best_epoch": epoch,
                "best_val": val_metrics,
            }, out_path)
            save_json({"history": history, "best": row}, out_path.with_suffix(".json"))
    print(f"?? val AUROC: {best:.6f}, checkpoint: {out_path}")


if __name__ == "__main__":
    main()


