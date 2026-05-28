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
from torch.utils.data import DataLoader
from tqdm import tqdm

from okk.condition_encoder import HybridConditionEncoder, LowFrequencyConditionEncoder
from okk.conditional_detector import ConditionalGaussianDetector, ConditionalResidualDetector, PatchScoreAggregator
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.dataset import ImageManifestDataset
from okk.lowlevel_encoder import BeyondPretextLowLevelEncoder, OnlineLNPLowLevelEncoder
from okk.metrics import compute_binary_metrics, format_binary_metrics, format_grouped_metrics, grouped_metrics, one_vs_real_grouped_metrics
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
            raise ValueError("checkpoint ??? lnp_checkpoint")
        return OnlineLNPLowLevelEncoder(lnp_checkpoint, feature_dim=out_dim, out_dim=out_dim), out_dim
    if kind == "beyond":
        if not beyond_checkpoint:
            raise ValueError("checkpoint ??? beyond_checkpoint")
        return BeyondPretextLowLevelEncoder(beyond_checkpoint, feature_dim=out_dim, out_dim=out_dim), out_dim
    raise ValueError(f"?? residual encoder: {kind}")


def build_detector(kind: str, c_dim: int, r_dim: int, cfg: ExperimentConfig):
    if kind == "mlp":
        return ConditionalResidualDetector(c_dim, r_dim, hidden_dim=cfg.detector_hidden_dim, depth=cfg.detector_depth, dropout=cfg.dropout)
    if kind == "gaussian":
        return ConditionalGaussianDetector(c_dim, r_dim, hidden_dim=cfg.detector_hidden_dim, depth=cfg.detector_depth, dropout=cfg.dropout)
    raise ValueError(f"unknown detector: {kind}")


@torch.no_grad()
def evaluate(c_encoder, r_encoder, detector, aggregator, loader, device):
    c_encoder.eval()
    r_encoder.eval()
    detector.eval()
    labels_all = []
    scores_all = []
    realness_all = []
    groups = []
    generators = []
    operations = []
    paths = []
    for batch in tqdm(loader, desc="eval_conditional"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].numpy()
        residual = r_encoder(x)
        condition = c_encoder(x, target_tokens=residual.shape[1])
        patch_scores = detector(condition, residual)
        realness = aggregator(patch_scores)
        suspicious = -realness
        labels_all.append(y)
        scores_all.append(suspicious.detach().cpu().numpy())
        realness_all.append(realness.detach().cpu().numpy())
        groups.extend(batch["group"])
        generators.extend(batch["generator"])
        operations.extend(batch["operation"])
        paths.extend(batch["path"])
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    realness = np.concatenate(realness_all)
    return labels, scores, realness, groups, generators, operations, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=str, default="outputs/conditional_eval.json")
    args = parser.parse_args()

    cfg = ExperimentConfig(batch_size=args.batch_size)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg.backbone_name = ckpt.get("backbone", cfg.backbone_name)
    dataset = ImageManifestDataset(args.manifest, split=args.split, image_size=cfg.image_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    c_encoder, c_dim = build_condition_encoder(ckpt["condition_kind"], cfg, ckpt["condition_dim_arg"])
    r_encoder, r_dim = build_residual_encoder(
        ckpt["residual_kind"],
        cfg,
        ckpt["residual_dim_arg"],
        ckpt.get("lnp_checkpoint", ""),
        ckpt.get("beyond_checkpoint", ""),
    )
    detector = build_detector(ckpt.get("detector_kind", "mlp"), ckpt["condition_dim"], ckpt["residual_dim"], cfg)
    aggregator = PatchScoreAggregator(cfg.lower_tail_ratio, cfg.agg_alpha, cfg.agg_beta)
    c_encoder.load_state_dict(ckpt["condition_encoder"])
    r_encoder.load_state_dict(ckpt["residual_encoder"])
    detector.load_state_dict(ckpt["detector"])
    c_encoder = c_encoder.to(device)
    r_encoder = r_encoder.to(device)
    detector = detector.to(device)
    aggregator = aggregator.to(device)

    labels, scores, realness, groups, generators, operations, paths = evaluate(c_encoder, r_encoder, detector, aggregator, loader, device)
    result = {
        "score_direction": "suspicious score = -realness_score???????label=1 ?? fake",
        "detector_kind": ckpt.get("detector_kind", "mlp"),
        "overall": compute_binary_metrics(labels, scores).to_dict(),
        "by_group": grouped_metrics(labels, scores, groups),
        "by_generator_vs_real": one_vs_real_grouped_metrics(labels, scores, generators),
        "by_operation_vs_real": one_vs_real_grouped_metrics(labels, scores, operations),
    }
    out = cfg.project_root / args.out
    save_json(result, out)
    np.savez(out.with_suffix(".npz"), labels=labels, suspicious=scores, realness=realness, paths=np.asarray(paths))
    print(format_binary_metrics("overall", result["overall"]))
    print(format_grouped_metrics("by_generator_vs_real", result["by_generator_vs_real"]))
    print(format_grouped_metrics("by_operation_vs_real", result["by_operation_vs_real"]))
    print(f"淇濆瓨璇勪及缁撴灉: {out}")


if __name__ == "__main__":
    main()


