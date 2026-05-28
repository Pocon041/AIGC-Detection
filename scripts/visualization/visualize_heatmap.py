from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from okk.condition_encoder import HybridConditionEncoder, LowFrequencyConditionEncoder
from okk.conditional_detector import ConditionalGaussianDetector, ConditionalResidualDetector
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.lowlevel_encoder import BeyondPretextLowLevelEncoder, OnlineLNPLowLevelEncoder
from okk.transforms import build_image_transform
from okk.utils import configure_torch, get_device, set_seed


def build_condition_encoder(kind: str, cfg: ExperimentConfig, out_dim: int):
    if kind == "lowfreq":
        return LowFrequencyConditionEncoder(out_dim=out_dim), out_dim
    if kind == "hybrid":
        enc = HybridConditionEncoder(cfg.backbone_name, pretrained=True, dino_dim=out_dim, low_dim=out_dim // 2)
        return enc, enc.out_dim
    raise ValueError(f"unknown condition encoder: {kind}")


def build_residual_encoder(kind: str, cfg: ExperimentConfig, out_dim: int, lnp_checkpoint: str = "", beyond_checkpoint: str = ""):
    if kind == "lnp":
        if not lnp_checkpoint:
            raise ValueError("missing lnp_checkpoint")
        return OnlineLNPLowLevelEncoder(lnp_checkpoint, feature_dim=out_dim, out_dim=out_dim), out_dim
    if kind == "beyond":
        if not beyond_checkpoint:
            raise ValueError("missing beyond_checkpoint")
        return BeyondPretextLowLevelEncoder(beyond_checkpoint, feature_dim=out_dim, out_dim=out_dim), out_dim
    raise ValueError(f"unknown residual encoder: {kind}")


def build_detector(kind: str, c_dim: int, r_dim: int, cfg: ExperimentConfig):
    if kind == "mlp":
        return ConditionalResidualDetector(c_dim, r_dim, hidden_dim=cfg.detector_hidden_dim, depth=cfg.detector_depth, dropout=cfg.dropout)
    if kind == "gaussian":
        return ConditionalGaussianDetector(c_dim, r_dim, hidden_dim=cfg.detector_hidden_dim, depth=cfg.detector_depth, dropout=cfg.dropout)
    raise ValueError(f"unknown detector: {kind}")


@torch.no_grad()
def compute_heatmap(image_path: str, ckpt_path: str, out_path: str):
    cfg = ExperimentConfig(batch_size=1)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg.backbone_name = ckpt.get("backbone", cfg.backbone_name)
    c_encoder, _ = build_condition_encoder(ckpt["condition_kind"], cfg, ckpt["condition_dim_arg"])
    r_encoder, _ = build_residual_encoder(
        ckpt["residual_kind"],
        cfg,
        ckpt["residual_dim_arg"],
        ckpt.get("lnp_checkpoint", ""),
        ckpt.get("beyond_checkpoint", ""),
    )
    detector = build_detector(ckpt.get("detector_kind", "mlp"), ckpt["condition_dim"], ckpt["residual_dim"], cfg)
    c_encoder.load_state_dict(ckpt["condition_encoder"])
    r_encoder.load_state_dict(ckpt["residual_encoder"])
    detector.load_state_dict(ckpt["detector"])
    c_encoder = c_encoder.to(device).eval()
    r_encoder = r_encoder.to(device).eval()
    detector = detector.to(device).eval()

    pil = Image.open(image_path).convert("RGB")
    transform = build_image_transform(cfg.image_size, train=False)
    x = transform(pil).unsqueeze(0).to(device)
    residual = r_encoder(x)
    condition = c_encoder(x, target_tokens=residual.shape[1])
    patch_realness = detector(condition, residual)
    suspicious = -patch_realness
    n = suspicious.shape[1]
    side = int(round(n ** 0.5))
    heat = suspicious.reshape(1, 1, side, side)
    heat = F.interpolate(heat, size=pil.size[::-1], mode="bilinear", align_corners=False)[0, 0]
    heat = heat.detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(pil)
    plt.imshow(heat, cmap="magma", alpha=0.45)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"saved heatmap: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    compute_heatmap(args.image, args.checkpoint, args.out)


if __name__ == "__main__":
    main()


