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

from okk.config import ExperimentConfig, ensure_project_dirs
from okk.dataset import ImageManifestDataset
from okk.features_dino_response import DINOResponseExtractor, ResponseProbe
from okk.metrics import compute_binary_metrics, format_binary_metrics, format_grouped_metrics, grouped_metrics, one_vs_real_grouped_metrics
from okk.utils import configure_torch, get_device, save_json, set_seed


@torch.no_grad()
def evaluate(model, extractor, loader, device):
    model.eval()
    labels_all = []
    scores_all = []
    groups = []
    generators = []
    operations = []
    paths = []
    for batch in tqdm(loader, desc="evaluate"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].numpy()
        feat = extractor(x)["image_features"]
        logits = model(feat)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels_all.append(y)
        scores_all.append(scores)
        groups.extend(batch["group"])
        generators.extend(batch["generator"])
        operations.extend(batch["operation"])
        paths.extend(batch["path"])
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    return labels, scores, groups, generators, operations, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=str, default="outputs/probe_eval.json")
    args = parser.parse_args()

    cfg = ExperimentConfig(batch_size=args.batch_size)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = ckpt.get("backbone", cfg.backbone_name)
    dataset = ImageManifestDataset(args.manifest, split=args.split, image_size=cfg.image_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    extractor = DINOResponseExtractor(backbone, pretrained=True).to(device)
    model = ResponseProbe(ckpt["response_dim"], hidden_dim=cfg.probe_hidden_dim, dropout=cfg.dropout).to(device)
    model.load_state_dict(ckpt["model"])

    labels, scores, groups, generators, operations, paths = evaluate(model, extractor, loader, device)
    result = {
        "score_direction": "higher suspicious score means more likely fake; label=1 means fake",
        "overall": compute_binary_metrics(labels, scores).to_dict(),
        "by_group": grouped_metrics(labels, scores, groups),
        "by_generator_vs_real": one_vs_real_grouped_metrics(labels, scores, generators),
        "by_operation_vs_real": one_vs_real_grouped_metrics(labels, scores, operations),
    }
    out = cfg.project_root / args.out
    save_json(result, out)
    np.savez(out.with_suffix(".npz"), labels=labels, scores=scores, paths=np.asarray(paths))
    print(format_binary_metrics("overall", result["overall"]))
    print(format_grouped_metrics("by_generator_vs_real", result["by_generator_vs_real"]))
    print(format_grouped_metrics("by_operation_vs_real", result["by_operation_vs_real"]))
    print(f"saved probe evaluation: {out}")


if __name__ == "__main__":
    main()


