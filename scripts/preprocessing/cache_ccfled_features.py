from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from okk.backbones import TimmPatchBackbone
from okk.beyond_lowlevel import BeyondLowLevelAdapter
from okk.ccfled import compute_proxy_features, pool_tokens, proxy_names, string_array
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.dataset import ImageManifestDataset
from okk.utils import configure_torch, get_device, set_seed


@torch.no_grad()
def extract_cache(args, cfg: ExperimentConfig):
    split = args.split if args.split else None
    dataset = ImageManifestDataset(args.manifest, split=split, image_size=args.image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = get_device(args.device)
    lowlevel = BeyondLowLevelAdapter(args.lowlevel_checkpoint, out_dim=args.lowlevel_dim).to(device)
    lowlevel.eval()

    semantic = None
    if args.semantic_backbone.lower() != "none":
        semantic = TimmPatchBackbone(args.semantic_backbone, pretrained=True).to(device)
        semantic.eval()

    z_f_all = []
    z_s_all = []
    z_c_all = []
    labels = []
    paths = []
    groups = []
    generators = []
    operations = []
    splits = []

    for batch in tqdm(loader, desc="cache_ccfled_features"):
        x = batch["image"].to(device, non_blocking=True)
        batch_paths = [str(path) for path in batch["path"]]
        tokens = lowlevel(x)
        z_f = pool_tokens(tokens, mode=args.pooling, topk_ratio=args.topk_ratio)
        if semantic is None:
            z_s = torch.zeros((x.shape[0], 1), device=device, dtype=z_f.dtype)
        else:
            z_s = semantic(x).cls
        z_c = compute_proxy_features(x, batch_paths)

        z_f_all.append(z_f.detach().cpu().numpy().astype(np.float32))
        z_s_all.append(z_s.detach().cpu().numpy().astype(np.float32))
        z_c_all.append(z_c.detach().cpu().numpy().astype(np.float32))
        labels.append(batch["label"].detach().cpu().numpy().astype(np.int64))
        paths.extend(batch_paths)
        groups.extend(batch["group"])
        generators.extend(batch["generator"])
        operations.extend(batch["operation"])
        splits.extend(batch["split"])

    label_array = np.concatenate(labels, axis=0)
    return {
        "z_f": np.concatenate(z_f_all, axis=0),
        "z_s": np.concatenate(z_s_all, axis=0),
        "z_c_proxy": np.concatenate(z_c_all, axis=0),
        "labels": label_array,
        "sample_index": np.arange(label_array.shape[0], dtype=np.int64),
        "paths": string_array(paths),
        "groups": string_array(groups),
        "generators": string_array(generators),
        "operations": string_array(operations),
        "splits": string_array(splits),
        "proxy_names": string_array(proxy_names(include_pipeline=True)),
        "proxy_frame": f"preprocessed_tensor_{args.image_size}_imagenet_denormalized",
        "lowlevel_projection": "none",
        "config_json": json.dumps(
            {
                "manifest": args.manifest,
                "split": args.split,
                "image_size": args.image_size,
                "lowlevel_checkpoint": args.lowlevel_checkpoint,
                "lowlevel_dim": args.lowlevel_dim,
                "lowlevel_projection": "none",
                "semantic_backbone": args.semantic_backbone,
                "pooling": args.pooling,
                "topk_ratio": args.topk_ratio,
                "proxy_frame": f"preprocessed_tensor_{args.image_size}_imagenet_denormalized",
            },
            ensure_ascii=False,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Cache z_f, z_s, and audited proxy features for CC-FLED Phase 1.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split", type=str, default="", help="Empty means cache all splits in the manifest.")
    parser.add_argument("--lowlevel-checkpoint", type=str, required=True)
    parser.add_argument("--lowlevel-dim", type=int, default=256)
    parser.add_argument("--semantic-backbone", type=str, default="timm/vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "topk_l2"])
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="cache/ccfled_features.npz")
    args = parser.parse_args()

    cfg = ExperimentConfig(batch_size=args.batch_size, image_size=args.image_size, num_workers=args.num_workers)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)

    cache = extract_cache(args, cfg)
    out = cfg.project_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cache)
    print(f"saved CC-FLED feature cache: {out}")
    print(f"samples={cache['labels'].shape[0]} z_f={cache['z_f'].shape} z_s={cache['z_s'].shape} z_c={cache['z_c_proxy'].shape}")


if __name__ == "__main__":
    main()
