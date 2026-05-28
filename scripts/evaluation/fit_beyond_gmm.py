from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse

import joblib
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

from okk.beyond_lowlevel import BeyondLowLevelAdapter
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.dataset import ImageManifestDataset
from okk.metrics import compute_binary_metrics, format_binary_metrics, format_grouped_metrics, one_vs_real_grouped_metrics, operating_metrics
from okk.utils import configure_torch, get_device, save_json, set_seed


@torch.no_grad()
def extract_features(adapter, loader, device, real_only: bool):
    adapter.eval()
    feats = []
    labels = []
    groups = []
    generators = []
    operations = []
    paths = []
    for batch in tqdm(loader, desc="extract_beyond_features"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].cpu().numpy()
        keep = np.ones_like(y, dtype=bool)
        if real_only:
            keep = y == 0
            if not keep.any():
                continue
            x = x[torch.from_numpy(keep).to(device)]
        tokens = adapter(x)
        pooled = tokens.mean(dim=1)
        feats.append(pooled.cpu().numpy())
        labels.extend(y[keep].tolist())
        groups.extend([g for g, k in zip(batch["group"], keep) if k])
        generators.extend([g for g, k in zip(batch["generator"], keep) if k])
        operations.extend([o for o, k in zip(batch["operation"], keep) if k])
        paths.extend([p for p, k in zip(batch["path"], keep) if k])
    if not feats:
        raise ValueError("no features were extracted; check manifest, split, and real_only filtering")
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.asarray(labels, dtype=np.int64),
        "groups": groups,
        "generators": generators,
        "operations": operations,
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calib-manifest", type=str, required=True)
    parser.add_argument("--eval-manifest", type=str, default="")
    parser.add_argument("--calib-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=5.0,
        help="Lower-tail percentile on real calibration likelihoods. 5.0 targets about 5%% false alarms on calibration.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--out", type=str, default="outputs/beyond_gmm_eval.json")
    parser.add_argument("--gmm-out", type=str, default="checkpoints/beyond_gmm.joblib")
    args = parser.parse_args()
    if not 0.0 < args.threshold_percentile < 100.0:
        raise ValueError("--threshold-percentile must be in the open interval (0, 100)")

    cfg = ExperimentConfig(batch_size=args.batch_size)
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    configure_torch(cfg.use_tf32)
    device = get_device(cfg.device)

    adapter = BeyondLowLevelAdapter(args.checkpoint, out_dim=args.out_dim).to(device)
    calib_set = ImageManifestDataset(args.calib_manifest, split=args.calib_split, image_size=cfg.image_size, train=False)
    calib_loader = DataLoader(calib_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    calib = extract_features(adapter, calib_loader, device, real_only=True)

    gmm = GaussianMixture(n_components=args.components, covariance_type="full", random_state=cfg.seed)
    gmm.fit(calib["features"])
    calib_likelihood = gmm.score_samples(calib["features"])
    threshold = float(np.percentile(calib_likelihood, args.threshold_percentile))

    gmm_path = cfg.project_root / args.gmm_out
    gmm_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "gmm": gmm,
        "threshold": threshold,
        "checkpoint": args.checkpoint,
        "components": args.components,
        "threshold_percentile": args.threshold_percentile,
        "threshold_false_alarm_target": args.threshold_percentile / 100.0,
    }, gmm_path)

    result = {
        "gmm_path": str(gmm_path),
        "threshold": threshold,
        "calib_count": int(calib["features"].shape[0]),
        "calib_likelihood_mean": float(calib_likelihood.mean()),
        "calib_likelihood_std": float(calib_likelihood.std()),
        "threshold_percentile": float(args.threshold_percentile),
        "threshold_false_alarm_target": float(args.threshold_percentile / 100.0),
    }

    if args.eval_manifest:
        eval_set = ImageManifestDataset(args.eval_manifest, split=args.eval_split, image_size=cfg.image_size, train=False)
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        eval_data = extract_features(adapter, eval_loader, device, real_only=False)
        likelihood = gmm.score_samples(eval_data["features"])
        suspicious = -likelihood
        metrics = compute_binary_metrics(eval_data["labels"], suspicious).to_dict()
        grouped = one_vs_real_grouped_metrics(eval_data["labels"], suspicious, eval_data["generators"])
        predicted_fake = likelihood < threshold
        calibrated_ops = operating_metrics(eval_data["labels"], suspicious, -threshold)
        metrics["threshold_acc"] = float((predicted_fake.astype(np.int64) == eval_data["labels"]).mean())
        metrics["false_alarm"] = float(predicted_fake[eval_data["labels"] == 0].mean()) if np.any(eval_data["labels"] == 0) else None
        metrics["recall"] = float(predicted_fake[eval_data["labels"] == 1].mean()) if np.any(eval_data["labels"] == 1) else None
        result.update({
            "eval_count": int(eval_data["features"].shape[0]),
            "metrics": metrics,
            "calibrated_operating_point": calibrated_ops,
            "grouped_by_generator_vs_real": grouped,
        })

    out = cfg.project_root / args.out
    save_json(result, out)
    if "metrics" in result:
        print(format_binary_metrics("overall", result["metrics"]))
        print(format_grouped_metrics("grouped_by_generator_vs_real", result["grouped_by_generator_vs_real"]))
        print(f"calibrated_threshold_suspicious={-threshold:.6f} recall={result['calibrated_operating_point']['recall'] * 100:.2f}% false_alarm={result['calibrated_operating_point']['false_alarm'] * 100:.2f}%")
    else:
        print(result)


if __name__ == "__main__":
    main()


