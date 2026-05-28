from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from okk.ccfled import (
    DEFAULT_CONDITION_PROXY_NAMES,
    build_weighted_condition_matrix,
    condition_blocks_from_cache,
    diagonal_global_energy,
    diagonal_local_energy,
    parse_name_list,
    remove_self_neighbors,
    standardize_from_bank,
)
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.metrics import compute_binary_metrics, format_binary_metrics, format_grouped_metrics, one_vs_real_grouped_metrics, operating_metrics
from okk.utils import save_json


def split_mask(splits: np.ndarray, value: str) -> np.ndarray:
    if not value or value == "all":
        return np.ones(len(splits), dtype=bool)
    allowed = {item.strip() for item in value.split(",") if item.strip()}
    return np.asarray([split in allowed for split in splits], dtype=bool)


def scalar_string(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return str(value.item())
    return str(value)


def calibrate_high_suspicion(scores: np.ndarray, false_alarm_percent: float) -> float:
    return float(np.percentile(scores, 100.0 - false_alarm_percent))


def fit_global_gmm(bank_z: np.ndarray, components: int, covariance_type: str, seed: int):
    components = min(int(components), int(bank_z.shape[0]))
    if components <= 0:
        raise ValueError("GMM components must be positive")
    model = GaussianMixture(n_components=components, covariance_type=covariance_type, random_state=seed)
    model.fit(bank_z)
    return model


def compute_local_scores(
    bank_z: np.ndarray,
    eval_z: np.ndarray,
    bank_condition: np.ndarray,
    eval_condition: np.ndarray,
    bank_positions: np.ndarray,
    eval_positions: np.ndarray,
    k: int,
    shrinkage: float,
    allow_self_neighbor: bool,
) -> tuple[np.ndarray, dict, np.ndarray]:
    has_overlap = bool(np.isin(eval_positions, bank_positions).any())
    exclude_self = has_overlap and not allow_self_neighbor
    available_neighbors = int(bank_condition.shape[0]) - (1 if exclude_self else 0)
    if available_neighbors <= 0:
        raise ValueError("memory bank has no usable non-self neighbors")
    k_eff = min(int(k), available_neighbors)
    query_k = min(k_eff + (1 if exclude_self else 0), int(bank_condition.shape[0]))

    knn = NearestNeighbors(n_neighbors=query_k, metric="euclidean")
    knn.fit(bank_condition)
    distances, indices = knn.kneighbors(eval_condition, return_distance=True)
    if exclude_self:
        indices, distances = remove_self_neighbors(
            indices=indices,
            distances=distances,
            bank_positions=bank_positions,
            eval_positions=eval_positions,
            max_k=k_eff,
        )
    else:
        indices = indices[:, :k_eff]
        distances = distances[:, :k_eff]

    scores = diagonal_local_energy(
        bank_z=bank_z,
        eval_z=eval_z,
        neighbor_idx=indices,
        shrinkage=shrinkage,
    )
    diagnostics = {
        "k": int(k_eff),
        "bank_eval_overlap": bool(has_overlap),
        "excluded_self_neighbors": bool(exclude_self),
        "mean_neighbor_distance": float(distances.mean()),
        "mean_kth_distance": float(distances[:, -1].mean()),
    }
    return scores, diagnostics, indices


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate cache-based feature scorers.")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument(
        "--scorer",
        type=str,
        default="global_diag",
        choices=["global_diag", "global_gmm", "local_semantic", "local_proxy", "local_semantic_proxy"],
    )
    parser.add_argument("--bank-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--allow-fake-bank", action="store_true")
    parser.add_argument("--proxy-columns", type=str, default="default")
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--proxy-weight", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--allow-self-neighbor", action="store_true")
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--covariance-type", type=str, default="full", choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--threshold-percentile", type=float, default=5.0)
    parser.add_argument("--out", type=str, default="outputs/feature_scorer_eval.json")
    parser.add_argument("--model-out", type=str, default="checkpoints/feature_scorer.joblib")
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.semantic_weight <= 0.0 or args.proxy_weight <= 0.0:
        raise ValueError("--semantic-weight and --proxy-weight must be positive")
    if not 0.0 < args.threshold_percentile < 100.0:
        raise ValueError("--threshold-percentile must be in the open interval (0, 100)")

    cfg = ExperimentConfig()
    ensure_project_dirs(cfg)
    data = np.load(args.cache, allow_pickle=True)
    labels = data["labels"].astype(np.int64)
    splits = data["splits"].astype(str)
    groups = data["groups"].astype(str)
    generators = data["generators"].astype(str)
    proxy_names = [str(x) for x in data["proxy_names"].tolist()]
    proxy_columns = parse_name_list(args.proxy_columns, proxy_names, DEFAULT_CONDITION_PROXY_NAMES)

    bank_mask = split_mask(splits, args.bank_split)
    if not args.allow_fake_bank:
        bank_mask &= labels == 0
    eval_mask = split_mask(splits, args.eval_split)
    if not bank_mask.any():
        raise ValueError(f"empty memory bank: split={args.bank_split}, allow_fake_bank={args.allow_fake_bank}")
    if not eval_mask.any():
        raise ValueError(f"empty eval split: {args.eval_split}")

    z_f = data["z_f"].astype(np.float64)
    z_f_std, zf_mean, zf_std = standardize_from_bank(z_f[bank_mask], z_f)
    bank_z = z_f_std[bank_mask]
    eval_z = z_f_std[eval_mask]
    bank_positions = np.flatnonzero(bank_mask)
    eval_positions = np.flatnonzero(eval_mask)

    scorer_model = {
        "type": args.scorer,
        "cache": str(args.cache),
        "bank_split": args.bank_split,
        "bank_real_only": not args.allow_fake_bank,
        "zf_mean": zf_mean,
        "zf_std": zf_std,
        "proxy_frame": scalar_string(data, "proxy_frame", "legacy_unknown"),
        "transform_protocol": scalar_string(data, "transform_protocol", "legacy_unknown"),
    }

    if args.scorer == "global_diag":
        bank_scores = diagonal_global_energy(bank_z, bank_z, shrinkage=args.shrinkage)
        eval_scores = diagonal_global_energy(bank_z, eval_z, shrinkage=args.shrinkage)
        scorer_model.update({
            "shrinkage": float(args.shrinkage),
            "bank_z": bank_z,
        })
        diagnostics = {}
    elif args.scorer == "global_gmm":
        gmm = fit_global_gmm(bank_z, args.components, args.covariance_type, cfg.seed)
        bank_scores = -gmm.score_samples(bank_z)
        eval_scores = -gmm.score_samples(eval_z)
        scorer_model.update({
            "gmm": gmm,
            "components": int(gmm.n_components),
            "covariance_type": args.covariance_type,
        })
        diagnostics = {}
    else:
        condition = args.scorer.removeprefix("local_")
        if bank_z.shape[0] < 2:
            raise ValueError("local scorers need at least two real bank samples for self-excluded calibration")
        fit_k = min(int(args.k), int(bank_z.shape[0] - 1))
        blocks, condition_label = condition_blocks_from_cache(data, condition, proxy_columns)
        condition_std, block_info = build_weighted_condition_matrix(
            blocks=blocks,
            bank_mask=bank_mask,
            semantic_weight=args.semantic_weight,
            proxy_weight=args.proxy_weight,
        )
        bank_condition = condition_std[bank_mask]
        eval_condition = condition_std[eval_mask]
        bank_scores, bank_diag, _ = compute_local_scores(
            bank_z=bank_z,
            eval_z=bank_z,
            bank_condition=bank_condition,
            eval_condition=bank_condition,
            bank_positions=bank_positions,
            eval_positions=bank_positions,
            k=fit_k,
            shrinkage=args.shrinkage,
            allow_self_neighbor=False,
        )
        eval_scores, eval_diag, _ = compute_local_scores(
            bank_z=bank_z,
            eval_z=eval_z,
            bank_condition=bank_condition,
            eval_condition=eval_condition,
            bank_positions=bank_positions,
            eval_positions=eval_positions,
            k=fit_k,
            shrinkage=args.shrinkage,
            allow_self_neighbor=args.allow_self_neighbor,
        )
        diagnostics = {
            "calibration_neighbors": bank_diag,
            "eval_neighbors": eval_diag,
        }
        scorer_model.update({
            "condition": condition_label,
            "condition_blocks": block_info,
            "condition_bank": bank_condition,
            "bank_z": bank_z,
            "k": int(eval_diag["k"]),
            "shrinkage": float(args.shrinkage),
            "proxy_columns": proxy_columns,
            "semantic_weight": float(args.semantic_weight),
            "proxy_weight": float(args.proxy_weight),
        })

    threshold = calibrate_high_suspicion(bank_scores, args.threshold_percentile)
    eval_labels = labels[eval_mask]
    metrics = compute_binary_metrics(eval_labels, eval_scores).to_dict()
    calibrated_ops = operating_metrics(eval_labels, eval_scores, threshold)
    metrics["threshold_acc"] = float(((eval_scores >= threshold).astype(np.int64) == eval_labels).mean())
    metrics["threshold_false_alarm_target"] = float(args.threshold_percentile / 100.0)
    grouped = one_vs_real_grouped_metrics(eval_labels, eval_scores, generators[eval_mask])

    model_out = cfg.project_root / args.model_out
    model_out.parent.mkdir(parents=True, exist_ok=True)
    scorer_model.update({
        "threshold": threshold,
        "threshold_percentile": float(args.threshold_percentile),
        "score_direction": "higher suspicious score means more likely fake; label=1 means fake",
    })
    joblib.dump(scorer_model, model_out)

    result = {
        "cache": str(args.cache),
        "model_path": str(model_out),
        "scorer": args.scorer,
        "bank_split": args.bank_split,
        "eval_split": args.eval_split,
        "bank_real_only": not args.allow_fake_bank,
        "bank_count": int(bank_mask.sum()),
        "eval_count": int(eval_mask.sum()),
        "proxy_frame": scalar_string(data, "proxy_frame", "legacy_unknown"),
        "transform_protocol": scalar_string(data, "transform_protocol", "legacy_unknown"),
        "threshold": threshold,
        "threshold_percentile": float(args.threshold_percentile),
        "score_direction": "higher suspicious score means more likely fake; label=1 means fake",
        "diagnostics": diagnostics,
        "overall": metrics,
        "calibrated_operating_point": calibrated_ops,
        "by_generator_vs_real": grouped,
    }
    out = cfg.project_root / args.out
    save_json(result, out)
    print(format_binary_metrics(args.scorer, metrics))
    print(format_grouped_metrics("by_generator_vs_real", grouped))
    print(f"saved feature scorer model: {model_out}")
    print(f"saved feature scorer evaluation: {out}")


if __name__ == "__main__":
    main()
