from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
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
from okk.metrics import compute_binary_metrics, format_binary_metrics, format_grouped_metrics, grouped_metrics, one_vs_real_grouped_metrics
from okk.utils import save_json


def parse_k_values(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(k <= 0 for k in values):
        raise ValueError(f"k values must be positive integers: {value}")
    return values


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


def neighbor_diagnostics(indices: np.ndarray, distances: np.ndarray, bank_meta: dict, eval_meta: dict) -> dict:
    diagnostics = {
        "mean_neighbor_distance": float(distances.mean()),
        "mean_kth_distance": float(distances[:, -1].mean()),
        "std_kth_distance": float(distances[:, -1].std()),
    }
    for key in ["groups", "operations", "generators"]:
        bank_values = bank_meta[key][indices]
        eval_values = eval_meta[key][:, None]
        diagnostics[f"same_{key[:-1]}_rate"] = float((bank_values == eval_values).mean())
    return diagnostics


def evaluate_scores(labels, scores, groups, generators, operations):
    overall = compute_binary_metrics(labels, scores).to_dict()
    return {
        "overall": overall,
        "by_group": grouped_metrics(labels, scores, groups),
        "by_generator_vs_real": one_vs_real_grouped_metrics(labels, scores, generators),
        "by_operation_vs_real": one_vs_real_grouped_metrics(labels, scores, operations),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate global vs local comparable real priors for CC-FLED Phase 1.")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--bank-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--allow-fake-bank", action="store_true")
    parser.add_argument("--conditions", type=str, default="semantic,proxy,semantic_proxy")
    parser.add_argument("--proxy-columns", type=str, default="default")
    parser.add_argument("--k-values", type=str, default="8,16,32,64")
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--proxy-weight", type=float, default=1.0)
    parser.add_argument("--allow-self-neighbor", action="store_true")
    parser.add_argument("--out", type=str, default="outputs/ccfled_local_prior_eval.json")
    args = parser.parse_args()
    if args.semantic_weight <= 0.0 or args.proxy_weight <= 0.0:
        raise ValueError("--semantic-weight and --proxy-weight must be positive")

    cfg = ExperimentConfig()
    ensure_project_dirs(cfg)
    data = np.load(args.cache, allow_pickle=True)
    labels = data["labels"].astype(np.int64)
    splits = data["splits"].astype(str)
    groups = data["groups"].astype(str)
    generators = data["generators"].astype(str)
    operations = data["operations"].astype(str)
    paths = data["paths"].astype(str)
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

    eval_labels = labels[eval_mask]
    eval_groups = groups[eval_mask]
    eval_generators = generators[eval_mask]
    eval_operations = operations[eval_mask]

    results = []
    global_scores = diagonal_global_energy(bank_z, eval_z, shrinkage=args.shrinkage)
    global_result = evaluate_scores(eval_labels, global_scores, eval_groups, eval_generators, eval_operations)
    results.append({
        "name": "global_diag",
        "condition": "none",
        "k": None,
        "shrinkage": float(args.shrinkage),
        **global_result,
    })
    print(format_binary_metrics("global_diag", global_result["overall"]))

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    k_values = parse_k_values(args.k_values)
    bank_positions = np.flatnonzero(bank_mask)
    eval_positions = np.flatnonzero(eval_mask)
    has_overlap = bool(np.isin(eval_positions, bank_positions).any())
    exclude_self_neighbor = has_overlap and not args.allow_self_neighbor
    available_neighbors = int(bank_mask.sum()) - (1 if exclude_self_neighbor else 0)
    if available_neighbors <= 0:
        raise ValueError("memory bank has no usable non-self neighbors")
    max_k = min(max(k_values), available_neighbors)
    effective_k_values = sorted({min(k, max_k) for k in k_values})
    query_k = max_k + 1 if exclude_self_neighbor else max_k
    query_k = min(query_k, int(bank_mask.sum()))
    bank_meta = {
        "groups": groups[bank_mask],
        "generators": generators[bank_mask],
        "operations": operations[bank_mask],
    }
    eval_meta = {
        "groups": eval_groups,
        "generators": eval_generators,
        "operations": eval_operations,
    }

    for condition in conditions:
        blocks, label = condition_blocks_from_cache(data, condition, proxy_columns)
        condition_std, block_info = build_weighted_condition_matrix(
            blocks=blocks,
            bank_mask=bank_mask,
            semantic_weight=args.semantic_weight,
            proxy_weight=args.proxy_weight,
        )
        bank_condition = condition_std[bank_mask]
        eval_condition = condition_std[eval_mask]
        knn = NearestNeighbors(n_neighbors=query_k, metric="euclidean")
        knn.fit(bank_condition)
        distances, indices = knn.kneighbors(eval_condition, return_distance=True)
        if exclude_self_neighbor:
            indices, distances = remove_self_neighbors(
                indices=indices,
                distances=distances,
                bank_positions=bank_positions,
                eval_positions=eval_positions,
                max_k=max_k,
            )
        for k in effective_k_values:
            k_eff = min(k, max_k)
            local_scores = diagonal_local_energy(
                bank_z=bank_z,
                eval_z=eval_z,
                neighbor_idx=indices[:, :k_eff],
                shrinkage=args.shrinkage,
            )
            score_result = evaluate_scores(eval_labels, local_scores, eval_groups, eval_generators, eval_operations)
            diagnostics = neighbor_diagnostics(indices[:, :k_eff], distances[:, :k_eff], bank_meta, eval_meta)
            row = {
                "name": f"{condition}_k{k_eff}",
                "condition": label,
                "k": int(k_eff),
                "shrinkage": float(args.shrinkage),
                "condition_blocks": block_info,
                "neighbor_diagnostics": diagnostics,
                **score_result,
            }
            results.append(row)
            print(format_binary_metrics(row["name"], row["overall"]))

    output = {
        "cache": str(args.cache),
        "bank_split": args.bank_split,
        "eval_split": args.eval_split,
        "bank_real_only": not args.allow_fake_bank,
        "bank_count": int(bank_mask.sum()),
        "eval_count": int(eval_mask.sum()),
        "requested_k_values": k_values,
        "effective_k_values": effective_k_values,
        "proxy_columns": proxy_columns,
        "proxy_frame": scalar_string(data, "proxy_frame", "legacy_unknown"),
        "transform_protocol": scalar_string(data, "transform_protocol", "legacy_unknown"),
        "condition_distance": {
            "semantic_weight": float(args.semantic_weight),
            "proxy_weight": float(args.proxy_weight),
            "block_normalization": "each standardized block is scaled by weight / sqrt(dim)",
        },
        "self_neighbor_policy": {
            "bank_eval_overlap": bool(has_overlap),
            "excluded_self_neighbors": bool(exclude_self_neighbor),
            "allow_self_neighbor": bool(args.allow_self_neighbor),
        },
        "zf_standardization": {
            "mean_shape": list(zf_mean.shape),
            "std_shape": list(zf_std.shape),
        },
        "score_direction": "higher energy means more suspicious; label=1 is fake",
        "results": results,
    }
    out = cfg.project_root / args.out
    save_json(output, out)
    print(format_grouped_metrics("best_result_by_generator_vs_real", max(results, key=lambda row: row["overall"]["auroc"])["by_generator_vs_real"]))
    print(f"saved local prior evaluation: {out}")


if __name__ == "__main__":
    main()
