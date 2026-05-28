from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from okk.ccfled import eta_squared, pearson_corr
from okk.config import ExperimentConfig, ensure_project_dirs
from okk.utils import save_json


def split_mask(splits: np.ndarray, value: str) -> np.ndarray:
    if not value or value == "all":
        return np.ones(len(splits), dtype=bool)
    allowed = {item.strip() for item in value.split(",") if item.strip()}
    return np.asarray([split in allowed for split in splits], dtype=bool)


def count_values(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values.astype(str), return_counts=True)
    return {str(key): int(count) for key, count in zip(unique, counts)}


def scalar_string(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return str(value.item())
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Audit CC-FLED proxy variables for label and pipeline confounding.")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/ccfled_proxy_audit.json")
    parser.add_argument("--audit-split", type=str, default="train")
    parser.add_argument("--flag-threshold", type=float, default=0.20)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    ensure_project_dirs(cfg)
    data = np.load(args.cache, allow_pickle=True)
    values = data["z_c_proxy"].astype(np.float64)
    names = [str(x) for x in data["proxy_names"].tolist()]
    labels = data["labels"].astype(np.float64)
    groups = data["groups"].astype(str)
    generators = data["generators"].astype(str)
    operations = data["operations"].astype(str)
    splits = data["splits"].astype(str)
    audit_mask = split_mask(splits, args.audit_split)
    if not audit_mask.any():
        raise ValueError(f"empty audit split: {args.audit_split}")

    audit_values = values[audit_mask]
    audit_labels = labels[audit_mask]
    audit_groups = groups[audit_mask]
    audit_generators = generators[audit_mask]
    audit_operations = operations[audit_mask]

    proxies = {}
    for idx, name in enumerate(names):
        x = audit_values[:, idx]
        label_corr = pearson_corr(x, audit_labels)
        item = {
            "label_corr": label_corr,
            "abs_label_corr": abs(label_corr) if np.isfinite(label_corr) else None,
            "generator_eta2": eta_squared(x, audit_generators),
            "group_eta2": eta_squared(x, audit_groups),
            "operation_eta2": eta_squared(x, audit_operations),
            "all_split_eta2": eta_squared(values[:, idx], splits),
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x)),
        }
        flags = []
        if item["abs_label_corr"] is not None and item["abs_label_corr"] >= args.flag_threshold:
            flags.append("label_correlated")
        for key in ["generator_eta2", "group_eta2", "operation_eta2"]:
            if np.isfinite(item[key]) and item[key] >= args.flag_threshold:
                flags.append(key.replace("_eta2", "_structured"))
        if np.isfinite(item["all_split_eta2"]) and item["all_split_eta2"] >= args.flag_threshold:
            flags.append("split_structured_all_splits")
        item["flags"] = flags
        proxies[name] = item

    if audit_values.shape[0] < 2:
        corr = np.eye(len(names), dtype=np.float64)
    else:
        corr = np.corrcoef(np.nan_to_num(audit_values, nan=0.0), rowvar=False)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(corr[i, j])
            pairs.append({"a": names[i], "b": names[j], "corr": value, "abs_corr": abs(value)})
    pairs.sort(key=lambda row: row["abs_corr"], reverse=True)

    result = {
        "cache": str(args.cache),
        "n": int(values.shape[0]),
        "audit_split": args.audit_split,
        "audit_count": int(audit_mask.sum()),
        "split_counts": count_values(splits),
        "flag_threshold": float(args.flag_threshold),
        "proxy_frame": scalar_string(data, "proxy_frame", "legacy_unknown"),
        "interpretation": (
            "Proxy variables are operational conditions, not complexity ground truth. "
            "Label/generator/group/operation flags are computed on audit_split only; "
            "all_split_eta2 is reported as a distribution-shift warning, not as a proxy-selection signal."
        ),
        "proxies": proxies,
        "top_proxy_correlations": pairs[:20],
    }
    out = cfg.project_root / args.out
    save_json(result, out)
    flagged = {name: item["flags"] for name, item in proxies.items() if item["flags"]}
    print(f"saved proxy audit: {out}")
    print(f"flagged proxies: {flagged if flagged else 'none'}")


if __name__ == "__main__":
    main()
