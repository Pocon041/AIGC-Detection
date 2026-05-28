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


def main():
    parser = argparse.ArgumentParser(description="Audit CC-FLED proxy variables for label and pipeline confounding.")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/ccfled_proxy_audit.json")
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

    proxies = {}
    for idx, name in enumerate(names):
        x = values[:, idx]
        label_corr = pearson_corr(x, labels)
        item = {
            "label_corr": label_corr,
            "abs_label_corr": abs(label_corr) if np.isfinite(label_corr) else None,
            "generator_eta2": eta_squared(x, generators),
            "group_eta2": eta_squared(x, groups),
            "operation_eta2": eta_squared(x, operations),
            "split_eta2": eta_squared(x, splits),
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x)),
        }
        flags = []
        if item["abs_label_corr"] is not None and item["abs_label_corr"] >= args.flag_threshold:
            flags.append("label_correlated")
        for key in ["generator_eta2", "group_eta2", "operation_eta2", "split_eta2"]:
            if np.isfinite(item[key]) and item[key] >= args.flag_threshold:
                flags.append(key.replace("_eta2", "_structured"))
        item["flags"] = flags
        proxies[name] = item

    corr = np.corrcoef(np.nan_to_num(values, nan=0.0), rowvar=False)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(corr[i, j])
            pairs.append({"a": names[i], "b": names[j], "corr": value, "abs_corr": abs(value)})
    pairs.sort(key=lambda row: row["abs_corr"], reverse=True)

    result = {
        "cache": str(args.cache),
        "n": int(values.shape[0]),
        "flag_threshold": float(args.flag_threshold),
        "interpretation": "Proxy variables are operational conditions, not complexity ground truth. Flagged variables need confound checks before use in z_c.",
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
