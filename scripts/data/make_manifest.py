from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import csv
from pathlib import Path

from okk.utils import list_images


def collect_split(root: Path, label: int, group: str, split: str, operation: str, generator: str):
    rows = []
    if not root.exists():
        return rows
    for path in sorted(list_images(root)):
        rows.append({
            "path": str(path),
            "label": label,
            "group": group,
            "pair_id": "",
            "mask_path": "",
            "operation": operation,
            "generator": generator,
            "split": split,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-train", type=str, default="")
    parser.add_argument("--fake-train", type=str, default="")
    parser.add_argument("--real-val", type=str, default="")
    parser.add_argument("--fake-val", type=str, default="")
    parser.add_argument("--real-test", type=str, default="")
    parser.add_argument("--fake-test", type=str, default="")
    parser.add_argument("--group", type=str, default="custom")
    parser.add_argument("--fake-generator", type=str, default="unknown_fake")
    parser.add_argument("--fake-operation", type=str, default="full_generation")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = []
    rows += collect_split(Path(args.real_train), 0, args.group, "train", "real", "real")
    rows += collect_split(Path(args.fake_train), 1, args.group, "train", args.fake_operation, args.fake_generator)
    rows += collect_split(Path(args.real_val), 0, args.group, "val", "real", "real")
    rows += collect_split(Path(args.fake_val), 1, args.group, "val", args.fake_operation, args.fake_generator)
    rows += collect_split(Path(args.real_test), 0, args.group, "test", "real", "real")
    rows += collect_split(Path(args.fake_test), 1, args.group, "test", args.fake_operation, args.fake_generator)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "pair_id", "mask_path", "operation", "generator", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"写入 manifest: {out}, 样本数: {len(rows)}")


if __name__ == "__main__":
    main()


