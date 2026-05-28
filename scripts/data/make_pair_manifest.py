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


def stem_map(root: Path):
    return {p.stem: p for p in sorted(list_images(root))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--fake-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--group", type=str, default="paired")
    parser.add_argument("--operation", type=str, default="aligned_fake")
    parser.add_argument("--generator", type=str, default="unknown_fake")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    real = stem_map(Path(args.real_dir))
    fake = stem_map(Path(args.fake_dir))
    common = sorted(set(real.keys()) & set(fake.keys()))
    rows = []
    for stem in common:
        rows.append({
            "path": str(real[stem]),
            "label": 0,
            "group": args.group,
            "pair_id": stem,
            "mask_path": "",
            "operation": "real",
            "generator": "real",
            "split": args.split,
        })
        rows.append({
            "path": str(fake[stem]),
            "label": 1,
            "group": args.group,
            "pair_id": stem,
            "mask_path": "",
            "operation": args.operation,
            "generator": args.generator,
            "split": args.split,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "pair_id", "mask_path", "operation", "generator", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"写入 paired manifest: {out}, pair 数: {len(common)}, 样本数: {len(rows)}")


if __name__ == "__main__":
    main()


