from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import csv

from okk.utils import list_images


FIELDNAMES = ["path", "label", "group", "pair_id", "mask_path", "operation", "generator", "split"]


def add_rows(root: Path, label: int, generator: str, split: str):
    rows = []
    for path in sorted(list_images(root)):
        rows.append({
            "path": str(path),
            "label": label,
            "group": "Chameleon",
            "pair_id": "",
            "mask_path": "",
            "operation": "real" if label == 0 else "full_generation",
            "generator": "real" if label == 0 else generator,
            "split": split,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/autodl-tmp/Chameleon")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fake-generator", type=str, default="Chameleon")
    parser.add_argument("--out", type=str, default="manifests/chameleon_test.csv")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    real_dir = root / "test" / "0_real"
    fake_dir = root / "test" / "1_fake"
    if not real_dir.exists():
        raise FileNotFoundError(f"real dir not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"fake dir not found: {fake_dir}")

    rows = []
    rows += add_rows(real_dir, label=0, generator=args.fake_generator, split=args.split)
    rows += add_rows(fake_dir, label=1, generator=args.fake_generator, split=args.split)
    if not rows:
        raise ValueError(f"no images found under {real_dir} or {fake_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    real_count = sum(1 for row in rows if row["label"] == 0)
    fake_count = sum(1 for row in rows if row["label"] == 1)
    print(f"wrote {out}")
    print(f"real: {real_count}, fake: {fake_count}, total: {len(rows)}")


if __name__ == "__main__":
    main()

