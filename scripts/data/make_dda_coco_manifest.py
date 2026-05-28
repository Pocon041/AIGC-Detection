from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import csv
import random
from pathlib import Path

from okk.utils import list_images


FIELDNAMES = ["path", "label", "group", "pair_id", "mask_path", "operation", "generator", "split"]


def image_map(root: Path):
    return {p.stem: p for p in sorted(list_images(root))}


def split_stems(stems, train_ratio: float, val_ratio: float, seed: int):
    stems = list(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = set(stems[:n_train])
    val = set(stems[n_train:n_train + n_val])
    test = set(stems[n_train + n_val:])
    return train, val, test


def get_split(stem: str, train: set[str], val: set[str]) -> str:
    if stem in train:
        return "train"
    if stem in val:
        return "val"
    return "test"


def collect_generators(fake_root: Path):
    gens = []
    for child in sorted(fake_root.iterdir()):
        if child.is_dir() and (child / "val2017").exists():
            gens.append(child.name)
    if not gens:
        raise ValueError(f"没有在 {fake_root} 下找到 */val2017 generator 目录")
    return gens


def parse_names(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def validate_generator_splits(train_gens, val_gens, test_gens):
    buckets = [set(train_gens), set(val_gens), set(test_gens)]
    names = ["train", "val", "test"]
    for i in range(len(buckets)):
        for j in range(i + 1, len(buckets)):
            overlap = buckets[i] & buckets[j]
            if overlap:
                raise ValueError(f"generator split 重复: {names[i]} 和 {names[j]} 都包含 {sorted(overlap)}")


def split_for_generator(gen: str, train_gens, val_gens, test_gens, default_split: str):
    if gen in train_gens:
        return "train"
    if gen in val_gens:
        return "val"
    if gen in test_gens:
        return "test"
    return default_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=str, default="D:/COCO/val2017")
    parser.add_argument("--fake-root", type=str, default="D:/DDA-COCO/DDA-COCO")
    parser.add_argument("--generators", type=str, default="")
    parser.add_argument("--train-generators", type=str, default="")
    parser.add_argument("--val-generators", type=str, default="")
    parser.add_argument("--test-generators", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_root = Path(args.fake_root)
    real = image_map(real_dir)
    if not real:
        raise ValueError(f"真实图目录没有图像: {real_dir}")

    train_gens = parse_names(args.train_generators)
    val_gens = parse_names(args.val_generators)
    test_gens = parse_names(args.test_generators)
    validate_generator_splits(train_gens, val_gens, test_gens)
    use_generator_splits = bool(train_gens or val_gens or test_gens)

    generators = parse_names(args.generators)
    if not generators:
        generators = sorted(set(train_gens + val_gens + test_gens)) if use_generator_splits else collect_generators(fake_root)
    missing_split_gens = sorted((set(train_gens) | set(val_gens) | set(test_gens)) - set(generators))
    if missing_split_gens:
        raise ValueError(f"--generators 不包含这些 split generator: {missing_split_gens}")

    all_common = set(real.keys())
    fake_maps = {}
    for gen in generators:
        fake_dir = fake_root / gen / "val2017"
        fake = image_map(fake_dir)
        if not fake:
            raise ValueError(f"fake generator 没有图像: {fake_dir}")
        fake_maps[gen] = fake
        all_common &= set(fake.keys())

    common = sorted(all_common)
    if not common:
        raise ValueError("real 与 fake generators 没有同名图像，无法配对")

    use_split_ratios = not use_generator_splits and (args.train_ratio > 0.0 or args.val_ratio > 0.0)
    if use_split_ratios:
        if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
            raise ValueError("要求 0 <= train_ratio, 0 <= val_ratio 且 train_ratio + val_ratio < 1")
        train_stems, val_stems, _ = split_stems(common, args.train_ratio, args.val_ratio, args.seed)
    else:
        train_stems, val_stems = set(), set()

    rows = []
    for stem in common:
        stem_split = get_split(stem, train_stems, val_stems) if use_split_ratios else args.split
        if args.paired:
            for gen in generators:
                split = split_for_generator(gen, train_gens, val_gens, test_gens, stem_split) if use_generator_splits else stem_split
                pair_id = f"{stem}__{gen}"
                rows.append({
                    "path": str(real[stem]),
                    "label": 0,
                    "group": "DDA-COCO",
                    "pair_id": pair_id,
                    "mask_path": "",
                    "operation": "real",
                    "generator": "real",
                    "split": split,
                })
                rows.append({
                    "path": str(fake_maps[gen][stem]),
                    "label": 1,
                    "group": "DDA-COCO",
                    "pair_id": pair_id,
                    "mask_path": "",
                    "operation": "full_generation",
                    "generator": gen,
                    "split": split,
                })
        else:
            if use_generator_splits:
                gen_iter = generators
            else:
                gen_iter = [""]
            for real_gen in gen_iter:
                split = split_for_generator(real_gen, train_gens, val_gens, test_gens, stem_split) if use_generator_splits else stem_split
                rows.append({
                    "path": str(real[stem]),
                    "label": 0,
                    "group": "DDA-COCO",
                    "pair_id": f"{stem}__{real_gen}" if use_generator_splits else stem,
                    "mask_path": "",
                    "operation": "real",
                    "generator": "real",
                    "split": split,
                })
            for gen in generators:
                split = split_for_generator(gen, train_gens, val_gens, test_gens, stem_split) if use_generator_splits else stem_split
                rows.append({
                    "path": str(fake_maps[gen][stem]),
                    "label": 1,
                    "group": "DDA-COCO",
                    "pair_id": f"{stem}__{gen}",
                    "mask_path": "",
                    "operation": "full_generation",
                    "generator": gen,
                    "split": split,
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"写入 DDA-COCO manifest: {out}")
    print(f"generators: {generators}")
    print(f"同名 real/fake 图像数: {len(common)}")
    print(f"样本行数: {len(rows)}")


if __name__ == "__main__":
    main()


