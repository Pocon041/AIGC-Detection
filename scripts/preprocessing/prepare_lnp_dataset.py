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

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from okk.dataset import read_manifest


def to_float_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_lnp(original: np.ndarray, denoised: np.ndarray, out_path: Path, gain: float = 8.0) -> None:
    residual = original - denoised
    residual = residual * gain + 0.5
    residual = np.clip(residual, 0.0, 1.0)
    img = Image.fromarray((residual * 255.0).round().astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def classical_denoise(path: str | Path, mode: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if mode == "median":
        den = img.filter(ImageFilter.MedianFilter(size=3))
    elif mode == "gaussian":
        den = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    else:
        raise ValueError(f"unknown denoise mode: {mode}")
    return np.asarray(den).astype(np.float32) / 255.0


def find_denoised_path(denoised_root: Path, original_path: str | Path, source_root: Path | None) -> Path | None:
    original_path = Path(original_path)
    if source_root is not None:
        try:
            rel = original_path.relative_to(source_root)
            candidate = denoised_root / rel
            if candidate.exists():
                return candidate
        except ValueError:
            pass
    for ext in [original_path.suffix, ".png", ".jpg", ".jpeg", ".webp"]:
        candidate = denoised_root / f"{original_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--out-manifest", type=str, required=True)
    parser.add_argument("--denoised-root", type=str, default="")
    parser.add_argument("--source-root", type=str, default="")
    parser.add_argument("--fallback-denoise", type=str, default="median", choices=["median", "gaussian"])
    parser.add_argument("--gain", type=float, default=8.0)
    args = parser.parse_args()

    items = read_manifest(args.manifest, split=args.split)
    out_root = Path(args.out_root)
    denoised_root = Path(args.denoised_root) if args.denoised_root else None
    source_root = Path(args.source_root) if args.source_root else None
    rows = []
    for item in tqdm(items, desc="prepare_lnp"):
        original = to_float_rgb(item.path)
        denoised = None
        if denoised_root is not None:
            den_path = find_denoised_path(denoised_root, item.path, source_root)
            if den_path is not None:
                denoised = to_float_rgb(den_path)
        if denoised is None:
            denoised = classical_denoise(item.path, args.fallback_denoise)
        if denoised.shape != original.shape:
            den_img = Image.fromarray((denoised * 255.0).round().astype(np.uint8)).resize((original.shape[1], original.shape[0]), Image.BICUBIC)
            denoised = np.asarray(den_img).astype(np.float32) / 255.0
        out_path = out_root / item.split / str(item.label) / f"{Path(item.path).stem}.png"
        save_lnp(original, denoised, out_path, gain=args.gain)
        rows.append({"path": str(out_path), "label": item.label, "group": item.group, "split": item.split})

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"LNP manifest: {out_manifest}, samples: {len(rows)}")


if __name__ == "__main__":
    main()


