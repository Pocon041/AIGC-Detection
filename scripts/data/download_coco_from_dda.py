from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

from okk.utils import IMAGE_EXTENSIONS, list_images


COCO_ID_RE = re.compile(r"(\d{12})")


def parse_names(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def infer_generators(fake_root: Path, split_dir: str):
    generators = []
    for child in sorted(fake_root.iterdir()):
        if child.is_dir() and (child / split_dir).exists():
            generators.append(child.name)
    if not generators:
        raise ValueError(f"没有在 {fake_root} 下找到 */{split_dir} 目录")
    return generators


def extract_coco_filename(path: Path):
    match = COCO_ID_RE.search(path.stem)
    if match:
        return f"{match.group(1)}.jpg"
    if path.stem.isdigit():
        return f"{int(path.stem):012d}.jpg"
    return ""


def collect_coco_filenames(fake_root: Path, split_dir: str, generators: list[str]):
    filenames = set()
    source_count = 0
    bad_names = []
    for gen in generators:
        image_dir = fake_root / gen / split_dir
        if not image_dir.exists():
            raise FileNotFoundError(f"generator 目录不存在: {image_dir}")
        for path in list_images(image_dir):
            source_count += 1
            name = extract_coco_filename(path)
            if name:
                filenames.add(name)
            else:
                bad_names.append(str(path))
    return sorted(filenames), source_count, bad_names


def download_one(filename: str, out_dir: Path, base_url: str, retries: int, timeout: int, min_bytes: int):
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size >= min_bytes:
        return filename, "exists", ""

    url = f"{base_url.rstrip('/')}/{filename}"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    last_error = ""
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as response:
                data = response.read()
            if len(data) < min_bytes:
                raise RuntimeError(f"文件过小: {len(data)} bytes")
            tmp_path.write_bytes(data)
            tmp_path.replace(out_path)
            return filename, "downloaded", ""
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    if tmp_path.exists():
        tmp_path.unlink()
    return filename, "failed", last_error


def write_report(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--generators", type=str, default="")
    parser.add_argument("--split-dir", type=str, default="val2017")
    parser.add_argument("--base-url", type=str, default="http://images.cocodataset.org/val2017")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=str, default="manifests/coco_download_report.csv")
    args = parser.parse_args()

    fake_root = Path(args.fake_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not fake_root.exists():
        raise FileNotFoundError(f"fake-root 不存在: {fake_root}")

    generators = parse_names(args.generators)
    if not generators:
        generators = infer_generators(fake_root, args.split_dir)

    filenames, source_count, bad_names = collect_coco_filenames(fake_root, args.split_dir, generators)
    print(f"fake_root: {fake_root}")
    print(f"generators: {generators}")
    print(f"扫描 fake 图像数: {source_count}")
    print(f"需要 COCO 原图数: {len(filenames)}")
    print(f"无法解析文件名数: {len(bad_names)}")

    if bad_names:
        print("前 10 个无法解析的文件名:")
        for item in bad_names[:10]:
            print(item)

    if args.dry_run:
        print("dry-run 模式，不下载。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, name, out_dir, args.base_url, args.retries, args.timeout, args.min_bytes)
            for name in filenames
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="download COCO"):
            filename, status, error = future.result()
            rows.append({"filename": filename, "status": status, "error": error})

    write_report(Path(args.report).expanduser(), rows)
    downloaded = sum(1 for row in rows if row["status"] == "downloaded")
    existed = sum(1 for row in rows if row["status"] == "exists")
    failed = sum(1 for row in rows if row["status"] == "failed")
    print(f"已存在: {existed}")
    print(f"新下载: {downloaded}")
    print(f"失败: {failed}")
    print(f"输出目录: {out_dir}")
    print(f"报告文件: {args.report}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


