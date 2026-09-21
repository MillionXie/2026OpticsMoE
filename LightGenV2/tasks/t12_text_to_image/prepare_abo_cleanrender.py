"""Build a small object-disjoint subset of the official ABO No-BG renders.

The upstream reconstruction benchmark is a 223 GB ZIP.  Its S3 endpoint
supports byte ranges, so ``remotezip`` can fetch only the selected PNG members
without downloading the archive.  Product identities, rather than views, are
split across train/validation/test.
"""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures
import contextlib
import gzip
import hashlib
import io
import json
import random
import re
import struct
import time
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageDraw, ImageOps

from .dataset import validate_split_contract, write_manifest


ARCHIVE_URL = (
    "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
    "archives/abo-release-renders.zip"
)
SOURCE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
LICENSE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-4.0.txt"
LICENSE = "CC BY 4.0"
PREFIX = "home/achleshwar/amazon_iccv21/ABO_RELEASE/"
CATEGORY_LABELS = {"CHAIR": "chair", "LAMP": "lamp", "TABLE": "table"}
MULTI_OBJECT = re.compile(
    r"\b(set of|pair of|pack of|[2-9][ -]pack|[2-9][ -]piece|bundle)\b",
    flags=re.IGNORECASE,
)


def _english(row: dict[str, Any], key: str) -> str:
    for value in row.get(key, []) or []:
        if value.get("language_tag") == "en_US" and str(value.get("value", "")).strip():
            return str(value["value"]).strip()
    return ""


def _short_attribute(value: str, words: int) -> str:
    cleaned = re.sub(r"[^A-Za-z -]+", " ", value).lower()
    tokens = [token for token in cleaned.split() if len(token) > 1]
    return " ".join(tokens[:words])


def _caption(row: dict[str, Any], category: str) -> str:
    color = _short_attribute(_english(row, "color"), 2)
    material = _short_attribute(_english(row, "material") or _english(row, "fabric_type"), 2)
    attributes: list[str] = []
    for token in f"{color} {material}".split():
        if token not in attributes:
            attributes.append(token)
    description = " ".join([*attributes, CATEGORY_LABELS[category]])
    article = "an" if description[:1] in "aeiou" else "a"
    return f"{article} {description} on a clean white studio background"


def _product_types(row: dict[str, Any]) -> set[str]:
    return {
        str(value.get("value", "") if isinstance(value, dict) else value)
        for value in row.get("product_type", []) or []
    }


def _listings(root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted((root / "listings" / "metadata").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


@contextlib.contextmanager
def _archive(source: str | Path):
    if isinstance(source, Path) or not str(source).startswith(("http://", "https://")):
        with zipfile.ZipFile(source) as handle:
            yield handle
        return
    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise RuntimeError("Remote ABO preparation requires `pip install remotezip`") from error
    with RemoteZip(str(source)) as handle:
        yield handle


def _render_index(names: list[str]) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = defaultdict(dict)
    pattern = re.compile(r"/([^/]+)/\1_(\d\d)\.png$")
    for name in names:
        match = pattern.search(name)
        if match:
            result[match.group(1)][int(match.group(2))] = name
    return result


def _view_subset(available: list[int], count: int) -> list[int]:
    if count > len(available):
        raise ValueError(f"Requested {count} views but only {len(available)} are available")
    if count == 1:
        return [available[len(available) // 2]]
    return [available[round(index * (len(available) - 1) / (count - 1))] for index in range(count)]


def _save_render(payload: bytes, output: Path, size: int) -> None:
    with Image.open(io.BytesIO(payload)) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGBA")
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image)
        rgb = white.convert("RGB")
        rgb = ImageOps.pad(rgb, (size, size), method=Image.Resampling.LANCZOS, color="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output, format="JPEG", quality=95, subsampling=0)


def _fetch_remote_member(url: str, info: zipfile.ZipInfo, retries: int = 5) -> bytes:
    """Fetch and verify one compressed ZIP member with a single HTTP range."""

    import requests

    overhead = 8192
    end = info.header_offset + 30 + len(info.filename.encode("utf-8")) + info.compress_size + overhead
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={info.header_offset}-{end}"},
                timeout=(20, 120),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise IOError(f"Server ignored ZIP member byte range: HTTP {response.status_code}")
            blob = response.content
            fields = struct.unpack("<IHHHHHIIIHH", blob[:30])
            if fields[0] != 0x04034B50:
                raise IOError(f"Invalid local ZIP header for {info.filename}")
            name_length, extra_length = fields[-2:]
            start = 30 + name_length + extra_length
            compressed = blob[start : start + info.compress_size]
            if len(compressed) != info.compress_size:
                raise IOError(f"Truncated compressed member {info.filename}")
            if info.compress_type == zipfile.ZIP_DEFLATED:
                payload = zlib.decompress(compressed, -15)
            elif info.compress_type == zipfile.ZIP_STORED:
                payload = compressed
            else:
                raise NotImplementedError(f"Unsupported ZIP compression {info.compress_type}")
            if len(payload) != info.file_size:
                raise IOError(f"Wrong uncompressed size for {info.filename}")
            if binascii.crc32(payload) & 0xFFFFFFFF != info.CRC:
                raise IOError(f"CRC mismatch for {info.filename}")
            return payload
        except (OSError, requests.RequestException, struct.error, zlib.error) as caught:
            error = caught
            if attempt + 1 < retries:
                time.sleep(min(16, 2**attempt))
    raise IOError(f"Failed to range-fetch {info.filename} after {retries} attempts") from error


def _download_remote_jobs(
    url: str,
    infos: dict[str, zipfile.ZipInfo],
    jobs: list[tuple[str, Path]],
    image_size: int,
    workers: int,
) -> None:
    def download(job: tuple[str, Path]) -> None:
        member, output = job
        _save_render(_fetch_remote_member(url, infos[member]), output, image_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if index == 1 or index % 50 == 0 or index == len(jobs):
                print(f"[ABO CleanRender] downloaded {index}/{len(jobs)}", flush=True)


def _contact_sheet(output_dir: Path, rows: list[dict[str, Any]], categories: list[str], size: int) -> None:
    columns, label_height = 8, 20
    tile = min(size, 160)
    canvas = Image.new("RGB", (columns * tile, len(categories) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, category in enumerate(categories):
        label = CATEGORY_LABELS[category]
        options = [row for row in rows if row["category"] == label]
        chosen = [options[round(i * (len(options) - 1) / (columns - 1))] for i in range(columns)]
        top = row_index * (tile + label_height)
        draw.text((4, top + 3), f"ABO No-BG / {label}", fill="black")
        for column, row in enumerate(chosen):
            with Image.open(output_dir / row["image_path"]) as handle:
                image = ImageOps.fit(handle.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
            canvas.paste(image, (column * tile, top + label_height))
    canvas.save(output_dir / "contact_sheet.jpg", quality=92, subsampling=0)


def prepare(
    abo_root: Path,
    output_dir: Path,
    categories: list[str],
    *,
    archive_source: str | Path = ARCHIVE_URL,
    train_instances: int = 48,
    val_instances: int = 8,
    test_instances: int = 8,
    train_views: int = 12,
    eval_views: int = 8,
    image_size: int = 128,
    seed: int = 42,
    download_workers: int = 16,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    license_path = abo_root / "LICENSE-CC-BY-4.0.txt"
    if not license_path.is_file():
        raise FileNotFoundError("ABO LICENSE-CC-BY-4.0.txt is required beside the metadata")
    unknown = set(categories) - CATEGORY_LABELS.keys()
    if unknown:
        raise ValueError(f"Unsupported CleanRender categories: {sorted(unknown)}")

    remote = str(archive_source).startswith(("http://", "https://"))
    download_jobs: list[tuple[str, Path]] = []
    with _archive(archive_source) as archive:
        render_index = _render_index(archive.namelist())
        info_by_name = {info.filename: info for info in archive.infolist()} if remote else {}
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[str] = set()
        for row in _listings(abo_root):
            item_id = str(row.get("item_id", ""))
            if item_id in seen or item_id not in render_index:
                continue
            category = next((value for value in categories if value in _product_types(row)), None)
            if category is None:
                continue
            title = _english(row, "item_name")
            if title and MULTI_OBJECT.search(title):
                continue
            if len(render_index[item_id]) < max(train_views, eval_views):
                continue
            seen.add(item_id)
            candidates[category].append({
                "item_id": item_id,
                "caption": _caption(row, category),
                "source_title": title,
            })

        needed = train_instances + val_instances + test_instances
        manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        eligible: dict[str, int] = {}
        split_counts = (("train", train_instances), ("val", val_instances), ("test", test_instances))
        for category in categories:
            rows = sorted(candidates[category], key=lambda value: value["item_id"])
            random.Random(f"{seed}:{category}").shuffle(rows)
            eligible[category] = len(rows)
            if len(rows) < needed:
                raise ValueError(f"{category} has {len(rows)} eligible renders; {needed} required")
            cursor = 0
            for split, count in split_counts:
                views = train_views if split == "train" else eval_views
                for row in rows[cursor : cursor + count]:
                    item_id = row["item_id"]
                    available = sorted(render_index[item_id])
                    offset = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()[0] % len(available)
                    rotated = available[offset:] + available[:offset]
                    chosen_views = _view_subset(rotated, views)
                    for view in chosen_views:
                        member = render_index[item_id][view]
                        relative = Path("images") / split / category.lower() / f"{item_id}_{view:02d}.jpg"
                        download_jobs.append((member, output_dir / relative))
                        manifests[split].append({
                            "sample_id": f"{category.lower()}-{item_id}-{view:02d}",
                            "sequence_id": item_id,
                            "category": CATEGORY_LABELS[category],
                            "caption": row["caption"],
                            "image_path": relative.as_posix(),
                            "license": LICENSE,
                            "source_url": SOURCE_URL,
                            "source_archive": str(archive_source),
                            "source_member": member,
                            "source_title": row["source_title"],
                            "modified": f"alpha-composited on white and resized to {image_size}x{image_size}",
                        })
                cursor += count

        if not remote:
            for index, (member, destination) in enumerate(download_jobs, 1):
                _save_render(archive.read(member), destination, image_size)
                if index == 1 or index % 50 == 0 or index == len(download_jobs):
                    print(f"[ABO CleanRender] extracted {index}/{len(download_jobs)}", flush=True)

    if remote:
        _download_remote_jobs(
            str(archive_source), info_by_name, download_jobs, image_size, download_workers
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        write_manifest(output_dir / f"{split}.jsonl", rows)
    (output_dir / "LICENSE-CC-BY-4.0.txt").write_bytes(license_path.read_bytes())
    _contact_sheet(output_dir, manifests["train"], categories, image_size)
    summary = validate_split_contract(output_dir)
    summary.update({
        "source_dataset": "ABO CVPR 2022 No-BG Blender render benchmark",
        "source_url": SOURCE_URL,
        "source_archive": str(archive_source),
        "source_license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        "attribution": "Amazon.com and the Amazon Berkeley Objects dataset creators",
        "seed": seed,
        "eligible_rendered_products": eligible,
        "views_per_train_product": train_views,
        "views_per_eval_product": eval_views,
        "image_size": image_size,
        "download_workers": download_workers,
        "one_object_per_image": True,
        "identity_split": True,
    })
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a range-fetched ABO No-BG render subset")
    parser.add_argument("--abo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", default=ARCHIVE_URL)
    parser.add_argument("--categories", default="CHAIR,LAMP,TABLE")
    parser.add_argument("--train-instances", type=int, default=48)
    parser.add_argument("--val-instances", type=int, default=8)
    parser.add_argument("--test-instances", type=int, default=8)
    parser.add_argument("--train-views", type=int, default=12)
    parser.add_argument("--eval-views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-workers", type=int, default=16)
    args = parser.parse_args()
    source: str | Path = args.archive
    if not str(source).startswith(("http://", "https://")):
        source = Path(source).expanduser().resolve()
    report = prepare(
        args.abo_root.expanduser().resolve(), args.output_dir.expanduser().resolve(),
        [value.strip() for value in args.categories.split(",") if value.strip()],
        archive_source=source, train_instances=args.train_instances,
        val_instances=args.val_instances, test_instances=args.test_instances,
        train_views=args.train_views, eval_views=args.eval_views,
        image_size=args.image_size, seed=args.seed, download_workers=args.download_workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
