"""Prepare a compact, clean CC BY 4.0 text-to-image subset from ABO.

The Amazon Berkeley Objects catalog archive contains one main product image
and structured text attributes for each listing.  T12 keeps one image per
product identity, selects a few exact product types, shortens the condition to
color/material/category, and places a trimmed product image on a deterministic
neutral canvas.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw, ImageOps

from .dataset import validate_split_contract, write_manifest


SOURCE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
LICENSE = "CC BY 4.0"
PALETTE = ((244, 244, 242), (238, 241, 246), (245, 241, 234), (235, 242, 239))
CATEGORY_LABELS = {
    "SHOES": "shoe",
    "BACKPACK": "backpack",
    "CHAIR": "chair",
    "LAMP": "lamp",
    "TABLE": "table",
}
MULTI_OBJECT = re.compile(
    r"\b(set of|pair of|pack of|[2-9][ -]pack|[2-9][ -]piece|swatch)\b",
    flags=re.IGNORECASE,
)


def _english(row: dict[str, Any], key: str) -> str:
    for value in row.get(key, []) or []:
        if value.get("language_tag") == "en_US" and str(value.get("value", "")).strip():
            return str(value["value"]).strip()
    return ""


def _product_types(row: dict[str, Any]) -> set[str]:
    result = set()
    for value in row.get("product_type", []) or []:
        result.add(str(value.get("value", "") if isinstance(value, dict) else value))
    return result


def _attribute(value: str, *, words: int = 3) -> str:
    value = re.sub(r"[^A-Za-z -]+", " ", value).lower()
    tokens = [token for token in value.split() if len(token) > 1]
    return " ".join(tokens[:words])


def _caption(row: dict[str, Any], category: str) -> str:
    color = _attribute(_english(row, "color"), words=3)
    material = _attribute(_english(row, "material") or _english(row, "fabric_type"), words=2)
    words: list[str] = []
    for token in f"{color} {material}".split():
        if token not in words:
            words.append(token)
    description = " ".join(words + [CATEGORY_LABELS[category]])
    article = "an" if description[:1] in "aeiou" else "a"
    return f"{article} {description} on a plain neutral background"


def _trim_product(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    corners = (
        image.getpixel((0, 0)), image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)), image.getpixel((image.width - 1, image.height - 1)),
    )
    background = tuple(sorted(pixel[channel] for pixel in corners)[len(corners) // 2] for channel in range(3))
    difference = ImageChops.difference(image, Image.new("RGB", image.size, background)).convert("L")
    foreground = difference.point(lambda value: 255 if value >= 14 else 0)
    box = foreground.getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    margin = max(2, round(0.05 * max(right - left, bottom - top)))
    return image.crop((
        max(0, left - margin), max(0, top - margin),
        min(image.width, right + margin), min(image.height, bottom + margin),
    ))


def _clean_image(source: Path, output: Path, size: int, color: tuple[int, int, int]) -> None:
    with Image.open(source) as handle:
        product = _trim_product(ImageOps.exif_transpose(handle))
        product.thumbnail((round(size * 0.84), round(size * 0.84)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), color)
        position = ((size - product.width) // 2, (size - product.height) // 2)
        canvas.paste(product, position)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, subsampling=0)


def _contact_sheet(output_dir: Path, rows: list[dict[str, Any]], categories: list[str]) -> None:
    tile, label_height, columns = 128, 18, 8
    canvas = Image.new("RGB", (columns * tile, len(categories) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, category in enumerate(categories):
        label = CATEGORY_LABELS[category]
        options = [row for row in rows if row["category"] == label]
        selected = [options[round(i * (len(options) - 1) / (columns - 1))] for i in range(columns)]
        top = row_index * (tile + label_height)
        draw.text((4, top + 2), f"{category} / {label}", fill="black")
        for column, row in enumerate(selected):
            with Image.open(output_dir / row["image_path"]) as handle:
                image = ImageOps.fit(handle.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
            canvas.paste(image, (column * tile, top + label_height))
    canvas.save(output_dir / "contact_sheet.jpg", quality=92, subsampling=0)


def _listings(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((root / "listings" / "metadata").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def _image_index(root: Path) -> dict[str, dict[str, str]]:
    with gzip.open(root / "images" / "metadata" / "images.csv.gz", "rt", newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def prepare(
    abo_root: Path,
    output_dir: Path,
    categories: list[str],
    *,
    train_instances: int = 200,
    val_instances: int = 25,
    test_instances: int = 25,
    image_size: int = 224,
    seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if not (abo_root / "LICENSE-CC-BY-4.0.txt").is_file():
        raise FileNotFoundError("ABO CC BY 4.0 license file is required")
    unknown = set(categories) - CATEGORY_LABELS.keys()
    if unknown:
        raise ValueError(f"Unsupported T12 ABO categories: {sorted(unknown)}")

    image_rows = _image_index(abo_root)
    image_root = abo_root / "images" / "small"
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _listings(abo_root):
        category = next((value for value in categories if value in _product_types(row)), None)
        if category is None:
            continue
        title = _english(row, "item_name")
        image = image_rows.get(str(row.get("main_image_id", "")))
        if not title or image is None or MULTI_OBJECT.search(title):
            continue
        width, height = int(image["width"]), int(image["height"])
        if min(width, height) < 128 or max(width, height) / min(width, height) > 1.8:
            continue
        image_path = image_root / image["path"]
        if not image_path.is_file():
            continue
        candidates[category].append({
            "item_id": str(row["item_id"]),
            "image_path": image_path,
            "caption": _caption(row, category),
            "source_title": title,
        })

    needed = train_instances + val_instances + test_instances
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    audit_counts: dict[str, int] = {}
    split_counts = (("train", train_instances), ("val", val_instances), ("test", test_instances))
    for category in categories:
        rows = sorted(candidates[category], key=lambda value: value["item_id"])
        random.Random(f"{seed}:{category}").shuffle(rows)
        audit_counts[category] = len(rows)
        if len(rows) < needed:
            raise ValueError(f"{category} has {len(rows)} eligible products; {needed} required")
        cursor = 0
        for split, count in split_counts:
            for row in rows[cursor : cursor + count]:
                item_id = row["item_id"]
                digest = hashlib.sha256(f"{category}:{item_id}".encode()).digest()
                relative = Path("images") / split / category.lower() / f"{item_id}.jpg"
                _clean_image(row["image_path"], output_dir / relative, image_size, PALETTE[digest[0] % len(PALETTE)])
                manifests[split].append({
                    "sample_id": f"{category.lower()}-{item_id}",
                    "sequence_id": item_id,
                    "category": CATEGORY_LABELS[category],
                    "caption": row["caption"],
                    "image_path": relative.as_posix(),
                    "license": LICENSE,
                    "source_url": SOURCE_URL,
                    "source_title": row["source_title"],
                    "modified": "trimmed and centered on a deterministic neutral 224x224 canvas",
                })
            cursor += count

    for split, rows in manifests.items():
        write_manifest(output_dir / f"{split}.jsonl", rows)
    _contact_sheet(output_dir, manifests["train"], categories)
    summary = validate_split_contract(output_dir)
    summary.update({
        "source_dataset": "Amazon Berkeley Objects catalog images-small",
        "source_root": str(abo_root.resolve()),
        "source_url": SOURCE_URL,
        "attribution": "Amazon.com and the ABO dataset creators",
        "seed": seed,
        "eligible_products": audit_counts,
        "one_main_image_per_product": True,
        "image_preprocessing": "corner-background trim; <=84% canvas; deterministic neutral background",
    })
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the curated ABO T12 subset")
    parser.add_argument("--abo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", default="SHOES,CHAIR,LAMP,TABLE")
    parser.add_argument("--train-instances", type=int, default=200)
    parser.add_argument("--val-instances", type=int, default=25)
    parser.add_argument("--test-instances", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare(
        args.abo_root.resolve(), args.output_dir.resolve(),
        [value.strip() for value in args.categories.split(",") if value.strip()],
        train_instances=args.train_instances, val_instances=args.val_instances,
        test_instances=args.test_instances, image_size=args.image_size, seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
