"""Build the small, clean, identity-disjoint T12 subset from exported uCO3D frames.

The official uCO3D downloader remains responsible for acquiring selected RGB
and mask modalities.  This script consumes a simple CSV index over those local
files, crops the single foreground object, composites a neutral background and
writes the three audited JSONL manifests used by T12.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .dataset import CC_BY_4_IDENTIFIERS, validate_split_contract, write_manifest


PALETTE = ((244, 244, 242), (238, 241, 246), (245, 241, 234), (235, 242, 239))
INDEX_FIELDS = {
    "sequence_id", "category", "caption", "frame_path", "mask_path", "source_url", "license",
}


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:48] or "object"


def _resolve(path: str, root: Path) -> Path:
    value = Path(path).expanduser()
    return (root / value).resolve() if not value.is_absolute() else value.resolve()


def _evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) < count:
        return []
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[index] for index in indices]


def _clean_frame(frame: Path, mask: Path, output: Path, size: int, color: tuple[int, int, int]) -> None:
    with Image.open(frame) as source_handle, Image.open(mask) as mask_handle:
        source = ImageOps.exif_transpose(source_handle).convert("RGB")
        alpha = ImageOps.exif_transpose(mask_handle).convert("L")
        if alpha.size != source.size:
            alpha = alpha.resize(source.size, Image.Resampling.NEAREST)
        alpha = alpha.point(lambda value: 255 if value >= 128 else 0)
        box = alpha.getbbox()
        if box is None:
            raise ValueError(f"Empty foreground mask: {mask}")
        left, top, right, bottom = box
        margin = max(4, round(0.12 * max(right - left, bottom - top)))
        box = (
            max(0, left - margin), max(0, top - margin),
            min(source.width, right + margin), min(source.height, bottom + margin),
        )
        foreground, cropped_alpha = source.crop(box), alpha.crop(box)
        foreground.thumbnail((round(size * 0.74), round(size * 0.74)), Image.Resampling.LANCZOS)
        cropped_alpha = cropped_alpha.resize(foreground.size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), color)
        position = ((size - foreground.width) // 2, (size - foreground.height) // 2)
        canvas.paste(foreground, position, cropped_alpha)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, subsampling=0)


def prepare(
    index_csv: Path,
    output_dir: Path,
    categories: list[str],
    *,
    train_instances: int = 160,
    val_instances: int = 20,
    test_instances: int = 20,
    frames_per_instance: int = 10,
    image_size: int = 224,
    seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    source_root = index_csv.parent.resolve()
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with index_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = INDEX_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Source index missing {sorted(missing)}")
        for row in reader:
            if row["category"] not in categories:
                continue
            if row["license"] not in CC_BY_4_IDENTIFIERS:
                raise ValueError(f"Sequence {row['sequence_id']} is not explicitly CC BY 4.0")
            row["frame_path"] = str(_resolve(row["frame_path"], source_root))
            row["mask_path"] = str(_resolve(row["mask_path"], source_root))
            grouped[row["category"]][row["sequence_id"]].append(row)

    needed = train_instances + val_instances + test_instances
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_counts = (("train", train_instances), ("val", val_instances), ("test", test_instances))
    for category in categories:
        sequences = [
            sequence for sequence, rows in grouped.get(category, {}).items()
            if len(rows) >= frames_per_instance
        ]
        random.Random(f"{seed}:{category}").shuffle(sequences)
        if len(sequences) < needed:
            raise ValueError(f"{category} has {len(sequences)} eligible sequences; {needed} required")
        cursor = 0
        for split, sequence_count in split_counts:
            for sequence in sequences[cursor : cursor + sequence_count]:
                rows = sorted(grouped[category][sequence], key=lambda row: row["frame_path"])
                for frame_index, row in enumerate(_evenly_spaced(rows, frames_per_instance)):
                    caption = row["caption"].strip()
                    if not caption:
                        raise ValueError(f"Empty caption for sequence {sequence}")
                    sequence_digest = hashlib.sha256(sequence.encode()).hexdigest()[:12]
                    sample_id = f"{_slug(category)}-{sequence_digest}-{frame_index:02d}"
                    digest = hashlib.sha256(sample_id.encode()).digest()
                    color = PALETTE[digest[0] % len(PALETTE)]
                    relative = Path("images") / split / _slug(category) / f"{sample_id}.jpg"
                    _clean_frame(
                        Path(row["frame_path"]), Path(row["mask_path"]),
                        output_dir / relative, image_size, color,
                    )
                    manifests[split].append({
                        "sample_id": sample_id,
                        "sequence_id": sequence,
                        "category": category,
                        "caption": caption,
                        "image_path": relative.as_posix(),
                        "license": "CC BY 4.0",
                        "source_url": row["source_url"],
                    })
            cursor += sequence_count

    for split, rows in manifests.items():
        write_manifest(output_dir / f"{split}.jsonl", rows)
    summary = validate_split_contract(output_dir)
    summary.update({
        "source_index": str(index_csv.resolve()),
        "seed": seed,
        "frames_per_instance": frames_per_instance,
        "image_preprocessing": "foreground mask crop; 12% margin; <=74% canvas; deterministic neutral background",
    })
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the curated uCO3D T12 subset")
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", required=True, help="Comma-separated exact uCO3D category names")
    parser.add_argument("--train-instances", type=int, default=160)
    parser.add_argument("--val-instances", type=int, default=20)
    parser.add_argument("--test-instances", type=int, default=20)
    parser.add_argument("--frames-per-instance", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare(
        args.index_csv.resolve(), args.output_dir.resolve(),
        [value.strip() for value in args.categories.split(",") if value.strip()],
        train_instances=args.train_instances, val_instances=args.val_instances,
        test_instances=args.test_instances, frames_per_instance=args.frames_per_instance,
        image_size=args.image_size, seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
