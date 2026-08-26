from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .settings import load_settings


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("--roi must be left,top,right,bottom")
    left, top, right, bottom = parts
    if min(left, top) < 0 or right <= left or bottom <= top:
        raise ValueError(f"Invalid ROI {parts}")
    return parts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty hardware prediction table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_capture(directory: Path, key: str) -> Path:
    matches = []
    for suffix in (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        candidate = directory / f"{key}{suffix}"
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CCD image for key={key!r} in {directory}, found {matches}"
        )
    return matches[0]


def evaluate_directory(
    *,
    config: Path,
    manifest: Path,
    ccd_dir: Path,
    output_dir: Path,
    roi: tuple[int, int, int, int] | None,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> dict[str, object]:
    settings = load_settings(config)
    rows = _read_manifest(manifest)
    if not rows:
        raise ValueError(f"Hardware manifest is empty: {manifest}")
    keys = [row.get("key", "") for row in rows]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Hardware manifest keys must be non-empty and unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    correct = 0
    confusion = np.zeros((4, 4), dtype=np.int64)
    for row in rows:
        key = row["key"]
        source = _find_capture(ccd_dir, key)
        with Image.open(source) as opened:
            image = opened.convert("L")
        if roi is not None:
            if roi[2] > image.width or roi[3] > image.height:
                raise ValueError(f"ROI {roi} exceeds {source.name} size {image.size}")
            image = image.crop(roi)
        if flip_vertical:
            image = ImageOps.flip(image)
        if flip_horizontal:
            image = ImageOps.mirror(image)
        image = image.resize(
            (settings.ccd_target_size, settings.ccd_target_size),
            resample=Image.Resampling.BOX,
        )
        array = np.asarray(image, dtype=np.float32)
        # No background subtraction. Dividing by total energy makes the four
        # detector scores invariant to one global exposure/gain factor.
        total = max(float(array.sum()), settings.loss_eps)
        scores = []
        for left, top, right, bottom in settings.detector_bounds():
            scores.append(float(array[top:bottom, left:right].sum()) / total)
        prediction = int(np.argmax(scores))
        label = int(row["label"])
        is_correct = prediction == label
        correct += int(is_correct)
        confusion[label, prediction] += 1
        processed_path = output_dir / "processed_478" / f"{key}.png"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(processed_path, optimize=True)
        result_rows.append(
            {
                "key": key,
                "label": label,
                "prediction": prediction,
                "correct": is_correct,
                "score_0": scores[0],
                "score_1": scores[1],
                "score_2": scores[2],
                "score_3": scores[3],
                "source": str(source),
                "processed": str(processed_path),
            }
        )
    _write_csv(output_dir / "hardware_predictions.csv", result_rows)
    per_class_accuracy = {}
    for label in range(4):
        count = int(confusion[label].sum())
        per_class_accuracy[str(label)] = (
            None if count == 0 else float(confusion[label, label] / count)
        )
    profiles = sorted({row.get("profile", "unspecified") for row in rows})
    summary = {
        "samples": len(result_rows),
        "correct": correct,
        "accuracy": correct / max(1, len(result_rows)),
        "roi_xyxy": None if roi is None else list(roi),
        "flip_vertical": flip_vertical,
        "flip_horizontal": flip_horizontal,
        "target_size": settings.ccd_target_size,
        "resize": "PIL BOX area resampling",
        "background_subtraction": False,
        "profiles": profiles,
        "selection_policy": sorted(
            {row.get("selection_policy", "unspecified") for row in rows}
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": per_class_accuracy,
    }
    _write_json(output_dir / "hardware_metrics.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate captured MNIST-4 CCD frames")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ccd-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi", default=None, help="left,top,right,bottom in raw CCD pixels")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    args = parser.parse_args()
    summary = evaluate_directory(
        config=Path(args.config).expanduser().resolve(),
        manifest=Path(args.manifest).expanduser().resolve(),
        ccd_dir=Path(args.ccd_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        roi=parse_roi(args.roi),
        flip_vertical=args.flip_vertical,
        flip_horizontal=args.flip_horizontal,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
