from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .io_utils import sha256_paths, write_json


@dataclass(frozen=True)
class BDDRecord:
    sample_id: str
    image_path: Path
    split: str
    labels: tuple[dict[str, Any], ...]
    drivable_mask_path: Path | None = None
    lane_mask_path: Path | None = None


class BDD100KSpatialDataset(Dataset):
    """Front RGB plus light road-structure targets.

    Geometry is deterministically resized to 224x224. Polygon coordinates and
    box2d annotations are rasterized in their original image coordinates before
    nearest-neighbour mask resizing.
    """

    def __init__(
        self,
        records: list[BDDRecord],
        image_size: int,
        lane_width: int,
        participant_categories: tuple[str, ...],
    ) -> None:
        self.records = records
        self.image_size = int(image_size)
        self.lane_width = int(lane_width)
        self.participant_categories = set(participant_categories)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        targets = rasterize_targets(
            image.size,
            record.labels,
            lane_width=self.lane_width,
            participant_categories=self.participant_categories,
            drivable_mask_path=record.drivable_mask_path,
            lane_mask_path=record.lane_mask_path,
        )
        resized = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        output_targets = torch.stack(
            [
                _resize_mask(targets["drivable"], self.image_size),
                _resize_mask(targets["lane"], self.image_size),
                _resize_mask(targets["road_participant"], self.image_size),
            ],
            dim=0,
        )
        return {
            "image": resized,
            "targets": output_targets,
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "split": record.split,
        }


def collate_bdd(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "targets": torch.stack([item["targets"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def build_bdd_records(settings: Any, split: str) -> list[BDDRecord]:
    image_root = settings.bdd_root / settings.bdd_image_dir / split
    if not image_root.is_dir():
        discovered = sorted(
            str(path.relative_to(settings.bdd_root))
            for path in settings.bdd_root.glob("**/*")
            if path.is_dir() and path.name == split
        ) if settings.bdd_root.is_dir() else []
        raise FileNotFoundError(
            f"BDD100K image split is missing: {image_root}. "
            f"Discovered candidate '{split}' directories: {discovered[:20]}. "
            "Expected the official images/100k/{train,val} layout."
        )
    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise RuntimeError(f"No BDD100K images found under {image_root}")
    annotations, annotation_paths = load_annotation_index(settings)
    drivable_root = _optional_root(settings.bdd_root, settings.bdd_drivable_mask_dir)
    lane_root = _optional_root(settings.bdd_root, settings.bdd_lane_mask_dir)
    records: list[BDDRecord] = []
    missing_aux = 0
    for image_path in images:
        labels = tuple(annotations.get(image_path.name, ()))
        drivable = _find_mask(drivable_root, split, image_path.stem)
        lane = _find_mask(lane_root, split, image_path.stem)
        if not labels and drivable is None and lane is None:
            missing_aux += 1
        records.append(
            BDDRecord(
                sample_id=f"{split}/{image_path.stem}",
                image_path=image_path,
                split=split,
                labels=labels,
                drivable_mask_path=drivable,
                lane_mask_path=lane,
            )
        )
    if settings.bdd_require_auxiliary_labels and missing_aux:
        raise RuntimeError(
            f"{missing_aux}/{len(records)} BDD100K {split} images have no lane, "
            "drivable, or detection annotations. Configure official annotation "
            "JSON/PNG mask paths, or explicitly set require_auxiliary_labels=false "
            "for a feature-distillation-only diagnostic."
        )
    limit = (
        settings.bdd_train_limit
        if split == settings.bdd_train_split
        else settings.bdd_test_limit
    )
    if limit is not None:
        records = records[:limit]
    metadata = {
        "dataset": "BDD100K",
        "split": split,
        "samples": len(records),
        "images_without_auxiliary_labels_before_limit": missing_aux,
        "image_root": str(image_root),
        "annotation_files": [str(path) for path in annotation_paths],
        "annotation_identity_sha256": (
            sha256_paths(annotation_paths) if annotation_paths else None
        ),
        "drivable_mask_root": str(drivable_root) if drivable_root else None,
        "lane_mask_root": str(lane_root) if lane_root else None,
        "resize": [settings.image_size, settings.image_size],
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
    }
    write_json(
        settings.artifact_cache_dir / "manifests" / f"bdd100k_{split}.json",
        metadata,
    )
    return records


def load_annotation_index(
    settings: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    paths: list[Path] = []
    for value in settings.bdd_annotation_jsons:
        path = Path(value)
        if not path.is_absolute():
            path = settings.bdd_root / path
        if path.is_file():
            paths.append(path.resolve())
    if not paths and settings.bdd_root.is_dir():
        paths = sorted(
            path.resolve()
            for path in settings.bdd_root.glob("**/*.json")
            if "label" in path.name.lower()
        )
    index: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload if isinstance(payload, list) else payload.get("frames", [])
        if not isinstance(frames, list):
            raise RuntimeError(f"BDD annotation root must contain a frame list: {path}")
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            name = frame.get("name") or frame.get("filename")
            if not name:
                continue
            labels = frame.get("labels", [])
            if isinstance(labels, list):
                index.setdefault(Path(str(name)).name, []).extend(labels)
    return index, paths


def rasterize_targets(
    image_size: tuple[int, int],
    labels: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    lane_width: int,
    participant_categories: set[str],
    drivable_mask_path: Path | None = None,
    lane_mask_path: Path | None = None,
) -> dict[str, Image.Image]:
    width, height = image_size
    drivable = _load_binary_mask(drivable_mask_path, image_size)
    lane = _load_binary_mask(lane_mask_path, image_size)
    participant = Image.new("L", image_size, 0)
    draw_drivable = ImageDraw.Draw(drivable)
    draw_lane = ImageDraw.Draw(lane)
    draw_participant = ImageDraw.Draw(participant)
    for label in labels:
        category = str(label.get("category", "")).strip().lower()
        poly2d = label.get("poly2d") or []
        if category in {"drivable area", "drivable", "road"}:
            for polygon in poly2d:
                vertices = _vertices(polygon)
                if len(vertices) >= 3:
                    draw_drivable.polygon(vertices, fill=255)
        elif category == "lane" or "lane" in category:
            for polygon in poly2d:
                vertices = _vertices(polygon)
                if len(vertices) >= 2:
                    closed = bool(polygon.get("closed", False))
                    if closed and len(vertices) >= 3:
                        draw_lane.polygon(vertices, fill=255)
                    else:
                        draw_lane.line(vertices, fill=255, width=max(1, lane_width))
        if category in participant_categories:
            box = label.get("box2d")
            if isinstance(box, dict):
                coordinates = [
                    float(box.get("x1", 0)),
                    float(box.get("y1", 0)),
                    float(box.get("x2", 0)),
                    float(box.get("y2", 0)),
                ]
                coordinates[0] = min(max(coordinates[0], 0), width - 1)
                coordinates[2] = min(max(coordinates[2], 0), width - 1)
                coordinates[1] = min(max(coordinates[1], 0), height - 1)
                coordinates[3] = min(max(coordinates[3], 0), height - 1)
                draw_participant.rectangle(coordinates, fill=255)
    return {
        "drivable": drivable,
        "lane": lane,
        "road_participant": participant,
    }


def _vertices(polygon: Any) -> list[tuple[float, float]]:
    if not isinstance(polygon, dict):
        return []
    vertices = polygon.get("vertices", [])
    result = []
    for vertex in vertices:
        if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
            result.append((float(vertex[0]), float(vertex[1])))
    return result


def _load_binary_mask(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None:
        return Image.new("L", size, 0)
    with Image.open(path) as source:
        mask = source.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    array = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(array, mode="L")


def _resize_mask(mask: Image.Image, size: int) -> torch.Tensor:
    resized = mask.resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32))


def _optional_root(root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else root / path
    return path.resolve() if path.is_dir() else None


def _find_mask(root: Path | None, split: str, stem: str) -> Path | None:
    if root is None:
        return None
    candidates = [
        root / split / f"{stem}.png",
        root / f"{stem}.png",
        root / split / f"{stem}_drivable_id.png",
        root / split / f"{stem}_lane_id.png",
    ]
    return next((path for path in candidates if path.is_file()), None)
