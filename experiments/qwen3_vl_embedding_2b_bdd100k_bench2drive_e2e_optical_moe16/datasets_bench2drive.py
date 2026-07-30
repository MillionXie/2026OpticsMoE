from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .io_utils import write_json


@dataclass(frozen=True)
class Bench2DriveRecord:
    sample_id: str
    route_id: str
    frame_id: str
    image_path: Path
    annotation_path: Path
    speed: float
    command: int
    target_x_local: float
    target_y_local: float
    steer: float
    throttle: float
    brake: float


class Bench2DriveBCDataset(Dataset):
    def __init__(self, records: list[Bench2DriveRecord], image_size: int) -> None:
        self.records = records
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        return {
            "image": image,
            "speed": record.speed,
            "command": record.command,
            "target_point": (record.target_x_local, record.target_y_local),
            "controls": (record.steer, record.throttle, record.brake),
            "sample_id": record.sample_id,
            "route_id": record.route_id,
            "image_path": str(record.image_path),
        }


def collate_bench2drive(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "speed": torch.tensor([item["speed"] for item in batch], dtype=torch.float32),
        "command": torch.tensor(
            [item["command"] for item in batch], dtype=torch.long
        ),
        "target_point": torch.tensor(
            [item["target_point"] for item in batch], dtype=torch.float32
        ),
        "controls": torch.tensor(
            [item["controls"] for item in batch], dtype=torch.float32
        ),
        "sample_ids": [item["sample_id"] for item in batch],
        "route_ids": [item["route_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def build_bench2drive_splits(
    settings: Any,
) -> tuple[list[Bench2DriveRecord], list[Bench2DriveRecord]]:
    root = settings.bench2drive_root
    if not root.is_dir():
        raise FileNotFoundError(
            f"Bench2Drive root does not exist: {root}. Extract an official "
            "Mini/Base/Full dataset so each clip contains camera/rgb_front and anno."
        )
    annotation_paths = sorted(root.glob("**/anno/*.json.gz"))
    if not annotation_paths:
        raise FileNotFoundError(
            f"No Bench2Drive anno/*.json.gz files found recursively under {root}. "
            "The official layout is <clip>/camera/rgb_front/*.jpg plus <clip>/anno/*.json.gz."
        )
    routes: dict[str, list[Path]] = {}
    for path in annotation_paths:
        route_root = path.parent.parent
        route_id = str(route_root.relative_to(root)).replace("\\", "/")
        routes.setdefault(route_id, []).append(path)
    route_ids = sorted(routes, key=lambda value: _stable_key(value, settings.random_seed))
    split_index = max(1, min(len(route_ids) - 1, round(len(route_ids) * settings.bench_train_fraction)))
    if len(route_ids) == 1:
        train_routes, test_routes = route_ids, route_ids
    else:
        train_routes, test_routes = route_ids[:split_index], route_ids[split_index:]
    train = _records_for_routes(root, routes, train_routes, settings.bench_frame_stride)
    test = _records_for_routes(root, routes, test_routes, settings.bench_frame_stride)
    if settings.bench_train_limit is not None:
        train = train[: settings.bench_train_limit]
    if settings.bench_test_limit is not None:
        test = test[: settings.bench_test_limit]
    overlap = {row.route_id for row in train} & {row.route_id for row in test}
    if overlap and len(route_ids) > 1:
        raise RuntimeError(f"Bench2Drive route leakage detected: {sorted(overlap)[:5]}")
    metadata = {
        "dataset": "Bench2Drive expert data",
        "root": str(root),
        "route_disjoint": len(route_ids) > 1,
        "route_count": len(route_ids),
        "train_routes": len({row.route_id for row in train}),
        "validation_routes": len({row.route_id for row in test}),
        "train_samples": len(train),
        "validation_samples": len(test),
        "split_policy": (
            "route-disjoint offline validation; not the official closed-loop test"
        ),
        "frame_stride": settings.bench_frame_stride,
        "seed": settings.random_seed,
        "official_fields_used": [
            "speed",
            "next_command",
            "x_target",
            "y_target",
            "x",
            "y",
            "theta",
            "steer",
            "throttle",
            "brake",
        ],
    }
    write_json(settings.bench_index_path, metadata)
    return train, test


def parse_annotation(
    annotation: dict[str, Any],
    *,
    sample_id: str,
    route_id: str,
    frame_id: str,
    image_path: Path,
    annotation_path: Path,
) -> Bench2DriveRecord:
    required = (
        "speed",
        "steer",
        "throttle",
        "brake",
        "x",
        "y",
        "theta",
        "x_target",
        "y_target",
    )
    missing = [key for key in required if key not in annotation]
    if missing:
        raise RuntimeError(
            f"Bench2Drive annotation {annotation_path} is missing {missing}; "
            f"available keys: {sorted(annotation)}"
        )
    command = int(
        annotation.get(
            "next_command",
            annotation.get("command_near", annotation.get("command_far", -1)),
        )
    )
    if command < 0:
        command = 4
    if command not in {1, 2, 3, 4, 5, 6}:
        raise RuntimeError(
            f"Unsupported Bench2Drive navigation command {command} in {annotation_path}"
        )
    theta = float(annotation["theta"])
    delta = np.array(
        [
            float(annotation["x_target"]) - float(annotation["x"]),
            float(annotation["y_target"]) - float(annotation["y"]),
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float64,
    )
    local = rotation @ delta
    controls = (
        max(-1.0, min(1.0, float(annotation["steer"]))),
        max(0.0, min(1.0, float(annotation["throttle"]))),
        max(0.0, min(1.0, float(annotation["brake"]))),
    )
    return Bench2DriveRecord(
        sample_id=sample_id,
        route_id=route_id,
        frame_id=frame_id,
        image_path=image_path,
        annotation_path=annotation_path,
        speed=max(0.0, float(annotation["speed"])),
        command=command - 1,
        target_x_local=float(local[0]),
        target_y_local=float(local[1]),
        steer=controls[0],
        throttle=controls[1],
        brake=controls[2],
    )


def _records_for_routes(
    root: Path,
    routes: dict[str, list[Path]],
    route_ids: list[str],
    stride: int,
) -> list[Bench2DriveRecord]:
    records: list[Bench2DriveRecord] = []
    for route_id in sorted(route_ids):
        for ordinal, annotation_path in enumerate(sorted(routes[route_id])):
            if ordinal % stride:
                continue
            route_root = annotation_path.parent.parent
            frame_id = annotation_path.name.removesuffix(".json.gz")
            candidates = [
                route_root / "camera" / "rgb_front" / f"{frame_id}.jpg",
                route_root / "camera" / "rgb_front" / f"{frame_id}.png",
            ]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(
                    f"Front RGB for {annotation_path} is missing. Tried: "
                    f"{[str(path) for path in candidates]}"
                )
            with gzip.open(annotation_path, "rt", encoding="utf-8") as handle:
                annotation = json.load(handle)
            records.append(
                parse_annotation(
                    annotation,
                    sample_id=f"{route_id}/{frame_id}",
                    route_id=route_id,
                    frame_id=frame_id,
                    image_path=image_path,
                    annotation_path=annotation_path,
                )
            )
    return records


def normalized_driving_state(
    speed: torch.Tensor,
    command: torch.Tensor,
    target_point: torch.Tensor,
    *,
    speed_scale: float,
    target_clip: float,
    num_commands: int = 6,
) -> torch.Tensor:
    speed_feature = (speed.float() / float(speed_scale)).clamp(0.0, 2.0).unsqueeze(-1)
    one_hot = torch.nn.functional.one_hot(
        command.long(), num_classes=num_commands
    ).float()
    target = target_point.float().clamp(-target_clip, target_clip) / float(target_clip)
    return torch.cat([speed_feature, one_hot, target], dim=-1)


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
