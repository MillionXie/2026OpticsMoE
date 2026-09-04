from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset


FRAME_FRACTIONS = (0.10, 0.37, 0.63, 0.90)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    video_path: str
    split: str
    spatial: float
    temporal: float


def read_manifest(path: str | Path) -> list[ManifestRow]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[ManifestRow] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise RuntimeError(f"Duplicate sample id {sample_id!r}")
        seen.add(sample_id)
        split = str(row["split"]).strip().lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Only train/test are allowed, got {split!r}")
        result.append(ManifestRow(sample_id, str(row["video_path"]), split, float(row["spatial"]), float(row["temporal"])))
    if not result:
        raise RuntimeError(f"Manifest is empty: {source}")
    return result


def decode_four_frames(path: Path, *, size: int, crop_fraction: float) -> torch.Tensor:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for LGVQ video decoding") from error
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")
    output: list[torch.Tensor] = []
    for fraction in FRAME_FRACTIONS:
        position = min(count - 1, max(0, round((count - 1) * fraction)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed to decode frame {position}: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        side = max(2, round(min(height, width) * crop_fraction))
        top, left = (height - side) // 2, (width - side) // 2
        resized = cv2.resize(rgb[top : top + side, left : left + side], (size, size), interpolation=cv2.INTER_AREA)
        output.append(torch.from_numpy(resized.copy()).permute(2, 0, 1))
    capture.release()
    return torch.stack(output).to(torch.uint8)


def build_frame_cache(manifest_path: Path, output_path: Path, *, frame_size: int, crop_fraction: float) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    frames = torch.empty(len(rows), 4, 3, frame_size, frame_size, dtype=torch.uint8)
    for index, row in enumerate(rows):
        frames[index] = decode_four_frames(Path(row.video_path), size=frame_size, crop_fraction=crop_fraction)
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f"[frame-cache] {index + 1}/{len(rows)}", flush=True)
    payload = {
        "schema_version": 1,
        "feature_contract": "raw_rgb_uint8_four_frames_center65_224_v1",
        "frames": frames,
        "frame_fractions": FRAME_FRACTIONS,
        "frame_size": frame_size,
        "crop_fraction": crop_fraction,
        "sample_ids": [row.sample_id for row in rows],
        "video_paths": [row.video_path for row in rows],
        "splits": [row.split for row in rows],
        "targets": torch.tensor([[row.spatial, row.temporal] for row in rows], dtype=torch.float32),
        "target_names": ["spatial", "temporal"],
        "alignment_target_present": False,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return cache_report(payload, output_path)


def load_frame_cache(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported frame cache: {source}")
    required = {"frames", "sample_ids", "video_paths", "splits", "targets", "target_names"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Frame cache is missing {sorted(missing)}")
    frames = payload["frames"]
    count = len(payload["sample_ids"])
    if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8 or frames.ndim != 5:
        raise ValueError("frames must be uint8 [N,4,3,H,W]")
    if tuple(frames.shape[:3]) != (count, 4, 3) or frames.shape[-1] != frames.shape[-2]:
        raise ValueError(f"Invalid frame tensor shape {tuple(frames.shape)}")
    if tuple(payload["targets"].shape) != (count, 2) or list(payload["target_names"]) != ["spatial", "temporal"]:
        raise ValueError("Targets must be [N,2] spatial/temporal")
    if any(len(payload[key]) != count for key in ("video_paths", "splits")):
        raise ValueError("Frame cache metadata lengths differ")
    if payload.get("alignment_target_present", False):
        raise ValueError("Image-text alignment is forbidden")
    return payload


def load_training_soft_targets(path: str | Path, expected_train_sample_ids: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and strictly align a training-only prediction file by sample id."""
    source = Path(path).expanduser().resolve()
    raw = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Training soft targets must be a .pt mapping")
    missing = {"sample_ids", "predictions", "target_names"}.difference(raw)
    if missing:
        raise ValueError(f"Training soft-target file is missing {sorted(missing)}")
    sample_ids = [str(value) for value in raw["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Training soft-target sample_ids contain duplicates")
    target_names = list(raw["target_names"])
    if target_names != ["spatial", "temporal"]:
        raise ValueError("Training soft-target target_names must be ['spatial', 'temporal']")
    predictions = torch.as_tensor(raw["predictions"], dtype=torch.float32).cpu()
    if predictions.shape != (len(sample_ids), 2):
        raise ValueError(f"Training soft-target predictions must be [N,2], got {tuple(predictions.shape)}")
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("Training soft-target predictions contain non-finite values")
    expected = [str(value) for value in expected_train_sample_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("Frame-cache training sample_ids contain duplicates")
    missing_ids = sorted(set(expected).difference(sample_ids))
    unexpected_ids = sorted(set(sample_ids).difference(expected))
    if missing_ids or unexpected_ids:
        raise ValueError(
            "Training soft-target IDs do not exactly match the train split: "
            f"missing={missing_ids[:5]} ({len(missing_ids)}), "
            f"unexpected={unexpected_ids[:5]} ({len(unexpected_ids)})"
        )
    lookup = {sample_id: predictions[index] for index, sample_id in enumerate(sample_ids)}
    aligned = torch.stack([lookup[sample_id] for sample_id in expected])
    provenance = {
        "usage": "training_only_scalar_soft_targets",
        "path": str(source),
        "sha256": file_sha256(source),
        "sample_count": len(expected),
        "target_names": target_names,
        "full_teacher_loaded_during_inference": False,
    }
    return {sample_id: aligned[index] for index, sample_id in enumerate(expected)}, provenance


def attach_training_soft_targets(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    sample_ids = [str(value) for value in payload["sample_ids"]]
    splits = [str(value) for value in payload["splits"]]
    train_ids = [sample_id for sample_id, split in zip(sample_ids, splits) if split == "train"]
    aligned, provenance = load_training_soft_targets(path, train_ids)
    soft_targets = torch.zeros(len(sample_ids), 2, dtype=torch.float32)
    present = torch.zeros(len(sample_ids), dtype=torch.bool)
    for index, (sample_id, split) in enumerate(zip(sample_ids, splits)):
        if split == "train":
            soft_targets[index] = aligned[sample_id]
            present[index] = True
    result = dict(payload)
    result["soft_targets"] = soft_targets
    result["soft_target_present"] = present
    result["training_soft_target_provenance"] = provenance
    return result


def cache_report(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    frames = payload["frames"]
    return {
        "path": str(path.resolve()),
        "shape": list(frames.shape),
        "dtype": str(frames.dtype),
        "storage_bytes": int(frames.numel() * frames.element_size()),
        "counts": {split: list(payload["splits"]).count(split) for split in ("train", "test")},
        "alignment_excluded": True,
    }


class LGVQFrameDataset(Dataset):
    def __init__(self, payload: Mapping[str, Any], split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        self.payload = payload
        self.indices = [index for index, value in enumerate(payload["splits"]) if value == split]
        if not self.indices:
            raise RuntimeError(f"No samples for split={split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.indices[index]
        item = {
            "frames": self.payload["frames"][source],
            "target": self.payload["targets"][source].float(),
            "sample_id": self.payload["sample_ids"][source],
            "video_path": self.payload["video_paths"][source],
        }
        if "soft_target_present" in self.payload and bool(self.payload["soft_target_present"][source]):
            item["soft_target"] = self.payload["soft_targets"][source].float()
        return item


__all__ = [
    "LGVQFrameDataset",
    "attach_training_soft_targets",
    "build_frame_cache",
    "cache_report",
    "load_frame_cache",
    "load_training_soft_targets",
    "read_manifest",
]
