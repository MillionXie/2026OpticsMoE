from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .io_utils import write_json
from .modeling import FrozenQwenVisionTeacher, preprocess_vision


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TeacherMaskCache:
    """Memory-mapped FP16 teacher logits, indexed by stable FSS sample id."""

    def __init__(self, directory: Path, split: str, expected: dict[str, Any]) -> None:
        metadata_path = directory / f"{split}_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Teacher mask cache is missing for {split}: {metadata_path}. "
                "Run --phase cache_teacher_masks first."
            )
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Teacher mask cache metadata mismatch for {split}: {mismatches}. "
                "Delete the cache and rebuild it."
            )
        self.metadata = metadata
        self.ids = list(metadata["sample_ids"])
        self.index = {sample_id: index for index, sample_id in enumerate(self.ids)}
        if len(self.index) != len(self.ids):
            raise RuntimeError("Teacher mask cache contains duplicate sample ids")
        self.array = np.load(directory / f"{split}_logits.npy", mmap_mode="r")
        expected_shape = (len(self.ids), 1, int(metadata["image_size"]), int(metadata["image_size"]))
        if self.array.shape != expected_shape or self.array.dtype != np.float16:
            raise RuntimeError(
                f"Teacher mask array is {self.array.shape}/{self.array.dtype}, "
                f"expected {expected_shape}/float16"
            )

    def fetch(self, sample_ids: list[str], device: torch.device) -> torch.Tensor:
        missing = [sample_id for sample_id in sample_ids if sample_id not in self.index]
        if missing:
            raise KeyError(f"Teacher mask cache has no samples: {missing[:10]}")
        values = np.stack(
            [np.asarray(self.array[self.index[sample_id]], dtype=np.float32) for sample_id in sample_ids]
        )
        return torch.from_numpy(values).to(device, non_blocking=True)


@torch.no_grad()
def build_teacher_mask_cache(
    teacher: FrozenQwenVisionTeacher,
    processor: Any,
    loader: DataLoader,
    directory: Path,
    *,
    split: str,
    settings: Any,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    count = len(loader.dataset)
    output_path = directory / f"{split}_logits.npy"
    values = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(count, 1, settings.image_size, settings.image_size),
    )
    sample_ids: list[str] = []
    offset = 0
    teacher.eval()
    for batch_index, batch in enumerate(loader, start=1):
        inputs = preprocess_vision(processor, batch["images"], device)
        logits, _ = teacher(inputs["pixel_values"], inputs["image_grid_thw"])
        batch_size = logits.shape[0]
        values[offset:offset + batch_size] = logits.detach().float().cpu().numpy().astype(np.float16)
        sample_ids.extend(batch["sample_ids"])
        offset += batch_size
        if batch_index % settings.log_interval_batches == 0 or offset == count:
            print(f"[teacher_mask_cache] {split} cached={offset:,}/{count:,}", flush=True)
    values.flush()
    if offset != count or len(sample_ids) != count:
        raise RuntimeError(f"Teacher mask cache wrote {offset}/{count} samples")
    metadata = {
        "dataset": "FSS-1000",
        "split": split,
        "samples": count,
        "sample_ids": sample_ids,
        "image_size": settings.image_size,
        "model_id": settings.model_id,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "dtype": "float16",
        "augmentation": False,
    }
    write_json(directory / f"{split}_metadata.json", metadata)
    return metadata


def expected_cache_identity(settings: Any, checkpoint_path: Path) -> dict[str, Any]:
    return {
        "dataset": "FSS-1000",
        "image_size": settings.image_size,
        "model_id": settings.model_id,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "teacher_checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "augmentation": False,
    }
