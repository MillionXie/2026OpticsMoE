from __future__ import annotations

import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .io_utils import atomic_torch_save, torch_load, write_json
from .modeling import preprocess_vision
from .pca import FixedPCAProjection


CACHE_VERSION = 1


def expected_cache_identity(
    settings: Any,
    *,
    split: str,
    dataset_manifest_digest: str,
    pca_metadata: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dataset": "COCO 2017",
        "split": split,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "sample_count": int(sample_count),
        "model_id": settings.model_id,
        "source_hidden": "final_native_qwen_vision_block_pre_merger",
        "teacher_hidden_size": 1024,
        "target_dimension": 224,
        "pca_projection_sha256": pca_metadata.get("projection_sha256"),
        "processor_min_pixels": int(settings.processor_min_pixels),
        "processor_max_pixels": int(settings.processor_max_pixels),
        "image_size": int(settings.image_size),
        "resize_mode": settings.coco_resize_mode,
        "dtype": settings.teacher_cache_dtype,
        "pca_is_student_module": False,
    }


@torch.no_grad()
def build_teacher_target_cache(
    split: str,
    teacher: Any,
    projection: FixedPCAProjection,
    processor: Any,
    loader: Any,
    settings: Any,
    *,
    dataset_manifest_digest: str,
) -> dict[str, Any]:
    directory = settings.teacher_cache_root / split
    identity = expected_cache_identity(
        settings,
        split=split,
        dataset_manifest_digest=dataset_manifest_digest,
        pca_metadata=getattr(projection, "metadata", {}),
        sample_count=len(loader.dataset),
    )
    # Projection metadata is attached by run.py after loading. Keep this
    # explicit fallback for direct unit usage.
    identity["pca_projection_sha256"] = getattr(
        projection,
        "projection_sha256",
        identity["pca_projection_sha256"],
    )
    metadata_path = directory / "metadata.json"
    index_path = directory / "index.json"
    if settings.rebuild_teacher_cache and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _read_json(metadata_path)
    index = _read_json(index_path) or {}
    if metadata is not None:
        mismatches = {
            key: (metadata.get("identity", {}).get(key), expected)
            for key, expected in identity.items()
            if metadata.get("identity", {}).get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Teacher cache identity mismatch for {split}: {mismatches}. "
                "Set cache.rebuild=true or delete this split cache."
            )
        if metadata.get("status") == "complete":
            if len(index) != len(loader.dataset):
                raise RuntimeError(
                    f"Teacher cache says complete but index has {len(index)} "
                    f"entries for {len(loader.dataset)} samples"
                )
            print(
                f"[teacher_cache] {split} already complete: {len(index):,} samples",
                flush=True,
            )
            return metadata
    else:
        metadata = {
            "status": "building",
            "identity": identity,
            "shards": [],
            "cached_samples": 0,
        }
        write_json(metadata_path, metadata)
        write_json(index_path, index)

    existing_shards = sorted(directory.glob("shard_*.pt"))
    next_shard = len(existing_shards)
    pending_ids: list[str] = []
    pending_targets: list[torch.Tensor] = []
    pending_grids: list[torch.Tensor] = []
    dtype = torch.float16 if settings.teacher_cache_dtype == "float16" else torch.float32

    def flush() -> None:
        nonlocal next_shard
        if not pending_ids:
            return
        max_tokens = max(int(target.shape[0]) for target in pending_targets)
        targets = torch.zeros(
            len(pending_targets),
            max_tokens,
            224,
            dtype=dtype,
        )
        lengths = torch.tensor(
            [target.shape[0] for target in pending_targets],
            dtype=torch.long,
        )
        for row, target in enumerate(pending_targets):
            targets[row, : target.shape[0]] = target.to(dtype)
        grids = torch.stack(pending_grids).long()
        filename = f"shard_{next_shard:06d}.pt"
        atomic_torch_save(
            directory / filename,
            {
                "sample_ids": list(pending_ids),
                "targets": targets,
                "lengths": lengths,
                "image_grid_thw": grids,
            },
        )
        for row, sample_id in enumerate(pending_ids):
            index[sample_id] = {"shard": filename, "row": row}
        metadata["shards"].append(
            {
                "filename": filename,
                "samples": len(pending_ids),
                "max_tokens": max_tokens,
            }
        )
        metadata["cached_samples"] = len(index)
        write_json(index_path, index)
        write_json(metadata_path, metadata)
        next_shard += 1
        pending_ids.clear()
        pending_targets.clear()
        pending_grids.clear()

    for batch_index, batch in enumerate(loader, start=1):
        missing_rows = [
            index_in_batch
            for index_in_batch, sample_id in enumerate(batch["sample_ids"])
            if sample_id not in index
        ]
        if not missing_rows:
            continue
        images = [batch["images"][row] for row in missing_rows]
        sample_ids = [batch["sample_ids"][row] for row in missing_rows]
        inputs = preprocess_vision(processor, images, teacher.device)
        packed, lengths = teacher.extract_packed(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        projected = projection.encode(packed)
        offset = 0
        grids = inputs["image_grid_thw"].detach().cpu()
        for sample_id, length, grid in zip(sample_ids, lengths, grids):
            expected = int(grid.long().prod())
            if length != expected:
                raise RuntimeError(
                    f"Teacher cache token/grid mismatch for {sample_id}: "
                    f"length={length}, grid={grid.tolist()}"
                )
            if length > settings.max_visual_tokens:
                raise RuntimeError(
                    f"visual token count {length} exceeds max_visual_tokens="
                    f"{settings.max_visual_tokens}; no truncation is allowed"
                )
            target = projected[offset : offset + length].detach().cpu()
            if target.shape != (length, 224):
                raise RuntimeError(
                    f"Projected target shape {tuple(target.shape)} is invalid"
                )
            if not torch.isfinite(target).all():
                raise RuntimeError(f"Teacher PCA target is non-finite: {sample_id}")
            pending_ids.append(sample_id)
            pending_targets.append(target)
            pending_grids.append(grid)
            offset += length
            if len(pending_ids) >= settings.teacher_cache_shard_size:
                flush()
        if batch_index % max(1, settings.log_interval_batches) == 0:
            print(
                f"[teacher_cache] {split} batch={batch_index:,} "
                f"cached={len(index) + len(pending_ids):,}/{len(loader.dataset):,}",
                flush=True,
            )
    flush()
    if len(index) != len(loader.dataset):
        missing = len(loader.dataset) - len(index)
        raise RuntimeError(
            f"Teacher cache build ended with {missing} missing {split} samples"
        )
    metadata["status"] = "complete"
    metadata["cached_samples"] = len(index)
    write_json(metadata_path, metadata)
    print(
        f"[teacher_cache] {split} complete: {len(index):,} samples, "
        f"{len(metadata['shards']):,} shards",
        flush=True,
    )
    return metadata


class TeacherTargetStore:
    def __init__(
        self,
        directory: Path,
        *,
        expected_identity: dict[str, Any] | None = None,
        lru_shards: int = 4,
    ) -> None:
        self.directory = directory
        self.metadata = _read_json(directory / "metadata.json")
        self.index = _read_json(directory / "index.json")
        if not isinstance(self.metadata, dict) or self.metadata.get("status") != "complete":
            raise FileNotFoundError(
                f"Complete teacher cache metadata is missing: {directory}"
            )
        if not isinstance(self.index, dict):
            raise RuntimeError(f"Teacher cache index is invalid: {directory}")
        if expected_identity is not None:
            mismatches = {
                key: (self.metadata["identity"].get(key), expected)
                for key, expected in expected_identity.items()
                if self.metadata["identity"].get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"Teacher cache cannot be reused because identity differs: {mismatches}"
                )
        self.lru_shards = max(1, int(lru_shards))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def get(self, sample_id: str) -> dict[str, torch.Tensor]:
        location = self.index.get(sample_id)
        if location is None:
            raise KeyError(f"Teacher cache has no target for {sample_id}")
        filename = str(location["shard"])
        shard = self._load_shard(filename)
        row = int(location["row"])
        if shard["sample_ids"][row] != sample_id:
            raise RuntimeError(
                f"Teacher cache index corruption for {sample_id}: "
                f"shard row contains {shard['sample_ids'][row]}"
            )
        length = int(shard["lengths"][row])
        return {
            "target": shard["targets"][row, :length].float(),
            "image_grid_thw": shard["image_grid_thw"][row].long(),
        }

    def _load_shard(self, filename: str) -> dict[str, Any]:
        if filename in self._cache:
            value = self._cache.pop(filename)
            self._cache[filename] = value
            return value
        value = torch_load(self.directory / filename)
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid teacher shard: {filename}")
        self._cache[filename] = value
        while len(self._cache) > self.lru_shards:
            self._cache.popitem(last=False)
        return value


class CocoTeacherTargetDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        image_dataset: Dataset[dict[str, Any]],
        cache_directory: Path,
        *,
        lru_shards: int,
        expected_identity: dict[str, Any] | None = None,
    ) -> None:
        self.image_dataset = image_dataset
        self.cache_directory = cache_directory
        self.lru_shards = int(lru_shards)
        self.expected_identity = expected_identity
        self._store: TeacherTargetStore | None = None

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.image_dataset[index])
        if self._store is None:
            self._store = TeacherTargetStore(
                self.cache_directory,
                expected_identity=self.expected_identity,
                lru_shards=self.lru_shards,
            )
        cached = self._store.get(item["sample_id"])
        item["teacher_target"] = cached["target"]
        item["teacher_image_grid_thw"] = cached["image_grid_thw"]
        return item


def collate_coco_targets(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "sample_indices": torch.tensor(
            [item["sample_index"] for item in batch],
            dtype=torch.long,
        ),
        "image_paths": [item["image_path"] for item in batch],
        "teacher_targets": [item["teacher_target"] for item in batch],
        "teacher_image_grid_thw": torch.stack(
            [item["teacher_image_grid_thw"] for item in batch]
        ),
    }


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))
