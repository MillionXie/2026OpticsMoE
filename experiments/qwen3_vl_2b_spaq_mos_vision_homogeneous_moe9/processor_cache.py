from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import torch

from .features import preprocess_images
from .io_utils import write_json


PROCESSOR_CACHE_SCHEMA_VERSION = 1


def expected_processor_metadata(split: str, samples: int, settings: Any) -> dict[str, Any]:
    return {
        "cache_schema_version": PROCESSOR_CACHE_SCHEMA_VERSION,
        "split": split,
        "sample_count": int(samples),
        "dataset": "spaq_mos",
        "task": "MOS",
        "data_root": str(settings.data_root),
        "annotations_file": settings.resolved_annotations_file,
        "split_digest": settings.split_digest,
        "model_id": str(settings.model_id),
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "storage_dtype": settings.cache_dtype,
        "cached_tensors": ["pixel_values", "image_grid_thw", "visual_token_counts"],
        "input_color_mode": "RGB",
        "source": "Qwen image_processor output; JPEG decode and resize are not repeated during student epochs",
    }


@torch.inference_mode()
def build_processor_cache(split: str, processor: Any, loader: Any, dataset_size: int,
                          settings: Any) -> Path:
    root = settings.output_dir / "processor_cache"
    manifest_path = root / f"{split}.pt"
    expected = expected_processor_metadata(split, dataset_size, settings)
    if manifest_path.is_file():
        _validate_manifest(manifest_path, expected)
        print(f"[processor_cache] validated existing cache: {manifest_path}", flush=True)
        return manifest_path

    shard_dir = root / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stored_dtype = torch.float16 if settings.cache_dtype == "float16" else torch.float32
    pending: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    pixel_feature_dim: int | None = None

    for batch_index, (images, _targets, indices) in enumerate(loader, start=1):
        inputs = preprocess_images(processor, images)
        grids = inputs["image_grid_thw"].cpu()
        pixel_values = inputs["pixel_values"].cpu()
        counts = [int(grid.long().prod()) for grid in grids]
        if sum(counts) != int(pixel_values.shape[0]):
            raise RuntimeError(
                "Qwen processor pixel_values cannot be split by image_grid_thw; "
                f"rows={pixel_values.shape[0]} grid_products={counts}"
            )
        pixel_feature_dim = int(pixel_values.shape[-1])
        groups = pixel_values.split(counts, dim=0)
        for local, group in enumerate(groups):
            pending.append({
                "sample_index": int(indices[local]),
                "image_grid_thw": grids[local],
                "visual_token_count": counts[local],
                "pixel_values": group.to(stored_dtype).contiguous(),
            })
            if len(pending) >= settings.teacher_cache_shard_size:
                shards.append(_flush_shard(shard_dir, len(shards), pending))
                pending = []
        if batch_index % settings.teacher_cache_log_interval_batches == 0 or batch_index == len(loader):
            cached = min(batch_index * settings.feature_batch_size, dataset_size)
            print(f"[processor_cache] {split} batch={batch_index}/{len(loader)} cached={cached}/{dataset_size}", flush=True)

    if pending:
        shards.append(_flush_shard(shard_dir, len(shards), pending))
    metadata = {
        **expected,
        "pixel_feature_dim": pixel_feature_dim,
        "shard_size": settings.teacher_cache_shard_size,
        "shard_count": len(shards),
        "total_cache_bytes": sum(row["bytes"] for row in shards),
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "shards": shards}, manifest_path)
    write_json(root / f"{split}_metadata.json", metadata)
    return manifest_path


def _flush_shard(directory: Path, number: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = directory / f"shard_{number:06d}.pt"
    payload = {
        "sample_indices": torch.tensor([row["sample_index"] for row in rows], dtype=torch.long),
        "image_grid_thw": torch.stack([row["image_grid_thw"] for row in rows]),
        "visual_token_counts": torch.tensor([row["visual_token_count"] for row in rows], dtype=torch.long),
        "pixel_values": [row["pixel_values"] for row in rows],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "count": len(rows), "bytes": path.stat().st_size, "sha256": _sha256(path)}


class ProcessorCacheStore:
    def __init__(self, manifest_path: Path, max_cached_shards: int = 8) -> None:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Processor input cache is missing: {manifest_path}. Run --phase input_precompute first."
            )
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        self.metadata = manifest["metadata"]
        self.shards = manifest["shards"]
        self.max_cached_shards = int(max_cached_shards)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._ranges: list[tuple[int, int, int]] = []
        self.cache_hits = 0
        self.cache_misses = 0
        offset = 0
        for number, record in enumerate(self.shards):
            self._ranges.append((offset, offset + int(record["count"]), number))
            offset += int(record["count"])

    def __len__(self) -> int:
        return int(self.metadata["sample_count"])

    def get(self, index: int) -> dict[str, Any]:
        for start, end, number in self._ranges:
            if start <= index < end:
                payload = self._load(number)
                position = index - start
                if int(payload["sample_indices"][position]) != index:
                    raise RuntimeError("Processor cache sample ordering mismatch")
                return {
                    "sample_index": payload["sample_indices"][position],
                    "image_grid_thw": payload["image_grid_thw"][position],
                    "visual_token_count": payload["visual_token_counts"][position],
                    # Restore the image processor's normal float32 output at batch assembly time.
                    "pixel_values": payload["pixel_values"][position],
                }
        raise IndexError(index)

    def _load(self, number: int) -> dict[str, Any]:
        if number in self._cache:
            self.cache_hits += 1
            payload = self._cache.pop(number)
            self._cache[number] = payload
            return payload
        self.cache_misses += 1
        payload = torch.load(self.shards[number]["path"], map_location="cpu", weights_only=True)
        self._cache[number] = payload
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return payload

    def reset_stats(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0

    def stats(self) -> dict[str, int | float]:
        requests = self.cache_hits + self.cache_misses
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / requests if requests else 0.0,
        }

    def iter_shards(self) -> Iterator[dict[str, Any]]:
        for record in self.shards:
            yield torch.load(record["path"], map_location="cpu", weights_only=True)


def validate_processor_cache(store: ProcessorCacheStore, split: str, samples: int, settings: Any) -> None:
    expected = expected_processor_metadata(split, samples, settings)
    changed = [key for key, value in expected.items() if store.metadata.get(key) != value]
    if changed:
        raise RuntimeError(
            f"Processor input cache metadata mismatch for {split}: {changed}. "
            "Delete processor_cache and rerun input_precompute."
        )


def _validate_manifest(path: Path, expected: dict[str, Any]) -> None:
    manifest = torch.load(path, map_location="cpu", weights_only=True)
    changed = [key for key, value in expected.items() if manifest["metadata"].get(key) != value]
    if changed:
        raise RuntimeError(
            f"Processor input cache metadata mismatch: {changed}. Delete {path.parent} and rebuild it."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
