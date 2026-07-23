from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import torch

from .features import preprocess_image_text
from .io_utils import write_json


PROCESSOR_CACHE_SCHEMA_VERSION = 2


def expected_processor_metadata(split: str, samples: int, settings: Any) -> dict[str, Any]:
    return {
        "cache_schema_version": PROCESSOR_CACHE_SCHEMA_VERSION,
        "split": split, "sample_count": int(samples), "dataset": "spaq_single_attribute",
        "task": settings.task_name, "data_root": str(settings.data_root),
        "annotations_file": settings.resolved_annotations_file, "split_digest": settings.split_digest,
        "model_id": str(settings.model_id), "classification_prompt": settings.classification_prompt,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels, "storage_dtype": settings.cache_dtype,
        "cached_tensors": ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"],
        "input_color_mode": "RGB", "source": "complete Qwen processor image+chat-template output",
    }


@torch.inference_mode()
def build_processor_cache(split: str, processor: Any, loader: Any, dataset_size: int, settings: Any) -> Path:
    root = settings.output_dir / "processor_cache"; manifest_path = root / f"{split}.pt"
    expected = expected_processor_metadata(split, dataset_size, settings)
    if manifest_path.is_file():
        _validate_manifest(manifest_path, expected)
        print(f"[processor_cache] validated existing cache: {manifest_path}", flush=True)
        return manifest_path
    shard_dir = root / f"{split}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    stored_dtype = torch.float16 if settings.cache_dtype == "float16" else torch.float32
    pending: list[dict[str, Any]] = []; shards: list[dict[str, Any]] = []; cached_count = 0
    padding_side = getattr(getattr(processor, "tokenizer", None), "padding_side", "left")
    pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0)
    if pad_token_id is None: pad_token_id = 0
    for batch_index, (images, _targets, indices) in enumerate(loader, start=1):
        inputs = preprocess_image_text(processor, images, settings.classification_prompt)
        grids = inputs["image_grid_thw"].cpu(); pixels = inputs["pixel_values"].cpu()
        patch_counts = [int(grid.long().prod()) for grid in grids]
        if sum(patch_counts) != int(pixels.shape[0]):
            raise RuntimeError("pixel_values rows do not match image_grid_thw products")
        pixel_groups = pixels.split(patch_counts, dim=0)
        for local, group in enumerate(pixel_groups):
            valid = inputs["attention_mask"][local].bool().cpu()
            ids = inputs["input_ids"][local].cpu()[valid].contiguous()
            pending.append({"sample_index": int(indices[local]), "input_ids": ids,
                            "pixel_values": group.to(stored_dtype).contiguous(),
                            "image_grid_thw": grids[local], "sequence_length": int(valid.sum())})
            if len(pending) >= settings.teacher_cache_shard_size:
                shards.append(_flush_shard(shard_dir, len(shards), pending)); pending = []
        cached_count += len(images)
        if batch_index % settings.teacher_cache_log_interval_batches == 0 or batch_index == len(loader):
            print(f"[processor_cache] {split} batch={batch_index}/{len(loader)} cached={cached_count}/{dataset_size}", flush=True)
    if pending: shards.append(_flush_shard(shard_dir, len(shards), pending))
    metadata = {**expected, "padding_side": padding_side, "pad_token_id": int(pad_token_id),
                "shard_count": len(shards), "shard_size": settings.teacher_cache_shard_size,
                "total_cache_bytes": sum(row["bytes"] for row in shards)}
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "shards": shards}, manifest_path)
    write_json(root / f"{split}_metadata.json", metadata)
    return manifest_path


def _flush_shard(directory: Path, number: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = directory / f"shard_{number:06d}.pt"
    payload = {"sample_indices": torch.tensor([row["sample_index"] for row in rows]),
               "input_ids": [row["input_ids"] for row in rows],
               "pixel_values": [row["pixel_values"] for row in rows],
               "image_grid_thw": torch.stack([row["image_grid_thw"] for row in rows]),
               "sequence_lengths": torch.tensor([row["sequence_length"] for row in rows])}
    temporary = path.with_suffix(".tmp"); torch.save(payload, temporary); temporary.replace(path)
    return {"path": str(path), "count": len(rows), "bytes": path.stat().st_size, "sha256": _sha256(path)}


class ProcessorCacheStore:
    def __init__(self, manifest_path: Path, max_cached_shards: int = 8) -> None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Processor cache missing: {manifest_path}. Run --phase input_precompute.")
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        self.metadata = manifest["metadata"]; self.shards = manifest["shards"]
        self.max_cached_shards = int(max_cached_shards); self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._ranges: list[tuple[int, int, int]] = []; self.cache_hits = 0; self.cache_misses = 0; offset = 0
        for number, record in enumerate(self.shards):
            self._ranges.append((offset, offset + int(record["count"]), number)); offset += int(record["count"])

    def __len__(self) -> int: return int(self.metadata["sample_count"])

    def get(self, index: int) -> dict[str, Any]:
        for start, end, number in self._ranges:
            if start <= index < end:
                payload = self._load(number); position = index - start
                if int(payload["sample_indices"][position]) != index: raise RuntimeError("Processor cache ordering mismatch")
                return {"sample_index": payload["sample_indices"][position],
                        "input_ids": payload["input_ids"][position],
                        "pixel_values": payload["pixel_values"][position],
                        "image_grid_thw": payload["image_grid_thw"][position],
                        "sequence_length": payload["sequence_lengths"][position]}
        raise IndexError(index)

    def _load(self, number: int) -> dict[str, Any]:
        if number in self._cache:
            self.cache_hits += 1; payload = self._cache.pop(number); self._cache[number] = payload; return payload
        self.cache_misses += 1
        payload = torch.load(self.shards[number]["path"], map_location="cpu", weights_only=True); self._cache[number] = payload
        while len(self._cache) > self.max_cached_shards: self._cache.popitem(last=False)
        return payload

    def reset_stats(self) -> None: self.cache_hits = self.cache_misses = 0
    def stats(self) -> dict[str, int | float]:
        requests = self.cache_hits + self.cache_misses
        return {"hits": self.cache_hits, "misses": self.cache_misses,
                "hit_rate": self.cache_hits / requests if requests else 0.0}


def collate_processor_samples(samples: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
    max_length = max(int(sample["sequence_length"]) for sample in samples)
    if max_length <= 0: raise RuntimeError("Cached sequence length must be positive")
    pad = int(metadata.get("pad_token_id", 0)); side = metadata.get("padding_side", "left")
    input_ids = torch.full((len(samples), max_length), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(samples), max_length), dtype=torch.long)
    for row, sample in enumerate(samples):
        ids = sample["input_ids"].long(); length = len(ids); start = max_length - length if side == "left" else 0
        input_ids[row, start:start + length] = ids; attention_mask[row, start:start + length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "pixel_values": torch.cat([sample["pixel_values"] for sample in samples]).float(),
            "image_grid_thw": torch.stack([sample["image_grid_thw"] for sample in samples]).long()}


def validate_processor_cache(store: ProcessorCacheStore, split: str, samples: int, settings: Any) -> None:
    expected = expected_processor_metadata(split, samples, settings)
    changed = [key for key, value in expected.items() if store.metadata.get(key) != value]
    if changed: raise RuntimeError(f"Processor cache metadata mismatch for {split}: {changed}. Rebuild it.")


def _validate_manifest(path: Path, expected: dict[str, Any]) -> None:
    manifest = torch.load(path, map_location="cpu", weights_only=True)
    changed = [key for key, value in expected.items() if manifest["metadata"].get(key) != value]
    if changed: raise RuntimeError(f"Processor cache metadata mismatch: {changed}. Delete {path.parent} and rebuild.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
