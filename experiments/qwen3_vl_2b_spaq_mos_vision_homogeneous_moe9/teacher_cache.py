from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from .features import move_inputs, preprocess_images, run_visual
from . import TASK_NAME
from .io_utils import write_json


CACHE_SCHEMA_VERSION = 2


def expected_metadata(split: str, samples: int, settings: Any, model: nn.Module | None) -> dict[str, Any]:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None) if model is not None else None
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "sample_count": int(samples),
        "model_id": str(settings.model_id),
        "dataset": "spaq_mos",
        "task": TASK_NAME,
        "data_root": str(settings.data_root),
        "annotations_file": settings.resolved_annotations_file,
        "image_dir": str(settings.image_dir) if settings.image_dir else None,
        "split_digest": settings.split_digest,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "cache_dtype": settings.cache_dtype,
        "dtype": settings.dtype,
        "attention_implementation": settings.attn_implementation,
        "vision_depth": int(getattr(vision_config, "depth", settings.vision_depth or 0)),
        "vision_hidden_size": int(getattr(vision_config, "hidden_size", settings.vision_hidden_size or 0)),
        "replacement_mode": "complete_vision_stack_homogeneous_moe9x5",
        "cached_tensors": ["targets", "image_grid_thw", "visual_token_counts", "teacher_vision_stack_output"],
        "input_color_mode": "RGB",
        "target_scale": [0.0, 1.0],
    }


@torch.inference_mode()
def build_teacher_cache(split: str, model: nn.Module, processor: Any, replacement: Any, loader: Any,
                        dataset_size: int, settings: Any, device: torch.device) -> Path:
    root = settings.output_dir / "teacher_cache"
    manifest_path = root / f"{split}.pt"
    expected = expected_metadata(split, dataset_size, settings, model)
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        changed = [key for key, value in expected.items() if manifest["metadata"].get(key) != value]
        if changed:
            raise RuntimeError(f"Teacher cache metadata mismatch for {split}: {changed}. Delete the cache and rerun teacher_precompute.")
        print(f"[teacher_precompute] validated existing cache: {manifest_path}", flush=True)
        return manifest_path
    shard_dir = root / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    replacement.use_teacher()
    stored_dtype = torch.float16 if settings.cache_dtype == "float16" else torch.float32
    pending: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for batch_index, (images, targets, indices) in enumerate(loader, start=1):
        cpu_inputs = preprocess_images(processor, images)
        inputs = move_inputs(cpu_inputs, device)
        replacement.teacher_output = None
        replacement.teacher_cu_seqlens = None
        run_visual(model, inputs)
        hidden = replacement.teacher_output
        if hidden is None:
            raise RuntimeError("Teacher hook did not capture the final electronic vision block output")
        counts = replacement.teacher_token_counts()
        groups = list(hidden.split(counts, dim=0))
        if len(groups) != len(images):
            raise RuntimeError("Teacher packed visual boundaries do not match the image batch")
        if max(counts) > settings.max_visual_tokens:
            raise RuntimeError(
                f"visual token count {max(counts)} exceeds max_visual_tokens={settings.max_visual_tokens}. "
                "Lower processor_max_pixels before generating the cache."
            )
        for local, group in enumerate(groups):
            pending.append({
                "sample_index": int(indices[local]),
                "target": float(targets[local]),
                "image_grid_thw": cpu_inputs["image_grid_thw"][local].cpu(),
                "visual_token_count": counts[local],
                "teacher_vision_stack_output": group.to(stored_dtype).cpu(),
            })
            if len(pending) >= settings.teacher_cache_shard_size:
                shards.append(_flush_shard(shard_dir, len(shards), pending))
                pending = []
        if batch_index % settings.teacher_cache_log_interval_batches == 0 or batch_index == len(loader):
            print(f"[teacher_precompute] {split} batch={batch_index}/{len(loader)} cached={min(batch_index * settings.feature_batch_size, dataset_size)}/{dataset_size}", flush=True)
    if pending:
        shards.append(_flush_shard(shard_dir, len(shards), pending))
    metadata = {**expected, "shard_count": len(shards), "total_cache_bytes": sum(row["bytes"] for row in shards)}
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "shards": shards}, manifest_path)
    write_json(root / f"{split}_metadata.json", metadata)
    return manifest_path


def _flush_shard(directory: Path, number: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = directory / f"shard_{number:06d}.pt"
    payload = {
        "sample_indices": torch.tensor([row["sample_index"] for row in rows], dtype=torch.long),
        "targets": torch.tensor([row["target"] for row in rows], dtype=torch.float32),
        "image_grid_thw": torch.stack([row["image_grid_thw"] for row in rows]),
        "visual_token_counts": torch.tensor([row["visual_token_count"] for row in rows], dtype=torch.long),
        "teacher_vision_stack_output": [row["teacher_vision_stack_output"] for row in rows],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "count": len(rows), "bytes": path.stat().st_size, "sha256": _sha256(path)}


class TeacherCacheStore:
    def __init__(self, manifest_path: Path, max_cached_shards: int = 8) -> None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Teacher cache is missing: {manifest_path}. Run --phase teacher_precompute first.")
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
                    raise RuntimeError("Teacher cache sample ordering mismatch")
                return {
                    "sample_index": payload["sample_indices"][position],
                    "target": payload["targets"][position],
                    "image_grid_thw": payload["image_grid_thw"][position],
                    "visual_token_count": payload["visual_token_counts"][position],
                    "teacher_vision_stack_output": payload["teacher_vision_stack_output"][position],
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


def pooled_teacher_features(store: TeacherCacheStore) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for shard in store.iter_shards():
        features.extend(group.float().mean(0) for group in shard["teacher_vision_stack_output"])
        targets.append(shard["targets"].float())
    return torch.stack(features), torch.cat(targets)


def write_teacher_predictions(output_dir: Path, split: str, predictions: torch.Tensor, targets: torch.Tensor,
                              head_specification: dict[str, Any]) -> None:
    torch.save({"sample_indices": torch.arange(len(targets)), "targets": targets.float(),
                "teacher_predictions": predictions.half(), "head": head_specification},
               output_dir / "teacher_cache" / f"{split}_teacher_predictions.pt")


def load_teacher_predictions(path: Path, expected_output_activation: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Teacher predictions missing: {path}. Run --phase teacher_predictions first.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved_activation = payload.get("head", {}).get("output_activation")
    if saved_activation != expected_output_activation:
        raise RuntimeError(
            f"Cached teacher predictions use output_activation={saved_activation!r}, but the current head uses "
            f"{expected_output_activation!r}. Rerun --phase teacher_train and --phase teacher_predictions. "
            "The expensive teacher_precompute feature cache can be reused."
        )
    return payload["teacher_predictions"].float()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
