from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from .features import move_inputs, multimodal_forward_features, pool_answer_hidden_state, preprocess_image_text
from .io_utils import write_json


CACHE_SCHEMA_VERSION = 3


def expected_metadata(split: str, samples: int, settings: Any, model: nn.Module | None) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION, "split": split, "sample_count": int(samples),
        "model_id": str(settings.model_id), "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "dataset": "spaq_single_attribute", "task": settings.task_name, "data_root": str(settings.data_root),
        "annotations_file": settings.resolved_annotations_file, "split_digest": settings.split_digest,
        "classification_prompt": settings.classification_prompt,
        "processor_min_pixels": settings.processor_min_pixels, "processor_max_pixels": settings.processor_max_pixels,
        "cache_dtype": settings.cache_dtype, "dtype": settings.dtype,
        "attention_implementation": settings.attn_implementation,
        "vision_depth": settings.vision_depth, "vision_hidden_size": settings.vision_hidden_size,
        "text_depth": settings.text_depth, "text_hidden_size": settings.text_hidden_size,
        "deepstack_visual_indexes": list(settings.deepstack_visual_indexes or []),
        "replacement_mode": "qwen3_vl_native_deepstack_teacher_targets",
        "cached_tensors": ["targets", "image_grid_thw", "visual_token_counts", "sequence_lengths",
                           "teacher_answer_hidden", "teacher_vision_taps"],
        "input_color_mode": "RGB", "target_scale": [0.0, 1.0],
    }


@torch.inference_mode()
def build_teacher_cache(split: str, model: nn.Module, processor: Any, replacement: Any, loader: Any,
                        dataset_size: int, settings: Any, device: torch.device) -> Path:
    root = settings.output_dir / "teacher_cache"; manifest_path = root / f"{split}.pt"
    expected = expected_metadata(split, dataset_size, settings, model)
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        changed = [key for key, value in expected.items() if manifest["metadata"].get(key) != value]
        if changed: raise RuntimeError(f"Teacher cache metadata mismatch for {split}: {changed}. Delete and rebuild it.")
        print(f"[teacher_precompute] validated existing cache: {manifest_path}", flush=True); return manifest_path
    shard_dir = root / f"{split}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    stored_dtype = torch.float16 if settings.cache_dtype == "float16" else torch.float32
    pending: list[dict[str, Any]] = []; shards: list[dict[str, Any]] = []; replacement.use_teacher()
    tap_indexes = [*replacement.deepstack_indexes, len(replacement.vision_blocks) - 1]
    for batch_index, (images, targets, indices) in enumerate(loader, start=1):
        cpu_inputs = preprocess_image_text(processor, images, settings.classification_prompt)
        inputs = move_inputs(cpu_inputs, device); replacement.teacher_vision_taps.clear(); replacement.teacher_cu_seqlens = None
        hidden = multimodal_forward_features(model, inputs); answer, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
        counts = replacement.teacher_token_counts(); sequence_lengths = cpu_inputs["attention_mask"].sum(1).long().tolist()
        if max(counts) > settings.max_visual_tokens: raise RuntimeError(f"visual token count {max(counts)} exceeds {settings.max_visual_tokens}")
        if max(sequence_lengths) > settings.max_language_tokens:
            raise RuntimeError(f"language sequence length {max(sequence_lengths)} exceeds {settings.max_language_tokens}; shorten prompt/pixel budget")
        missing = [index for index in tap_indexes if index not in replacement.teacher_vision_taps]
        if missing: raise RuntimeError(f"Teacher hooks missed vision blocks: {missing}")
        tap_groups = {index: list(replacement.teacher_vision_taps[index].split(counts)) for index in tap_indexes}
        for local in range(len(images)):
            pending.append({"sample_index": int(indices[local]), "target": float(targets[local]),
                            "image_grid_thw": cpu_inputs["image_grid_thw"][local].cpu(),
                            "visual_token_count": counts[local], "sequence_length": sequence_lengths[local],
                            "teacher_answer_hidden": answer[local].to(stored_dtype).cpu(),
                            "teacher_vision_taps": [tap_groups[index][local].to(stored_dtype).cpu() for index in tap_indexes]})
            if len(pending) >= settings.teacher_cache_shard_size:
                shards.append(_flush_shard(shard_dir, len(shards), pending)); pending = []
        if batch_index % settings.teacher_cache_log_interval_batches == 0 or batch_index == len(loader):
            print(f"[teacher_precompute] {split} batch={batch_index}/{len(loader)} cached={min(batch_index * settings.feature_batch_size, dataset_size)}/{dataset_size}", flush=True)
    if pending: shards.append(_flush_shard(shard_dir, len(shards), pending))
    metadata = {**expected, "shard_count": len(shards), "total_cache_bytes": sum(row["bytes"] for row in shards)}
    root.mkdir(parents=True, exist_ok=True); torch.save({"metadata": metadata, "shards": shards}, manifest_path)
    write_json(root / f"{split}_metadata.json", metadata); return manifest_path


def _flush_shard(directory: Path, number: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = directory / f"shard_{number:06d}.pt"
    payload = {"sample_indices": torch.tensor([row["sample_index"] for row in rows]),
               "targets": torch.tensor([row["target"] for row in rows], dtype=torch.float32),
               "image_grid_thw": torch.stack([row["image_grid_thw"] for row in rows]),
               "visual_token_counts": torch.tensor([row["visual_token_count"] for row in rows]),
               "sequence_lengths": torch.tensor([row["sequence_length"] for row in rows]),
               "teacher_answer_hidden": torch.stack([row["teacher_answer_hidden"] for row in rows]),
               "teacher_vision_taps": [row["teacher_vision_taps"] for row in rows]}
    temporary = path.with_suffix(".tmp"); torch.save(payload, temporary); temporary.replace(path)
    return {"path": str(path), "count": len(rows), "bytes": path.stat().st_size, "sha256": _sha256(path)}


class TeacherCacheStore:
    def __init__(self, manifest_path: Path, max_cached_shards: int = 8) -> None:
        if not manifest_path.is_file(): raise FileNotFoundError(f"Teacher cache missing: {manifest_path}")
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        self.metadata = manifest["metadata"]; self.shards = manifest["shards"]; self.max_cached_shards = max_cached_shards
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict(); self._ranges = []; self.cache_hits = self.cache_misses = 0; offset = 0
        for number, record in enumerate(self.shards):
            self._ranges.append((offset, offset + int(record["count"]), number)); offset += int(record["count"])

    def __len__(self): return int(self.metadata["sample_count"])
    def get(self, index: int) -> dict[str, Any]:
        for start, end, number in self._ranges:
            if start <= index < end:
                payload = self._load(number); position = index - start
                if int(payload["sample_indices"][position]) != index: raise RuntimeError("Teacher cache ordering mismatch")
                return {"sample_index": payload["sample_indices"][position], "target": payload["targets"][position],
                        "image_grid_thw": payload["image_grid_thw"][position],
                        "visual_token_count": payload["visual_token_counts"][position],
                        "sequence_length": payload["sequence_lengths"][position],
                        "teacher_answer_hidden": payload["teacher_answer_hidden"][position],
                        "teacher_vision_taps": payload["teacher_vision_taps"][position]}
        raise IndexError(index)

    def _load(self, number: int):
        if number in self._cache:
            self.cache_hits += 1; payload = self._cache.pop(number); self._cache[number] = payload; return payload
        self.cache_misses += 1; payload = torch.load(self.shards[number]["path"], map_location="cpu", weights_only=True)
        self._cache[number] = payload
        while len(self._cache) > self.max_cached_shards: self._cache.popitem(last=False)
        return payload

    def reset_stats(self): self.cache_hits = self.cache_misses = 0
    def stats(self):
        total = self.cache_hits + self.cache_misses
        return {"hits": self.cache_hits, "misses": self.cache_misses, "hit_rate": self.cache_hits / total if total else 0.0}
    def iter_shards(self) -> Iterator[dict[str, Any]]:
        for record in self.shards: yield torch.load(record["path"], map_location="cpu", weights_only=True)


def cached_answer_features(store: TeacherCacheStore, targets: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.cat([shard["teacher_answer_hidden"].float() for shard in store.iter_shards()])
    return features, torch.tensor(targets, dtype=torch.float32)


def write_teacher_predictions(output_dir: Path, split: str, predictions: torch.Tensor, activation: str) -> Path:
    path = output_dir / "teacher_cache" / f"{split}_teacher_predictions.pt"
    torch.save({"sample_indices": torch.arange(len(predictions)), "teacher_predictions": predictions.half(),
                "head_output_activation": activation}, path); return path


def load_teacher_predictions(path: Path, activation: str) -> torch.Tensor:
    if not path.is_file(): raise FileNotFoundError(f"Teacher predictions missing: {path}. Run teacher_predictions.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("head_output_activation") != activation: raise RuntimeError("Teacher prediction activation mismatch")
    return payload["teacher_predictions"].float()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
