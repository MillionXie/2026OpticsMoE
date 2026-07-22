from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from .features import move_inputs, multimodal_forward_features, pool_answer_hidden_state
from .io_utils import write_json
from .processor_cache import ProcessorCacheStore, collate_processor_samples


CACHE_SCHEMA_VERSION = 6
TEACHER_LOGIT_SCHEMA_VERSION = 1


def expected_metadata(split: str, samples: int, settings: Any, model: nn.Module | None) -> dict[str, Any]:
    purpose = getattr(settings, "cache_purpose", "flickr30k_binary_fine_tuning")
    generic = purpose == "generic_multimodal_hidden_distillation"
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION, "split": split, "sample_count": int(samples),
        "model_id": str(settings.model_id), "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "dataset": settings.dataset, "dataset_repo_id": settings.dataset_repo_id,
        "dataset_revision": settings.dataset_revision,
        "dataset_fingerprints": settings.resolved_dataset_fingerprints,
        "pair_manifest_digest": (settings.pair_manifest_digests or {}).get(split),
        "prompt_template": settings.prompt_template,
        "negative_sampling_algorithm": settings.negative_sampling_algorithm,
        "captions_per_image": settings.captions_per_image,
        "negatives_per_positive": settings.negatives_per_positive,
        "seed": settings.seed,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "max_visual_tokens": settings.max_visual_tokens,
        "max_language_tokens": settings.max_language_tokens,
        "cache_dtype": settings.cache_dtype, "dtype": settings.dtype,
        "attention_implementation": settings.attn_implementation,
        "vision_depth": settings.vision_depth, "vision_hidden_size": settings.vision_hidden_size,
        "text_depth": settings.text_depth, "text_hidden_size": settings.text_hidden_size,
        "deepstack_visual_indexes": list(settings.deepstack_visual_indexes or []),
        "replacement_mode": "qwen3_vl_native_deepstack_teacher_targets",
        "cache_purpose": purpose,
        "cached_tensors": ["labels", "image_grid_thw", "visual_token_counts", "sequence_lengths",
                           "teacher_answer_hidden", "teacher_vision_taps"],
        "teacher_logits_cached_separately": not generic,
        "teacher_logit_semantics": None if generic else "raw_binary_logit",
        "teacher_fine_tuned": False,
        "input_color_mode": "RGB",
    }


@torch.inference_mode()
def build_teacher_cache(split: str, model: nn.Module, replacement: Any,
                        processor_inputs: ProcessorCacheStore, labels: Sequence[float],
                        dataset_size: int, settings: Any, device: torch.device) -> Path:
    root = settings.output_dir / "teacher_cache"; manifest_path = root / f"{split}.pt"
    expected = {
        **expected_metadata(split, dataset_size, settings, model),
        "teacher_input_source": "validated_processor_cache",
        "processor_cache_schema_version": processor_inputs.metadata.get("cache_schema_version"),
    }
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        changed = [key for key, value in expected.items() if manifest["metadata"].get(key) != value]
        if changed: raise RuntimeError(f"Teacher cache metadata mismatch for {split}: {changed}. Delete and rebuild it.")
        print(f"[teacher_precompute] validated existing cache: {manifest_path}", flush=True); return manifest_path
    shard_dir = root / f"{split}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    stored_dtype = torch.float16 if settings.cache_dtype == "float16" else torch.float32
    pending: list[dict[str, Any]] = []; replacement.use_teacher()
    writer = _AsyncShardWriter(shard_dir, max_pending=2)
    tap_indexes = [*replacement.deepstack_indexes, len(replacement.vision_blocks) - 1]
    batches = (dataset_size + settings.feature_batch_size - 1) // settings.feature_batch_size
    try:
        for batch_index, (cpu_inputs, batch_labels, indices) in enumerate(
                iter_cached_input_batches(processor_inputs, labels, settings.feature_batch_size), start=1):
            inputs = move_inputs(cpu_inputs, device); replacement.teacher_vision_taps.clear(); replacement.teacher_cu_seqlens = None
            hidden = multimodal_forward_features(model, inputs); answer, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
            counts = replacement.teacher_token_counts(); sequence_lengths = cpu_inputs["attention_mask"].sum(1).long().tolist()
            if max(counts) > settings.max_visual_tokens:
                raise RuntimeError(f"visual token count {max(counts)} exceeds max_visual_tokens={settings.max_visual_tokens}")
            if max(sequence_lengths) > settings.max_language_tokens:
                raise RuntimeError(
                    f"language sequence length {max(sequence_lengths)} exceeds max_language_tokens={settings.max_language_tokens}; "
                    "shorten the caption/prompt. Silent truncation is forbidden."
                )
            missing = [index for index in tap_indexes if index not in replacement.teacher_vision_taps]
            if missing: raise RuntimeError(f"Teacher hooks missed vision blocks: {missing}")
            # Transfer each packed target once. The old per-sample `.cpu()` calls
            # introduced (1 + number_of_taps) * batch_size CUDA synchronizations.
            answer_cpu, tap_groups = packed_teacher_targets_to_cpu(
                answer, replacement.teacher_vision_taps, tap_indexes, counts, stored_dtype
            )
            for local in range(len(indices)):
                pending.append({"sample_index": int(indices[local]), "label": float(batch_labels[local]),
                                "image_grid_thw": cpu_inputs["image_grid_thw"][local].cpu(),
                                "visual_token_count": counts[local], "sequence_length": sequence_lengths[local],
                                "teacher_answer_hidden": answer_cpu[local],
                                "teacher_vision_taps": [tap_groups[index][local] for index in tap_indexes]})
                if len(pending) >= settings.teacher_cache_shard_size:
                    writer.submit(pending); pending = []
            if batch_index % settings.teacher_cache_log_interval_batches == 0 or batch_index == batches:
                cached = int(indices[-1]) + 1
                print(f"[teacher_precompute] {split} batch={batch_index}/{batches} cached={cached}/{dataset_size}", flush=True)
        if pending: writer.submit(pending)
        shards = writer.finish()
    except BaseException:
        writer.abort()
        raise
    metadata = {**expected, "shard_count": len(shards), "total_cache_bytes": sum(row["bytes"] for row in shards)}
    root.mkdir(parents=True, exist_ok=True); torch.save({"metadata": metadata, "shards": shards}, manifest_path)
    write_json(root / f"{split}_metadata.json", metadata); return manifest_path


def iter_cached_input_batches(processor_inputs: ProcessorCacheStore, labels: Sequence[float],
                              batch_size: int) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
    """Yield the exact persisted processor tensors without decoding images again."""
    sample_count = len(processor_inputs)
    if sample_count != len(labels):
        raise RuntimeError(f"Processor cache/label length mismatch: {sample_count} != {len(labels)}")
    if batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        indices = torch.arange(start, stop, dtype=torch.long)
        samples = [processor_inputs.get(index) for index in range(start, stop)]
        yield (collate_processor_samples(samples, processor_inputs.metadata),
               torch.tensor(labels[start:stop], dtype=torch.float32), indices)


def packed_teacher_targets_to_cpu(answer: torch.Tensor, taps: dict[int, torch.Tensor],
                                  tap_indexes: Sequence[int], counts: Sequence[int],
                                  dtype: torch.dtype) -> tuple[torch.Tensor, dict[int, list[torch.Tensor]]]:
    """Copy packed GPU targets in O(number of taps), then split on CPU."""
    answer_cpu = answer.to(dtype).cpu()
    groups = {index: list(taps[index].to(dtype).cpu().split(list(counts))) for index in tap_indexes}
    return answer_cpu, groups


def _flush_shard(directory: Path, number: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = directory / f"shard_{number:06d}.pt"
    payload = pack_teacher_rows(rows)
    temporary = path.with_suffix(".tmp"); torch.save(payload, temporary); temporary.replace(path)
    return {"path": str(path), "count": len(rows), "bytes": path.stat().st_size}


class _AsyncShardWriter:
    """Overlap one bounded teacher-shard write with subsequent GPU forwards."""

    def __init__(self, directory: Path, max_pending: int) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.directory = directory; self.max_pending = int(max_pending)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher-cache-writer")
        self.futures: list[Future[dict[str, Any]]] = []; self.records: list[dict[str, Any]] = []
        self.next_number = 0; self.closed = False

    def submit(self, rows: list[dict[str, Any]]) -> None:
        if self.closed:
            raise RuntimeError("Cannot submit to a closed teacher-cache writer")
        self.futures.append(self.executor.submit(_flush_shard, self.directory, self.next_number, rows))
        self.next_number += 1
        if len(self.futures) >= self.max_pending:
            self.records.append(self.futures.pop(0).result())

    def finish(self) -> list[dict[str, Any]]:
        if not self.closed:
            for future in self.futures:
                self.records.append(future.result())
            self.futures.clear(); self.executor.shutdown(wait=True); self.closed = True
        return self.records

    def abort(self) -> None:
        if not self.closed:
            self.executor.shutdown(wait=True, cancel_futures=True); self.closed = True


def pack_teacher_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pack variable-token vision targets into four contiguous shard tensors."""
    if not rows:
        raise ValueError("Cannot pack an empty teacher-cache shard")
    tap_count = len(rows[0]["teacher_vision_taps"])
    if any(len(row["teacher_vision_taps"]) != tap_count for row in rows):
        raise RuntimeError("Teacher tap count changed within a cache shard")
    for row in rows:
        expected_tokens = int(row["visual_token_count"])
        if any(int(tap.shape[0]) != expected_tokens for tap in row["teacher_vision_taps"]):
            raise RuntimeError("Teacher tap token count does not match visual_token_count")
    counts = torch.tensor([row["visual_token_count"] for row in rows], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return {"sample_indices": torch.tensor([row["sample_index"] for row in rows]),
            "labels": torch.tensor([row["label"] for row in rows], dtype=torch.float32),
            "image_grid_thw": torch.stack([row["image_grid_thw"] for row in rows]),
            "visual_token_counts": counts,
            "visual_token_offsets": offsets,
            "sequence_lengths": torch.tensor([row["sequence_length"] for row in rows]),
            "teacher_answer_hidden": torch.stack([row["teacher_answer_hidden"] for row in rows]),
            "teacher_vision_taps": [
                torch.cat([row["teacher_vision_taps"][tap] for row in rows], dim=0)
                for tap in range(tap_count)
            ]}


class TeacherCacheStore:
    def __init__(self, manifest_path: Path, max_cached_shards: int = 8) -> None:
        if not manifest_path.is_file(): raise FileNotFoundError(f"Teacher cache missing: {manifest_path}")
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        self.metadata = manifest["metadata"]; self.shards = manifest["shards"]; self.max_cached_shards = max_cached_shards
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict(); self._ranges = []
        self.cache_hits = self.cache_misses = 0; offset = 0
        for number, record in enumerate(self.shards):
            self._ranges.append((offset, offset + int(record["count"]), number)); offset += int(record["count"])

    def __len__(self) -> int: return int(self.metadata["sample_count"])

    def get(self, index: int) -> dict[str, Any]:
        for start, end, number in self._ranges:
            if start <= index < end:
                payload = self._load(number); position = index - start
                if int(payload["sample_indices"][position]) != index: raise RuntimeError("Teacher cache ordering mismatch")
                start = int(payload["visual_token_offsets"][position])
                stop = int(payload["visual_token_offsets"][position + 1])
                return {"sample_index": payload["sample_indices"][position], "label": payload["labels"][position],
                        "image_grid_thw": payload["image_grid_thw"][position],
                        "visual_token_count": payload["visual_token_counts"][position],
                        "sequence_length": payload["sequence_lengths"][position],
                        "teacher_answer_hidden": payload["teacher_answer_hidden"][position],
                        "teacher_vision_taps": [tap[start:stop] for tap in payload["teacher_vision_taps"]]}
        raise IndexError(index)

    def _load(self, number: int) -> dict[str, Any]:
        if number in self._cache:
            self.cache_hits += 1; payload = self._cache.pop(number); self._cache[number] = payload; return payload
        self.cache_misses += 1; payload = torch.load(self.shards[number]["path"], map_location="cpu", weights_only=True)
        self._cache[number] = payload
        while len(self._cache) > self.max_cached_shards: self._cache.popitem(last=False)
        return payload

    def reset_stats(self) -> None: self.cache_hits = self.cache_misses = 0
    def stats(self) -> dict[str, int | float]:
        total = self.cache_hits + self.cache_misses
        return {"hits": self.cache_hits, "misses": self.cache_misses,
                "hit_rate": self.cache_hits / total if total else 0.0}

    def iter_shards(self) -> Iterator[dict[str, Any]]:
        for record in self.shards:
            yield torch.load(record["path"], map_location="cpu", weights_only=True)


def cached_answer_features(store: TeacherCacheStore) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.cat([shard["teacher_answer_hidden"].float() for shard in store.iter_shards()])
    labels = torch.cat([shard["labels"].float() for shard in store.iter_shards()])
    return features, labels


def write_teacher_logits(output_dir: Path, split: str, logits: torch.Tensor, settings: Any) -> Path:
    path = output_dir / "teacher_cache" / f"{split}_teacher_logits.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_schema_version": TEACHER_LOGIT_SCHEMA_VERSION,
        "sample_indices": torch.arange(len(logits)),
        "teacher_logits": logits.float().cpu(),
        "semantics": "raw_binary_logit",
        "head_type": settings.head_type,
        "pair_manifest_digest": (settings.pair_manifest_digests or {}).get(split),
    }
    torch.save(payload, path); return path


def load_teacher_logits(path: Path, settings: Any, split: str) -> torch.Tensor:
    if not path.is_file(): raise FileNotFoundError(f"Teacher logits missing: {path}. Run --phase teacher_logits.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"cache_schema_version": TEACHER_LOGIT_SCHEMA_VERSION, "semantics": "raw_binary_logit",
                "head_type": settings.head_type,
                "pair_manifest_digest": (settings.pair_manifest_digests or {}).get(split)}
    changed = [key for key, value in expected.items() if payload.get(key) != value]
    if changed: raise RuntimeError(f"Teacher logit cache mismatch for {split}: {changed}. Rebuild teacher_logits.")
    logits = payload["teacher_logits"].float()
    if logits.ndim != 1: raise RuntimeError(f"Teacher raw logits must be rank-1, got {tuple(logits.shape)}")
    return logits
