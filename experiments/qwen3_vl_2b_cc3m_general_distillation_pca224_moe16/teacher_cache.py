from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import torch

from .cache_paths import cache_identity, cache_identity_digest, teacher_cache_root
from .datasets import DatasetBundle, ImageCaptionDataset, make_loader
from .features import move_inputs, preprocess_image_text, run_multimodal_forward
from .io_utils import write_json
from .optics.replacement import TeacherTapCapture
from .pca import load_projection, projection_paths


CACHE_SCHEMA_VERSION = 1


def expected_metadata(
    split: str,
    sample_count: int,
    settings: Any,
    model: torch.nn.Module | None,
) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "sample_count": int(sample_count),
        "cache_identity": cache_identity(settings),
        "cache_identity_digest": cache_identity_digest(settings),
        "model_id": str(settings.model_id),
        "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "dataset": settings.dataset,
        "manifest_digest": settings.manifest_digest,
        "caption_prompt_template": settings.caption_prompt_template,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "vision_hidden_size": settings.vision_hidden_size,
        "text_hidden_size": settings.text_hidden_size,
        "deepstack_visual_indexes": list(settings.deepstack_visual_indexes),
        "language_tap_indexes": list(settings.language_tap_indexes),
        "latent_dim": settings.latent_dim,
        "cache_dtype": settings.cache_dtype,
        "cached_tensors": [
            "input_ids",
            "pixel_values",
            "image_grid_thw",
            "visual_token_count",
            "language_token_mask",
            "teacher_vision_input_pca",
            "teacher_vision_stage_taps_pca",
            "teacher_language_input_pca",
            "teacher_language_stage_taps_pca",
        ],
        "contains_task_labels": False,
        "contains_teacher_logits": False,
    }


@torch.inference_mode()
def build_projected_teacher_cache(
    model: torch.nn.Module,
    processor: Any,
    data: DatasetBundle,
    settings: Any,
    device: torch.device,
) -> dict[str, Path]:
    vision_path, language_path = projection_paths(settings)
    vision_pca = load_projection(vision_path, settings.vision_hidden_size).to(device)
    language_pca = load_projection(language_path, settings.text_hidden_size).to(device)
    capture = TeacherTapCapture(model, tuple(settings.language_tap_indexes))
    paths: dict[str, Path] = {}
    try:
        for split, dataset in (("train", data.train), ("validation", data.validation)):
            paths[split] = _build_split(
                split,
                dataset,
                model,
                processor,
                capture,
                vision_pca,
                language_pca,
                settings,
                device,
            )
        return paths
    finally:
        capture.close()


def _build_split(
    split: str,
    dataset: ImageCaptionDataset,
    model: torch.nn.Module,
    processor: Any,
    capture: TeacherTapCapture,
    vision_pca: torch.nn.Module,
    language_pca: torch.nn.Module,
    settings: Any,
    device: torch.device,
) -> Path:
    root = teacher_cache_root(settings)
    manifest_path = root / f"{split}.pt"
    expected = expected_metadata(split, len(dataset), settings, model)
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        changed = [
            key for key, value in expected.items()
            if manifest["metadata"].get(key) != value
        ]
        if changed:
            raise RuntimeError(
                f"Teacher cache metadata mismatch for {split}: {changed}. "
                f"Delete {manifest_path.parent} and rebuild."
            )
        print(f"[precompute_teacher] validated existing cache: {manifest_path}", flush=True)
        return manifest_path
    loader = make_loader(
        dataset,
        settings.feature_batch_size,
        settings.num_workers,
        False,
        settings.seed,
    )
    shard_dir = root / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stored_dtype = torch.float16 if settings.cache_dtype == "float16" else torch.float32
    pending: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    cached = 0
    for batch_index, (images, captions, indexes) in enumerate(loader, start=1):
        cpu_inputs = preprocess_image_text(
            processor, images, captions, settings.caption_prompt_template
        )
        inputs = move_inputs(cpu_inputs, device)
        capture.clear()
        run_multimodal_forward(model, inputs)
        capture.validate_complete()
        vision_lengths = capture.vision_lengths()
        language_lengths = [int(value) for value in cpu_inputs["attention_mask"].sum(1).tolist()]
        _check_lengths(vision_lengths, language_lengths, settings)
        assert capture.vision_input is not None and capture.language_input is not None
        vision_inputs = list(capture.vision_input.split(vision_lengths))
        vision_taps = {
            index: list(capture.vision_taps[index].split(vision_lengths))
            for index in capture.vision_tap_indexes
        }
        patch_counts = [
            int(grid.long().prod())
            for grid in cpu_inputs["image_grid_thw"]
        ]
        pixel_groups = cpu_inputs["pixel_values"].split(patch_counts, dim=0)
        for local, dataset_index in enumerate(indexes):
            language_mask = inputs["attention_mask"][local].bool()
            valid_ids = cpu_inputs["input_ids"][local][
                cpu_inputs["attention_mask"][local].bool()
            ].cpu()
            language_input = capture.language_input[local][language_mask]
            row = {
                "sample_index": int(dataset_index),
                "input_ids": valid_ids,
                "pixel_values": pixel_groups[local].to(stored_dtype).cpu(),
                "image_grid_thw": cpu_inputs["image_grid_thw"][local].cpu(),
                "visual_token_count": int(vision_lengths[local]),
                "language_token_mask": torch.ones(
                    language_lengths[local], dtype=torch.bool
                ),
                "teacher_vision_input_pca": vision_pca.encode(
                    vision_inputs[local]
                ).to(stored_dtype).cpu(),
                "teacher_vision_stage_taps_pca": [
                    vision_pca.encode(vision_taps[index][local]).to(stored_dtype).cpu()
                    for index in capture.vision_tap_indexes
                ],
                "teacher_language_input_pca": language_pca.encode(
                    language_input
                ).to(stored_dtype).cpu(),
                "teacher_language_stage_taps_pca": [
                    language_pca.encode(
                        capture.language_taps[index][local][language_mask]
                    ).to(stored_dtype).cpu()
                    for index in capture.language_tap_indexes
                ],
            }
            pending.append(row)
            if len(pending) >= settings.teacher_cache_shard_size:
                shards.append(_flush_shard(shard_dir, len(shards), pending))
                pending = []
        cached += len(images)
        if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
            print(
                f"[precompute_teacher] {split} batch={batch_index}/{len(loader)} "
                f"cached={cached}/{len(dataset)}",
                flush=True,
            )
    if pending:
        shards.append(_flush_shard(shard_dir, len(shards), pending))
    metadata = {
        **expected,
        "shard_count": len(shards),
        "total_cache_bytes": sum(record["bytes"] for record in shards),
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "shards": shards}, manifest_path)
    write_json(root / f"{split}_metadata.json", metadata)
    return manifest_path


def _flush_shard(
    directory: Path,
    number: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = directory / f"shard_{number:06d}.pt"
    payload = {
        "sample_indices": torch.tensor([row["sample_index"] for row in rows]),
        "input_ids": [row["input_ids"] for row in rows],
        "pixel_values": [row["pixel_values"] for row in rows],
        "image_grid_thw": torch.stack([row["image_grid_thw"] for row in rows]),
        "visual_token_counts": torch.tensor([row["visual_token_count"] for row in rows]),
        "language_token_masks": [row["language_token_mask"] for row in rows],
        "teacher_vision_input_pca": [row["teacher_vision_input_pca"] for row in rows],
        "teacher_vision_stage_taps_pca": [
            row["teacher_vision_stage_taps_pca"] for row in rows
        ],
        "teacher_language_input_pca": [row["teacher_language_input_pca"] for row in rows],
        "teacher_language_stage_taps_pca": [
            row["teacher_language_stage_taps_pca"] for row in rows
        ],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path),
        "count": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


class ProjectedTeacherCacheStore:
    def __init__(self, manifest_path: Path, max_cached_shards: int = 8) -> None:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Projected teacher cache is missing: {manifest_path}. "
                "Run --phase precompute_teacher."
            )
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        self.metadata = manifest["metadata"]
        self.shards = manifest["shards"]
        self.max_cached_shards = int(max_cached_shards)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._ends: list[int] = []
        offset = 0
        for record in self.shards:
            offset += int(record["count"])
            self._ends.append(offset)

    def __len__(self) -> int:
        return int(self.metadata["sample_count"])

    def get_many(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        located: dict[int, list[tuple[int, int, int]]] = {}
        for output_position, raw_index in enumerate(indices):
            index = int(raw_index)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            shard = bisect_right(self._ends, index)
            start = 0 if shard == 0 else self._ends[shard - 1]
            located.setdefault(shard, []).append((output_position, index, index - start))
        result: list[dict[str, Any] | None] = [None] * len(indices)
        for shard, requests in located.items():
            payload = self._load(shard)
            for output_position, index, position in requests:
                if int(payload["sample_indices"][position]) != index:
                    raise RuntimeError("Teacher cache sample ordering mismatch")
                result[output_position] = {
                    key: payload[key][position]
                    for key in (
                        "sample_indices",
                        "input_ids",
                        "pixel_values",
                        "image_grid_thw",
                        "visual_token_counts",
                        "language_token_masks",
                        "teacher_vision_input_pca",
                        "teacher_vision_stage_taps_pca",
                        "teacher_language_input_pca",
                        "teacher_language_stage_taps_pca",
                    )
                }
        return [row for row in result if row is not None]

    def _load(self, shard: int) -> dict[str, Any]:
        if shard in self._cache:
            payload = self._cache.pop(shard)
            self._cache[shard] = payload
            return payload
        payload = torch.load(self.shards[shard]["path"], map_location="cpu", weights_only=True)
        self._cache[shard] = payload
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return payload


def collate_cached_rows(
    rows: Sequence[dict[str, Any]],
    pad_token_id: int = 0,
    padding_side: str = "left",
) -> dict[str, Any]:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    language_targets = [
        torch.zeros((len(rows), max_length, rows[0]["teacher_language_stage_taps_pca"][0].shape[-1]))
        for _ in range(4)
    ]
    for batch_index, row in enumerate(rows):
        length = len(row["input_ids"])
        start = max_length - length if padding_side == "left" else 0
        input_ids[batch_index, start : start + length] = row["input_ids"].long()
        attention_mask[batch_index, start : start + length] = 1
        for stage in range(4):
            language_targets[stage][batch_index, start : start + length] = (
                row["teacher_language_stage_taps_pca"][stage].float()
            )
    return {
        "sample_indices": torch.tensor([int(row["sample_indices"]) for row in rows]),
        "inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.cat([row["pixel_values"] for row in rows]),
            "image_grid_thw": torch.stack([row["image_grid_thw"] for row in rows]).long(),
        },
        "visual_token_counts": [int(row["visual_token_counts"]) for row in rows],
        "vision_targets": [
            torch.cat([
                row["teacher_vision_stage_taps_pca"][stage].float()
                for row in rows
            ])
            for stage in range(4)
        ],
        "language_targets": language_targets,
        "language_mask": attention_mask.bool(),
    }


def load_cache_stores(settings: Any) -> dict[str, ProjectedTeacherCacheStore]:
    root = teacher_cache_root(settings)
    return {
        split: ProjectedTeacherCacheStore(
            root / f"{split}.pt", settings.teacher_cache_lru_shards
        )
        for split in ("train", "validation")
    }


def _check_lengths(
    vision_lengths: list[int],
    language_lengths: list[int],
    settings: Any,
) -> None:
    if max(vision_lengths) > settings.max_visual_tokens:
        raise RuntimeError(
            f"visual token count {max(vision_lengths)} exceeds 224. Lower "
            "processor_max_pixels; silent truncation is forbidden."
        )
    if max(language_lengths) > settings.max_language_tokens:
        raise RuntimeError(
            f"language sequence length {max(language_lengths)} exceeds 224. Shorten "
            "the caption/prompt; silent truncation is forbidden."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
