from __future__ import annotations

import bisect
import hashlib
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .features import (
    move_inputs,
    multimodal_forward_features,
    pool_answer_hidden_state,
    preprocess_image_text,
)
from .optics import VisionBlockReplacement


CACHE_SCHEMA_VERSION = 1


def expected_teacher_cache_metadata(
    *,
    split: str,
    samples: int,
    settings: Any,
    model: nn.Module,
    replacement: VisionBlockReplacement,
    class_names: Sequence[str],
) -> dict[str, Any]:
    vision_config = model.config.vision_config
    text_config = model.config.text_config
    teacher_checkpoint = settings.output_dir / "checkpoints" / "teacher_mlp.pt"
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(
            f"Teacher MLP checkpoint is required for teacher cache metadata: {teacher_checkpoint}"
        )
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "samples": int(samples),
        "model_id": str(settings.model_id),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "teacher_mlp_sha256": _sha256_file(teacher_checkpoint),
        "classification_prompt": settings.classification_prompt,
        "data_root": str(settings.data_root),
        "class_names": list(class_names),
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "resize_to": settings.resize_to,
        "vision_depth": int(vision_config.depth),
        "vision_hidden_size": int(vision_config.hidden_size),
        "text_hidden_size": int(text_config.hidden_size),
        "groups": [list(group) for group in replacement.block_groups],
        "dtype": settings.dtype,
        "attention_implementation": settings.attn_implementation,
        "cache_dtype": settings.cache_dtype,
        "feature_source": "full_electronic_qwen3_vl_2b_teacher",
    }


def build_teacher_cache(
    *,
    split: str,
    model: nn.Module,
    processor: Any,
    replacement: VisionBlockReplacement,
    teacher_head: nn.Module,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    dataset_size: int,
    class_names: Sequence[str],
    settings: Any,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> Path:
    manifest_path = settings.output_dir / "teacher_cache" / f"{split}.pt"
    expected = expected_teacher_cache_metadata(
        split=split,
        samples=dataset_size,
        settings=settings,
        model=model,
        replacement=replacement,
        class_names=class_names,
    )
    valid, changed = validate_teacher_cache(manifest_path, expected)
    if valid:
        log(f"reusing validated teacher group cache: split={split} path={manifest_path}")
        return manifest_path
    if manifest_path.exists():
        log(f"teacher group cache invalidated: split={split} changed_fields={changed}")

    shard_dir = manifest_path.parent / f"{split}_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    replacement.use_teacher()
    replacement.clear_captures()
    model.eval()
    teacher_head.eval()
    group_names = [_group_name(group) for group in replacement.block_groups]
    pending = _empty_pending(group_names)
    shard_records: list[dict[str, Any]] = []
    grid_values: Counter[tuple[int, int, int]] = Counter()
    grid_shapes: Counter[tuple[int, ...]] = Counter()
    sample_index = 0
    shard_index = 0
    iterator: Any = loader
    if settings.progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc=f"Teacher cache {split}")
        except ImportError:
            pass

    with torch.inference_mode():
        for images, labels in iterator:
            inputs = move_inputs(
                preprocess_image_text(processor, images, settings.classification_prompt),
                device,
            )
            replacement.clear_captures()
            hidden = multimodal_forward_features(model, inputs)
            answer_hidden, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
            teacher_logits = teacher_head(answer_hidden).float()
            token_counts = replacement.teacher_token_counts()
            teacher_inputs = replacement.teacher_inputs()
            teacher_outputs = replacement.teacher_outputs()
            _validate_group_shapes(
                token_counts, teacher_inputs, teacher_outputs, len(images), replacement.block_groups
            )
            split_inputs = [list(value.split(token_counts, dim=0)) for value in teacher_inputs]
            split_outputs = [list(value.split(token_counts, dim=0)) for value in teacher_outputs]
            grids = inputs["image_grid_thw"].detach().cpu()
            grid_shapes[tuple(grids.shape)] += 1
            for row in grids.tolist():
                grid_values[tuple(int(value) for value in row)] += 1

            for batch_index, label in enumerate(labels.tolist()):
                pending["sample_indices"].append(sample_index)
                pending["labels"].append(int(label))
                pending["teacher_logits"].append(teacher_logits[batch_index].detach().cpu())
                pending["teacher_answer_hidden"].append(
                    answer_hidden[batch_index].detach().cpu()
                )
                pending["image_grid_thw"].append(grids[batch_index])
                pending["token_counts"].append(int(token_counts[batch_index]))
                for group_index, group_name in enumerate(group_names):
                    pending["groups"][group_name]["input"].append(
                        _cache_tensor(split_inputs[group_index][batch_index], settings.cache_dtype)
                    )
                    pending["groups"][group_name]["output"].append(
                        _cache_tensor(split_outputs[group_index][batch_index], settings.cache_dtype)
                    )
                sample_index += 1
                if len(pending["labels"]) >= settings.teacher_cache_shard_size:
                    shard_records.append(
                        _flush_shard(pending, shard_dir, shard_index, manifest_path.parent)
                    )
                    shard_index += 1
                    pending = _empty_pending(group_names)

    if pending["labels"]:
        shard_records.append(_flush_shard(pending, shard_dir, shard_index, manifest_path.parent))
    if sample_index != dataset_size:
        raise RuntimeError(
            f"Teacher cache sample count mismatch for {split}: {sample_index} != {dataset_size}"
        )
    metadata = dict(expected)
    metadata["image_grid_thw_summary"] = {
        "batch_shapes": [
            {"shape": list(shape), "batches": count}
            for shape, count in sorted(grid_shapes.items())
        ],
        "values": [
            {"value": list(value), "samples": count}
            for value, count in sorted(grid_values.items())
        ],
    }
    manifest = {
        "metadata": metadata,
        "sample_count": sample_index,
        "group_names": group_names,
        "shards": shard_records,
    }
    _atomic_torch_save(manifest, manifest_path)
    return manifest_path


def validate_teacher_cache(
    manifest_path: Path, expected: dict[str, Any]
) -> tuple[bool, list[str]]:
    if not manifest_path.is_file():
        return False, ["manifest_missing"]
    try:
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    except Exception:
        return False, ["manifest_unreadable"]
    cached = manifest.get("metadata", {})
    changed = sorted(key for key, value in expected.items() if cached.get(key) != value)
    if int(manifest.get("sample_count", -1)) != int(expected["samples"]):
        changed.append("sample_count")
    base = manifest_path.parent
    for shard in manifest.get("shards", []):
        if not (base / shard["path"]).is_file():
            changed.append(f"missing_shard:{shard['path']}")
            break
    return not changed, changed


class TeacherCacheStore:
    def __init__(self, manifest_path: Path, expected: dict[str, Any]) -> None:
        valid, changed = validate_teacher_cache(manifest_path, expected)
        if not valid:
            raise RuntimeError(
                f"Teacher cache is missing or stale: {manifest_path}; changed_fields={changed}. "
                "Run --phase teacher_cache first."
            )
        self.manifest_path = manifest_path
        self.manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        self.shards = list(self.manifest["shards"])
        self.ends = [int(shard["end_index_exclusive"]) for shard in self.shards]
        self.group_names = list(self.manifest["group_names"])
        self._loaded_index: int | None = None
        self._loaded: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_loaded_index"] = None
        state["_loaded"] = None
        return state

    def get(self, index: int) -> dict[str, Any]:
        shard_index = bisect.bisect_right(self.ends, int(index))
        if shard_index >= len(self.shards):
            raise IndexError(index)
        shard_info = self.shards[shard_index]
        start = int(shard_info["start_index"])
        end = int(shard_info["end_index_exclusive"])
        if not start <= index < end:
            raise IndexError(index)
        if self._loaded_index != shard_index:
            path = self.manifest_path.parent / shard_info["path"]
            self._loaded = torch.load(path, map_location="cpu", weights_only=True)
            self._loaded_index = shard_index
        assert self._loaded is not None
        local = index - start
        return {
            "label": int(self._loaded["labels"][local]),
            "teacher_logits": self._loaded["teacher_logits"][local],
            "teacher_answer_hidden": self._loaded["teacher_answer_hidden"][local],
            "token_count": int(self._loaded["token_counts"][local]),
            "group_outputs": {
                name: self._loaded["groups"][name]["output"][local]
                for name in self.group_names
            },
        }


class CachedTeacherDataset(Dataset[dict[str, Any]]):
    def __init__(self, base: Dataset[Any], store: TeacherCacheStore) -> None:
        if len(base) != int(store.manifest["sample_count"]):
            raise ValueError("Dataset and teacher cache lengths differ")
        self.base = base
        self.store = store

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.base[index]
        cached = self.store.get(index)
        if int(label) != cached["label"]:
            raise RuntimeError(
                f"Teacher cache label mismatch at sample {index}: {cached['label']} != {label}"
            )
        return {"image": image, "label": int(label), "index": int(index), **cached}


class CacheAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        shards: Sequence[dict[str, Any]],
        eligible_indices: Sequence[int],
        batch_size: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        eligible = set(int(index) for index in eligible_indices)
        self.by_shard = [
            [
                index
                for index in range(
                    int(shard["start_index"]), int(shard["end_index_exclusive"])
                )
                if index in eligible
            ]
            for shard in shards
        ]
        self.by_shard = [indices for indices in self.by_shard if indices]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [list(indices) for indices in self.by_shard]
        rng.shuffle(groups)
        for indices in groups:
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.by_shard)


def make_cached_teacher_loader(
    dataset: CachedTeacherDataset,
    train_indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader[Any]:
    sampler = CacheAwareBatchSampler(
        dataset.store.shards, train_indices, batch_size, seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=cached_teacher_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def cached_teacher_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    group_names = list(batch[0]["group_outputs"])
    return {
        "images": [sample["image"] for sample in batch],
        "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
        "sample_indices": torch.tensor(
            [sample["index"] for sample in batch], dtype=torch.long
        ),
        "teacher_logits": torch.stack([sample["teacher_logits"] for sample in batch]),
        "teacher_answer_hidden": torch.stack(
            [sample["teacher_answer_hidden"] for sample in batch]
        ),
        "teacher_token_counts": [int(sample["token_count"]) for sample in batch],
        "teacher_group_outputs": [
            torch.cat([sample["group_outputs"][name] for sample in batch], dim=0)
            for name in group_names
        ],
        "group_names": group_names,
    }


def _validate_group_shapes(
    token_counts: Sequence[int],
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    batch_size: int,
    groups: Sequence[tuple[int, int]],
) -> None:
    if len(token_counts) != batch_size or any(count <= 0 for count in token_counts):
        raise RuntimeError(
            f"Invalid per-image teacher token counts: batch={batch_size}, counts={token_counts}"
        )
    packed_tokens = sum(token_counts)
    for group, input_hidden, output_hidden in zip(groups, inputs, outputs):
        if input_hidden.ndim != 2 or output_hidden.ndim != 2:
            raise RuntimeError(f"Teacher group {group} hidden states must be packed 2D tensors")
        if input_hidden.shape != output_hidden.shape:
            raise RuntimeError(f"Teacher group {group} input/output shapes differ")
        if input_hidden.shape[0] != packed_tokens:
            raise RuntimeError(
                f"Teacher group {group} packed tokens {input_hidden.shape[0]} != {packed_tokens}"
            )


def _group_name(group: tuple[int, int]) -> str:
    return f"group_{group[0]}_{group[1]}"


def _cache_tensor(value: torch.Tensor, cache_dtype: str) -> torch.Tensor:
    value = value.detach().cpu()
    return value.half() if cache_dtype == "float16" else value.float()


def _empty_pending(group_names: Sequence[str]) -> dict[str, Any]:
    return {
        "sample_indices": [],
        "labels": [],
        "teacher_logits": [],
        "teacher_answer_hidden": [],
        "image_grid_thw": [],
        "token_counts": [],
        "groups": {
            name: {"input": [], "output": []} for name in group_names
        },
    }


def _flush_shard(
    pending: dict[str, Any],
    shard_dir: Path,
    shard_index: int,
    manifest_dir: Path,
) -> dict[str, Any]:
    start = int(pending["sample_indices"][0])
    end = int(pending["sample_indices"][-1]) + 1
    payload = {
        "sample_indices": torch.tensor(pending["sample_indices"], dtype=torch.long),
        "labels": torch.tensor(pending["labels"], dtype=torch.long),
        "teacher_logits": torch.stack(pending["teacher_logits"]),
        "teacher_answer_hidden": torch.stack(pending["teacher_answer_hidden"]),
        "image_grid_thw": torch.stack(pending["image_grid_thw"]),
        "token_counts": torch.tensor(pending["token_counts"], dtype=torch.long),
        "groups": pending["groups"],
    }
    path = shard_dir / f"shard_{shard_index:06d}.pt"
    _atomic_torch_save(payload, path)
    return {
        "path": str(path.relative_to(manifest_dir)),
        "start_index": start,
        "end_index_exclusive": end,
        "samples": end - start,
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
