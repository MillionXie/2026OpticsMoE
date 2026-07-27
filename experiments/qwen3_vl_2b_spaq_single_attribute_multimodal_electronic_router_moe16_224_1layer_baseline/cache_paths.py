from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


CACHE_IDENTITY_SCHEMA_VERSION = 1


def cache_identity(settings: Any) -> dict[str, Any]:
    """Return the immutable identity shared by teacher and processor caches.

    The optical geometry and training hyperparameters are deliberately absent:
    both caches describe the frozen electronic Qwen teacher inputs/targets and
    can therefore be reused across optical debug runs with different output
    directories.  Every field that changes Qwen input tokens or target rows is
    included.
    """

    if not settings.split_digest:
        raise RuntimeError("Dataset split_digest must be resolved before selecting a precompute cache")
    return {
        "cache_identity_schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
        "dataset": settings.dataset,
        "task": settings.task_name,
        "data_root": str(settings.data_root),
        "annotations_file": settings.resolved_annotations_file,
        "split_digest": settings.split_digest,
        "train_fraction": settings.train_fraction,
        "train_image_limit": settings.train_image_limit,
        "test_image_limit": settings.test_image_limit,
        "model_id": str(settings.model_id),
        "classification_prompt": settings.classification_prompt,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "cache_dtype": settings.cache_dtype,
        "dtype": settings.dtype,
        "attention_implementation": settings.attn_implementation,
        "input_color_mode": "RGB",
    }


def cache_identity_digest(settings: Any) -> str:
    encoded = json.dumps(
        cache_identity(settings),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def precompute_cache_root(settings: Any) -> Path:
    task = re.sub(r"[^a-z0-9]+", "_", settings.task_name.lower()).strip("_")
    return Path(settings.precompute_cache_dir) / task / cache_identity_digest(settings)[:20]


def teacher_cache_root(settings: Any) -> Path:
    return precompute_cache_root(settings) / "teacher_cache"


def processor_cache_root(settings: Any) -> Path:
    return precompute_cache_root(settings) / "processor_cache"
