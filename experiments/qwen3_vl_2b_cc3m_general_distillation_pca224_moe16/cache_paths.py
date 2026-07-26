from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .pca import file_sha256, projection_paths


def cache_identity(settings: Any) -> dict[str, Any]:
    vision_path, language_path = projection_paths(settings)
    if not settings.manifest_digest:
        raise RuntimeError("Dataset manifest digest has not been resolved")
    return {
        "schema_version": 1,
        "dataset": settings.dataset,
        "manifest_digest": settings.manifest_digest,
        "caption_prompt_template": settings.caption_prompt_template,
        "model_id": str(settings.model_id),
        "dtype": settings.dtype,
        "cache_dtype": settings.cache_dtype,
        "attention_implementation": settings.attn_implementation,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "max_visual_tokens": settings.max_visual_tokens,
        "max_language_tokens": settings.max_language_tokens,
        "vision_pca_sha256": file_sha256(vision_path),
        "language_pca_sha256": file_sha256(language_path),
        "deepstack_visual_indexes": list(settings.deepstack_visual_indexes or []),
        "language_tap_indexes": list(settings.language_tap_indexes or []),
    }


def cache_identity_digest(settings: Any) -> str:
    value = json.dumps(
        cache_identity(settings),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def teacher_cache_root(settings: Any) -> Path:
    return (
        Path(settings.precompute_cache_dir)
        / "teacher_cache"
        / str(settings.manifest_digest)
        / cache_identity_digest(settings)[:20]
    )
