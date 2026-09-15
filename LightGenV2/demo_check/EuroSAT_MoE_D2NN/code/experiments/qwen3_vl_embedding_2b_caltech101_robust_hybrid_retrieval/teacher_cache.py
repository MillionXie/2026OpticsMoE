from __future__ import annotations

from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.cache_teacher_embeddings import (
    CACHE_VERSION,
    LOW_DIMENSION_METHOD,
    TeacherEmbeddingStore,
    build_teacher_embedding_cache as build_full_teacher_embedding_cache,
    cache_identity,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    write_json,
)


def build_teacher_embedding_cache(
    loaded,
    bundle,
    settings,
    *,
    force: bool = False,
) -> Path:
    destination = settings.teacher_cache_path
    if destination.is_file() and not force:
        TeacherEmbeddingStore(destination, bundle, settings)
        print(f"Teacher cache already valid: {destination}")
        return destination
    if not force and settings.teacher_cache_source_path is not None:
        derived = _derive_subset_cache(
            settings.teacher_cache_source_path,
            destination,
            bundle,
            settings,
        )
        if derived is not None:
            return derived
    path = build_full_teacher_embedding_cache(
        loaded, bundle, settings, force=force
    )
    _write_cache_metadata(path)
    return path


def _derive_subset_cache(
    source_path: Path,
    destination: Path,
    bundle,
    settings,
) -> Path | None:
    if not source_path.is_file():
        print(f"Shared all-class cache is absent; computing normally: {source_path}")
        return None
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = source.get("metadata", {})
    required = {
        "cache_version": CACHE_VERSION,
        "model_id": settings.model_id,
        "instruction": settings.instruction,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "embedding_dim": settings.embedding_dim,
        "low_dimension_method": LOW_DIMENSION_METHOD,
    }
    mismatch = {
        key: (metadata.get(key), expected)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"All-class Teacher cache is incompatible: {mismatch}")
    records = source.get("records")
    embeddings = source.get("teacher_embeddings")
    if not isinstance(records, list) or not torch.is_tensor(embeddings):
        raise RuntimeError(f"Malformed Teacher cache: {source_path}")
    if embeddings.shape != (len(records), settings.embedding_dim):
        raise RuntimeError("All-class Teacher cache record/tensor shape mismatch")
    by_path = {
        str(Path(record["image_path"]).resolve()): index
        for index, record in enumerate(records)
    }
    samples = bundle.all_samples()
    missing = [
        str(sample.image_path.resolve())
        for sample in samples
        if str(sample.image_path.resolve()) not in by_path
    ]
    if missing:
        raise RuntimeError(
            f"All-class Teacher cache lacks {len(missing)} target images; first={missing[0]}"
        )
    indexes = [by_path[str(sample.image_path.resolve())] for sample in samples]
    values = embeddings[indexes].to(torch.float16).contiguous()
    target_records = [sample.manifest_record() for sample in samples]
    target_metadata = {
        **cache_identity(bundle, settings),
        "embedding_shape": list(values.shape),
        "embedding_dtype": str(values.dtype),
        "teacher_parameters_trainable": 0,
        "model_mode": "eval",
        "teacher_embedding_actual_output_shape": [settings.embedding_dim],
        "derived_without_teacher_forward": True,
        "derived_from": str(source_path),
        "source_manifest_sha256": metadata.get("manifest_sha256"),
        "norm_min": float(values.float().norm(dim=-1).min()),
        "norm_max": float(values.float().norm(dim=-1).max()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "metadata": target_metadata,
            "records": target_records,
            "teacher_embeddings": values,
        },
        temporary,
    )
    temporary.replace(destination)
    write_json(destination.parent / "metadata.json", target_metadata)
    TeacherEmbeddingStore(destination, bundle, settings)
    print(
        f"Derived reusable target-10 cache without Qwen forward: "
        f"{len(target_records):,} images -> {destination}"
    )
    return destination


def _write_cache_metadata(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    write_json(path.parent / "metadata.json", payload.get("metadata", {}))
