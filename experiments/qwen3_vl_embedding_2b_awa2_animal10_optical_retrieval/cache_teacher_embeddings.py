from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .features import (
    move_inputs,
    preprocess_images,
    teacher_embeddings,
    validate_token_budgets,
)
from .io_utils import write_json
from .modeling import LoadedBackbone, load_backbone
from .prepare_awa2_retrieval_subset import (
    AwA2RetrievalBundle,
    AwA2RetrievalDataset,
    AwA2Sample,
    collate_awa2,
    prepare_awa2_subset,
)
from .settings import Settings, load_settings


CACHE_VERSION = 1
LOW_DIMENSION_METHOD = "last_valid_token_first_64_matryoshka_dimensions_l2_normalized"


class TeacherEmbeddingStore:
    def __init__(
        self,
        path: Path,
        bundle: AwA2RetrievalBundle,
        settings: Settings,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Teacher embedding cache is missing: {path}. Run cache_teacher_embeddings."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = cache_identity(bundle, settings)
        metadata = payload.get("metadata", {})
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                "Teacher embedding cache identity mismatch. Delete/rebuild the cache. "
                f"Differences: {mismatch}"
            )
        embeddings = payload.get("teacher_embeddings")
        records = payload.get("records")
        if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
            raise RuntimeError("Teacher cache has no 2-D teacher_embeddings tensor")
        if embeddings.shape != (len(records), settings.embedding_dim):
            raise RuntimeError(
                f"Teacher cache shape {tuple(embeddings.shape)} does not match records/dimension"
            )
        normalized = embeddings.float()
        norms = normalized.norm(dim=-1)
        if not torch.isfinite(normalized).all() or torch.any((norms - 1.0).abs() > 2e-3):
            raise RuntimeError("Teacher cache contains non-finite or non-normalized embeddings")
        self.metadata = metadata
        self.records = records
        self.embeddings = normalized
        self.by_path: dict[str, int] = {}
        for index, record in enumerate(records):
            key = str(Path(record["image_path"]).resolve())
            if key in self.by_path:
                raise RuntimeError(f"Duplicate image path in teacher cache: {key}")
            self.by_path[key] = index

    def lookup(self, samples: Sequence[AwA2Sample]) -> torch.Tensor:
        try:
            indexes = [self.by_path[str(sample.image_path.resolve())] for sample in samples]
        except KeyError as exc:
            raise RuntimeError(f"Image is absent from teacher cache: {exc.args[0]}") from exc
        return self.embeddings[indexes]

    def split(
        self, samples: Sequence[AwA2Sample]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lookup(samples), torch.tensor(
            [sample.class_index for sample in samples], dtype=torch.long
        )


def cache_identity(
    bundle: AwA2RetrievalBundle, settings: Settings
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dataset": f"AwA2-{settings.dataset_variant}-class-retrieval",
        "manifest_sha256": bundle.manifest_digest,
        "selected_classes": list(bundle.class_names),
        "model_id": settings.model_id,
        "instruction": settings.instruction,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "embedding_dim": settings.embedding_dim,
        "low_dimension_method": LOW_DIMENSION_METHOD,
    }


@torch.no_grad()
def build_teacher_embedding_cache(
    loaded: LoadedBackbone,
    bundle: AwA2RetrievalBundle,
    settings: Settings,
    *,
    force: bool = False,
) -> Path:
    path = settings.teacher_cache_path
    if path.is_file() and not force:
        TeacherEmbeddingStore(path, bundle, settings)
        print(f"Teacher cache already valid: {path}")
        return path
    if not force and settings.teacher_cache_source_path is not None:
        derived = _derive_subset_cache(
            settings.teacher_cache_source_path, path, bundle, settings
        )
        if derived is not None:
            return derived
    samples = bundle.all_samples()
    dataset = AwA2RetrievalDataset(samples, settings.image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=settings.teacher_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_awa2,
    )
    loaded.model.eval().requires_grad_(False)
    chunks: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, 1):
        inputs = preprocess_images(
            loaded.processor, batch["images"], settings.instruction
        )
        validate_token_budgets(inputs, settings)
        inputs = move_inputs(inputs, loaded.device)
        embeddings = teacher_embeddings(
            loaded.model, inputs, settings.embedding_dim
        ).detach().cpu()
        if embeddings.shape != (len(batch["samples"]), settings.embedding_dim):
            raise RuntimeError(
                f"Teacher embedding shape is {tuple(embeddings.shape)}, expected "
                f"({len(batch['samples'])},{settings.embedding_dim})"
            )
        chunks.append(embeddings.to(torch.float16))
        for sample in batch["samples"]:
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": str(sample.image_path.resolve()),
                    "class_id": sample.class_id,
                    "class_name": sample.class_name,
                    "class_index": sample.class_index,
                    "split": sample.split,
                    "source_split": sample.source_split,
                    "is_gallery": sample.is_gallery,
                }
            )
        if batch_index % 10 == 0 or len(records) == len(samples):
            print(f"[teacher_cache] cached={len(records):,}/{len(samples):,}")
    values = torch.cat(chunks, dim=0)
    identity = cache_identity(bundle, settings)
    metadata = {
        **identity,
        "embedding_shape": list(values.shape),
        "embedding_dtype": str(values.dtype),
        "teacher_parameters_trainable": 0,
        "model_mode": "eval",
        "elapsed_sec": time.perf_counter() - started,
        "norm_min": float(values.float().norm(dim=-1).min()),
        "norm_max": float(values.float().norm(dim=-1).max()),
        "teacher_embedding_actual_output_shape": [settings.embedding_dim],
    }
    payload = {
        "metadata": metadata,
        "records": records,
        "teacher_embeddings": values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    write_json(settings.output_dir / "teacher_cache" / "metadata.json", metadata)
    return path


def _sample_record(sample: AwA2Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path.resolve()),
        "class_id": sample.class_id,
        "class_name": sample.class_name,
        "class_index": sample.class_index,
        "split": sample.split,
        "source_split": sample.source_split,
        "is_gallery": sample.is_gallery,
    }


def _derive_subset_cache(
    source_path: Path,
    destination: Path,
    bundle: AwA2RetrievalBundle,
    settings: Settings,
) -> Path | None:
    """Slice a target-class cache from the all-class cache without rerunning Qwen."""
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        print(
            f"Teacher cache source is not available yet; computing normally: {source_path}"
        )
        return None
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = source.get("metadata", {})
    required_identity = {
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
        for key, expected in required_identity.items()
        if metadata.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(
            "Configured all-class teacher cache is incompatible with the target stage: "
            f"{mismatch}"
        )
    source_records = source.get("records")
    source_embeddings = source.get("teacher_embeddings")
    if not isinstance(source_records, list) or not torch.is_tensor(source_embeddings):
        raise RuntimeError(f"Malformed source teacher cache: {source_path}")
    if source_embeddings.shape != (len(source_records), settings.embedding_dim):
        raise RuntimeError(
            f"Source teacher cache shape/record mismatch: {tuple(source_embeddings.shape)}"
        )
    by_path = {
        str(Path(record["image_path"]).resolve()): index
        for index, record in enumerate(source_records)
    }
    samples = bundle.all_samples()
    missing = [
        str(sample.image_path.resolve())
        for sample in samples
        if str(sample.image_path.resolve()) not in by_path
    ]
    if missing:
        raise RuntimeError(
            f"All-class teacher cache lacks {len(missing)} target images; first={missing[0]}"
        )
    indexes = [by_path[str(sample.image_path.resolve())] for sample in samples]
    values = source_embeddings[indexes].to(torch.float16).contiguous()
    records = [_sample_record(sample) for sample in samples]
    derived_metadata = {
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
            "metadata": derived_metadata,
            "records": records,
            "teacher_embeddings": values,
        },
        temporary,
    )
    temporary.replace(destination)
    write_json(destination.parent / "metadata.json", derived_metadata)
    TeacherEmbeddingStore(destination, bundle, settings)
    print(
        f"Derived target teacher cache from all-class cache: {len(records):,} images -> {destination}"
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_awa2_subset(settings, persist=True)
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    loaded = load_backbone(settings, device)
    path = build_teacher_embedding_cache(
        loaded, bundle, settings, force=args.force
    )
    print(f"Teacher embeddings saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
