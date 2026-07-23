from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .clip_teacher import FrozenClipTeacher, save_text_prototypes
from .datasets import CLIP_MEAN, CLIP_STD, ImageNetBundle
from .settings import ExperimentSettings


CACHE_SCHEMA_VERSION = 1


def cache_directory(settings: ExperimentSettings) -> Path:
    return settings.training.output_dir / "clip_cache"


def expected_cache_metadata(
    split: str,
    dataset,
    bundle: ImageNetBundle,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset": settings.dataset.name,
        "dataset_source": settings.dataset.source,
        "data_root": str(settings.dataset.root),
        "hf_dataset_id": (
            settings.dataset.hf_dataset_id
            if settings.dataset.source == "huggingface"
            else None
        ),
        "hf_revision": (
            settings.dataset.hf_revision
            if settings.dataset.source == "huggingface"
            else None
        ),
        "source_fingerprint": getattr(dataset, "fingerprint", None),
        "dataset_digest": bundle.digest,
        "split": split,
        "base_samples": dataset.base_sample_count,
        "views_per_image": dataset.views,
        "feature_dim": settings.model.clip_projection_dim,
        "cache_dtype": settings.clip.cache_dtype,
        "clip_model": settings.clip.model_name,
        "clip_normalization_mean": list(CLIP_MEAN),
        "clip_normalization_std": list(CLIP_STD),
        "image_size": settings.model.image_size,
        "patch_size": settings.model.patch_size,
        "train_transform_version": settings.clip.train_transform_version,
        "transform_schema_version": 1,
        "augmentation_seed": settings.dataset.seed,
        "random_resized_crop_scale": list(settings.clip.random_resized_crop_scale),
        "random_resized_crop_ratio": list(settings.clip.random_resized_crop_ratio),
        "horizontal_flip_probability": settings.clip.horizontal_flip_probability,
        "randaugment_enabled": settings.clip.randaugment_enabled,
        "randaugment_num_ops": settings.clip.randaugment_num_ops,
        "randaugment_magnitude": settings.clip.randaugment_magnitude,
        "class_names": bundle.class_names,
        "folder_classes": bundle.folder_classes,
        "torchvision_version": torchvision_version,
    }


def _paths(split: str, settings: ExperimentSettings) -> dict[str, Path]:
    root = cache_directory(settings)
    return {
        "features": root / f"{split}_clip_embeddings.npy",
        "complete": root / f"{split}_complete.npy",
        "labels": root / f"{split}_labels.npy",
        "metadata": root / f"{split}_metadata.json",
    }


def _metadata_matches(saved: dict, expected: dict) -> dict[str, tuple[Any, Any]]:
    return {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }


@torch.no_grad()
def build_clip_cache(
    split: str,
    dataset: ImageNetViewDataset,
    bundle: ImageNetBundle,
    teacher: FrozenClipTeacher,
    settings: ExperimentSettings,
    device: torch.device,
) -> dict:
    paths = _paths(split, settings)
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    expected = expected_cache_metadata(split, dataset, bundle, settings)
    shape = (
        dataset.base_sample_count,
        dataset.views,
        settings.model.clip_projection_dim,
    )
    dtype = np.float16 if settings.clip.cache_dtype == "float16" else np.float32
    if paths["metadata"].is_file():
        saved = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        mismatches = _metadata_matches(saved, expected)
        if mismatches:
            raise RuntimeError(
                f"Existing CLIP {split} cache is incompatible: {mismatches}. "
                f"Delete {paths['metadata'].parent} and rebuild; silent reuse is forbidden."
            )
        for key in ("features", "complete", "labels"):
            if not paths[key].is_file():
                raise RuntimeError(f"Cache metadata exists but {paths[key]} is missing")
        features = np.lib.format.open_memmap(paths["features"], mode="r+")
        complete = np.lib.format.open_memmap(paths["complete"], mode="r+")
    else:
        features = np.lib.format.open_memmap(
            paths["features"], mode="w+", dtype=dtype, shape=shape
        )
        complete = np.lib.format.open_memmap(
            paths["complete"],
            mode="w+",
            dtype=np.bool_,
            shape=(dataset.base_sample_count, dataset.views),
        )
        complete[:] = False
        labels = np.lib.format.open_memmap(
            paths["labels"],
            mode="w+",
            dtype=np.int64,
            shape=(dataset.base_sample_count,),
        )
        labels[:] = np.asarray(dataset.targets, dtype=np.int64)
        labels.flush()
        initial = {
            **expected,
            "status": "building",
            "completed_entries": 0,
            "total_entries": int(dataset.base_sample_count * dataset.views),
            "features_path": str(paths["features"]),
            "labels_path": str(paths["labels"]),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        paths["metadata"].write_text(
            json.dumps(initial, indent=2) + "\n", encoding="utf-8"
        )
    if tuple(features.shape) != shape:
        raise RuntimeError(f"Feature cache shape {features.shape} does not match expected {shape}")

    pending = np.flatnonzero(~np.asarray(complete).reshape(-1))
    if len(pending):
        pending_dataset = _PendingCompositeDataset(dataset, pending.tolist())
        loader = DataLoader(
            pending_dataset,
            batch_size=settings.clip.cache_batch_size,
            shuffle=False,
            num_workers=settings.training.num_workers,
            pin_memory=settings.training.pin_memory,
            persistent_workers=(
                settings.training.persistent_workers and settings.training.num_workers > 0
            ),
            prefetch_factor=(
                settings.training.prefetch_factor
                if settings.training.num_workers > 0
                else None
            ),
        )
        processed = int(np.asarray(complete).sum())
        total = dataset.base_sample_count * dataset.views
        for batch_index, batch in enumerate(loader, 1):
            embeddings = teacher.encode_images(batch["image"]).cpu().numpy().astype(dtype)
            sample_indices = batch["sample_index"].numpy()
            view_indices = batch["view_index"].numpy()
            features[sample_indices, view_indices] = embeddings
            complete[sample_indices, view_indices] = True
            processed += len(sample_indices)
            if batch_index % 25 == 0 or processed == total:
                features.flush()
                complete.flush()
                print(
                    f"[clip_cache] {split} cached={processed:,}/{total:,}",
                    flush=True,
                )
    features.flush()
    complete.flush()
    if not bool(np.asarray(complete).all()):
        raise RuntimeError(f"CLIP {split} cache is incomplete after extraction")
    final = {
        **expected,
        "status": "complete",
        "completed_entries": int(np.asarray(complete).sum()),
        "total_entries": int(dataset.base_sample_count * dataset.views),
        "features_path": str(paths["features"]),
        "labels_path": str(paths["labels"]),
        "feature_file_size_bytes": paths["features"].stat().st_size,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    paths["metadata"].write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


class _PendingCompositeDataset(Dataset):
    def __init__(self, dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class ClipFeatureStore:
    def __init__(
        self,
        split: str,
        dataset,
        bundle: ImageNetBundle,
        settings: ExperimentSettings,
    ) -> None:
        paths = _paths(split, settings)
        if not paths["metadata"].is_file():
            raise FileNotFoundError(
                f"CLIP cache is missing for {split}: {paths['metadata']}. Run --phase clip_cache."
            )
        saved = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        expected = expected_cache_metadata(split, dataset, bundle, settings)
        mismatches = _metadata_matches(saved, expected)
        if mismatches or saved.get("status") != "complete":
            raise RuntimeError(
                f"CLIP {split} cache metadata mismatch/incomplete: {mismatches}, "
                f"status={saved.get('status')!r}"
            )
        self.features = np.load(paths["features"], mmap_mode="r")
        self.labels = np.load(paths["labels"], mmap_mode="r")
        self.metadata = saved

    def feature(self, sample_index: int, view_index: int) -> torch.Tensor:
        # Copy detaches the read-only memmap and avoids PyTorch's non-writable
        # numpy warning in DataLoader workers.
        return torch.from_numpy(
            np.array(self.features[sample_index, view_index], dtype=np.float32, copy=True)
        )


class DistillationViewDataset(Dataset):
    def __init__(self, base, store: ClipFeatureStore) -> None:
        self.base = base
        self.store = store

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        if int(self.store.labels[item["sample_index"]]) != item["label"]:
            raise RuntimeError(
                f"CLIP cache label mismatch at sample {item['sample_index']}"
            )
        item["teacher_embedding"] = self.store.feature(
            item["sample_index"], item["view_index"]
        )
        return item


def build_all_clip_caches(
    bundle: ImageNetBundle,
    settings: ExperimentSettings,
    device: torch.device,
) -> dict[str, dict]:
    teacher = FrozenClipTeacher(settings, device)
    root = cache_directory(settings)
    prototypes_path = root / "imagenet_text_prototypes.pt"
    prototypes = teacher.build_text_prototypes(
        bundle.class_names, settings.clip.text_prompt_templates
    )
    save_text_prototypes(
        prototypes_path,
        prototypes,
        bundle.class_names,
        settings,
        teacher.logit_scale,
    )
    reports = {}
    for split, dataset in (("train", bundle.train), ("validation", bundle.validation)):
        reports[split] = build_clip_cache(
            split, dataset, bundle, teacher, settings, device
        )
    return reports
