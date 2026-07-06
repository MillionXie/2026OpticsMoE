from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .data_prepare import SCENE4_CLASSES, ensure_scene4_dataset


SCENE4_CLASS_NAMES = list(SCENE4_CLASSES)


@dataclass
class DatasetBundle:
    train: Dataset[Any]
    test: Dataset[Any]
    class_names: list[str]
    metadata: dict[str, Any]


class RGBImageFolder(Dataset[Any]):
    def __init__(self, base: Any, mapping: dict[int, int]) -> None:
        self.base = base
        self.mapping = mapping
        self.labels = [mapping[int(value)] for value in base.targets]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image.convert("RGB"), self.mapping[int(label)]


def load_scene4(settings: Any) -> DatasetBundle:
    try:
        from torchvision.datasets import ImageFolder
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc
    manifest = ensure_scene4_dataset(settings.data_root, settings.imagefolder_train, settings.imagefolder_test) if settings.download else None
    train_base = ImageFolder(str(settings.data_root / settings.imagefolder_train))
    test_base = ImageFolder(str(settings.data_root / settings.imagefolder_test))
    _validate(train_base.classes)
    _validate(test_base.classes)
    train: Dataset[Any] = RGBImageFolder(train_base, {train_base.class_to_idx[name]: SCENE4_CLASS_NAMES.index(name) for name in SCENE4_CLASS_NAMES})
    test: Dataset[Any] = RGBImageFolder(test_base, {test_base.class_to_idx[name]: SCENE4_CLASS_NAMES.index(name) for name in SCENE4_CLASS_NAMES})
    train = _per_class_limit(train, settings.train_limit_per_class, settings.seed)
    test = _per_class_limit(test, settings.test_limit_per_class, settings.seed + 1)
    train = _total_limit(train, settings.train_limit, settings.seed + 2)
    test = _total_limit(test, settings.test_limit, settings.seed + 3)
    train_indices, validation_indices = stratified_split_indices(train, settings.validation_fraction, settings.seed)
    train_counts = class_counts(Subset(train, train_indices))
    epoch_counts = {
        name: _epoch_count(count, settings.train_samples_per_class_per_epoch, settings.oversample_minority_classes)
        for name, count in train_counts.items()
    }
    full_counts = class_counts(train)
    test_counts = class_counts(test)
    return DatasetBundle(train, test, list(SCENE4_CLASS_NAMES), {
        "dataset": "bdd100k_scene4",
        "root": str(settings.data_root),
        "class_names": list(SCENE4_CLASS_NAMES),
        "full_train_samples": len(train),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test),
        "per_class_full_train_counts": full_counts,
        "per_class_train_counts": train_counts,
        "per_class_epoch_sample_counts": epoch_counts,
        "epoch_train_samples": sum(epoch_counts.values()),
        "per_class_validation_counts": class_counts(Subset(train, validation_indices)),
        "per_class_test_counts": test_counts,
        "class_imbalance": _imbalance(full_counts, test_counts),
        "train_limit": settings.train_limit,
        "test_limit": settings.test_limit,
        "train_limit_per_class": settings.train_limit_per_class,
        "test_limit_per_class": settings.test_limit_per_class,
        "train_samples_per_class_per_epoch": settings.train_samples_per_class_per_epoch,
        "oversample_minority_classes": settings.oversample_minority_classes,
        "validation_fraction": settings.validation_fraction,
        "manifest": manifest,
    })


def labels_of(dataset: Dataset[Any]) -> list[int]:
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if isinstance(dataset, Subset):
        parent = labels_of(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset has no labels")


def class_counts(dataset: Dataset[Any]) -> dict[str, int]:
    counts = Counter(labels_of(dataset))
    return {name: int(counts.get(index, 0)) for index, name in enumerate(SCENE4_CLASS_NAMES)}


def stratified_split_indices(dataset: Dataset[Any], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    train: list[int] = []
    validation: list[int] = []
    for class_index in range(len(SCENE4_CLASS_NAMES)):
        indices = [index for index, value in enumerate(labels) if value == class_index]
        order = torch.randperm(len(indices), generator=generator).tolist()
        count = min(max(round(len(indices) * fraction), 1), len(indices) - 1) if len(indices) > 1 else 0
        validation.extend(indices[position] for position in order[:count])
        train.extend(indices[position] for position in order[count:])
    return sorted(train), sorted(validation)


class IndexedDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


def indexed_collate(batch: Sequence[Any]):
    images, labels, indices = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), torch.tensor(indices, dtype=torch.long)


def make_indexed_loader(dataset: Dataset[Any], batch_size: int, workers: int, shuffle: bool, seed: int):
    return DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=indexed_collate, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed))


def _validate(classes: list[str]) -> None:
    if set(classes) != set(SCENE4_CLASS_NAMES):
        raise ValueError(f"Expected classes {SCENE4_CLASS_NAMES}, found {classes}")


def _per_class_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None:
        return dataset
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for class_index in range(len(SCENE4_CLASS_NAMES)):
        indices = [index for index, value in enumerate(labels) if value == class_index]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[:limit])
    return Subset(dataset, sorted(selected))


def _total_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected = []
    base, remainder = divmod(limit, len(SCENE4_CLASS_NAMES))
    for class_index in range(len(SCENE4_CLASS_NAMES)):
        indices = [index for index, value in enumerate(labels) if value == class_index]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[:base + int(class_index < remainder)])
    return Subset(dataset, sorted(selected))


def _epoch_count(count: int, limit: int | None, oversample: bool) -> int:
    if limit is None:
        return count
    return limit if oversample else min(count, limit)


def _imbalance(train_counts: dict[str, int], test_counts: dict[str, int]) -> dict[str, Any]:
    positive = [value for value in train_counts.values() if value > 0]
    return {
        "minority_class": min(train_counts, key=train_counts.get),
        "majority_class": max(train_counts, key=train_counts.get),
        "train_majority_to_minority_ratio": max(positive) / min(positive),
        "train_counts": train_counts,
        "test_counts": test_counts,
    }
