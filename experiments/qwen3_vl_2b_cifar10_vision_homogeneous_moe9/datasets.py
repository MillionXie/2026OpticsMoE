from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@dataclass
class DatasetBundle:
    train: Dataset[Any]
    test: Dataset[Any]
    class_names: list[str]
    metadata: dict[str, Any]


class RGBDataset(Dataset[tuple[Image.Image, int]]):
    """Preserve CIFAR color information for the Qwen image processor."""

    def __init__(self, base: Dataset[Any]) -> None:
        self.base = base
        self.labels = [int(value) for value in getattr(base, "targets")]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image, label = self.base[index]
        return image.convert("RGB"), int(label)


def load_cifar10(settings: Any) -> DatasetBundle:
    try:
        from torchvision.datasets import CIFAR10
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc
    train: Dataset[Any] = RGBDataset(CIFAR10(str(settings.data_root), train=True, download=settings.download))
    test: Dataset[Any] = RGBDataset(CIFAR10(str(settings.data_root), train=False, download=settings.download))
    train = _per_class_limit(train, settings.train_limit_per_class, settings.seed)
    test = _per_class_limit(test, settings.test_limit_per_class, settings.seed + 1)
    train = _total_limit(train, settings.train_limit, settings.seed + 2)
    test = _total_limit(test, settings.test_limit, settings.seed + 3)
    train_indices, validation_indices = stratified_split_indices(train, settings.validation_fraction, settings.seed)
    train_counts = class_counts(train)
    per_epoch_counts = {
        name: min(count, settings.train_samples_per_class_per_epoch)
        if settings.train_samples_per_class_per_epoch is not None else count
        for name, count in train_counts.items()
    }
    metadata = {
        "dataset": "cifar10",
        "input_color_mode": "RGB",
        "root": str(settings.data_root),
        "class_names": list(CIFAR10_CLASSES),
        "full_train_samples": len(train),
        "train_samples": len(train),
        "student_train_samples": len(train),
        "student_validation_samples": 0,
        "student_checkpoint_selection_split": "test",
        "teacher_head_train_samples": len(train_indices),
        "teacher_head_validation_samples": len(validation_indices),
        "test_samples": len(test),
        "per_class_train_counts": train_counts,
        "per_class_epoch_sample_counts": per_epoch_counts,
        "epoch_train_samples": sum(per_epoch_counts.values()),
        "per_class_teacher_head_validation_counts": class_counts(Subset(train, validation_indices)),
        "per_class_test_counts": class_counts(test),
        "validation_fraction": settings.validation_fraction,
        "train_limit": settings.train_limit,
        "test_limit": settings.test_limit,
        "train_limit_per_class": settings.train_limit_per_class,
        "test_limit_per_class": settings.test_limit_per_class,
        "train_samples_per_class_per_epoch": settings.train_samples_per_class_per_epoch,
        "sampling_note": "The per-epoch limit rotates through retained samples; it does not shrink the stored dataset.",
    }
    return DatasetBundle(train, test, list(CIFAR10_CLASSES), metadata)


def labels_of(dataset: Dataset[Any]) -> list[int]:
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if isinstance(dataset, Subset):
        parent = labels_of(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset does not expose labels")


def stratified_split_indices(dataset: Dataset[Any], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    train: list[int] = []
    validation: list[int] = []
    for class_index in range(10):
        indices = [index for index, label in enumerate(labels) if label == class_index]
        order = torch.randperm(len(indices), generator=generator).tolist()
        count = min(max(int(round(len(indices) * fraction)), 1), len(indices) - 1) if len(indices) > 1 else 0
        validation.extend(indices[position] for position in order[:count])
        train.extend(indices[position] for position in order[count:])
    return sorted(train), sorted(validation)


def class_counts(dataset: Dataset[Any]) -> dict[str, int]:
    labels = labels_of(dataset)
    return {name: labels.count(index) for index, name in enumerate(CIFAR10_CLASSES)}


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


def make_indexed_loader(dataset: Dataset[Any], batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader[Any]:
    return DataLoader(
        IndexedDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=indexed_collate, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def _per_class_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None:
        return dataset
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for class_index in range(10):
        indices = [index for index, label in enumerate(labels) if label == class_index]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[:limit])
    return Subset(dataset, sorted(selected))


def _total_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    base, remainder = divmod(limit, 10)
    for class_index in range(10):
        indices = [index for index, label in enumerate(labels) if label == class_index]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[: base + int(class_index < remainder)])
    return Subset(dataset, sorted(selected))
