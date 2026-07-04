from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .data_prepare import ensure_weather4_dataset


WEATHER4_CLASSES = ["clear", "rainy", "snowy", "foggy"]


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset[tuple[Image.Image, int]]
    test: Dataset[tuple[Image.Image, int]]
    class_names: list[str]
    metadata: dict[str, Any]


class Weather4Dataset(Dataset[tuple[Image.Image, int]]):
    def __init__(self, base: Dataset[Any], old_to_new: dict[int, int], resize_to: int | None) -> None:
        self.base = base
        self.old_to_new = old_to_new
        self.resize_to = resize_to
        self.labels = [old_to_new[int(label)] for label in getattr(base, "targets")]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image, old_label = self.base[index]
        image = image.convert("RGB")
        if self.resize_to is not None:
            image = image.resize((self.resize_to, self.resize_to), Image.Resampling.BICUBIC)
        return image, self.old_to_new[int(old_label)]


def load_weather4(
    root: Path,
    resize_to: int | None,
    train_limit: int | None,
    test_limit: int | None,
    train_limit_per_class: int | None,
    test_limit_per_class: int | None,
    imagefolder_train: str,
    imagefolder_test: str,
    seed: int,
    download: bool = False,
) -> DatasetBundle:
    try:
        from torchvision.datasets import ImageFolder
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc

    train_dir = root / imagefolder_train
    test_dir = root / imagefolder_test
    preparation: dict[str, Any] | None = None
    if download:
        preparation = ensure_weather4_dataset(root, imagefolder_train, imagefolder_test)
    for path in (train_dir, test_dir):
        if not path.is_dir():
            raise FileNotFoundError(
                f"BDD100K Weather-4 directory is missing: {path}. Expected class folders: "
                f"{', '.join(WEATHER4_CLASSES)}. Set download=true to download and prepare "
                "BDD100K automatically, or prepare the ImageFolder directories manually."
            )
    train_base = ImageFolder(str(train_dir))
    test_base = ImageFolder(str(test_dir))
    _validate_classes(train_base.classes, "train")
    _validate_classes(test_base.classes, "test")
    train_mapping = {
        train_base.class_to_idx[name]: WEATHER4_CLASSES.index(name)
        for name in WEATHER4_CLASSES
    }
    test_mapping = {
        test_base.class_to_idx[name]: WEATHER4_CLASSES.index(name)
        for name in WEATHER4_CLASSES
    }
    train: Dataset[tuple[Image.Image, int]] = Weather4Dataset(
        train_base, train_mapping, resize_to
    )
    test: Dataset[tuple[Image.Image, int]] = Weather4Dataset(
        test_base, test_mapping, resize_to
    )
    train = _balanced_limit(train, train_limit_per_class, seed)
    test = _balanced_limit(test, test_limit_per_class, seed + 1)
    train = _total_limit(train, train_limit, seed + 2)
    test = _total_limit(test, test_limit, seed + 3)
    if not len(train) or not len(test):
        raise ValueError("Weather-4 train and test datasets must both be non-empty")
    train_counts = class_counts(train)
    test_counts = class_counts(test)
    first_image, _ = train[0]
    return DatasetBundle(
        train=train,
        test=test,
        class_names=list(WEATHER4_CLASSES),
        metadata={
            "name": "bdd100k_weather4",
            "root": str(root),
            "train_samples": len(train),
            "test_samples": len(test),
            "num_classes": 4,
            "class_names": list(WEATHER4_CLASSES),
            "first_image_size": list(first_image.size),
            "resize_to": resize_to,
            "train_limit_per_class": train_limit_per_class,
            "test_limit_per_class": test_limit_per_class,
            "per_class_full_train_counts": train_counts,
            "per_class_test_counts": test_counts,
            "class_imbalance": imbalance_summary(train_counts, test_counts),
            "automatic_preparation": preparation,
        },
    )


def dataset_labels(dataset: Dataset[tuple[Image.Image, int]]) -> list[int]:
    return _dataset_labels(dataset)


def class_counts(dataset: Dataset[tuple[Image.Image, int]]) -> dict[str, int]:
    counts = [0] * len(WEATHER4_CLASSES)
    for label in _dataset_labels(dataset):
        counts[int(label)] += 1
    return {name: counts[index] for index, name in enumerate(WEATHER4_CLASSES)}


def stratified_split_indices(
    dataset: Dataset[tuple[Image.Image, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    labels = _dataset_labels(dataset)
    by_class: list[list[int]] = [[] for _ in WEATHER4_CLASSES]
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in by_class:
        order = torch.randperm(len(indices), generator=generator).tolist()
        validation_size = int(round(len(indices) * validation_fraction))
        if len(indices) > 1:
            validation_size = min(max(validation_size, 1), len(indices) - 1)
        else:
            validation_size = 0
        validation_indices.extend(indices[position] for position in order[:validation_size])
        train_indices.extend(indices[position] for position in order[validation_size:])
    return sorted(train_indices), sorted(validation_indices)


def stratified_split(
    dataset: Dataset[tuple[Image.Image, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[Subset[tuple[Image.Image, int]], Subset[tuple[Image.Image, int]]]:
    train_indices, validation_indices = stratified_split_indices(
        dataset, validation_fraction, seed
    )
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def imbalance_summary(
    train_counts: dict[str, int], test_counts: dict[str, int]
) -> dict[str, Any]:
    positive = [value for value in train_counts.values() if value > 0]
    total = sum(train_counts.values())
    majority = max(train_counts, key=train_counts.get)
    minority = min(train_counts, key=train_counts.get)
    return {
        "is_imbalanced": len(set(train_counts.values())) > 1,
        "majority_class": majority,
        "minority_class": minority,
        "majority_fraction": train_counts[majority] / total if total else 0.0,
        "train_max_to_min_ratio": max(positive) / min(positive) if positive else 0.0,
        "train_counts": train_counts,
        "test_counts": test_counts,
    }


def make_loader(
    dataset: Dataset[tuple[Image.Image, int]],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pil_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def pil_collate(
    batch: Sequence[tuple[Image.Image, int]],
) -> tuple[list[Image.Image], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def _validate_classes(classes: Sequence[str], split: str) -> None:
    actual = set(classes)
    expected = set(WEATHER4_CLASSES)
    if actual != expected:
        raise ValueError(
            f"Weather-4 {split} classes must be exactly {WEATHER4_CLASSES}; "
            f"found {sorted(actual)}"
        )


def _balanced_limit(
    dataset: Dataset[tuple[Image.Image, int]], limit: int | None, seed: int
) -> Dataset[tuple[Image.Image, int]]:
    if limit is None:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    by_class: list[list[int]] = [[] for _ in WEATHER4_CLASSES]
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise TypeError("Balanced limiting requires a dataset with precomputed labels")
    for index, label in enumerate(labels):
        by_class[label].append(index)
    selected: list[int] = []
    for indices in by_class:
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[:limit])
    return Subset(dataset, sorted(selected))


def _total_limit(
    dataset: Dataset[tuple[Image.Image, int]], limit: int | None, seed: int
) -> Dataset[tuple[Image.Image, int]]:
    if limit is None:
        return dataset
    limit = min(limit, len(dataset))
    labels = _dataset_labels(dataset)
    generator = torch.Generator().manual_seed(seed)
    by_class: list[list[int]] = [[] for _ in WEATHER4_CLASSES]
    for index, label in enumerate(labels):
        by_class[label].append(index)
    selected: list[int] = []
    base_quota, remainder = divmod(limit, len(WEATHER4_CLASSES))
    for class_index, indices in enumerate(by_class):
        quota = base_quota + int(class_index < remainder)
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[:quota])
    if len(selected) < limit:
        remaining = sorted(set(range(len(dataset))) - set(selected))
        selected.extend(remaining[: limit - len(selected)])
    return Subset(dataset, sorted(selected))


def _dataset_labels(dataset: Dataset[tuple[Image.Image, int]]) -> list[int]:
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return list(labels)
    if isinstance(dataset, Subset):
        parent = _dataset_labels(dataset.dataset)
        return [parent[index] for index in dataset.indices]
    raise TypeError("Dataset does not expose labels for balanced limiting")
