from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset


WEATHER4_CLASSES = ["clear", "rainy", "snowy", "foggy"]


class GrayscaleWeatherDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, base: Dataset[Any], old_to_new: dict[int, int], input_size: int) -> None:
        self.base = base
        self.old_to_new = old_to_new
        self.input_size = int(input_size)
        self.labels = [old_to_new[int(value)] for value in getattr(base, "targets")]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, old_label = self.base[index]
        image = ImageOps.grayscale(image.convert("RGB"))
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), self.old_to_new[int(old_label)]


@dataclass(frozen=True)
class DataBundle:
    train: Dataset[tuple[torch.Tensor, int]]
    validation: Dataset[tuple[torch.Tensor, int]]
    test: Dataset[tuple[torch.Tensor, int]]
    class_names: list[str]
    metadata: dict[str, Any]


def load_weather4(settings: object) -> DataBundle:
    try:
        from torchvision.datasets import ImageFolder
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc

    root = Path(settings.data_root)
    train_dir = root / settings.imagefolder_train
    test_dir = root / settings.imagefolder_test
    validate_dataset_layout(root, settings.imagefolder_train, settings.imagefolder_test)
    train_base = ImageFolder(str(train_dir))
    test_base = ImageFolder(str(test_dir))
    _validate_classes(train_base.classes, "train")
    _validate_classes(test_base.classes, "test")
    train_mapping = {train_base.class_to_idx[name]: WEATHER4_CLASSES.index(name) for name in WEATHER4_CLASSES}
    test_mapping = {test_base.class_to_idx[name]: WEATHER4_CLASSES.index(name) for name in WEATHER4_CLASSES}
    train_full: Dataset[tuple[torch.Tensor, int]] = GrayscaleWeatherDataset(train_base, train_mapping, settings.input_size)
    test: Dataset[tuple[torch.Tensor, int]] = GrayscaleWeatherDataset(test_base, test_mapping, settings.input_size)
    train_full = _per_class_limit(train_full, settings.train_limit_per_class, settings.seed)
    test = _per_class_limit(test, settings.test_limit_per_class, settings.seed + 1)
    train_full = _total_limit(train_full, settings.train_limit, settings.seed + 2)
    test = _total_limit(test, settings.test_limit, settings.seed + 3)
    train, validation = _stratified_split(train_full, settings.validation_fraction, settings.seed + 4)
    counts_train = _counts(train)
    counts_validation = _counts(validation)
    counts_test = _counts(test)
    all_train_counts = [counts_train[name] + counts_validation[name] for name in WEATHER4_CLASSES]
    nonzero = [value for value in all_train_counts if value > 0]
    imbalance_ratio = max(nonzero) / min(nonzero) if nonzero else None
    return DataBundle(
        train=train,
        validation=validation,
        test=test,
        class_names=list(WEATHER4_CLASSES),
        metadata={
            "dataset": "bdd100k_weather4",
            "root": str(root.resolve()),
            "class_names": list(WEATHER4_CLASSES),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "test_samples": len(test),
            "per_class_train_counts": counts_train,
            "per_class_validation_counts": counts_validation,
            "per_class_test_counts": counts_test,
            "class_imbalance": {
                "train_max_to_min_ratio_including_validation": imbalance_ratio,
                "note": "Weather-4 is strongly imbalanced; macro-F1 and balanced accuracy are primary diagnostics.",
            },
            "validation_fraction": settings.validation_fraction,
            "train_limit": settings.train_limit,
            "test_limit": settings.test_limit,
            "train_limit_per_class": settings.train_limit_per_class,
            "test_limit_per_class": settings.test_limit_per_class,
            "input_preprocessing": "RGB -> grayscale -> resize -> [0,1]; RMS amplitude normalization is performed by the model",
        },
    )


def validate_dataset_layout(root: Path, train_name: str, test_name: str) -> dict[str, Any]:
    missing: list[str] = []
    for split in (train_name, test_name):
        for name in WEATHER4_CLASSES:
            path = root / split / name
            if not path.is_dir():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "BDD100K Weather-4 ImageFolder data is not prepared. Missing directories:\n- "
            + "\n- ".join(missing)
            + "\nExpected data_root/{train,test}/{clear,rainy,snowy,foggy}."
        )
    return {"root": str(root.resolve()), "status": "ready", "classes": list(WEATHER4_CLASSES)}


def make_loader(dataset: Dataset[Any], batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _validate_classes(classes: list[str], split: str) -> None:
    if set(classes) != set(WEATHER4_CLASSES):
        raise ValueError(f"{split} must contain exactly {WEATHER4_CLASSES}; found {classes}")


def dataset_labels(dataset: Dataset[Any]) -> list[int]:
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return [int(value) for value in labels]
    if isinstance(dataset, Subset):
        parent = dataset_labels(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset does not expose labels")


def _per_class_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    labels = dataset_labels(dataset)
    selected: list[int] = []
    for class_index in range(len(WEATHER4_CLASSES)):
        indices = [index for index, label in enumerate(labels) if label == class_index]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[: int(limit)])
    return Subset(dataset, sorted(selected))


def _total_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    labels = dataset_labels(dataset)
    generator = torch.Generator().manual_seed(seed)
    by_class = [[index for index, label in enumerate(labels) if label == class_index] for class_index in range(4)]
    selected: list[int] = []
    base, remainder = divmod(int(limit), 4)
    for class_index, indices in enumerate(by_class):
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[: base + int(class_index < remainder)])
    if len(selected) < limit:
        remaining = sorted(set(range(len(dataset))) - set(selected))
        selected.extend(remaining[: int(limit) - len(selected)])
    return Subset(dataset, sorted(selected))


def _stratified_split(dataset: Dataset[Any], fraction: float, seed: int) -> tuple[Dataset[Any], Dataset[Any]]:
    labels = dataset_labels(dataset)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index in range(4):
        indices = [index for index, label in enumerate(labels) if label == class_index]
        if not indices:
            continue
        order = torch.randperm(len(indices), generator=generator).tolist()
        validation_count = max(1, int(round(len(indices) * fraction))) if len(indices) > 1 else 0
        validation_indices.extend(indices[position] for position in order[:validation_count])
        train_indices.extend(indices[position] for position in order[validation_count:])
    if not train_indices or not validation_indices:
        raise ValueError("Training and validation splits must both be non-empty")
    return Subset(dataset, sorted(train_indices)), Subset(dataset, sorted(validation_indices))


def _counts(dataset: Dataset[Any]) -> dict[str, int]:
    counts = Counter(dataset_labels(dataset))
    return {name: int(counts.get(index, 0)) for index, name in enumerate(WEATHER4_CLASSES)}
