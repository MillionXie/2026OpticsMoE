from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .settings import Settings


@dataclass
class DatasetBundle:
    train: Dataset
    validation: Dataset
    test: Dataset
    metadata: dict[str, object]


def _class_indices(targets: torch.Tensor, classes: tuple[int, ...]) -> list[int]:
    allowed = torch.zeros_like(targets, dtype=torch.bool)
    for label in classes:
        allowed |= targets == int(label)
    return torch.nonzero(allowed, as_tuple=False).flatten().tolist()


def _stratified_train_validation(
    targets: torch.Tensor,
    classes: tuple[int, ...],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(int(seed))
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in classes:
        indices = torch.nonzero(targets == int(label), as_tuple=False).flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        validation_count = max(1, int(round(len(indices) * float(val_fraction))))
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())
    return train_indices, validation_indices


def _limit(indices: list[int], limit: int | None) -> list[int]:
    return indices if limit is None else indices[: int(limit)]


def build_datasets(settings: Settings) -> DatasetBundle:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (settings.input_size, settings.input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )
    full_train = datasets.MNIST(
        root=str(settings.dataset_root),
        train=True,
        download=settings.download,
        transform=transform,
    )
    full_test = datasets.MNIST(
        root=str(settings.dataset_root),
        train=False,
        download=settings.download,
        transform=transform,
    )
    train_indices, validation_indices = _stratified_train_validation(
        full_train.targets,
        settings.classes,
        settings.val_fraction,
        settings.random_seed,
    )
    test_indices = _class_indices(full_test.targets, settings.classes)
    train_indices = _limit(train_indices, settings.train_limit)
    validation_indices = _limit(validation_indices, settings.val_limit)
    test_indices = _limit(test_indices, settings.test_limit)
    counts = {
        str(label): {
            "train": sum(int(full_train.targets[index]) == label for index in train_indices),
            "validation": sum(
                int(full_train.targets[index]) == label for index in validation_indices
            ),
            "test": sum(int(full_test.targets[index]) == label for index in test_indices),
        }
        for label in settings.classes
    }
    return DatasetBundle(
        train=Subset(full_train, train_indices),
        validation=Subset(full_train, validation_indices),
        test=Subset(full_test, test_indices),
        metadata={
            "name": "MNIST",
            "classes": list(settings.classes),
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "per_class": counts,
            "official_test_split_untouched": settings.test_limit is None,
            "input_resize": [settings.input_size, settings.input_size],
            "active_padding": settings.input_guard,
        },
    )


def build_loaders(
    bundle: DatasetBundle, settings: Settings
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": settings.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(settings.random_seed)
    train = DataLoader(
        bundle.train,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation = DataLoader(
        bundle.validation,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        **common,
    )
    test = DataLoader(
        bundle.test,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        **common,
    )
    return train, validation, test

