from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .settings import Settings


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    validation: Dataset
    test: Dataset
    classes: tuple[str, ...]


def _dataset_class(name: str):
    from torchvision.datasets import CIFAR10, CIFAR100

    return CIFAR10 if name == "cifar10" else CIFAR100


def _split(targets: Sequence[int], validation_per_class: int, seed: int) -> tuple[list[int], list[int]]:
    generator = np.random.default_rng(int(seed))
    values = np.asarray(targets)
    train: list[int] = []
    validation: list[int] = []
    for label in sorted(set(int(value) for value in targets)):
        indices = np.flatnonzero(values == label)
        generator.shuffle(indices)
        validation.extend(int(value) for value in indices[:validation_per_class])
        train.extend(int(value) for value in indices[validation_per_class:])
    return train, validation


def prepare_data(settings: Settings) -> dict[str, object]:
    dataset_class = _dataset_class(settings.data.dataset)
    settings.data.root.mkdir(parents=True, exist_ok=True)
    train = dataset_class(settings.data.root, train=True, download=True)
    test = dataset_class(settings.data.root, train=False, download=True)
    return {
        "dataset": settings.data.dataset,
        "train_samples": len(train),
        "test_samples": len(test),
        "classes": list(train.classes),
    }


def load_datasets(settings: Settings, *, download: bool = False) -> DatasetBundle:
    from torchvision import transforms

    dataset_class = _dataset_class(settings.data.dataset)
    train_transforms: list[object] = []
    if settings.data.crop_padding > 0:
        train_transforms.append(transforms.RandomCrop(32, padding=settings.data.crop_padding, padding_mode="reflect"))
    if settings.data.horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    evaluation_transform = transforms.ToTensor()
    train_base = dataset_class(
        settings.data.root,
        train=True,
        download=download,
        transform=transforms.Compose(train_transforms),
    )
    evaluation_base = dataset_class(
        settings.data.root,
        train=True,
        download=False,
        transform=evaluation_transform,
    )
    test = dataset_class(settings.data.root, train=False, download=False, transform=evaluation_transform)
    train_indices, validation_indices = _split(
        train_base.targets,
        settings.data.validation_per_class,
        settings.data.split_seed,
    )
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("Train/validation leakage detected")
    return DatasetBundle(
        train=Subset(train_base, train_indices),
        validation=Subset(evaluation_base, validation_indices),
        test=test,
        classes=tuple(train_base.classes),
    )


def make_loader(
    dataset: Dataset,
    settings: Settings,
    *,
    train: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=settings.training.batch_size if train else settings.training.evaluation_batch_size,
        shuffle=train,
        num_workers=settings.data.num_workers,
        pin_memory=settings.data.pin_memory,
        persistent_workers=settings.data.num_workers > 0,
        drop_last=False,
        generator=generator,
    )
