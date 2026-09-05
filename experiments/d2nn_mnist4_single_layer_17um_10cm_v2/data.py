"""MNIST split plus the notebook-exact 336-to-400 amplitude encoding."""

from __future__ import annotations

from torch.utils.data import Subset
from torchvision import datasets, transforms

from .base_data import (
    DatasetBundle,
    _balanced_limit,
    _class_indices,
    _stratified_train_validation,
    build_loaders,
)

from .settings import V2Settings


def build_input_transform(settings: V2Settings) -> transforms.Compose:
    content_guard = (settings.input_size - settings.input_content_size) // 2
    return transforms.Compose(
        [
            transforms.Resize(
                (settings.input_content_size, settings.input_content_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.Pad(content_guard, fill=0, padding_mode="constant"),
            transforms.ToTensor(),
        ]
    )


def build_datasets(settings: V2Settings) -> DatasetBundle:
    content_guard = (settings.input_size - settings.input_content_size) // 2
    transform = build_input_transform(settings)
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
    train_indices = _balanced_limit(
        train_indices, full_train.targets, settings.classes, settings.train_limit
    )
    validation_indices = _balanced_limit(
        validation_indices,
        full_train.targets,
        settings.classes,
        settings.val_limit,
    )
    test_indices = _balanced_limit(
        test_indices, full_test.targets, settings.classes, settings.test_limit
    )
    counts = {
        str(label): {
            "train": sum(
                int(full_train.targets[index]) == label for index in train_indices
            ),
            "validation": sum(
                int(full_train.targets[index]) == label
                for index in validation_indices
            ),
            "test": sum(
                int(full_test.targets[index]) == label for index in test_indices
            ),
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
            "input_content_resize": [
                settings.input_content_size,
                settings.input_content_size,
            ],
            "input_content_zero_padding": content_guard,
            "input_field_size": [settings.input_size, settings.input_size],
            "active_padding": settings.input_guard,
            "amplitude_encoding": "ToTensor [0,1], no sqrt or normalization",
        },
    )


__all__ = [
    "DatasetBundle",
    "build_datasets",
    "build_input_transform",
    "build_loaders",
]
