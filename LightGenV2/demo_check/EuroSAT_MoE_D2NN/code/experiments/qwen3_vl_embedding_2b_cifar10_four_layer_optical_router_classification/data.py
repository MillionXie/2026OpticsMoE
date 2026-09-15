from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CIFAR10DataBundle:
    train: Subset
    validation: Subset
    test: Subset
    train_labels: tuple[int, ...]
    metadata: dict[str, Any]


def _digest_indices(*groups: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        digest.update(json.dumps(list(group), separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _class_indices(labels: Sequence[int], num_classes: int) -> list[list[int]]:
    values = [[] for _ in range(num_classes)]
    for index, label in enumerate(labels):
        values[int(label)].append(index)
    if any(not group for group in values):
        raise RuntimeError("CIFAR-10 data is missing at least one class")
    return values


def _stratified_split(
    labels: Sequence[int], per_class: int, seed: int, num_classes: int
) -> tuple[list[int], list[int]]:
    train: list[int] = []
    validation: list[int] = []
    for label, indices in enumerate(_class_indices(labels, num_classes)):
        generator = torch.Generator().manual_seed(seed + 104729 * label)
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected = [indices[position] for position in order]
        validation.extend(selected[:per_class])
        train.extend(selected[per_class:])
    train.sort()
    validation.sort()
    return train, validation


def _stratified_subset(
    labels: Sequence[int], per_class: int | None, seed: int, num_classes: int
) -> list[int]:
    if per_class is None:
        return list(range(len(labels)))
    selected: list[int] = []
    for label, indices in enumerate(_class_indices(labels, num_classes)):
        generator = torch.Generator().manual_seed(seed + 130363 * label)
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[:per_class])
    selected.sort()
    return selected


def prepare_cifar10(settings: Any, *, persist: bool = True) -> CIFAR10DataBundle:
    root = Path(settings.dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    augmentation: list[Any] = []
    if settings.train_augmentation:
        augmentation.extend(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_transform = transforms.Compose(augmentation) if augmentation else None

    train_augmented = datasets.CIFAR10(
        root=str(root), train=True, transform=train_transform, download=settings.data_download
    )
    train_plain = datasets.CIFAR10(
        root=str(root), train=True, transform=None, download=False
    )
    test_plain = datasets.CIFAR10(
        root=str(root), train=False, transform=None, download=settings.data_download
    )
    if tuple(train_plain.classes) != tuple(settings.class_names):
        raise RuntimeError(
            f"Unexpected CIFAR-10 class order: {tuple(train_plain.classes)}"
        )

    train_indices, validation_indices = _stratified_split(
        train_plain.targets,
        settings.validation_samples_per_class,
        settings.split_seed,
        settings.num_classes,
    )
    test_indices = _stratified_subset(
        test_plain.targets,
        settings.test_samples_per_class,
        settings.split_seed + 1,
        settings.num_classes,
    )
    train_labels = tuple(int(train_plain.targets[index]) for index in train_indices)
    split_digest = _digest_indices(train_indices, validation_indices, test_indices)
    metadata = {
        "dataset": "torchvision.datasets.CIFAR10",
        "root": str(root.resolve()),
        "class_names": list(settings.class_names),
        "split_seed": int(settings.split_seed),
        "split_sha256": split_digest,
        "counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "validation_samples_per_class": int(settings.validation_samples_per_class),
        "test_samples_per_class": settings.test_samples_per_class,
        "train_augmentation": bool(settings.train_augmentation),
    }
    if persist:
        output = settings.output_dir / "cifar10_split.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return CIFAR10DataBundle(
        train=Subset(train_augmented, train_indices),
        validation=Subset(train_plain, validation_indices),
        test=Subset(test_plain, test_indices),
        train_labels=train_labels,
        metadata=metadata,
    )


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Deterministic balanced batches over Subset-local CIFAR indices."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        classes_per_batch: int,
        samples_per_class: int,
        steps: int,
        seed: int,
    ) -> None:
        self.labels = tuple(int(value) for value in labels)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.steps = int(steps)
        self.seed = int(seed)
        self.epoch = 0
        self.pools = _class_indices(self.labels, max(self.labels) + 1)
        if not 1 <= self.classes_per_batch <= len(self.pools):
            raise ValueError("classes_per_batch exceeds available classes")
        if self.samples_per_class <= 0 or self.steps <= 0:
            raise ValueError("samples_per_class and steps must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + 1000003 * self.epoch)
        orders = [
            torch.randperm(len(pool), generator=generator).tolist() for pool in self.pools
        ]
        offsets = [0 for _ in self.pools]
        for _ in range(self.steps):
            class_order = torch.randperm(len(self.pools), generator=generator).tolist()
            chosen = class_order[: self.classes_per_batch]
            batch: list[int] = []
            for label in chosen:
                pool = self.pools[label]
                for _ in range(self.samples_per_class):
                    if offsets[label] >= len(orders[label]):
                        orders[label] = torch.randperm(
                            len(pool), generator=generator
                        ).tolist()
                        offsets[label] = 0
                    batch.append(pool[orders[label][offsets[label]]])
                    offsets[label] += 1
            shuffle = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[position] for position in shuffle]


def _collate(samples: Sequence[tuple[Any, int]]) -> tuple[list[Any], torch.Tensor]:
    images, labels = zip(*samples)
    return list(images), torch.tensor(labels, dtype=torch.long)


def build_train_loader(
    bundle: CIFAR10DataBundle, settings: Any, *, epoch: int
) -> DataLoader:
    steps = settings.max_train_steps_per_epoch
    if steps is None:
        steps = math.ceil(len(bundle.train) / settings.batch_size)
    sampler = BalancedClassBatchSampler(
        bundle.train_labels,
        classes_per_batch=settings.pk_skus_per_batch,
        samples_per_class=settings.pk_images_per_sku,
        steps=steps,
        seed=settings.split_seed,
    )
    sampler.set_epoch(epoch)
    return DataLoader(
        bundle.train,
        batch_sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=settings.num_workers > 0,
        collate_fn=_collate,
    )


def build_eval_loader(dataset: Subset, settings: Any) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=False,
        collate_fn=_collate,
    )


__all__ = [
    "BalancedClassBatchSampler",
    "CIFAR10DataBundle",
    "build_eval_loader",
    "build_train_loader",
    "prepare_cifar10",
]
