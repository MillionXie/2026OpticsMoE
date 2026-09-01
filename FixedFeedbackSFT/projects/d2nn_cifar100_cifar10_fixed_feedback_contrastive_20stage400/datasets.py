from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .settings import BalancedBatchConfig, Settings


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _gray_tensor(image: object) -> torch.Tensor:
    value = torch.from_numpy(np.array(image, copy=True))
    if value.ndim == 3:
        value = value.permute(2, 0, 1)
    value = value.float() / 255.0
    if value.shape[0] == 3:
        value = (0.299 * value[0] + 0.587 * value[1] + 0.114 * value[2]).unsqueeze(0)
    if tuple(value.shape) != (1, 32, 32):
        raise ValueError(f"Expected a CIFAR image, got {tuple(value.shape)}")
    return value


def _augment(image: torch.Tensor, *, seed: int, padding: int, horizontal_flip: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    result = image
    if padding > 0:
        padded = F.pad(result, (padding, padding, padding, padding), mode="reflect")
        y0 = int(torch.randint(0, 2 * padding + 1, (), generator=generator))
        x0 = int(torch.randint(0, 2 * padding + 1, (), generator=generator))
        result = padded[:, y0 : y0 + 32, x0 : x0 + 32]
    if horizontal_flip and bool(torch.rand((), generator=generator) < 0.5):
        result = torch.flip(result, dims=(-1,))
    return result.contiguous()


class CIFARContrastiveView(Dataset):
    def __init__(
        self,
        base: Dataset,
        indices: Sequence[int],
        *,
        views_per_image: int,
        augment: bool,
        seed: int,
        crop_padding: int,
        horizontal_flip: bool,
        dataset_name: str,
    ) -> None:
        self.base = base
        self.indices = tuple(int(index) for index in indices)
        self.targets = tuple(int(base.targets[index]) for index in self.indices)  # type: ignore[attr-defined]
        self.views_per_image = int(views_per_image)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.crop_padding = int(crop_padding)
        self.horizontal_flip = bool(horizontal_flip)
        self.dataset_name = str(dataset_name)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        base_index = self.indices[index]
        image, target = self.base[base_index]
        original = _gray_tensor(image)
        views = []
        for view_index in range(self.views_per_image):
            if self.augment:
                view = _augment(
                    original,
                    seed=stable_seed(self.seed, self.epoch, base_index, view_index),
                    padding=self.crop_padding,
                    horizontal_flip=self.horizontal_flip,
                )
            else:
                view = original
            views.append(view)
        return {
            "views": torch.stack(views, dim=0),
            "target": int(target),
            "sample_id": f"{self.dataset_name}_{base_index:05d}",
            "base_index": base_index,
        }


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Deterministic P x K batches with per-class queues and reproducible refills."""

    def __init__(
        self,
        targets: Sequence[int],
        config: BalancedBatchConfig,
        *,
        seed: int,
        epoch: int,
    ) -> None:
        self.targets = tuple(int(value) for value in targets)
        self.config = config
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.class_to_indices: dict[int, list[int]] = {}
        for index, target in enumerate(self.targets):
            self.class_to_indices.setdefault(target, []).append(index)
        if config.classes_per_batch > len(self.class_to_indices):
            raise ValueError("classes_per_batch exceeds available classes")

    def __len__(self) -> int:
        return self.config.batches_per_epoch

    def _permutation(self, values: Sequence[int], generator: torch.Generator) -> list[int]:
        order = torch.randperm(len(values), generator=generator).tolist()
        return [int(values[index]) for index in order]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(stable_seed(self.seed, self.epoch, "balanced_sampler"))
        classes = sorted(self.class_to_indices)
        queues = {label: self._permutation(indices, generator) for label, indices in self.class_to_indices.items()}
        offsets = {label: 0 for label in classes}
        for _ in range(self.config.batches_per_epoch):
            # A fresh class permutation prevents duplicate class labels inside
            # one P x K batch, including at permutation-cycle boundaries.
            selected_classes = self._permutation(classes, generator)[: self.config.classes_per_batch]
            batch: list[int] = []
            for label in selected_classes:
                needed = self.config.images_per_class
                while needed > 0:
                    available = len(queues[label]) - offsets[label]
                    if available == 0:
                        queues[label] = self._permutation(self.class_to_indices[label], generator)
                        offsets[label] = 0
                        available = len(queues[label])
                    take = min(needed, available)
                    batch.extend(queues[label][offsets[label] : offsets[label] + take])
                    offsets[label] += take
                    needed -= take
            yield batch


def batch_order_digest(sampler: BalancedClassBatchSampler) -> str:
    digest = hashlib.sha256()
    for batch in sampler:
        digest.update(np.asarray(batch, dtype=np.int64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    cifar100_classes: tuple[str, ...]
    cifar10_classes: tuple[str, ...]
    pretrain_train: CIFARContrastiveView
    pretrain_validation: CIFARContrastiveView
    finetune_train: CIFARContrastiveView
    prototype_support: CIFARContrastiveView
    finetune_validation: CIFARContrastiveView
    finetune_test: CIFARContrastiveView


def _stratified_split(targets: Sequence[int], *, validation_per_class: int, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    target_array = np.asarray(targets)
    for label in sorted(set(int(value) for value in targets)):
        indices = np.flatnonzero(target_array == label)
        rng.shuffle(indices)
        validation.extend(int(value) for value in indices[:validation_per_class])
        train.extend(int(value) for value in indices[validation_per_class:])
    return train, validation


def _cifar10_split(targets: Sequence[int], settings: Settings) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(settings.data.split_seed)
    train: list[int] = []
    support: list[int] = []
    validation: list[int] = []
    values = np.asarray(targets)
    for label in range(10):
        indices = np.flatnonzero(values == label)
        rng.shuffle(indices)
        n_support = settings.data.cifar10_support_per_class
        n_validation = settings.data.cifar10_validation_per_class
        support.extend(int(value) for value in indices[:n_support])
        validation.extend(int(value) for value in indices[n_support : n_support + n_validation])
        train.extend(int(value) for value in indices[n_support + n_validation :])
    if set(train) & set(support) or set(train) & set(validation) or set(support) & set(validation):
        raise RuntimeError("CIFAR-10 train/support/validation leakage detected")
    return train, support, validation


def prepare_data(settings: Settings) -> dict[str, object]:
    from torchvision.datasets import CIFAR10, CIFAR100

    settings.data.torchvision_root.mkdir(parents=True, exist_ok=True)
    cifar100_train = CIFAR100(settings.data.torchvision_root, train=True, download=True)
    cifar10_train = CIFAR10(settings.data.torchvision_root, train=True, download=True)
    cifar10_test = CIFAR10(settings.data.torchvision_root, train=False, download=True)
    metadata = {
        "cifar100_train": len(cifar100_train),
        "cifar10_train": len(cifar10_train),
        "cifar10_test": len(cifar10_test),
        "cifar100_classes": list(cifar100_train.classes),
        "cifar10_classes": list(cifar10_train.classes),
    }
    settings.data.root.mkdir(parents=True, exist_ok=True)
    (settings.data.root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_datasets(settings: Settings, *, prepare: bool = True) -> DatasetBundle:
    from torchvision.datasets import CIFAR10, CIFAR100

    if prepare:
        prepare_data(settings)
    cifar100 = CIFAR100(settings.data.torchvision_root, train=True, download=False)
    cifar10_train = CIFAR10(settings.data.torchvision_root, train=True, download=False)
    cifar10_test = CIFAR10(settings.data.torchvision_root, train=False, download=False)
    pretrain_indices, pretrain_validation = _stratified_split(
        cifar100.targets,
        validation_per_class=settings.data.cifar100_validation_per_class,
        seed=settings.training.pretrain_seed,
    )
    finetune_indices, support_indices, validation_indices = _cifar10_split(cifar10_train.targets, settings)

    def view(
        base: Dataset,
        indices: Sequence[int],
        *,
        views: int,
        augment: bool,
        seed: int,
        name: str,
    ) -> CIFARContrastiveView:
        return CIFARContrastiveView(
            base,
            indices,
            views_per_image=views,
            augment=augment,
            seed=seed,
            crop_padding=settings.data.crop_padding,
            horizontal_flip=settings.data.horizontal_flip,
            dataset_name=name,
        )

    bundle = DatasetBundle(
        cifar100_classes=tuple(cifar100.classes),
        cifar10_classes=tuple(cifar10_train.classes),
        pretrain_train=view(
            cifar100,
            pretrain_indices,
            views=settings.training.pretrain_batch.views_per_image,
            augment=True,
            seed=settings.training.pretrain_seed,
            name="cifar100_train",
        ),
        pretrain_validation=view(
            cifar100,
            pretrain_validation,
            views=settings.training.pretrain_batch.views_per_image,
            augment=True,
            seed=settings.training.pretrain_seed + 1,
            name="cifar100_validation",
        ),
        finetune_train=view(
            cifar10_train,
            finetune_indices,
            views=settings.training.finetune_batch.views_per_image,
            augment=True,
            seed=settings.data.split_seed,
            name="cifar10_finetune",
        ),
        prototype_support=view(cifar10_train, support_indices, views=1, augment=False, seed=0, name="cifar10_support"),
        finetune_validation=view(
            cifar10_train, validation_indices, views=1, augment=False, seed=0, name="cifar10_validation"
        ),
        finetune_test=view(
            cifar10_test, list(range(len(cifar10_test))), views=1, augment=False, seed=0, name="cifar10_test"
        ),
    )
    manifest = {
        "pretrain": {"dataset": "CIFAR-100", "train": len(bundle.pretrain_train), "validation": len(bundle.pretrain_validation)},
        "downstream": {
            "dataset": "CIFAR-10",
            "finetune_train": len(bundle.finetune_train),
            "prototype_support": len(bundle.prototype_support),
            "validation": len(bundle.finetune_validation),
            "test": len(bundle.finetune_test),
            "classes": list(bundle.cifar10_classes),
        },
        "no_image_overlap_between_cifar10_splits": True,
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle
