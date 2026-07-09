from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from ..qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.data_prepare import (
    TIMEOFDAY3_CLASSES,
    ensure_timeofday3_dataset,
)


CLASS_NAMES = list(TIMEOFDAY3_CLASSES)
RGB_MODELS = {"resnet18", "vgg11_bn", "mobilenet_v2"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class DataBundle:
    train: Dataset[Any]
    validation: Dataset[Any]
    test: Dataset[Any]
    class_names: list[str]
    metadata: dict[str, Any]


class TimeOfDayTensorFolder(Dataset[Any]):
    def __init__(
        self,
        base: Any,
        mapping: dict[int, int],
        image_size: int,
        channels: int,
        normalization: str,
    ) -> None:
        self.base = base
        self.mapping = mapping
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.normalization = normalization
        self.labels = [mapping[int(value)] for value in base.targets]
        self.paths = [str(path) for path, _ in base.samples]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str]:
        image, old_label = self.base[index]
        tensor = self._to_tensor(image)
        return tensor, self.mapping[int(old_label)], index, self.paths[index]

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.channels == 1:
            gray = ImageOps.grayscale(image).resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            return torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0).unsqueeze(0)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        value = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.normalization == "imagenet":
            value = (value - IMAGENET_MEAN) / IMAGENET_STD
        return value


def load_data(settings: Any) -> DataBundle:
    try:
        from torchvision.datasets import ImageFolder
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc

    manifest = (
        ensure_timeofday3_dataset(settings.data_root, settings.imagefolder_train, settings.imagefolder_test)
        if settings.download
        else None
    )
    train_base = ImageFolder(str(settings.data_root / settings.imagefolder_train))
    test_base = ImageFolder(str(settings.data_root / settings.imagefolder_test))
    _validate(train_base.classes)
    _validate(test_base.classes)
    channels = 3 if settings.model_type in RGB_MODELS else 1
    mapping_train = {train_base.class_to_idx[name]: CLASS_NAMES.index(name) for name in CLASS_NAMES}
    mapping_test = {test_base.class_to_idx[name]: CLASS_NAMES.index(name) for name in CLASS_NAMES}
    train: Dataset[Any] = TimeOfDayTensorFolder(
        train_base, mapping_train, settings.image_size, channels, settings.image_normalization
    )
    test: Dataset[Any] = TimeOfDayTensorFolder(
        test_base, mapping_test, settings.image_size, channels, settings.image_normalization
    )
    train = _per_class_limit(train, settings.train_limit_per_class, settings.seed)
    test = _per_class_limit(test, settings.test_limit_per_class, settings.seed + 1)
    train = _total_limit(train, settings.train_limit, settings.seed + 2)
    test = _total_limit(test, settings.test_limit, settings.seed + 3)
    train_indices, validation_indices = stratified_split_indices(
        train, settings.validation_fraction, settings.seed
    )
    train_split = Subset(train, train_indices)
    validation = Subset(train, validation_indices)
    train_counts = class_counts(train_split)
    epoch_counts = {
        name: min(count, settings.train_samples_per_class_per_epoch)
        if settings.train_samples_per_class_per_epoch is not None
        else count
        for name, count in train_counts.items()
    }
    return DataBundle(
        train_split,
        validation,
        test,
        list(CLASS_NAMES),
        {
            "dataset": "bdd100k_timeofday3",
            "root": str(settings.data_root),
            "class_names": list(CLASS_NAMES),
            "image_size": settings.image_size,
            "channels": channels,
            "image_normalization": settings.image_normalization,
            "alignment_reference": settings.reference_experiment,
            "reference_processor_min_pixels": settings.reference_processor_min_pixels,
            "reference_processor_max_pixels": settings.reference_processor_max_pixels,
            "split_seed": settings.seed,
            "limit_seeds": {
                "train_limit_per_class": settings.seed,
                "test_limit_per_class": settings.seed + 1,
                "train_limit": settings.seed + 2,
                "test_limit": settings.seed + 3,
            },
            "full_train_samples": len(train),
            "train_samples": len(train_split),
            "validation_samples": len(validation),
            "test_samples": len(test),
            "per_class_full_train_counts": class_counts(train),
            "per_class_train_counts": train_counts,
            "per_class_epoch_sample_counts": epoch_counts,
            "epoch_train_samples": sum(epoch_counts.values()),
            "per_class_validation_counts": class_counts(validation),
            "per_class_test_counts": class_counts(test),
            "train_limit": settings.train_limit,
            "test_limit": settings.test_limit,
            "train_limit_per_class": settings.train_limit_per_class,
            "test_limit_per_class": settings.test_limit_per_class,
            "train_samples_per_class_per_epoch": settings.train_samples_per_class_per_epoch,
            "validation_fraction": settings.validation_fraction,
            "manifest": manifest,
        },
    )


def make_loader(
    dataset: Dataset[Any],
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    sampler: Sampler[int] | None = None,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def collate(batch: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    images, labels, indices, paths = zip(*batch)
    return (
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
        list(paths),
    )


def labels_of(dataset: Dataset[Any]) -> list[int]:
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if isinstance(dataset, Subset):
        parent = labels_of(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset has no labels")


def stratified_split_indices(dataset: Dataset[Any], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    train: list[int] = []
    validation: list[int] = []
    for cls in range(3):
        indices = [index for index, value in enumerate(labels) if value == cls]
        order = torch.randperm(len(indices), generator=generator).tolist()
        count = min(max(round(len(indices) * fraction), 1), len(indices) - 1) if len(indices) > 1 else 0
        validation.extend(indices[position] for position in order[:count])
        train.extend(indices[position] for position in order[count:])
    return sorted(train), sorted(validation)


def class_counts(dataset: Dataset[Any]) -> dict[str, int]:
    counts = Counter(labels_of(dataset))
    return {name: int(counts.get(index, 0)) for index, name in enumerate(CLASS_NAMES)}


def _validate(classes: list[str]) -> None:
    if set(classes) != set(CLASS_NAMES):
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {classes}")


def _per_class_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None:
        return dataset
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for cls in range(3):
        indices = [index for index, value in enumerate(labels) if value == cls]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[:limit])
    return Subset(dataset, sorted(selected))


def _total_limit(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    labels = labels_of(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    base, remainder = divmod(limit, 3)
    for cls in range(3):
        indices = [index for index, value in enumerate(labels) if value == cls]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[: base + int(cls < remainder)])
    return Subset(dataset, sorted(selected))

