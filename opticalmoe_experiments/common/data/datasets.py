from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from .loader_utils import dataloader_kwargs
from .transforms import build_image_transform


DATASET_REGISTRY = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
    "emnist": datasets.EMNIST,
    "cifar10": datasets.CIFAR10,
}

EMNIST_NUM_CLASSES = {
    "balanced": 47,
    "digits": 10,
    "letters": 26,
    "byclass": 62,
    "bymerge": 47,
}


class SubtractOne:
    """Target transform for EMNIST letters labels 1..26 -> 0..25."""

    def __call__(self, target):
        return int(target) - 1


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: List[str]


def dataset_key(name: str) -> str:
    key = str(name).lower().replace("-", "")
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset {name!r}. Use MNIST, FashionMNIST, KMNIST, EMNIST, or CIFAR10."
        )
    return key


def _base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _labels_for_dataset(dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        labels = _labels_for_dataset(dataset.dataset)
        return labels[torch.as_tensor(dataset.indices, dtype=torch.long)]
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError("Dataset does not expose targets/labels.")
    return torch.as_tensor(labels, dtype=torch.long)


def _class_balanced_indices(dataset, size: int, seed: int) -> List[int]:
    size = int(size)
    labels = _labels_for_dataset(dataset)
    if size <= 0 or size > len(labels):
        raise ValueError(f"Invalid requested sample size {size} for dataset of length {len(labels)}.")
    generator = torch.Generator().manual_seed(int(seed))
    classes = torch.unique(labels).sort().values.tolist()
    base = size // len(classes)
    remainder = size % len(classes)
    selected = []
    for index, class_value in enumerate(classes):
        class_indices = torch.where(labels == int(class_value))[0]
        quota = base + (1 if index < remainder else 0)
        if quota > len(class_indices):
            raise ValueError(f"Class {class_value} has only {len(class_indices)} samples; need {quota}.")
        permuted = class_indices[torch.randperm(len(class_indices), generator=generator)]
        selected.extend(permuted[:quota].tolist())
    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    selected_tensor = selected_tensor[torch.randperm(len(selected_tensor), generator=generator)]
    return selected_tensor.tolist()


def _random_indices(dataset, size: int, seed: int) -> List[int]:
    size = int(size)
    if size <= 0 or size > len(dataset):
        raise ValueError(f"Invalid requested sample size {size} for dataset of length {len(dataset)}.")
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(len(dataset), generator=generator)[:size].tolist()


def _subset(dataset, size: Optional[int]):
    if size is None:
        return dataset
    return Subset(dataset, list(range(min(int(size), len(dataset)))))


def _max_samples(dataset_cfg: Dict, split: str) -> Optional[int]:
    value = dataset_cfg.get(f"max_{split}_samples")
    if value is None:
        return None
    return int(value)


def _apply_sampling_protocol(train_full, test_dataset, dataset_cfg: Dict, seed: int):
    protocol = dataset_cfg.get("sampling_protocol", {}) or {}
    if not bool(protocol.get("enabled", False)):
        return train_full, test_dataset
    if protocol.get("total_size") is None:
        raise ValueError("sampling_protocol.enabled=true requires total_size for torchvision datasets.")
    total_size = int(protocol["total_size"])
    ratio = protocol.get("train_test_ratio", [4, 1])
    train_parts, test_parts = int(ratio[0]), int(ratio[1])
    if train_parts <= 0 or test_parts <= 0:
        raise ValueError("sampling_protocol.train_test_ratio must contain positive values.")
    train_pool_size = total_size * train_parts // (train_parts + test_parts)
    test_size = total_size - train_pool_size
    offset = int(protocol.get("seed_offset", 0))
    balanced = bool(protocol.get("class_balanced", True))
    index_fn = _class_balanced_indices if balanced else _random_indices
    train_indices = index_fn(train_full, train_pool_size, seed + offset)
    test_indices = index_fn(test_dataset, test_size, seed + offset + 100_000)
    return Subset(train_full, train_indices), Subset(test_dataset, test_indices)


def _split_train_val(dataset, val_split: float, seed: int):
    val_split = float(val_split)
    if val_split <= 0.0:
        return dataset, _subset(dataset, min(1, len(dataset)))
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_size, val_size], generator=generator)


def _class_names(name: str, split: str, base_dataset, num_classes: int) -> List[str]:
    if name == "emnist" and split == "letters":
        return [chr(ord("A") + index) for index in range(26)]
    if hasattr(base_dataset, "classes"):
        return [str(item) for item in base_dataset.classes]
    return [str(index) for index in range(num_classes)]


def create_dataloaders(dataset_cfg: Dict, seed: int = 7) -> DataBundle:
    name = dataset_key(dataset_cfg.get("name", "mnist"))
    root = dataset_cfg.get("root", "./data")
    input_size = int(dataset_cfg.get("input_size", 134))
    grayscale = bool(dataset_cfg.get("grayscale", True))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    smoke_test = bool(dataset_cfg.get("smoke_test", False))
    download = bool(dataset_cfg.get("download", True))

    transform = build_image_transform(
        input_size=input_size,
        grayscale=grayscale,
        fix_emnist_orientation=bool(name == "emnist" and dataset_cfg.get("fix_orientation", True)),
    )
    cls = DATASET_REGISTRY[name]
    kwargs = {"root": root, "transform": transform, "download": download}
    split = str(dataset_cfg.get("split", "letters" if name == "emnist" else "")).lower()
    if name == "emnist":
        if split not in EMNIST_NUM_CLASSES:
            raise ValueError("EMNIST split must be balanced, digits, letters, byclass, or bymerge.")
        kwargs["split"] = split
        if split == "letters":
            kwargs["target_transform"] = SubtractOne()

    train_full = cls(train=True, **kwargs)
    test_dataset = cls(train=False, **kwargs)
    train_full, test_dataset = _apply_sampling_protocol(train_full, test_dataset, dataset_cfg, seed)

    if smoke_test:
        train_full = _subset(train_full, int(dataset_cfg.get("smoke_train_size", 128)))
        test_dataset = _subset(test_dataset, int(dataset_cfg.get("smoke_test_size", 64)))

    train_dataset, val_dataset = _split_train_val(train_full, val_split, seed + 200_000)
    train_dataset = _subset(train_dataset, _max_samples(dataset_cfg, "train"))
    val_dataset = _subset(val_dataset, _max_samples(dataset_cfg, "val"))
    test_dataset = _subset(test_dataset, _max_samples(dataset_cfg, "test"))
    train_loader = DataLoader(train_dataset, **dataloader_kwargs(dataset_cfg, shuffle=True, seed=seed + 300_000))
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(dataset_cfg, shuffle=False))
    test_loader = DataLoader(test_dataset, **dataloader_kwargs(dataset_cfg, shuffle=False))

    if name == "emnist":
        num_classes = EMNIST_NUM_CLASSES[split]
    elif name == "cifar10":
        num_classes = 10
    else:
        num_classes = len(getattr(_base_dataset(train_full), "classes", list(range(10))))
    class_names = _class_names(name, split, _base_dataset(train_full), num_classes)
    return DataBundle(train_loader, val_loader, test_loader, num_classes, class_names)
