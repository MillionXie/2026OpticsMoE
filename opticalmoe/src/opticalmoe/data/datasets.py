from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from .dsprites import create_dsprites_dataloaders


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


class _SubtractOne:
    """Pickle-safe target transform for EMNIST letters labels 1..26."""

    def __call__(self, target):
        return int(target) - 1


class _PILToFloatTensorNoNumpy:
    """Convert a PIL image to [1,H,W] float tensor without using NumPy.

    Some server environments combine a PyTorch build compiled against one
    NumPy ABI with a different installed NumPy version. In that case
    torchvision.transforms.ToTensor can fail inside torch.from_numpy even
    though the object is a numpy.ndarray. This converter avoids that path.
    """

    def __call__(self, image):
        image = image.convert("L")
        width, height = image.size
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(height, width).unsqueeze(0)
        return tensor.to(dtype=torch.float32).div_(255.0)


def _dataset_class(name: str):
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset '{name}'. Use MNIST, FashionMNIST, KMNIST, "
            "EMNIST, or CIFAR10."
        )
    return DATASET_REGISTRY[key]


class _FixEMNISTOrientation:
    """Rotate/flip EMNIST glyphs into the usual upright image convention."""

    def __call__(self, image):
        return TF.hflip(TF.rotate(image, -90))


def _transform(input_size: int, fix_emnist_orientation: bool = False):
    steps = []
    if fix_emnist_orientation:
        steps.append(_FixEMNISTOrientation())
    steps.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            _PILToFloatTensorNoNumpy(),
        ]
    )
    return transforms.Compose(
        steps
    )


def _subset(dataset, size: int):
    if size is None:
        return dataset
    return Subset(dataset, range(min(int(size), len(dataset))))


def _labels_for_dataset(dataset) -> torch.Tensor:
    """Return labels for a torchvision dataset or Subset without loading images."""

    if isinstance(dataset, Subset):
        labels = _labels_for_dataset(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return labels[indices]
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError(
            "Class-balanced sampling requires a dataset with targets or labels."
        )
    return torch.as_tensor(labels, dtype=torch.long)


def _base_dataset(dataset):
    """Unwrap nested Subset objects to the underlying torchvision dataset."""

    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _class_balanced_indices(dataset, size: int, seed: int) -> list:
    """Pick a deterministic, approximately equal number of samples per class."""

    size = int(size)
    if size <= 0:
        raise ValueError("sample size must be positive.")
    labels = _labels_for_dataset(dataset)
    if size > len(labels):
        raise ValueError(
            f"Requested {size} samples, but dataset only has {len(labels)}."
        )

    classes = torch.unique(labels).sort().values.tolist()
    class_count = len(classes)
    base = size // class_count
    remainder = size % class_count
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for order, class_value in enumerate(classes):
        class_indices = torch.where(labels == int(class_value))[0]
        quota = base + (1 if order < remainder else 0)
        if quota > len(class_indices):
            raise ValueError(
                f"Class {class_value} has {len(class_indices)} samples, "
                f"but {quota} were requested."
            )
        permuted = class_indices[torch.randperm(len(class_indices), generator=generator)]
        selected.extend(permuted[:quota].tolist())

    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    selected_tensor = selected_tensor[
        torch.randperm(len(selected_tensor), generator=generator)
    ]
    return selected_tensor.tolist()


def _split_subset_train_val(dataset, val_split: float, seed: int):
    """Split a sampled train pool into train/validation subsets."""

    val_split = float(val_split)
    if val_split <= 0.0:
        return dataset, _subset(dataset, min(1, len(dataset)))
    val_size = max(1, int(len(dataset) * val_split))
    val_indices = set(_class_balanced_indices(dataset, val_size, seed))
    train_indices = [idx for idx in range(len(dataset)) if idx not in val_indices]
    if not train_indices:
        raise ValueError("Validation split consumed the whole training pool.")
    return Subset(dataset, train_indices), Subset(dataset, sorted(val_indices))


def _apply_sampling_protocol(train_full, test_dataset, dataset_cfg: Dict, seed: int):
    """Apply paper-style class-balanced total sampling and train:test splitting.

    Protocol semantics:
    - total_size counts the selected train-pool plus selected test samples.
    - train_test_ratio, for example [4, 1], splits total_size into train-pool
      and test sizes. A validation set is then carved from the train-pool using
      the normal val_split field.
    - Sampling is class-balanced and deterministic with the run seed.
    """

    protocol = dataset_cfg.get("sampling_protocol", {})
    if not protocol or not bool(protocol.get("enabled", False)):
        return train_full, test_dataset

    total_size = int(protocol["total_size"])
    ratio = protocol.get("train_test_ratio", [4, 1])
    if len(ratio) != 2:
        raise ValueError("sampling_protocol.train_test_ratio must have two values.")
    train_parts = int(ratio[0])
    test_parts = int(ratio[1])
    if train_parts <= 0 or test_parts <= 0:
        raise ValueError("train_test_ratio values must be positive.")
    train_pool_size = total_size * train_parts // (train_parts + test_parts)
    test_size = total_size - train_pool_size

    train_indices = _class_balanced_indices(
        train_full,
        train_pool_size,
        seed + int(protocol.get("seed_offset", 0)),
    )
    test_indices = _class_balanced_indices(
        test_dataset,
        test_size,
        seed + int(protocol.get("seed_offset", 0)) + 100_000,
    )
    return Subset(train_full, train_indices), Subset(test_dataset, test_indices)


def create_dataloaders(dataset_cfg: Dict, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    name = dataset_cfg.get("name", "mnist").lower()
    if name == "dsprites":
        return create_dsprites_dataloaders(dataset_cfg, seed)

    root = dataset_cfg.get("root", "./data")
    input_size = int(dataset_cfg.get("input_size", 200))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    batch_size = int(dataset_cfg.get("batch_size", 8))
    num_workers = int(dataset_cfg.get("num_workers", 2))
    smoke_test = bool(dataset_cfg.get("smoke_test", False))

    cls = _dataset_class(name)
    transform = _transform(
        input_size,
        fix_emnist_orientation=bool(
            name == "emnist" and dataset_cfg.get("fix_orientation", False)
        ),
    )

    dataset_kwargs = {
        "root": root,
        "transform": transform,
        "download": bool(dataset_cfg.get("download", True)),
    }
    if name == "emnist":
        split = dataset_cfg.get("split", "balanced").lower()
        if split not in EMNIST_NUM_CLASSES:
            raise ValueError(
                "EMNIST split must be balanced, digits, letters, byclass, or bymerge."
            )
        dataset_kwargs["split"] = split
        if split == "letters":
            dataset_kwargs["target_transform"] = _SubtractOne()

    train_full = cls(train=True, **dataset_kwargs)
    test_dataset = cls(train=False, **dataset_kwargs)
    train_full, test_dataset = _apply_sampling_protocol(
        train_full,
        test_dataset,
        dataset_cfg,
        seed,
    )

    if smoke_test:
        train_full = _subset(train_full, int(dataset_cfg.get("smoke_train_size", 256)))
        test_dataset = _subset(test_dataset, int(dataset_cfg.get("smoke_test_size", 128)))

    if dataset_cfg.get("sampling_protocol", {}).get("enabled", False):
        train_dataset, val_dataset = _split_subset_train_val(
            train_full,
            val_split,
            seed + 200_000,
        )
    else:
        val_size = max(1, int(len(train_full) * val_split))
        train_size = len(train_full) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_full, [train_size, val_size], generator=generator
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if name == "emnist":
        num_classes = EMNIST_NUM_CLASSES[dataset_cfg.get("split", "balanced").lower()]
    elif name == "cifar10":
        num_classes = 10
    else:
        base_dataset = _base_dataset(train_full)
        num_classes = len(base_dataset.classes)
    return train_loader, val_loader, test_loader, num_classes
