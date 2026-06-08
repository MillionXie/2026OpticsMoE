from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


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


def _transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            _PILToFloatTensorNoNumpy(),
        ]
    )


def _subset(dataset, size: int):
    if size is None:
        return dataset
    return Subset(dataset, range(min(int(size), len(dataset))))


def create_dataloaders(dataset_cfg: Dict, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    name = dataset_cfg.get("name", "mnist").lower()
    root = dataset_cfg.get("root", "./data")
    input_size = int(dataset_cfg.get("input_size", 200))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    batch_size = int(dataset_cfg.get("batch_size", 8))
    num_workers = int(dataset_cfg.get("num_workers", 2))
    smoke_test = bool(dataset_cfg.get("smoke_test", False))

    cls = _dataset_class(name)
    transform = _transform(input_size)

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

    if smoke_test:
        train_full = _subset(train_full, int(dataset_cfg.get("smoke_train_size", 256)))
        test_dataset = _subset(test_dataset, int(dataset_cfg.get("smoke_test_size", 128)))

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
        base_dataset = train_full.dataset if isinstance(train_full, Subset) else train_full
        num_classes = len(base_dataset.classes)
    return train_loader, val_loader, test_loader, num_classes
