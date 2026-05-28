from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


DATASET_REGISTRY = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}


def _dataset_class(name: str):
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{name}'. Use MNIST, FashionMNIST, or KMNIST.")
    return DATASET_REGISTRY[key]


def _transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )


def _subset(dataset, size: int):
    if size is None:
        return dataset
    return Subset(dataset, range(min(int(size), len(dataset))))


def create_dataloaders(dataset_cfg: Dict, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    name = dataset_cfg.get("name", "mnist")
    root = dataset_cfg.get("root", "./data")
    input_size = int(dataset_cfg.get("input_size", 200))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    batch_size = int(dataset_cfg.get("batch_size", 8))
    num_workers = int(dataset_cfg.get("num_workers", 2))
    smoke_test = bool(dataset_cfg.get("smoke_test", False))

    cls = _dataset_class(name)
    transform = _transform(input_size)

    train_full = cls(root=root, train=True, transform=transform, download=True)
    test_dataset = cls(root=root, train=False, transform=transform, download=True)

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

    num_classes = len(train_full.dataset.classes) if isinstance(train_full, Subset) else len(train_full.classes)
    return train_loader, val_loader, test_loader, num_classes
