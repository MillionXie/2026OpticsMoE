from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets

from .datasets import _transform


class MixedMNISTFashion(Dataset):
    """Concatenate MNIST and FashionMNIST with shared 0-9 labels.

    task_id is diagnostic metadata only:
    0 = MNIST, 1 = FashionMNIST. The class label remains 0-9, not 0-19.
    """

    def __init__(self, root: str, train: bool, input_size: int, download: bool = True) -> None:
        transform = _transform(input_size)
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        self.fashion = datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self) -> int:
        return len(self.mnist) + len(self.fashion)

    def __getitem__(self, index: int):
        if index < len(self.mnist):
            image, label = self.mnist[index]
            return image, int(label), 0
        image, label = self.fashion[index - len(self.mnist)]
        return image, int(label), 1


def _subset(dataset, size: int):
    if size is None:
        return dataset
    return Subset(dataset, range(min(int(size), len(dataset))))


def _balanced_mixed_subset(dataset: MixedMNISTFashion, size: int):
    """Take a balanced smoke subset from the concatenated mixed dataset."""

    if size is None:
        return dataset
    size = min(int(size), len(dataset))
    mnist_count = min(size // 2, len(dataset.mnist))
    fashion_count = min(size - mnist_count, len(dataset.fashion))
    mnist_indices = list(range(mnist_count))
    fashion_offset = len(dataset.mnist)
    fashion_indices = list(range(fashion_offset, fashion_offset + fashion_count))
    return Subset(dataset, mnist_indices + fashion_indices)


def create_mixed_mnist_fashion_dataloaders(dataset_cfg: Dict, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    root = dataset_cfg.get("root", "./data")
    input_size = int(dataset_cfg.get("input_size", 200))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    batch_size = int(dataset_cfg.get("batch_size", 4))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    smoke_test = bool(dataset_cfg.get("smoke_test", False))

    train_full = MixedMNISTFashion(root=root, train=True, input_size=input_size, download=True)
    test_dataset = MixedMNISTFashion(root=root, train=False, input_size=input_size, download=True)

    if smoke_test:
        train_full = _balanced_mixed_subset(train_full, int(dataset_cfg.get("smoke_train_size", 256)))
        test_dataset = _balanced_mixed_subset(test_dataset, int(dataset_cfg.get("smoke_test_size", 128)))

    val_size = max(1, int(len(train_full) * val_split))
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size], generator=generator)

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, 10
