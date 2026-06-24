import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import datasets as data_mod
from common.data.datasets import create_dataloaders


class TinyVisionDataset(torch.utils.data.Dataset):
    classes = [str(i) for i in range(10)]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, split=None):
        self.transform = transform
        self.target_transform = target_transform
        self.targets = [i % len(self.classes) for i in range(40 if train else 20)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.new("RGB", (16, 16), color=(index % 255, 30, 200))
        target = self.targets[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


class TinyEMNIST(TinyVisionDataset):
    classes = [chr(ord("A") + i) for i in range(26)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = [(i % 26) + 1 for i in range(len(self.targets))]


def test_supported_dataloaders(monkeypatch):
    monkeypatch.setitem(data_mod.DATASET_REGISTRY, "mnist", TinyVisionDataset)
    monkeypatch.setitem(data_mod.DATASET_REGISTRY, "fashionmnist", TinyVisionDataset)
    monkeypatch.setitem(data_mod.DATASET_REGISTRY, "kmnist", TinyVisionDataset)
    monkeypatch.setitem(data_mod.DATASET_REGISTRY, "cifar10", TinyVisionDataset)
    monkeypatch.setitem(data_mod.DATASET_REGISTRY, "emnist", TinyEMNIST)

    for name in ["mnist", "fashionmnist", "kmnist", "cifar10"]:
        bundle = create_dataloaders({"name": name, "input_size": 32, "batch_size": 4, "download": False}, seed=7)
        images, _ = next(iter(bundle.train_loader))
        assert images.shape[1:] == (1, 32, 32)
        assert bundle.num_classes == 10

    bundle = create_dataloaders(
        {"name": "emnist", "split": "letters", "input_size": 32, "batch_size": 4, "download": False},
        seed=7,
    )
    images, targets = next(iter(bundle.train_loader))
    assert images.shape[1:] == (1, 32, 32)
    assert bundle.num_classes == 26
    assert int(targets.min()) >= 0

