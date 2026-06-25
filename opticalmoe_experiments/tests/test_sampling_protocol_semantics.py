import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.datasets import _apply_sampling_protocol, _labels_for_dataset, _split_train_val, _subset


class ToyClassDataset(Dataset):
    def __init__(self, count: int, num_classes: int = 10):
        self.targets = torch.arange(count) % num_classes

    def __len__(self):
        return int(len(self.targets))

    def __getitem__(self, index):
        return torch.zeros(1, 8, 8), int(self.targets[index])


def test_torchvision_sampling_total_size_semantics():
    train_full = ToyClassDataset(20000, 10)
    test_full = ToyClassDataset(10000, 10)
    cfg = {
        "val_split": 0.1,
        "sampling_protocol": {
            "enabled": True,
            "total_size": 10000,
            "train_test_ratio": [4, 1],
            "class_balanced": True,
            "seed_offset": 0,
        },
    }
    train_pool, test = _apply_sampling_protocol(train_full, test_full, cfg, seed=7)
    train, val = _split_train_val(train_pool, 0.1, seed=8)
    assert len(train_pool) == 8000
    assert len(test) == 2000
    assert len(val) == 800
    assert len(train) == 7200
    labels = _labels_for_dataset(test)
    counts = torch.bincount(labels, minlength=10)
    assert int(counts.max() - counts.min()) <= 1


def test_max_split_override_caps_each_split():
    dataset = ToyClassDataset(1000, 10)
    assert len(_subset(dataset, 100)) == 100
    assert len(_subset(dataset, None)) == 1000
