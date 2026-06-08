import torch
from PIL import Image
from torch.utils.data import Dataset

from opticalmoe.data.datasets import (
    DATASET_REGISTRY,
    EMNIST_NUM_CLASSES,
    _SubtractOne,
    _apply_sampling_protocol,
    _split_subset_train_val,
    _transform,
)


def test_extended_dataset_registry():
    assert {
        "mnist",
        "fashionmnist",
        "kmnist",
        "emnist",
        "cifar10",
    }.issubset(DATASET_REGISTRY)
    assert EMNIST_NUM_CLASSES["balanced"] == 47
    assert EMNIST_NUM_CLASSES["letters"] == 26


def test_emnist_letters_targets_are_remapped_to_zero_based_indices():
    transform = _SubtractOne()
    assert transform(1) == 0
    assert transform(26) == 25


def test_pil_transform_does_not_require_numpy_conversion():
    image = Image.new("L", (4, 3), color=128)
    tensor = _transform(8)(image)

    assert tensor.shape == (1, 8, 8)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


class _TargetOnlyDataset(Dataset):
    def __init__(self, class_count: int, samples_per_class: int):
        self.targets = []
        for class_index in range(class_count):
            self.targets.extend([class_index] * samples_per_class)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.zeros(1, 2, 2), int(self.targets[index])


def test_sampling_protocol_uses_total_size_and_four_to_one_split():
    train_source = _TargetOnlyDataset(class_count=10, samples_per_class=300)
    test_source = _TargetOnlyDataset(class_count=10, samples_per_class=100)
    config = {
        "sampling_protocol": {
            "enabled": True,
            "total_size": 2000,
            "train_test_ratio": [4, 1],
        }
    }

    train_pool, test_subset = _apply_sampling_protocol(
        train_source,
        test_source,
        config,
        seed=7,
    )

    assert len(train_pool) == 1600
    assert len(test_subset) == 400


def test_sampled_train_pool_can_be_split_for_validation():
    train_source = _TargetOnlyDataset(class_count=10, samples_per_class=300)
    config = {
        "sampling_protocol": {
            "enabled": True,
            "total_size": 2000,
            "train_test_ratio": [4, 1],
        }
    }
    train_pool, _ = _apply_sampling_protocol(
        train_source,
        _TargetOnlyDataset(class_count=10, samples_per_class=100),
        config,
        seed=7,
    )

    train_subset, val_subset = _split_subset_train_val(
        train_pool,
        val_split=0.1,
        seed=17,
    )

    assert len(train_subset) == 1440
    assert len(val_subset) == 160
