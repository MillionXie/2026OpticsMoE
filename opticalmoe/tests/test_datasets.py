import torch
from PIL import Image

from opticalmoe.data.datasets import (
    DATASET_REGISTRY,
    EMNIST_NUM_CLASSES,
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


def test_pil_transform_does_not_require_numpy_conversion():
    image = Image.new("L", (4, 3), color=128)
    tensor = _transform(8)(image)

    assert tensor.shape == (1, 8, 8)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
