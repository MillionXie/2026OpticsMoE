from opticalmoe.data.datasets import DATASET_REGISTRY, EMNIST_NUM_CLASSES


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
