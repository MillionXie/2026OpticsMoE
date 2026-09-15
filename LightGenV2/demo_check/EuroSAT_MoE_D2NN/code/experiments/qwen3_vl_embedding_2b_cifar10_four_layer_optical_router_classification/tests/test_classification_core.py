from __future__ import annotations

from collections import Counter

import pytest
import torch

try:
    from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import (
        BalancedClassBatchSampler,
    )
    from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.modeling import (
        CIFAR10ClassificationHead,
    )
except ModuleNotFoundError:
    from qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import (
        BalancedClassBatchSampler,
    )
    from qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.modeling import (
        CIFAR10ClassificationHead,
    )


def test_classification_head_returns_raw_ten_class_logits() -> None:
    torch.manual_seed(7)
    head = CIFAR10ClassificationHead(384, 10)
    features = torch.randn(4, 384, requires_grad=True)
    logits = head(features)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits.norm(dim=1), torch.ones(4), atol=1.0e-3)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert head.classifier.weight.grad is not None
    assert torch.isfinite(head.classifier.weight.grad).all()


def test_classification_head_rejects_wrong_detector_width() -> None:
    head = CIFAR10ClassificationHead(384, 10)
    with pytest.raises(RuntimeError, match="384"):
        head(torch.randn(2, 224))


def test_balanced_sampler_is_deterministic_and_balanced() -> None:
    labels = tuple(label for label in range(10) for _ in range(11))
    left = BalancedClassBatchSampler(
        labels,
        classes_per_batch=10,
        samples_per_class=3,
        steps=4,
        seed=42,
    )
    right = BalancedClassBatchSampler(
        labels,
        classes_per_batch=10,
        samples_per_class=3,
        steps=4,
        seed=42,
    )
    left.set_epoch(3)
    right.set_epoch(3)
    left_batches = list(left)
    assert left_batches == list(right)
    for batch in left_batches:
        counts = Counter(labels[index] for index in batch)
        assert counts == Counter({label: 3 for label in range(10)})
