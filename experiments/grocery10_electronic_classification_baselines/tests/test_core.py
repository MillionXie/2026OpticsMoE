from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from experiments.grocery10_electronic_classification_baselines.modeling import (
    ElectronicTenClassClassifier,
    parameter_report,
)
from experiments.grocery10_electronic_classification_baselines.settings import (
    load_settings,
)
from experiments.grocery10_electronic_classification_baselines.training import (
    classification_metrics,
    load_checkpoint,
    save_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_formal_configs_share_the_same_ten_skus_and_direct_task() -> None:
    values = []
    for name in (
        "resnet18_classification.yaml",
        "efficientnet_b0_classification.yaml",
        "mobilenet_v3_small_classification.yaml",
    ):
        settings = load_settings(ROOT / "configs" / name)
        values.append(settings.selected_skus)
        assert settings.weights == "imagenet1k"
        assert len(settings.selected_skus) == 10
        assert settings.output_dir.is_relative_to(ROOT)
    assert values[0] == values[1] == values[2]


def test_backbones_return_ten_raw_logits_without_similarity_normalization() -> None:
    for name in ("resnet18", "efficientnet_b0", "mobilenet_v3_small"):
        model = ElectronicTenClassClassifier(name, 10, weights="none")
        model.eval()
        logits = model(torch.randn(2, 3, 64, 64))
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()
        assert not torch.allclose(logits.norm(dim=1), torch.ones(2), atol=1e-3)
        assert isinstance(model.classifier, nn.Linear)


def test_direct_cross_entropy_backpropagates() -> None:
    model = ElectronicTenClassClassifier(
        "mobilenet_v3_small", 10, weights="none"
    )
    logits = model(torch.randn(4, 3, 64, 64))
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_parameter_report_identifies_classification_head() -> None:
    model = ElectronicTenClassClassifier("resnet18", 10, weights="none")
    report = parameter_report(model)
    assert report["task"] == "grocery10_direct_classification"
    assert report["similarity_matching_used"] is False
    assert report["classifier_parameters"] == 512 * 10 + 10
    assert report["classification_head_parameters"] == (
        report["feature_norm_parameters"] + report["classifier_parameters"]
    )
    assert report["output_shape"] == [None, 10]


def test_classification_metrics() -> None:
    labels = torch.arange(10)
    logits = torch.eye(10) * 4
    metrics = classification_metrics(labels, logits, 0.25)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    settings = load_settings(ROOT / "configs" / "mobilenet_v3_small_smoke.yaml")
    source = ElectronicTenClassClassifier(
        "mobilenet_v3_small", 10, weights="none"
    )
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source, optimizer, None, 2, 0.75, settings, "digest")
    target = ElectronicTenClassClassifier(
        "mobilenet_v3_small", 10, weights="none"
    )
    payload = load_checkpoint(path, target)
    assert payload["epoch"] == 2
    for left, right in zip(source.state_dict().values(), target.state_dict().values()):
        assert torch.equal(left, right)
