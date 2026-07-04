from __future__ import annotations

import torch

from experiments.bdd100k_weather4_optical5_mlp.metrics import classification_metrics
from experiments.bdd100k_weather4_optical5_mlp.model import Optical5MLPWeatherClassifier
from experiments.bdd100k_weather4_optical5_mlp.settings import PhaseDropoutSettings


def tiny_model() -> Optical5MLPWeatherClassifier:
    return Optical5MLPWeatherClassifier(
        input_size=12,
        optical_field_size=12,
        optical_padding_size=16,
        detector_pool_size=4,
        mlp_hidden_dim=8,
        phase_init="uniform",
        optical_layers=5,
        phase_dropout=PhaseDropoutSettings(enabled=True, p=0.2, start_epoch=1, block_size=2),
    )


def test_forward_shape_and_optical_gradients() -> None:
    model = tiny_model()
    logits = model(torch.rand(2, 1, 12, 12))
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert model.optical_layers[0].raw_phase.grad is not None
    assert model.optical_layers[0].raw_amplitude.grad is not None


def test_diagnostics_have_five_fields() -> None:
    model = tiny_model().eval()
    logits, diagnostics = model(torch.rand(1, 1, 12, 12), return_diagnostics=True)
    assert logits.shape == (1, 4)
    assert len(diagnostics["layer_intensities"]) == 5
    assert diagnostics["detector_input"].shape == (1, 12, 12)


def test_imbalance_metrics() -> None:
    metrics = classification_metrics([0, 0, 1, 2, 3], [0, 1, 1, 2, 2], ["clear", "rainy", "snowy", "foggy"])
    assert metrics["top1_accuracy"] == 0.6
    assert len(metrics["confusion_matrix"]) == 4
    assert set(metrics["per_class_f1"]) == {"clear", "rainy", "snowy", "foggy"}
