import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation_distillation.electronic_baselines import FeatureDistilledLeNetClassifier


def _model():
    return FeatureDistilledLeNetClassifier(
        num_classes=10,
        teacher_feature_dim=32,
        lenet_config={
            "input_channels": 1,
            "channels": [32, 64, 128],
            "activation": "gelu",
            "pooling": "avg",
            "adaptive_pool_size": 5,
            "output_feature_dim": 900,
        },
        feature_preprocess_config={"norm": "layernorm", "norm_affine": True, "activation": "gelu"},
        projector_config={
            "type": "mlp", "input_dim": 900, "output_dim": "auto_teacher_dim",
            "hidden_layers": 1, "hidden_dim": 64, "output_l2_normalize": True,
        },
        classifier_config={
            "input": "semantic_feature", "input_dim": "auto_teacher_dim",
            "hidden_layers": 1, "hidden_dim": 16, "activation": "gelu", "dropout": 0.0,
        },
    )


def test_lenet_feature_distilled_forward_shapes():
    model = _model()
    outputs = model(torch.rand(2, 1, 120, 120), return_intermediates=True)
    logits, raw, processed, semantic, semantic_normalized, intermediates = outputs
    assert logits.shape == (2, 10)
    assert raw.shape == processed.shape == (2, 900)
    assert semantic.shape == semantic_normalized.shape == (2, 32)
    assert intermediates["lenet_feature_raw"].shape == (2, 900)
    assert model.optical_parameter_count() == 0
    assert model.lenet_parameter_count() > 0


def test_lenet_classifier_receives_projected_semantic_feature():
    model = _model().eval()
    captured = {}
    handle = model.classifier.register_forward_pre_hook(
        lambda _module, args: captured.setdefault("classifier_input", args[0].detach())
    )
    with torch.inference_mode():
        _logits, _raw, _processed, _semantic, semantic_normalized, intermediates = model(
            torch.rand(2, 1, 64, 64), return_intermediates=True
        )
    handle.remove()
    assert captured["classifier_input"].shape == (2, 32)
    assert torch.allclose(captured["classifier_input"], semantic_normalized)
    assert torch.allclose(intermediates["classifier_feature"], semantic_normalized)
    assert captured["classifier_input"].shape[1] != model.student_feature_dim
