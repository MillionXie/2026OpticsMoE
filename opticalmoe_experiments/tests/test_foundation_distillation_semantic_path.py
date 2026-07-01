import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.optics.distilled_moe import build_projector


def test_linear_and_mlp_projectors_resolve_auto_dimensions():
    linear, linear_cfg = build_projector(
        900, 32, {"type": "linear", "input_dim": "auto_feature_dim", "output_dim": "auto_teacher_dim"}
    )
    mlp, mlp_cfg = build_projector(
        900, 32, {
            "type": "mlp", "input_dim": "auto_feature_dim", "output_dim": "auto_teacher_dim",
            "hidden_layers": 1, "hidden_dim": 64, "hidden_norm": "layernorm",
        },
    )
    assert linear(torch.rand(2, 900)).shape == (2, 32)
    assert mlp(torch.rand(2, 900)).shape == (2, 32)
    assert linear_cfg["input_dim"] == mlp_cfg["input_dim"] == 900
    assert linear_cfg["output_dim"] == mlp_cfg["output_dim"] == 32


def test_classifier_and_feature_loss_share_semantic_representation():
    from test_camera_crop_feature_detector import _model

    model = _model()
    captured = {}
    handle = model.classifier.register_forward_pre_hook(lambda _module, args: captured.setdefault("input", args[0].detach()))
    with torch.inference_mode():
        logits, _raw, _processed, semantic, semantic_normalized, intermediates = model(
            torch.rand(2, 1, 16, 16), return_intermediates=True
        )
    handle.remove()
    assert logits.shape == (2, 3)
    assert semantic.shape == semantic_normalized.shape == (2, 8)
    assert torch.allclose(captured["input"], semantic_normalized)
    assert torch.allclose(intermediates["semantic_feature_normalized"], semantic_normalized)
    assert model.projector_parameter_count() > 0
    assert model.electronic_parameter_count() == (
        model.projector_parameter_count() + model.classifier_parameter_count()
    )
    assert model.total_parameter_count() == (
        model.optical_parameter_count()
        + model.prompt_parameter_count()
        + model.feature_preprocess_parameter_count()
        + model.electronic_parameter_count()
    )
