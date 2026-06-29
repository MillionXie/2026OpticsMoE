import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.optics.distilled_moe import DetectorFeatureASGlobalRouterMoEClassifier
from common.optics.expert_layout import ExpertLayout


def test_end_to_end_moe_has_no_projector_and_returns_detector_feature():
    layout = ExpertLayout(
        num_experts=9,
        canvas_size=128,
        input_size=16,
        expert_size=12,
        expert_pitch=30,
        padding=19,
        prompt_aperture_size=90,
    )
    model = DetectorFeatureASGlobalRouterMoEClassifier(
        num_classes=10,
        layout=layout,
        feature_detector_config={"grid_size": 4, "feature_dim": 16},
        classifier_config={"hidden_dim": 8, "hidden_layers": 1},
        num_layers=1,
        global_fc_phase_size=90,
        distances_m={
            key: 0.01
            for key in ("input_to_prompt", "prompt_to_expert", "inter_layer", "layer5_to_fc", "fc_to_detector")
        },
    )
    logits, optical_feature, intermediates = model(torch.rand(2, 1, 16, 16), return_intermediates=True)
    assert logits.shape == (2, 10)
    assert optical_feature.shape == (2, 16)
    assert "detector_intensity" in intermediates
    assert "after_global_fc" in intermediates
    assert not hasattr(model, "projector")
    assert model.total_parameter_count() == (
        model.optical_parameter_count() + model.prompt_parameter_count() + model.classifier_parameter_count()
    )

