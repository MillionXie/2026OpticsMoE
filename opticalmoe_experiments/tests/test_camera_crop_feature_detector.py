import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.optics.distilled_moe import FeatureDistilledASGlobalRouterMoEClassifier, GridPoolFeatureDetector
from common.optics.expert_layout import ExpertLayout


def _model(feature_dim=900):
    layout = ExpertLayout(
        num_experts=9, canvas_size=96, input_size=16, expert_size=10,
        expert_pitch=24, padding=12, prompt_aperture_size=72,
    )
    return FeatureDistilledASGlobalRouterMoEClassifier(
        num_classes=3,
        teacher_feature_dim=8,
        layout=layout,
        feature_detector_config={"source_region": "camera_active_window", "grid_size": 30, "feature_dim": feature_dim},
        feature_preprocess_config={"norm": "layernorm", "activation": "gelu"},
        classifier_config={"input": "semantic_feature", "input_dim": "auto_teacher_dim", "hidden_dim": 4},
        projector_config={"type": "linear", "input_dim": "auto_feature_dim", "output_dim": "auto_teacher_dim"},
        num_layers=1,
        global_fc_phase_size=72,
        distances_m={key: 0.01 for key in ("input_to_prompt", "prompt_to_expert", "inter_layer", "layer5_to_fc", "fc_to_detector")},
    )


def test_camera_crop_and_grid30_feature_shape():
    model = _model()
    with torch.inference_mode():
        outputs = model(torch.rand(2, 1, 16, 16), return_intermediates=True)
    intermediates = outputs[-1]
    assert intermediates["detector_intensity"].shape == (2, 96, 96)
    assert intermediates["camera_intensity"].shape == (2, 72, 72)
    assert intermediates["camera_region"] == [12, 84, 12, 84]
    assert intermediates["camera_feature_raw"].shape == (2, 900)
    assert intermediates["outside_camera_energy_ratio"].shape == (2,)


def test_grid_pool_only_uses_supplied_camera_crop():
    detector = GridPoolFeatureDetector(grid_size=30, pooling="sum")
    camera = torch.ones(1, 450, 450)
    feature = detector(camera)
    assert feature.shape == (1, 900)
    assert torch.all(feature == 225.0)


def test_feature_dim_mismatch_is_rejected():
    with pytest.raises(ValueError, match=r"must equal 30\^2=900"):
        _model(feature_dim=899)
