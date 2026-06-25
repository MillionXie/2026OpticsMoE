import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.expert_layout import ExpertLayout
from common.optics.optical_models import ASGlobalRouterMoEClassifier, GeneralD2NNClassifier


def test_moe_optical_parameter_count_uses_windowed_global_fc():
    layout = ExpertLayout(
        num_experts=9,
        canvas_size=1000,
        input_size=134,
        expert_size=134,
        expert_pitch=200,
        padding=200,
        prompt_aperture_size=600,
    )
    model = ASGlobalRouterMoEClassifier(
        num_classes=10,
        layout=layout,
        wavelength_m=5.32e-7,
        pixel_size_m=8.0e-6,
        focal_length_m=0.10,
        distances_m={
            "input_to_prompt": 0.20,
            "prompt_to_expert": 0.20,
            "inter_layer": 0.05,
            "layer5_to_fc": 0.05,
            "fc_to_detector": 0.05,
        },
        num_layers=5,
        detector_size=8,
        global_fc_phase_size=600,
        readout_type="linear",
    )
    assert model.expert_phase_parameter_count() == 5 * 9 * 134 * 134
    assert model.global_fc_parameter_count() == 600 * 600
    assert model.optical_parameter_count() == 808020 + 360000
    assert model.global_fc.trainable_parameter_count() == 360000
    assert model.prompt_parameter_count() == 18
    assert model.global_fc.phase_region() == [200, 800, 200, 800]


def test_general_d2nn_optical_parameter_count_uses_windowed_global_fc():
    model = GeneralD2NNClassifier(
        num_classes=10,
        canvas_size=1000,
        input_size=134,
        d2nn_phase_grid_size=402,
        num_layers=5,
        wavelength_m=5.32e-7,
        pixel_size_m=8.0e-6,
        distances_m={"input_to_prompt": 0.20, "inter_layer": 0.05, "layer5_to_fc": 0.05, "fc_to_detector": 0.05},
        detector_size=8,
        global_fc_phase_size=600,
        readout_type="linear",
    )
    assert model.d2nn_local_phase_parameter_count() == 5 * 402 * 402
    assert model.d2nn_global_fc_parameter_count() == 600 * 600
    assert model.optical_parameter_count() == 808020 + 360000
    assert model.global_fc.phase_region() == [200, 800, 200, 800]
