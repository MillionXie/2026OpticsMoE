import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.expert_layout import ExpertLayout
from common.optics.optical_models import ASGlobalRouterMoEClassifier


def test_fast_geometry_phase_parameter_counts():
    model = ASGlobalRouterMoEClassifier(
        num_classes=10,
        layout=ExpertLayout(),
        num_layers=5,
        global_fc_phase_size=450,
        detector_size=4,
        readout_type="optical_only",
    )
    assert model.layout.to_dict()["expert_phase_params_per_layer"] == 129600
    assert model.expert_phase_parameter_count() == 648000
    assert model.global_fc_parameter_count() == 202500
    assert model.optical_parameter_count() == 850500
    actual_phase_parameters = sum(parameter.numel() for parameter in model.expert_layers.parameters())
    actual_phase_parameters += model.global_fc.trainable_parameter_count()
    assert actual_phase_parameters == model.optical_parameter_count()
