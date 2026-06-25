import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.optical_models import GeneralD2NNClassifier
from common.reporting.run_manifest import architecture_report


def test_general_d2nn_accounting_includes_global_fc():
    model = GeneralD2NNClassifier(
        num_classes=10,
        canvas_size=1000,
        input_size=134,
        d2nn_phase_grid_size=402,
        num_layers=5,
        detector_size=8,
        readout_type="optical_only",
    )
    assert model.d2nn_local_phase_parameter_count() == 5 * 402 * 402
    assert model.d2nn_global_fc_parameter_count() == 600 * 600
    assert model.optical_parameter_count() == 5 * 402 * 402 + 600 * 600
    assert model.prompt_parameter_count() == 0


def test_general_d2nn_architecture_report_records_accounting(tmp_path):
    model = GeneralD2NNClassifier(
        num_classes=10,
        canvas_size=64,
        input_size=16,
        d2nn_phase_grid_size=16,
        num_layers=2,
        detector_size=4,
        readout_type="optical_only",
    )
    report = architecture_report(
        model,
        {"model": {"type": "general_d2nn"}, "readout": {"type": "optical_only"}},
        tmp_path,
    )
    assert report["d2nn_local_phase_params"] == 2 * 16 * 16
    assert report["d2nn_global_fc_params"] == 64 * 64
    assert report["optical_parameter_count"] == 2 * 16 * 16 + 64 * 64
    assert report["global_fc_is_full_canvas"] is False
