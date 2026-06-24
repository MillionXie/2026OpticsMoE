import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.optical_models import GeneralD2NNClassifier
from common.visualization.mask_viz import save_expert_phase_layers


def test_general_d2nn_phase_masks_are_saved(tmp_path):
    model = GeneralD2NNClassifier(
        num_classes=10,
        canvas_size=64,
        input_size=16,
        d2nn_phase_grid_size=16,
        num_layers=2,
        detector_size=4,
        readout_type="optical_only",
    )
    save_expert_phase_layers(model, tmp_path)
    assert (tmp_path / "d2nn_phase_layer_1.png").exists()
    assert (tmp_path / "d2nn_phase_layer_2.png").exists()
    assert (tmp_path / "d2nn_all_phase_layers.png").exists()
    assert (tmp_path / "global_fc_phase.png").exists()
