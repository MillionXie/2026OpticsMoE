import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import D2NNClassifier
from optics import PhaseLayer


def base_config(phase_init="zeros"):
    return {
        "dataset": {"class_digits": [0, 1, 2, 3], "input_size": 400},
        "optics": {
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 16.0e-6,
            "canvas_size": 400,
            "input_size": 400,
            "phase_mask_size": 400,
            "phase_mask_mode": "full_canvas",
            "num_layers": 1,
            "phase_param": "sigmoid",
            "phase_init": phase_init,
            "init_std": 0.02,
            "evanescent_mode": "zero",
            "input_to_layer_distance_m": 0.03,
            "inter_layer_distance_m": 0.03,
            "detector_distance_m": 0.20,
        },
        "detector": {
            "detector_size": 50,
            "layout": "fixed_2x2",
            "start_pos_x": 75,
            "start_pos_y": 75,
            "N_det_sets": [2, 2],
            "det_steps_x": [150, 150],
            "det_steps_y": 150,
            "normalize_detector_energy": True,
        },
        "readout": {"type": "detector_only"},
        "regularization": {"phase_dropout": {"enabled": False}},
    }


def test_model_is_one_full_canvas_phase_layer_and_detector_only():
    model = D2NNClassifier(base_config(), num_classes=4)
    assert model.canvas_size == 400
    assert model.input_size == 400
    assert model.phase_mask_size == 400
    assert len(model.phase_layers) == 1
    assert tuple(model.phase_layers[0].raw_phase.shape) == (400, 400)
    assert model.phase_mask_region() == [0, 400, 0, 400]
    assert model.optical_parameter_count() == 400 * 400
    assert model.electronic_parameter_count() == 0


def test_forward_outputs_four_detector_energies_without_electronic_readout():
    model = D2NNClassifier(base_config(), num_classes=4)
    x = torch.rand(2, 1, 400, 400)
    logits, intermediates = model(x, return_intermediates=True)
    assert tuple(logits.shape) == (2, 4)
    assert tuple(intermediates["detector_energies"].shape) == (2, 4)
    assert torch.allclose(logits, intermediates["detector_energies"])
    assert torch.all(logits >= 0)


def test_fixed_detector_layout_matches_requested_regions():
    model = D2NNClassifier(base_config(), num_classes=4)
    masks = model.detector.masks
    expected = [(75, 75), (75, 225), (225, 75), (225, 225)]
    for index, (y0, x0) in enumerate(expected):
        assert masks[index, y0 : y0 + 50, x0 : x0 + 50].sum().item() == 50 * 50
        assert masks[index].sum().item() == 50 * 50


def test_sigmoid_phase_constraint_is_between_zero_and_two_pi():
    for init in ["zeros", "uniform", "gaussian"]:
        layer = PhaseLayer(400, parameterization="sigmoid", init=init, init_std=0.02)
        phase = layer.get_phase()
        assert torch.all(phase >= 0)
        assert torch.all(phase <= 2.0 * math.pi)
