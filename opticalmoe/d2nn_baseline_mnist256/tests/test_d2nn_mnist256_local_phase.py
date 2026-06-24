import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import CenteredLocalPhaseLayer, D2NNClassifier


def base_config():
    return {
        "optics": {
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 8.0e-6,
            "canvas_size": 400,
            "input_size": 256,
            "phase_mask_size": 256,
            "phase_mask_mode": "centered_local",
            "num_layers": 5,
            "phase_param": "unconstrained",
            "phase_init": "identity",
            "init_std": 0.02,
            "evanescent_mode": "zero",
            "input_to_layer_distance_m": 0.05,
            "inter_layer_distance_m": 0.05,
            "detector_distance_m": 0.05,
        },
        "detector": {
            "detector_size": 32,
            "layout": "grid",
            "normalize_detector_energy": True,
        },
        "readout": {
            "type": "optical_only",
            "input_norm": "none",
            "norm_affine": False,
            "logit_scale": 10.0,
        },
        "regularization": {
            "phase_dropout": {
                "enabled": True,
                "mode": "block_phase_bypass",
                "p": 0.05,
                "block_size": 8,
                "batch_shared": True,
                "start_epoch": 10,
            }
        },
    }


def test_local_phase_layer_shapes_and_param_count():
    model = D2NNClassifier(base_config(), num_classes=10)
    assert model.phase_mask_size == 256
    assert model.canvas_size == 400
    assert len(model.phase_layers) == 5
    for layer in model.phase_layers:
        assert tuple(layer.raw_phase.shape) == (256, 256)
    assert model.optical_parameter_count() == 5 * 256 * 256
    assert model.phase_mask_region() == [72, 328, 72, 328]


def test_forward_intermediates_include_input_to_layer_and_local_layers():
    model = D2NNClassifier(base_config(), num_classes=10)
    x = torch.rand(2, 1, 256, 256)
    logits, intermediates = model(x, return_intermediates=True)
    assert tuple(logits.shape) == (2, 10)
    for key in [
        "canvas_input_400",
        "after_input_to_layer",
        "after_phase_modulation_1",
        "after_propagation_1",
        "detector_field",
        "detector_energies",
    ]:
        assert key in intermediates
    assert tuple(intermediates["detector_energies"].shape) == (2, 10)


def test_centered_local_phase_padding_is_transparent():
    layer = CenteredLocalPhaseLayer(
        canvas_size=400,
        phase_mask_size=256,
        init="identity",
        phase_dropout_mode="none",
    )
    with torch.no_grad():
        layer.raw_phase.fill_(1.0)
    field = torch.ones(1, 400, 400, dtype=torch.complex64)
    output = layer(field)
    y0, y1, x0, x1 = layer.phase_mask_region()
    outside = output.clone()
    outside[:, y0:y1, x0:x1] = 1.0 + 0j
    assert torch.allclose(outside, torch.ones_like(outside))
    center = output[:, y0:y1, x0:x1]
    assert not torch.allclose(center, torch.ones_like(center))

