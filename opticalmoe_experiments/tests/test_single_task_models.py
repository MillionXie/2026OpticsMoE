import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SINGLE = ROOT / "single_task"
if str(SINGLE) not in sys.path:
    sys.path.insert(0, str(SINGLE))

from baselines.model_factory import build_model
from common.training.phase_dropout import phase_dropout_active_for_epoch, phase_dropout_settings


def tiny_config(model_type="learnable_route_moe"):
    cfg = {
        "model": {
            "type": model_type,
            "num_experts": 9,
            "input_size": 32,
            "canvas_size": 256,
            "expert_size": 24,
            "expert_pitch": 50,
            "padding": 38,
            "prompt_aperture_size": 180,
            "num_layers": 1,
            "d2nn_phase_grid_size": 64,
            "d2nn_num_layers": 1,
        },
        "optics": {
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 8.0e-6,
            "focal_length_m": 0.10,
            "expert_phase_init": "identity",
            "global_fc_phase_init": "identity",
            "distances_m": {
                "input_to_prompt": 0.01,
                "prompt_to_expert": 0.01,
                "inter_layer": 0.01,
                "layer5_to_fc": 0.01,
                "fc_to_detector": 0.01,
            },
        },
        "prompt": {"mode": "complex_order_router", "train_amplitudes": True, "train_phase_biases": True},
        "detector": {"detector_size": 8, "layout": "grid"},
        "readout": {"type": "mlp", "hidden_dim": 8, "hidden_layers": 1, "activation": "relu", "dropout": 0.0},
        "regularization": {
            "phase_dropout": {
                "enabled": True,
                "mode": "block_phase_bypass",
                "expert_p": 0.05,
                "global_fc_p": 0.0,
                "block_size": 4,
                "batch_shared": True,
                "apply_to_experts": True,
                "apply_to_global_fc": False,
                "start_epoch": 2,
            }
        },
    }
    return cfg


def test_model_forwards():
    x = torch.rand(2, 1, 32, 32)
    for model_type in ["learnable_route_moe", "fixed_route_moe", "general_d2nn", "lenet5"]:
        model = build_model(tiny_config(model_type), num_classes=10)
        out = model(x, return_intermediates=True)
        logits, intermediates = out if isinstance(out, tuple) else (out, {})
        assert logits.shape == (2, 10)
        if model_type != "lenet5":
            assert "detector_energies" in intermediates


def test_phase_dropout_config_and_schedule():
    cfg = tiny_config("learnable_route_moe")
    model = build_model(cfg, num_classes=10)
    for layer in model.expert_layers:
        for local in layer.local_phases:
            assert local.phase_dropout_mode == "block_phase_bypass"
            assert local.phase_dropout_p == 0.05
            assert local.phase_dropout_block_size == 4
    settings = phase_dropout_settings(cfg)
    assert phase_dropout_active_for_epoch(settings, 1) is False
    assert phase_dropout_active_for_epoch(settings, 2) is True
