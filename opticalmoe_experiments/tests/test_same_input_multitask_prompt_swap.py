import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from same_input_multitask.scripts.train_same_input_multitask import build_model


def _config():
    return {
        "model": {"type": "learnable_route_moe", "num_experts": 9},
        "layout": {"canvas_height": 1000, "input_size": 134, "expert_size": 134, "expert_pitch": 200, "padding": 200, "prompt_aperture_size": 600},
        "optics": {"num_layers": 1, "distances_m": {"input_to_prompt": 0.20, "prompt_to_expert": 0.20, "inter_layer": 0.05, "layer5_to_fc": 0.05, "fc_to_detector": 0.05}},
        "prompt": {"mode": "complex_order_router", "amplitude_init_logits": 2.0, "train_amplitudes": True, "train_phase_biases": True},
        "detector": {"detector_size": 8, "layout": "grid"},
        "readout": {"type": "linear", "normalize_detector_energy": True},
        "regularization": {"phase_dropout": {"enabled": False}},
    }


def test_same_input_model_forward_and_prompt_swap():
    tasks = ["shape", "scale", "x_position_4bin", "y_position_4bin"]
    classes = {"shape": 3, "scale": 6, "x_position_4bin": 4, "y_position_4bin": 4}
    model = build_model(_config(), tasks, classes)
    images = torch.rand(2, 1, 134, 134)
    assert model(images, task_name="shape").shape == (2, 3)
    assert model(images, task_name="scale").shape == (2, 6)
    assert model(images, task_name="x_position_4bin").shape == (2, 4)
    logits, ints = model(images, task_name="shape", prompt_task_name="scale", readout_task_name="shape", return_intermediates=True)
    assert logits.shape == (2, 3)
    assert ints["prompt_task_name"] == "scale"
    assert ints["readout_task_name"] == "shape"
    assert "prompt_amplitudes" in ints
    assert "expert_energy_ratios" in ints
    assert "detector_energies" in ints
