import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model


def tiny_config(model_type="learnable_route_moe"):
    return {
        "model": {
            "type": model_type,
            "num_experts": 9,
            "prompt_type": "complex_amplitude",
            "routing_type": "learnable" if model_type == "learnable_route_moe" else "fixed_uniform",
        },
        "layout": {
            "canvas_height": 128,
            "canvas_width": 128,
            "input_size": 16,
            "expert_size": 12,
            "expert_pitch": 30,
            "padding": 19,
            "prompt_aperture_size": 90,
        },
        "optics": {
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 8.0e-6,
            "num_layers": 1,
            "focal_length_m": 0.01,
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
        "prompt": {"mode": "complex_order_router", "train_amplitudes": model_type != "fixed_route_moe", "train_phase_biases": model_type != "fixed_route_moe"},
        "detector": {"detector_size": 4, "layout": "grid"},
        "readout": {"type": "mlp", "hidden_dim": 8, "hidden_layers": 1, "activation": "relu", "dropout": 0.0},
        "regularization": {"phase_dropout": {"enabled": False}},
        "training": {
            "multitask": {
                "tasks": [
                    {"name": "mnist", "head": {"hidden_dim": 8}},
                    {"name": "fashionmnist", "head": {"hidden_dim": 8}},
                    {"name": "emnist_letters", "head": {"hidden_dim": 8}},
                ]
            }
        },
    }


def test_dataset_switching_moe_forward_and_prompt_swap():
    task_names = ["mnist", "fashionmnist", "emnist_letters"]
    num_classes = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(tiny_config("learnable_route_moe"), task_names, num_classes)
    x = torch.rand(2, 1, 16, 16)
    assert model(x, task_name="mnist").shape == (2, 10)
    assert model(x, task_name="fashionmnist").shape == (2, 10)
    assert model(x, task_name="emnist_letters").shape == (2, 26)
    logits, intermediates = model(
        x,
        task_name="mnist",
        prompt_task_name="fashionmnist",
        readout_task_name="mnist",
        return_intermediates=True,
    )
    assert logits.shape == (2, 10)
    assert intermediates["prompt_task_name"] == "fashionmnist"
    assert intermediates["readout_task_name"] == "mnist"
    assert "prompt_amplitudes" in intermediates
    assert "expert_energy_ratios" in intermediates
    assert "detector_energies" in intermediates


def test_fixed_route_prompt_is_not_trainable():
    task_names = ["mnist", "fashionmnist", "emnist_letters"]
    num_classes = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(tiny_config("fixed_route_moe"), task_names, num_classes)
    for prompt in model.prompt_bank.prompts.values():
        assert prompt.amplitude_logits.requires_grad is False
        assert prompt.phase_biases.requires_grad is False
