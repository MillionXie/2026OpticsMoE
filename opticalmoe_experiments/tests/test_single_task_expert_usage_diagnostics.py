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
from single_task.scripts.train_single_task import collect_optical_diagnostics, expert_usage_row, optical_energy_rows_from_intermediates


def _tiny_moe_config():
    return {
        "model": {
            "type": "learnable_route_moe",
            "num_experts": 9,
            "input_size": 32,
            "canvas_size": 256,
            "expert_size": 24,
            "expert_pitch": 50,
            "padding": 53,
            "prompt_aperture_size": 180,
            "num_layers": 1,
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
                "enabled": False,
                "mode": "none",
                "expert_p": 0.0,
                "global_fc_p": 0.0,
            }
        },
    }


def test_expert_usage_rows_include_energy_diagnostics():
    model = build_model(_tiny_moe_config(), num_classes=10)
    batch = (torch.rand(2, 1, 32, 32), torch.zeros(2, dtype=torch.long))
    diagnostics = collect_optical_diagnostics(model, batch, torch.device("cpu"))
    rows = expert_usage_row("run", 1, "mnist", "learnable_route_moe", model, diagnostics)
    assert len(rows) == 9
    assert rows[0]["expert_entrance_energy_ratio"] != ""
    assert rows[0]["expert_output_energy_ratio"] != ""
    energy_rows = optical_energy_rows_from_intermediates("run", 1, diagnostics["intermediates"], model)
    assert energy_rows
    assert any(row["stage"] == "expert_entrance_before_aperture" for row in energy_rows)
    assert energy_rows[0]["total_energy"] != ""
