import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.visualization.curve_viz import save_training_curves
from common.visualization.lightfield_viz import save_light_fields
from common.visualization.mask_viz import PHASE_CMAP, PHASE_TICKS, save_phase_masks
from common.visualization.prompt_viz import save_prompt_maps, save_task_expert_weights_grouped


def test_phase_visualization_uses_twilight_and_shared_moe_layout(tmp_path):
    assert PHASE_CMAP == "twilight"
    assert len(PHASE_TICKS) == 3

    class Layer:
        def get_phase_wrapped(self):
            return torch.linspace(0.0, 2.0 * torch.pi, 9 * 4 * 4).reshape(9, 4, 4)

    apertures = [SimpleNamespace(name=f"E{index // 3}{index % 3}") for index in range(9)]
    model = SimpleNamespace(expert_layers=[Layer(), Layer()], layout=SimpleNamespace(expert_apertures=apertures))
    save_phase_masks(model, tmp_path / "masks")
    output = tmp_path / "masks" / "expert_phase_layers.png"
    assert output.exists() and output.stat().st_size > 0


def test_mask_aware_prompt_maps_and_grouped_task_weights(tmp_path):
    mask = torch.zeros(20, 20, dtype=torch.float32)
    mask[4:16, 4:16] = 1.0
    phase = torch.remainder(torch.rand(20, 20) * 2.0 * torch.pi, 2.0 * torch.pi)
    amplitude = torch.rand(20, 20) * mask
    intermediates = {
        "prompt_router_amplitude": amplitude,
        "prompt_router_phase": phase * mask,
        "prompt_total_amplitude": amplitude,
        "prompt_total_phase": phase * mask,
        "prompt_aperture_mask": mask,
        "prompt_amplitudes": torch.ones(9),
        "normalized_prompt_powers": torch.full((9,), 1.0 / 9.0),
    }
    save_prompt_maps(intermediates, tmp_path / "prompt")
    phase_output = tmp_path / "prompt" / "prompt_total_phase.png"
    assert phase_output.exists() and phase_output.stat().st_size > 0
    assert (tmp_path / "prompt" / "prompt_aperture_region_on_canvas.png").exists()

    grouped_output = tmp_path / "task_expert_weights_grouped.png"
    saved = save_task_expert_weights_grouped(
        {
            "mnist": torch.full((9,), 1.0 / 9.0),
            "fashionmnist": torch.tensor([0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
            "emnist_letters": torch.tensor([0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.20]),
        },
        grouped_output,
    )
    assert saved
    assert grouped_output.exists() and grouped_output.stat().st_size > 0


def test_all_intermediate_light_fields_overview_and_training_curves(tmp_path):
    field = torch.ones(1, 12, 12, dtype=torch.complex64)
    intermediates = {
        "input_amplitude": field.real,
        "after_input_to_prompt": field,
        "after_prompt": field,
        "expert_entrance_before_aperture": field,
        "expert_entrance_after_aperture": field,
        "after_each_layer": [field * (index + 1) for index in range(3)],
        "after_layer5_to_fc": field,
        "after_global_fc": field,
        "detector_field": field,
    }
    light_dir = tmp_path / "light_fields"
    save_light_fields(intermediates, light_dir)
    assert (light_dir / "05_after_expert_layer_1.png").exists()
    assert (light_dir / "06_after_expert_layer_2.png").exists()
    assert (light_dir / "07_after_expert_layer_3.png").exists()
    assert (light_dir / "overview.png").exists()

    curve_path = tmp_path / "figures" / "training_curves.png"
    save_training_curves(
        [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "train_acc": 0.4, "val_acc": 0.35},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.9, "train_acc": 0.6, "val_acc": 0.55},
        ],
        curve_path,
    )
    assert curve_path.exists() and curve_path.stat().st_size > 0
