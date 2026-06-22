import math

import torch

from opticalmoe.optics.as_global_router_prompt import ASGlobalRouterPromptBank
from opticalmoe.optics.nine_expert_as_multitask_moe import (
    NineExpertASGlobalRouterMultitaskMoEClassifier,
)
from opticalmoe.optics.nine_expert_geometry import NineExpertFair134Layout


def test_nine_expert_fair134_geometry():
    layout = NineExpertFair134Layout()
    layout.validate()

    assert layout.canvas_shape == (1000, 1000)
    assert layout.canvas_center == (500, 500)
    assert layout.padding == 200
    assert layout.prompt_aperture.y0 == 200
    assert layout.prompt_aperture.y1 == 800
    assert layout.prompt_aperture.x0 == 200
    assert layout.prompt_aperture.x1 == 800
    assert layout.expert_coords == [300, 500, 700]
    assert layout.gap_px == 66

    masks = layout.expert_masks()
    assert masks.shape == (9, 1000, 1000)
    assert torch.all(masks.sum(dim=0) <= 1.0)
    assert int(masks.sum().item()) == 9 * 134 * 134


def test_as_global_router_prompt_channel_table():
    layout = NineExpertFair134Layout()
    prompt = ASGlobalRouterPromptBank(
        task_names=["mnist", "fashionmnist"],
        layout=layout,
        wavelength_m=532e-9,
        pixel_size_m=8e-6,
        prompt_to_expert_m=0.20,
        focal_length_m=0.10,
    )

    transmission = prompt.transmission("mnist")
    amplitudes = prompt.amplitudes("mnist")
    table = prompt.channel_table()

    assert transmission.shape == (1000, 1000)
    assert amplitudes.shape == (9,)
    assert prompt.router("mnist").abs().max().item() <= 1.0 + 1e-5
    assert sorted({row["dx_px"] for row in table}) == [-200.0, 0.0, 200.0]
    assert sorted({row["dy_px"] for row in table}) == [-200.0, 0.0, 200.0]

    finite_periods = [
        row["grating_period_x_px"]
        for row in table
        if math.isfinite(row["grating_period_x_px"])
    ]
    assert finite_periods
    assert abs(finite_periods[0] - 8.3125) < 0.05


def test_nine_expert_model_forward_and_intermediates():
    model = NineExpertASGlobalRouterMultitaskMoEClassifier(
        task_names=["mnist", "emnist"],
        task_num_classes={"mnist": 10, "emnist": 26},
        num_layers=1,
        expert_phase_init="identity",
        global_fc_phase_init="identity",
        task_head_configs={
            "mnist": {"readout_type": "mlp", "hidden_dim": 8},
            "emnist": {"readout_type": "mlp", "hidden_dim": 8},
        },
    )
    logits, intermediates = model(
        torch.rand(1, 1, 134, 134),
        task_name="mnist",
        return_intermediates=True,
    )

    assert logits.shape == (1, 10)
    assert intermediates["expert_energy_ratios"].shape == (1, 9)
    assert intermediates["prompt_amplitudes"].shape == (9,)
    assert intermediates["prompt_router_amplitude"].shape == (1000, 1000)
    assert intermediates["prompt_total_phase"].shape == (1000, 1000)
    assert "expert_entrance_after_aperture" in intermediates
