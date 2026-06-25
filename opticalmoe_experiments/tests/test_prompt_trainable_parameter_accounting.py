import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.expert_layout import ExpertLayout
from common.optics.global_router_prompt import GlobalRouterPrompt


def test_global_router_prompt_has_only_channel_trainable_parameters():
    layout = ExpertLayout(
        num_experts=9,
        canvas_size=1000,
        input_size=134,
        expert_size=134,
        expert_pitch=200,
        padding=200,
        prompt_aperture_size=600,
    )
    prompt = GlobalRouterPrompt(
        layout=layout,
        wavelength_m=5.32e-7,
        pixel_size_m=8.0e-6,
        prompt_to_expert_m=0.20,
        focal_length_m=0.10,
    )
    named_params = dict(prompt.named_parameters())
    assert set(named_params) == {"amplitude_logits", "phase_biases"}
    assert named_params["amplitude_logits"].numel() == 9
    assert named_params["phase_biases"].numel() == 9
    assert sum(p.numel() for p in prompt.parameters()) == 18

    named_buffers = dict(prompt.named_buffers())
    assert "lens_phase" in named_buffers
    assert "grating_phases" in named_buffers
    assert "prompt_mask" in named_buffers
    assert "lens_phase" not in named_params
    assert "grating_phases" not in named_params


def test_prompt_transmission_is_masked_to_prompt_aperture():
    layout = ExpertLayout(
        num_experts=9,
        canvas_size=1000,
        input_size=134,
        expert_size=134,
        expert_pitch=200,
        padding=200,
        prompt_aperture_size=600,
    )
    prompt = GlobalRouterPrompt(
        layout=layout,
        wavelength_m=5.32e-7,
        pixel_size_m=8.0e-6,
        prompt_to_expert_m=0.20,
        focal_length_m=0.10,
    )
    maps = prompt.prompt_maps()
    total_amp = maps["prompt_total_amplitude"]
    mask = maps["prompt_aperture_mask"].bool()
    assert maps["prompt_aperture_region"]["y0"] == 200
    assert maps["prompt_aperture_region"]["y1"] == 800
    assert torch.count_nonzero(total_amp[~mask]) == 0
    assert torch.count_nonzero(total_amp[mask]) > 0
