import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config.layout_config import layout_from_config
from common.optics.expert_layout import ExpertLayout
from common.optics.global_router_prompt import GlobalRouterPrompt


def test_fast120_520_default_geometry_is_exact():
    layout = ExpertLayout()
    assert layout.geometry_profile == "fast120_520"
    assert layout.canvas_size == 520
    assert layout.input_size == 120
    assert layout.expert_size == 120
    assert layout.expert_pitch == 150
    assert layout.padding == 35
    assert layout.prompt_aperture_size == 450
    assert layout.expert_coords == [110, 260, 410]
    assert layout.expert_centers == [(y, x) for y in (110, 260, 410) for x in (110, 260, 410)]
    assert [(ap.y0, ap.y1, ap.x0, ap.x1) for ap in layout.expert_apertures] == [
        (y0, y0 + 120, x0, x0 + 120)
        for y0 in (50, 200, 350)
        for x0 in (50, 200, 350)
    ]
    assert layout.expert_union_bounds == [50, 470, 50, 470]
    assert layout.expert_union_size == 420
    assert layout.gap_px == 30
    assert [layout.active_window_aperture.y0, layout.active_window_aperture.y1, layout.active_window_aperture.x0, layout.active_window_aperture.x1] == [35, 485, 35, 485]
    assert [layout.prompt_aperture.y0, layout.prompt_aperture.y1, layout.prompt_aperture.x0, layout.prompt_aperture.x1] == [35, 485, 35, 485]


def test_explicit_legacy_geometry_resolves_without_fast_override():
    config = {
        "layout": {
            "canvas_height": 1000,
            "canvas_width": 1000,
            "input_size": 134,
            "expert_size": 134,
            "expert_pitch": 200,
            "padding": 200,
            "prompt_aperture_size": 600,
        }
    }
    layout = layout_from_config(config)
    assert layout.geometry_profile == "fair134_1000"
    assert layout.canvas_size == 1000
    assert layout.prompt_aperture_size == 600


def test_fast_prompt_is_zero_outside_active_aperture():
    layout = ExpertLayout()
    prompt = GlobalRouterPrompt(
        layout=layout,
        wavelength_m=5.32e-7,
        pixel_size_m=8.0e-6,
        prompt_to_expert_m=0.20,
        focal_length_m=0.10,
    )
    outside = ~layout.prompt_aperture_mask().bool()
    transmission = prompt.transmission()
    maps = prompt.prompt_maps()
    assert torch.count_nonzero(transmission[outside]) == 0
    assert torch.count_nonzero(maps["prompt_total_amplitude"][outside]) == 0
    field = torch.ones(1, 520, 520, dtype=torch.complex64)
    assert torch.count_nonzero(prompt(field)[0][outside]) == 0
    assert maps["prompt_aperture_bounds"] == [35, 485, 35, 485]
