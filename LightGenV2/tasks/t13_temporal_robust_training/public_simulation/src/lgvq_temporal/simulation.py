"""Small CPU-only graph check independent of external datasets."""

from dataclasses import replace

import torch

from .model import build_model
from .settings import MultiVideoGeometry


def synthetic_smoke(settings) -> dict:
    grid = settings.geometry.video_grid
    active_size = 2 + 28 + (grid - 1) * 29
    geometry = MultiVideoGeometry(
        canvas_size=active_size + 4,
        active_size=active_size,
        video_grid=grid,
        video_count=grid**2,
        video_tile_size=28,
        video_tile_pitch=29,
        video_tile_offset=1,
        frame_grid=2,
        frames_per_video=4,
        frame_lane_size=14,
        frame_lane_pitch=14,
        frame_expert_size=6,
        frame_expert_pitch=8,
        video_field_size=12,
        video_expert_pitch=16,
        video_phase_tile_size=26,
    )
    small = replace(
        settings,
        geometry=geometry,
        vision_input_width=32,
        quality_input_width=6,
        language_input_width=40,
        model_width=24,
        detector_projection_size=8,
        head_width=32,
        temporal_readout_hidden_width=64,
        temporal_readout_rank=16,
        maximum_language_tokens=12,
        frame_router_intervals=((2, 6), (8, 12)),
        video_router_intervals=((4, 10), (18, 24)),
        input_shift_pixels=0,
        phase_shift_pixels=0,
        ccd_shift_pixels=0,
        phase_dropout_cell_size=2,
        k_space_enabled=False,
        initialization_checkpoint=None,
        batch_size=1,
        num_workers=0,
        synthetic=True,
        device="cpu",
    )
    small.validate()
    model = build_model(small).train()
    vision = torch.randn(1, grid**2, 4, 49, 32)
    quality = torch.randn(1, grid**2, 4, 49, 6)
    language = torch.randn(1, 8, 40)
    mask = torch.ones(1, 8, dtype=torch.bool)
    result = model(vision, quality, language, mask, optical_enabled=True)
    loss = result["normalized_prediction"].square().mean() + result["router_balance_loss"]
    loss.backward()
    gradients = {
        name: parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.named_parameters()
        if "phase" in name
    }
    if result["prediction"].shape != (1, grid**2) or not all(gradients.values()):
        raise RuntimeError("Full-field simulation graph check failed")
    return {"status": "passed", "prediction_shape": list(result["prediction"].shape),
            "all_phase_gradients_finite": True}
