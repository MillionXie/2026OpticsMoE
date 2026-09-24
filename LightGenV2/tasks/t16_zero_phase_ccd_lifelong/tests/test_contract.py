import numpy as np
import torch

from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.data import rgb_luma_field
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics


def test_all_phase_parameters_begin_at_raw_zero_and_no_linear_head():
    for architecture in ("moe", "d2nn"):
        model = DirectCCDOptics(architecture)
        phases = [model.global_phase, *model.additional_phases]
        if architecture == "moe":
            phases.extend([model.router_phase, *model.first_phase])
        else:
            phases.append(model.first_phase)
        assert all(torch.count_nonzero(phase) == 0 for phase in phases)
        assert all(torch.allclose(model.transmission(phase).real,
                                  torch.full_like(phase, -1)) for phase in phases)
        assert not any(isinstance(layer, torch.nn.Linear) for layer in model.modules())
        assert len(model.output_centers) == 10
        assert len(set(model.output_centers)) == 10


def test_rgb_luma_fills_fourth_quadrant_without_training_parameters():
    source = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    source[..., 0] = 255
    source[..., 1] = 128
    source[..., 2] = 64
    field = rgb_luma_field(source)
    assert field.shape == (1, 224, 224)
    assert field[:, 112:, 112:].min() > 0
    assert torch.allclose(field.square().sum((-2, -1)), torch.ones(1), atol=1e-5)
