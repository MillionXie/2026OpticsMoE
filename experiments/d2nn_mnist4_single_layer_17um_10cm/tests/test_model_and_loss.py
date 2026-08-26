from pathlib import Path

import torch
from torch.nn import functional as F

from ..modeling import SingleLayerMNIST4D2NN
from ..settings import load_settings


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "mnist4_single_layer_17um_10cm_notebook_mse.yaml"
)


def test_release_geometry_and_notebook_loss_contract() -> None:
    settings = load_settings(CONFIG)
    assert settings.detector_distance_m == 0.10
    assert settings.logical_pixel_pitch_um == 17.0
    assert settings.canvas_size == 518
    assert settings.propagation_grid_size == 1024
    assert settings.active_size == 478
    assert settings.phase_learning_rate == 0.01
    assert settings.template_mse_loss_weight == 1.0
    assert settings.detector_ce_loss_weight == 0.0
    assert settings.amplitude_invert_before_export is False


def test_raw_zero_is_uniform_pi_and_reference_mse_is_exact() -> None:
    settings = load_settings(CONFIG)
    model = SingleLayerMNIST4D2NN(settings)
    torch.testing.assert_close(
        model.phase(),
        torch.full_like(model.raw_phase, torch.pi),
        atol=1.0e-6,
        rtol=0.0,
    )
    output = model(torch.rand(1, 1, 28, 28))
    targets = torch.tensor([2])
    total, template_mse, detector_ce = model.optical_routing_loss(output, targets)
    target_active = model.detector_masks[targets]
    guard = settings.canvas_guard
    target_canvas = F.pad(target_active, (guard, guard, guard, guard))
    expected = 100.0 * F.mse_loss(
        output["detector_intensity_canvas"], target_canvas
    )
    torch.testing.assert_close(template_mse, expected)
    torch.testing.assert_close(total, expected)
    assert torch.isfinite(detector_ce)
    total.backward()
    assert model.raw_phase.grad is not None
    assert torch.isfinite(model.raw_phase.grad).all()
    assert torch.any(model.raw_phase.grad != 0)
