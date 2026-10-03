from pathlib import Path

import numpy as np
import pytest
import torch

from LightGenV2.tasks.t03_saliency.reproduce_baseline import independent_cc
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import MeanOnlyCCDNormalizer, architecture_label

TASK = Path(__file__).resolve().parents[1]


def test_independent_cc_is_per_image_not_global():
    x = np.arange(12).reshape(2, 2, 3)
    y = np.stack([x[0] * 7 + 13, -x[1] + 100])
    np.testing.assert_allclose(independent_cc(x, y), [1, -1], atol=1e-12)
    np.testing.assert_allclose(independent_cc(np.zeros_like(x), y), [0, 0])


def test_mean_only_keeps_intensity_ratios_without_log_or_clipping():
    normalizer = MeanOnlyCCDNormalizer(5)
    x = torch.zeros(1, 5, 5)
    x[0, 0, :2] = torch.tensor([1., 100.])
    y = normalizer(x)
    assert y[0, 0, 1] / y[0, 0, 0] == pytest.approx(100.)
    assert y.max() > 12
    assert y.mean() == pytest.approx(1.)
    assert torch.equal(normalizer(torch.zeros_like(x)), torch.zeros_like(x))
    s = load_settings(TASK / "configs/moe_dc20_mean_only_continue.yaml")
    assert architecture_label(s).endswith("_mean_only")


@pytest.mark.parametrize("name", ["moe_dc20_no_shift_continue.yaml", "moe_dc20_cc_continue.yaml"])
def test_continuation_keeps_router_and_dc(name):
    s = load_settings(TASK / "configs" / name)
    assert s.router_backend == "optical" and s.top_k == 2
    assert s.student_epochs == 100
    assert s.initialization_checkpoint.name == "best_checkpoint.pt"
    assert s.language_optical_zero_order_enabled
    assert s.language_optical_amplitude_zero_order_intensity_min == pytest.approx(0.2)
    assert s.language_optical_max_shift_pixels == 0
    assert s.language_optical_phase_shift_pixels == 0
    assert s.language_optical_ccd_shift_pixels == 0
    assert s.optical_router_input_shift_pixels == 0
    assert s.optical_router_phase_shift_pixels == 0
    assert s.optical_router_ccd_shift_pixels == 0
    assert s.router_hard_load_balance_weight > 0
