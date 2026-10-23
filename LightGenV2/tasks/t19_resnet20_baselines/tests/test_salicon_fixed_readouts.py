import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines.salicon_fixed_readouts import cc_per_image, energy_density
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import correlation_coefficient


def test_fixed_readout_density_and_cc_match_original_metric_per_image():
    torch.manual_seed(42)
    energies = torch.rand(3, 1, 7, 7)
    before = energies.clone()
    predicted = energy_density(energies)
    assert predicted.shape == (3, 1, 224, 224)
    torch.testing.assert_close(predicted.sum((1, 2, 3)), torch.ones(3))
    target = torch.rand_like(predicted)
    actual = cc_per_image(predicted, target)
    expected = torch.stack([correlation_coefficient(p[None], y[None]) for p, y in zip(predicted, target)])
    # FP32 reductions can round differently for a batch versus singleton calls.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    assert torch.equal(energies, before)
    with pytest.raises(ValueError, match='nonnegative'):
        energy_density(-energies)


def test_zero_energy_is_finite_uniform_and_has_zero_cc_with_nonconstant_target():
    predicted = energy_density(torch.zeros(1, 1, 7, 7))
    assert torch.isfinite(predicted).all() and predicted.min() > 0
    target = torch.arange(224 * 224).reshape(1, 1, 224, 224).float()
    assert abs(float(cc_per_image(predicted, target))) < 1e-6
