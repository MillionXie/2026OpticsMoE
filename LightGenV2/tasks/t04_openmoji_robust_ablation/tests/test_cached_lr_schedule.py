import pytest

from LightGenV2.tasks.t04_openmoji_robust_ablation.lab_cached_decoder_regularize import epoch_learning_rate


def test_constant_default_and_single_epoch():
    assert all(epoch_learning_rate(e, 120) == 1e-5 for e in range(1, 121))
    assert epoch_learning_rate(1, 1, True) == 1e-5


def test_cosine_endpoints_monotonic_and_bounded():
    values = [epoch_learning_rate(e, 120, True) for e in range(1, 121)]
    assert values[0] == pytest.approx(1e-5)
    assert values[-1] == pytest.approx(1e-6)
    assert all(1e-6 <= x <= 1e-5 for x in values)
    assert all(x >= y for x, y in zip(values, values[1:]))


@pytest.mark.parametrize('epoch,epochs', [(0, 120), (121, 120), (1, 0)])
def test_invalid_budget(epoch, epochs):
    with pytest.raises(ValueError):
        epoch_learning_rate(epoch, epochs, True)
