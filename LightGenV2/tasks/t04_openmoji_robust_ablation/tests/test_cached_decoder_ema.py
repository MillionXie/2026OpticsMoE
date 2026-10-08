import pytest
import torch

from LightGenV2.tasks.t04_openmoji_robust_ablation.lab_cached_decoder_regularize import update_ema, evaluation_weights


def test_ema_train_update_detached_and_integer_buffer():
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.register_buffer('count', torch.tensor(1))
    with torch.no_grad():
        layer.weight.fill_(2.)
    shadow = {k: v.detach().clone() for k, v in layer.state_dict().items()}
    with torch.no_grad():
        layer.weight.fill_(4.)
        layer.count.fill_(3)
    update_ema(shadow, layer, .99)
    assert torch.allclose(shadow['weight'], torch.full_like(layer.weight, 2.02))
    assert shadow['count'].item() == 3
    assert layer.weight.grad is None and not shadow['weight'].requires_grad
    assert torch.all(layer.weight == 4.)


def test_evaluation_swap_restores_live_weights_on_failure():
    layer = torch.nn.Linear(2, 1)
    original = {k: v.detach().clone() for k, v in layer.state_dict().items()}
    shadow = {k: torch.zeros_like(v) for k, v in original.items()}
    with pytest.raises(RuntimeError):
        with evaluation_weights(layer, shadow):
            assert torch.all(layer.weight == 0)
            raise RuntimeError('evaluation failed')
    assert all(torch.equal(v, original[k]) for k, v in layer.state_dict().items())


def test_disabled_evaluation_does_not_replace_parameters():
    layer = torch.nn.Linear(2, 1)
    identity = id(layer.weight)
    with evaluation_weights(layer, None):
        assert id(layer.weight) == identity
