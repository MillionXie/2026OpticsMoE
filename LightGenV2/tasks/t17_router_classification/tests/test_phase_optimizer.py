import copy
import pytest
import torch
from LightGenV2.tasks.t17_router_classification.phase_optimizer import WrappedPhaseAdam
from LightGenV2.tasks.t17_router_classification.model_four import FourRouterClassification
from LightGenV2.tasks.t17_router_classification.train_boundary_development import configure_phase_optimizer


def test_phase_adam_matches_direct_angular_adam_across_periodic_seam():
    raw = torch.nn.Parameter(torch.tensor([-12., 0., 12.], dtype=torch.float64))
    angle = torch.nn.Parameter(2 * torch.pi * raw.detach().sigmoid())
    wrapped = WrappedPhaseAdam([raw], .02)
    reference = torch.optim.Adam([angle], .02)
    for _ in range(5):
        wrapped.zero_grad(); reference.zero_grad()
        (2 * torch.pi * raw.sigmoid()).sin().sum().backward()
        angle.sin().sum().backward()
        wrapped.step(); reference.step()
        with torch.no_grad():angle.remainder_(2 * torch.pi)
        assert torch.allclose(torch.exp(2j * torch.pi * raw.sigmoid()), torch.exp(1j * angle), atol=1e-7)
    assert torch.isfinite(raw).all()


def test_nonfinite_gradient_restores_canonical_raw():
    raw = torch.nn.Parameter(torch.tensor([0., 2.]))
    before = raw.detach().clone(); raw.grad = torch.tensor([1., float('nan')])
    with pytest.raises(RuntimeError):WrappedPhaseAdam([raw], .01).step()
    assert torch.equal(raw, before)


def test_optical_step_keeps_canonical_model_and_frozen_router():
    model = FourRouterClassification('optical', 'ccd_grid', num_classes=9)
    before = copy.deepcopy(model.state_dict())
    optimizer, parameters = configure_phase_optimizer(model, .003, dict(phase_optimizer='wrapped_phase_adam'))
    result = model(torch.rand(1, 224, 224))
    torch.nn.functional.nll_loss(result['logits'], torch.tensor([2])).backward()
    optimizer.step()
    assert model.router_phase.grad is None and torch.equal(model.router_phase, before['router_phase'])
    assert sum(p.numel() for p in parameters) == 429188
    assert set(model.state_dict()) == set(before)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert not torch.equal(before['global_phase'], model.global_phase)
    clone = FourRouterClassification('optical', 'ccd_grid', num_classes=9)
    clone.load_state_dict(model.state_dict()); x = torch.rand(1, 224, 224)
    with torch.no_grad():assert torch.equal(model(x)['probabilities'], clone(x)['probabilities'])
