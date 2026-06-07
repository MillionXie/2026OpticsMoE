import torch

from opticalmoe.optics.four_expert_moe_v2 import FourExpertMoEClassifierV2
from opticalmoe.training.progressive_schedule import ProgressiveUnfreezingSchedule


def _model(num_layers=2):
    return FourExpertMoEClassifierV2(
        num_classes=10,
        num_layers=num_layers,
        detector_size=24,
        expert_phase_init="identity",
        global_fc_phase_init="identity",
    )


def test_four_expert_moe_v2_forward_and_diagnostics():
    model = _model(num_layers=1)
    images = torch.rand(1, 1, 32, 32)
    logits, intermediates = model(images, return_intermediates=True)

    assert logits.shape == (1, 10)
    assert model.layout.prompt_cell_size == 300
    assert model.layout.expert_size == 200
    assert intermediates["prompt_amplitudes"].shape == (4,)
    assert intermediates["expert_energy"].shape == (1, 4)
    assert intermediates["detector_energies"].shape == (1, 10)
    assert "expert_entrance_after_aperture" in intermediates
    assert torch.all(intermediates["outside_energy_ratio"] >= 0.0)
    assert len(intermediates["after_each_layer"]) == 1


def test_prompt_amplitude_logits_are_trainable():
    model = _model(num_layers=1)
    logits = model(torch.rand(1, 1, 32, 32))
    logits.sum().backward()

    assert model.prompt.amplitude_logits.requires_grad
    assert model.prompt.amplitude_logits.grad is not None
    assert torch.all(model.prompt.amplitudes() > 0.5)


def test_progressive_backward_unfreezing():
    model = _model(num_layers=5)
    schedule = ProgressiveUnfreezingSchedule(
        num_layers=5,
        enabled=True,
        order="backward",
        stage_epochs=[1, 1, 1, 1, 1, 1],
    )

    stage0 = schedule.apply(model, 0)
    assert stage0["active_layers"] == []
    assert model.prompt.amplitude_logits.requires_grad
    assert not model.prompt.phase_biases.requires_grad
    assert not any(
        parameter.requires_grad for parameter in model.expert_layers[0].parameters()
    )

    stage2 = schedule.apply(model, 2)
    assert stage2["active_layers"] == [4, 5]
    assert model.prompt.phase_biases.requires_grad
    assert all(
        parameter.requires_grad for parameter in model.expert_layers[4].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.expert_layers[0].parameters()
    )


def test_progressive_can_be_disabled():
    model = _model(num_layers=2)
    schedule = ProgressiveUnfreezingSchedule(
        num_layers=2,
        enabled=False,
        stage_epochs=[1],
    )
    result = schedule.apply(model, 0)
    assert result["active_layers"] == [1, 2]
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_progressive_forward_unfreezing():
    model = _model(num_layers=3)
    schedule = ProgressiveUnfreezingSchedule(
        num_layers=3,
        enabled=True,
        order="forward",
        stage_epochs=[1, 1, 1, 1],
    )
    result = schedule.apply(model, 2)
    assert result["active_layers"] == [1, 2]
    assert all(
        parameter.requires_grad for parameter in model.expert_layers[0].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.expert_layers[2].parameters()
    )
