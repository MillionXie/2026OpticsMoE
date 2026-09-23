import inspect

import torch

from LightGenV2.tasks.t14_shared_readout_lifelong.model import SharedReadoutOptics


def test_shared_readout_is_identical_and_frozen_across_architectures():
    moe = SharedReadoutOptics("moe", max_experts=4)
    d2nn = SharedReadoutOptics("d2nn", max_experts=4)
    assert torch.equal(moe.shared_head.weight, d2nn.shared_head.weight)
    assert tuple(moe.shared_head.weight.shape) == (10, 784)
    assert not moe.shared_head.weight.requires_grad
    assert not d2nn.shared_head.weight.requires_grad
    assert not moe.heads and not d2nn.heads
    assert "task" not in inspect.signature(moe.forward).parameters


def test_fixed_gain_only_scales_the_same_frozen_linear_rows():
    unit = SharedReadoutOptics("moe", max_experts=4, readout_gain=1.0)
    amplified = SharedReadoutOptics("moe", max_experts=4, readout_gain=8.0)
    assert torch.allclose(amplified.shared_head.weight,
                          8 * unit.shared_head.weight)
    assert not amplified.shared_head.weight.requires_grad


def test_fixed_window_readout_is_identical_and_frozen_for_both_optics():
    moe = SharedReadoutOptics("moe", max_experts=4,
                              readout_gain=8, readout_design="windows")
    d2nn = SharedReadoutOptics("d2nn", max_experts=4,
                               readout_gain=8, readout_design="windows")
    assert torch.equal(moe.shared_head.weight, d2nn.shared_head.weight)
    assert not moe.shared_head.weight.requires_grad
    assert torch.count_nonzero(moe.shared_head.weight[0] -
                               moe.shared_head.weight[1]) > 0


def test_new_stage_freezes_old_experts_but_not_router_or_global():
    model = SharedReadoutOptics("moe", max_experts=16)
    model.configure_stage(1, warmup=True)
    assert all(not p.requires_grad for p in model.first_phase[:4])
    assert all(p.requires_grad for p in model.first_phase[4:8])
    assert not model.router_phase.requires_grad
    assert not model.global_phase.requires_grad
    model.configure_stage(1, warmup=False)
    assert all(not p.requires_grad for p in model.first_phase[:4])
    assert model.router_phase.requires_grad
    assert model.global_phase.requires_grad
    assert not model.shared_head.weight.requires_grad


def test_optical_forward_produces_ten_logits_and_optical_gradients():
    torch.set_num_threads(2)
    model = SharedReadoutOptics("moe", max_experts=4)
    model.configure_stage(0)
    output = model(torch.rand(1, 224, 224))
    assert output["logits"].shape == (1, 10)
    output["logits"][0, 0].backward()
    assert model.first_phase[0].grad is not None
    assert model.shared_head.weight.grad is None
