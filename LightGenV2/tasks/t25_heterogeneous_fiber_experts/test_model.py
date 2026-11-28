import pytest
import torch

from LightGenV2.tasks.t24_continuous_spatial_routing.model import SpatialRoutingLG
from LightGenV2.tasks.t25_heterogeneous_fiber_experts.model import (
    CELL_SIDE, HeterogeneousFiberLG, SLOT_SIDE, SLOTS_YX, lp01_mode, project_fiber_cells,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_independent_normalized_core_modes_and_orthogonal_rejection():
    u = lp01_mode(torch.float64)
    torch.testing.assert_close(u.square().sum(), torch.tensor(1., dtype=torch.float64))
    # Four arbitrary complex coefficients, one per cell, including a dark core.
    c = torch.tensor([[1+2j, -3j], [0j, -1+.2j]], dtype=torch.complex128)
    matched = (c[:, None, :, None] * u[None, :, None, :]).reshape(16, 16)
    torch.testing.assert_close(project_fiber_cells(matched, u), matched, atol=1e-14, rtol=1e-14)
    odd = u * (torch.arange(CELL_SIDE, dtype=torch.float64) - 3.5)
    rejected = (c[:, None, :, None] * odd[None, :, None, :]).reshape(16, 16)
    torch.testing.assert_close(project_fiber_cells(rejected, u), torch.zeros_like(rejected), atol=1e-14, rtol=0)


def test_complex_linearity_idempotence_and_passivity():
    gen = torch.Generator().manual_seed(91)
    u = lp01_mode(torch.float64)
    a, b = torch.randn(2, 5, 16, 24, generator=gen, dtype=torch.complex128)
    phase = torch.exp(torch.tensor(.73j, dtype=torch.complex128))
    pa, pb = project_fiber_cells(a, u), project_fiber_cells(b, u)
    torch.testing.assert_close(project_fiber_cells(phase * a + (.2-1.1j) * b, u), phase * pa + (.2-1.1j) * pb)
    torch.testing.assert_close(project_fiber_cells(pa, u), pa)
    power = lambda f: f.abs().square().sum((-2, -1))
    torch.testing.assert_close(power(a), power(pa) + power(a-pa))


@pytest.mark.parametrize("slots", [(0, 3), (1, 2)])
def test_selective_projection_preserves_other_slots_and_cross_gap(slots):
    model = HeterogeneousFiberLG("mixed", fiber_slots=slots)
    field = torch.randn(1, 478, 478, dtype=torch.complex64)
    result = model.filter_experts(field)
    for s, (y, x) in enumerate(SLOTS_YX):
        index = (slice(None), slice(y, y+SLOT_SIDE), slice(x, x+SLOT_SIDE))
        if s in slots:
            torch.testing.assert_close(result[index], project_fiber_cells(field[index], model.core_mode))
            assert result[index].abs().square().sum() < field[index].abs().square().sum()
        else:
            assert torch.equal(result[index], field[index])
    assert torch.equal(result[:, 224:254], field[:, 224:254])
    assert torch.equal(result[:, :, 224:254], field[:, :, 224:254])


def test_diffraction_matches_archived_forward_and_backward_bitwise():
    torch.manual_seed(37)
    parent = SpatialRoutingLG("fixed").eval()
    torch.manual_seed(37)
    model = HeterogeneousFiberLG("diffraction").eval()
    assert set(model.state_dict()) == set(parent.state_dict()) == {
        "first_phase", "global_phase", "router_phase", "shared_head.weight"}
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, parent.state_dict()[name])
    fields = torch.rand(2, 224, 224)
    reference = parent(fields)
    output = model(fields, diagnostics=True)
    ordinary = model(fields)
    for name in reference:
        assert torch.equal(reference[name], output[name]) and torch.equal(reference[name], ordinary[name])
    assert torch.equal(output["transmitted_power"], output["pre_fiber_power"])
    output["logits"].square().mean().backward()
    reference["logits"].square().mean().backward()
    for name, param in model.named_parameters():
        assert torch.equal(param.grad, dict(parent.named_parameters())[name].grad)


@pytest.mark.parametrize("arm,slots", [("mixed", (0, 3)), ("fiber", (0, 1, 2, 3)),
                                      ("uniform_half", (0, 1, 2, 3))])
def test_optical_top2_and_classification_gradients(arm, slots):
    torch.manual_seed(53)
    model = HeterogeneousFiberLG(arm, fiber_slots=slots)
    fields = torch.rand(3, 224, 224)
    out = model(fields, diagnostics=True)
    indices = out["selected_slots"]
    assert torch.equal(indices, out["dense_power"].topk(2, dim=1).indices)
    assert torch.equal(out["positions"], model.corners[indices].float())
    assert torch.all(out["slot_gate_power"].gt(0).sum(1) == 2)
    torch.testing.assert_close(out["router_port_energy"].gather(1, indices).sum(1), out["router_capture"])
    torch.testing.assert_close(out["slot_gate_power"].sum(1), torch.ones(len(fields)))
    torch.testing.assert_close(out["pre_fiber_power"], out["slot_gate_power"], atol=2e-5, rtol=2e-5)
    assert torch.all(out["transmitted_power"] <= out["pre_fiber_power"] + 2e-6)
    with torch.no_grad():
        ordinary = model(fields)
    for name in ordinary:
        assert torch.equal(ordinary[name], out[name])
    out["logits"].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0, name


def test_arms_require_explicit_supported_mixed_slot_assignment():
    with pytest.raises(ValueError):
        HeterogeneousFiberLG("mixed")
    with pytest.raises(ValueError):
        HeterogeneousFiberLG("mixed", fiber_slots=(0, 1))
    with pytest.raises(ValueError):
        HeterogeneousFiberLG("diffraction", fiber_slots=(0, 3))
    with pytest.raises(ValueError):
        HeterogeneousFiberLG("uniform_half", fiber_slots=(0, 3))


@pytest.mark.parametrize("seed,mixed_slots", [(37, (0, 3)), (53, (1, 2))])
def test_uniform_half_matches_total_cores_and_preserves_unwired_cells(seed, mixed_slots):
    torch.manual_seed(seed)
    uniform = HeterogeneousFiberLG("uniform_half")
    torch.manual_seed(seed)
    mixed = HeterogeneousFiberLG("mixed", fiber_slots=mixed_slots)
    for name, value in uniform.state_dict().items():
        assert torch.equal(value, mixed.state_dict()[name])
    assert uniform.fiber_slots == (0, 1, 2, 3)
    assert uniform.fiber_core_counts == (392,) * 4
    assert mixed.fiber_core_counts == tuple(784 if i in mixed_slots else 0 for i in range(4))
    assert sum(uniform.fiber_core_counts) == sum(mixed.fiber_core_counts) == 1568
    grid = torch.arange(28)
    check = (grid[:, None] + grid[None, :]).remainder(2) == 0
    assert torch.equal(uniform.fiber_cell_mask, check.expand(4, 28, 28))
    pixel_mask = check.repeat_interleave(CELL_SIDE, dim=0).repeat_interleave(CELL_SIDE, dim=1)
    original = torch.randn(2, 478, 478, dtype=torch.complex64)
    relayed = uniform.filter_experts(original)
    for y, x in SLOTS_YX:
        a = original[:, y:y+SLOT_SIDE, x:x+SLOT_SIDE]
        b = relayed[:, y:y+SLOT_SIDE, x:x+SLOT_SIDE]
        assert torch.equal(b[:, ~pixel_mask], a[:, ~pixel_mask])
        torch.testing.assert_close(b[:, pixel_mask], project_fiber_cells(a, uniform.core_mode)[:, pixel_mask])
    assert torch.all(uniform.slot_power(relayed) <= uniform.slot_power(original))
