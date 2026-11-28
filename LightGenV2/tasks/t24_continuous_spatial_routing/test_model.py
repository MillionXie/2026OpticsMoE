import torch

from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG
from LightGenV2.tasks.t24_continuous_spatial_routing.model import SpatialRoutingLG


def test_positions_are_subpixel_and_not_confined_to_quadrants():
    model = SpatialRoutingLG("continuous")
    scores = torch.ones(1, 17, 17)
    # Both winners in upper left quarter, further apart than one detector width.
    scores[0, 2, 2] = 30
    scores[0, 2, 3] = 7
    scores[0, 2, 7] = 20
    scores.requires_grad_()
    positions, selected = model.continuous_coordinates(scores)
    assert selected.tolist() == [[30, 20]]
    assert ((positions > 0) & (positions < 127)).all()
    assert (positions[0, 1, 1] - positions[0, 0, 1]) > 60
    assert (positions - positions.round()).abs().max() > 0.05
    positions.square().sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_router_integrals_at_corners_and_interior():
    model = SpatialRoutingLG("continuous")
    ccd = torch.rand(2, 518, 518, dtype=torch.float64)
    scores = model.window_scores(ccd)
    for i, j in ((0, 0), (0, 16), (16, 0), (16, 16), (6, 9)):
        top, left = 102 + int(model.axis[i]), 102 + int(model.axis[j])
        expected = ccd[:, top:top + 60, left:left + 60].sum((-2, -1))
        torch.testing.assert_close(scores[:, i, j], expected)


def test_overlapping_copies_and_subpixel_translation_gradient():
    model = SpatialRoutingLG("continuous")
    amplitude = torch.zeros(1, 224, 224)
    amplitude[:, 0, 0] = 1
    positions = torch.tensor([[[112.25, 120.0], [112.25, 120.0]]], requires_grad=True)
    plane = model.multiplex(amplitude, positions, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(plane.square().sum(), torch.tensor(1.0))
    torch.testing.assert_close(plane[0, 112, 120] / plane[0, 113, 120], torch.tensor(3.0))
    plane[0, 113, 120].backward()
    assert torch.isfinite(positions.grad).all()
    assert (positions.grad[..., 0] > 0).all()


def test_fixed_mode_reproduces_t17_with_same_phases():
    model = SpatialRoutingLG("fixed")
    baseline = LinearReadoutLG(main_layers=2, num_classes=8)
    with torch.no_grad():
        model.first_phase.normal_(0, 0.1)
        model.global_phase.normal_(0, 0.1)
        model.router_phase.normal_(0, 0.1)
        baseline.global_phase.copy_(model.global_phase)
        baseline.router_phase.copy_(model.router_phase)
        baseline.shared_head.weight.copy_(model.shared_head.weight)
        for i, (y, x) in enumerate(model.corners.tolist()):
            baseline.first_phase[i].copy_(model.first_phase[y:y + 224, x:x + 224])
        amplitude = torch.rand(2, 224, 224)
        actual = model(amplitude)
        reference = baseline(amplitude)
    expected_power, indices = reference["route_power"].topk(2, dim=1)
    torch.testing.assert_close(actual["positions"], model.corners[indices].float())
    torch.testing.assert_close(actual["power"], expected_power)
    torch.testing.assert_close(actual["logits"], reference["raw_logits"], atol=2e-5, rtol=5e-5)


def test_real_classifier_backpropagates_to_router_and_both_full_planes():
    model = SpatialRoutingLG("continuous")
    assert set(model.state_dict()) == {"first_phase", "global_phase", "router_phase", "shared_head.weight"}
    result = model(torch.rand(2, 224, 224))
    assert result["positions"].shape == (2, 2, 2)
    assert ((result["positions"] >= 0) & (result["positions"] <= 254)).all()
    torch.testing.assert_close(result["power"].sum(1), torch.ones(2))
    assert (result["router_capture"] > 0).all()
    result["logits"].sum().backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
