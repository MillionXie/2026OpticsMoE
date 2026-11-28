import torch

from LightGenV2.tasks.t24_continuous_spatial_routing.constrained_routing import FootprintRoutingLG
from LightGenV2.tasks.t24_continuous_spatial_routing.model import SpatialRoutingLG


def overlap(positions):
    displacement = (positions[:, 0] - positions[:, 1]).abs()
    return (224 - displacement).clamp_min(0).prod(-1) / 224**2


def broad_peaks(size, peaks):
    u = torch.arange(size)
    y, x = torch.meshgrid(u, u, indexing="ij")
    return sum(height * torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
               for cy, cx, sigma, height in peaks)


def test_geometric_bound_for_broad_flat_edge_and_dark_inputs():
    model = FootprintRoutingLG()
    broad = broad_peaks(17, [(8, 8, 4, 7), (7, 9, 3, 4)])
    border = broad_peaks(17, [(0, 1, 0.6, 2), (16, 16, 1, 3)])
    scores = torch.stack((broad, torch.ones_like(broad), border, torch.zeros_like(broad)))
    positions, energy = model.continuous_coordinates(scores)
    assert torch.isfinite(energy).all() and torch.isfinite(positions).all()
    assert (positions >= 0).all() and (positions <= 254).all()
    assert (overlap(positions) <= 0.25).all()
    assert ((positions[2] == 0) | (positions[2] == 254)).any()


def test_pair_maximizes_smoothed_joint_readings_under_actual_footprint_bound():
    model = FootprintRoutingLG()
    # Two nearby 60px CCD windows can be distinct even while two 224px input
    # copies overlap heavily. The old peak separation allows both bright spots.
    scores = broad_peaks(17, [(5.1, 5.2, 0.7, 3), (5.1, 9.7, 0.8, 2.7),
                               (14.3, 12.6, 0.8, 2.2)])[None]
    old_positions, _ = SpatialRoutingLG("continuous").continuous_coordinates(scores)
    assert overlap(old_positions).item() > 0.25
    candidates, ranking = model._candidate_positions(scores)
    pair = model._select_pair(candidates, ranking)[0]
    i, j = torch.triu_indices(289, 289, offset=1)
    all_pairs = torch.stack((candidates[0, i], candidates[0, j]), 1)
    products = ranking[0, i] * ranking[0, j]
    expected = products[overlap(all_pairs) <= 0.25].max()
    torch.testing.assert_close(ranking[0, pair].prod(), expected)
    positions, measured = model.continuous_coordinates(scores)
    assert (positions > 0).all() and (positions < 254).all()
    assert (positions - positions.round()).abs().max() > 0.05
    assert overlap(positions).item() <= 0.25
    torch.testing.assert_close(measured, scores.flatten(1)[:, pair])


def test_small_ccd_perturbation_on_separated_peaks_preserves_locations():
    model = FootprintRoutingLG()
    ccd = broad_peaks(518, [(181.1, 184.4, 19, 1), (350.8, 335.0, 23, .8)])
    noise = torch.rand(ccd.shape, generator=torch.Generator().manual_seed(5)) * 0.001
    pos, power, capture = model.route_from_ccd(torch.stack((ccd, ccd + noise)))
    assert (pos[0] - pos[1]).abs().max() < 0.1
    assert (overlap(pos) <= 0.25).all()
    assert torch.isfinite(capture).all()
    torch.testing.assert_close(power.sum(1), torch.ones(2))


def test_identical_model_state_and_real_optical_gradients():
    torch.manual_seed(17)
    model = FootprintRoutingLG()
    torch.manual_seed(17)
    baseline = SpatialRoutingLG("continuous")
    assert model.state_dict().keys() == baseline.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, baseline.state_dict()[name])
    amp = torch.rand(2, 224, 224)
    result = model(amp)
    assert (overlap(result["positions"]) <= 0.25).all()
    torch.nn.functional.cross_entropy(result["logits"], torch.tensor([2, 3])).backward()
    for param in model.parameters():
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
