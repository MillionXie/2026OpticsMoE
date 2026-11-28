import numpy as np
import torch

from LightGenV2.tasks.t24_continuous_spatial_routing.audit_routes import (
    derangement, effective_pixels, evaluate_intervention, rank_matched_donors,
    shifted_router_input, unordered_distance,
)
from LightGenV2.tasks.t24_continuous_spatial_routing.model import SpatialRoutingLG


def test_unordered_matching_and_null():
    a = np.array([[[2., 0.], [8., 0.]], [[5., 0.], [10., 0.]]])
    distance, swapped = unordered_distance(a, a[:, ::-1])
    np.testing.assert_array_equal(distance, [0., 0.])
    assert swapped.all()
    p = derangement(100, 4423)
    assert np.array_equal(np.sort(p), np.arange(100)) and np.all(p != np.arange(100))
    np.testing.assert_array_equal(p, derangement(100, 4423))
    assert effective_pixels(torch.ones(4, 4)) == 16
    gates = np.array([[.3, .7], [.6, .4]])
    donor, swaps = rank_matched_donors(a, gates, np.array([1, 0]))
    assert swaps.all()
    np.testing.assert_array_equal(donor[0, 1], a[1, 0])


def test_real_frozen_interventions():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    fields = torch.rand(4, 224, 224)
    model = SpatialRoutingLG('continuous').eval()
    device = torch.device('cpu')
    clean, usage = evaluate_intervention(model, fields, 2, device)
    perm = derangement(len(fields), 4423)
    shuffled, _ = evaluate_intervention(model, fields, 2, device, positions=clean['positions'][perm])
    np.testing.assert_array_equal(shuffled['positions'], clean['positions'][perm])
    np.testing.assert_array_equal(shuffled['power'], clean['power'])
    assert 0 <= usage['mean_power_fraction_in_original_cross_gap'] <= 1
    noisy, _ = evaluate_intervention(model, fields, 2, device, photons=1e6, noise_seed=1069)
    replay, _ = evaluate_intervention(model, fields, 2, device, photons=1e6, noise_seed=1069)
    np.testing.assert_array_equal(noisy['positions'], replay['positions'])
    assert np.any(noisy['positions'] != clean['positions'])
    shifted, _ = evaluate_intervention(model, fields, 2, device, shift_router=True)
    assert np.any(shifted['positions'] != clean['positions'])
    quantized, _ = evaluate_intervention(model, fields, 2, device, positions=np.rint(clean['positions']))
    np.testing.assert_array_equal(quantized['positions'], np.rint(clean['positions']))
    np.testing.assert_array_equal(quantized['power'], clean['power'])
    clean2, _ = evaluate_intervention(model, fields, 2, device)
    np.testing.assert_array_equal(clean['scores'], clean2['scores'])


def test_right_shift_has_no_tile_leak_and_conserves_router_input_power():
    fields = torch.arange(224 * 224, dtype=torch.float32).reshape(1, 224, 224)
    fields[:, 112:, 112:] = 0
    expected = fields.clone()
    for left in (0, 112):
        expected[:, :, left + 1:left + 112] = fields[:, :, left:left + 111]
    result = shifted_router_input(fields)
    torch.testing.assert_close(result, expected / expected.norm())
    torch.testing.assert_close(result.norm(), torch.tensor(1.), rtol=1e-4, atol=1e-5)
