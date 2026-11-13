from types import SimpleNamespace
import unittest

import torch

from ..model import ModelConfig, Propagation
from ..tools.spatial_locality import isolated_video_response


class SpatialLocalityTests(unittest.TestCase):
    def make_stage(self, propagation):
        return SimpleNamespace(origins=[(23 + 119 * row, 23 + 119 * column)
                                        for row in range(4) for column in range(4)],
                               propagation=propagation)

    def test_identity_propagation_keeps_each_source_in_its_own_region(self):
        stage = self.make_stage(torch.nn.Identity())
        field = torch.zeros(1, 518, 518, dtype=torch.complex64)
        for row, column in stage.origins:
            field[:, row + 30, column + 30] = 1
        response = isolated_video_response(stage, field, range(16))
        self.assertTrue(torch.equal(response['fractions'][0], torch.eye(16)))
        self.assertTrue(torch.equal(response['foreign_fraction'], torch.zeros(1, 16)))

    def test_whole_field_propagation_has_finite_phase_gradients(self):
        stage = self.make_stage(Propagation(ModelConfig()))
        phase = torch.randn(1, 518, 518, requires_grad=True)
        amplitude = torch.zeros_like(phase)
        amplitude[:, 60:87, 60:87] = 1
        response = isolated_video_response(stage, amplitude * torch.exp(1j * phase), [0])
        response['foreign_fraction'].mean().backward()
        self.assertTrue(bool(torch.isfinite(phase.grad).all()))
        self.assertGreater(float(phase.grad.abs().sum()), 0)

    def test_dark_source_is_reported_without_spurious_guard_energy(self):
        stage = self.make_stage(torch.nn.Identity())
        response = isolated_video_response(stage, torch.zeros(1, 518, 518, dtype=torch.complex64), [0])
        self.assertEqual(float(response['dark_source_fraction'].item()), 1.)
        self.assertEqual(float(response['guard_fraction'].item()), 0.)
        self.assertEqual(float(response['filter_transmission'].item()), 0.)


if __name__ == '__main__':
    unittest.main()
