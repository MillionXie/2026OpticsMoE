from types import SimpleNamespace
import unittest

import torch

from ..model import ModelConfig, Propagation
from ..tools.spatial_locality import isolated_video_response, audit_spatial_locality, spatial_locality_loss
from ..tools.analyze_spatial_coupling import own_video_propagation
from ..tools.audit_mask_illumination import field_statistics


class SpatialLocalityTests(unittest.TestCase):
    def test_illumination_area_counts_pixels_and_detects_power_concentration(self):
        fields = torch.ones(1, 16, 4, 27, 27)
        uniform = field_statistics(fields)
        expected = 100 * 4 * 27 * 27 / (111 * 111)
        self.assertAlmostEqual(float(uniform[0, 0, 0]), expected)
        self.assertAlmostEqual(float(uniform[0, 0, 1]), expected)
        fields.zero_()
        fields[:, :, 0, 0, 0] = 1
        concentrated = field_statistics(fields)
        self.assertAlmostEqual(float(concentrated[0, 0, 0]), 100 / (111 * 111))
        fields.zero_()
        self.assertTrue(torch.equal(field_statistics(fields), torch.zeros(1, 16, 3, dtype=torch.float64)))

    def make_stage(self, propagation):
        return SimpleNamespace(origins=[(23 + 119 * row, 23 + 119 * column)
                                        for row in range(4) for column in range(4)],
                               propagation=propagation)

    def test_coupling_diagnostic_preserves_identity_inside_video_regions(self):
        stage = self.make_stage(torch.nn.Identity())
        field = torch.zeros(1, 518, 518, dtype=torch.complex64)
        for row, column in stage.origins:
            field[:, row + 30, column + 30] = 1 + 1j
        actual = own_video_propagation(stage, stage.propagation.forward, field)
        self.assertTrue(torch.equal(actual, field))

    def test_coupling_diagnostic_removes_a_source_arriving_in_another_video(self):
        stage = self.make_stage(torch.nn.Identity())
        field = torch.zeros(1, 518, 518, dtype=torch.complex64)
        field[:, 60, 60] = 1
        translate = lambda value: torch.roll(value, 119, dims=-1)
        self.assertEqual(float(translate(field).abs().sum()), 1.)
        isolated = own_video_propagation(stage, translate, field)
        self.assertEqual(float(isolated.abs().sum()), 0.)

    def test_cross_video_loss_stops_below_the_requested_energy_threshold(self):
        class Mixing(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.amount = torch.nn.Parameter(torch.tensor(.1))

            def forward(self, value):
                return value + self.amount * torch.roll(value, 119, dims=-1)

        propagation = Mixing()
        stage = self.make_stage(propagation)
        model = SimpleNamespace(frame_global=stage, video_global=stage)
        field = torch.zeros(1, 518, 518, dtype=torch.complex64)
        field[:, 60, 60] = 1
        captured = {'vision': field, 'language': field}
        valid = torch.ones(1, 16, dtype=torch.bool)
        loss = spatial_locality_loss(model, captured, valid, [0], foreign_threshold=.01)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(propagation.amount.grad), 0.)
        with torch.no_grad():
            propagation.amount.fill_(.2)
        propagation.zero_grad(set_to_none=True)
        loss = spatial_locality_loss(model, captured, valid, [0], foreign_threshold=.01)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        self.assertGreater(float(propagation.amount.grad), 0.)

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

    def test_per_video_audit_arrays_keep_ids_and_exclude_padding(self):
        stage = self.make_stage(torch.nn.Identity())

        class PointModel(torch.nn.Module):
            frame_global = stage
            video_global = stage

            def forward(self, tokens):
                field = torch.zeros(len(tokens), 518, 518, dtype=torch.complex64)
                for row, column in stage.origins:
                    field[:, row + 30, column + 30] = 1
                self.frame_global.propagation(field)
                self.video_global.propagation(field)

        payload = {'ids': [str(index) for index in range(18)], 'splits': ['test'] * 18,
                   'vision': torch.zeros(18, 1), 'target': torch.zeros(18)}
        summary, arrays = audit_spatial_locality(PointModel(), payload, 'cpu', field_count=2)
        self.assertEqual(summary['videos'], 18)
        self.assertEqual(arrays['sample_ids'].tolist(), payload['ids'])
        self.assertEqual(arrays['vision_per_video_fractions'].shape, (18, 16))
        self.assertEqual(arrays['language_per_video_foreign_fraction'].shape, (18,))
        self.assertEqual(summary['vision']['foreign_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
