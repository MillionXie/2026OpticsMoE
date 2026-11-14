from types import SimpleNamespace
import unittest

import torch

from ..tools.optical_training import illumination_coverage_loss, registered_optics, registration_condition


class OpticalTrainingTests(unittest.TestCase):
    def test_illumination_penalty_increases_for_concentrated_energy(self):
        valid = torch.ones(1, 16, dtype=torch.bool)
        minimum = {'vision': .05, 'language': .05}
        broad = torch.ones(1, 16, 30, 30)
        captured = {branch + '_incident': broad for branch in minimum}
        self.assertEqual(float(illumination_coverage_loss(captured, valid, minimum)), 0.)
        concentrated = torch.zeros_like(broad)
        concentrated[:, :, 10:15, 10:15] = 6
        captured = {branch + '_incident': concentrated for branch in minimum}
        self.assertGreater(float(illumination_coverage_loss(captured, valid, minimum)), .9)

    def test_illumination_penalty_excludes_padding_and_preserves_gradients(self):
        field = torch.rand(1, 16, 10, 10, requires_grad=True)
        captured = {branch + '_incident': field for branch in ('vision', 'language')}
        valid = torch.ones(1, 16, dtype=torch.bool)
        valid[:, -1] = False
        loss = illumination_coverage_loss(captured, valid, {'vision': .1, 'language': .1})
        loss.backward()
        self.assertTrue(bool(torch.isfinite(field.grad).all()))
        self.assertGreater(float(field.grad[:, :-1].abs().sum()), 0.)
        self.assertEqual(float(field.grad[:, -1].abs().sum()), 0.)

    def test_registration_sampling_is_bounded_and_has_a_clean_default(self):
        self.assertEqual(registration_condition(0), {'kind': 'clean'})
        with self.assertRaises(ValueError):
            registration_condition(-1)
        torch.manual_seed(815)
        conditions = [registration_condition(1) for _ in range(40)]
        self.assertEqual({value['kind'] for value in conditions}, {'clean', 'phase_shift', 'ccd_shift'})
        for value in conditions:
            if value['kind'] != 'clean':
                self.assertEqual(abs(value['dy']) + abs(value['dx']), 1)

    def test_phase_registration_restores_methods_and_allows_phase_gradients(self):
        class Stage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.phase = torch.nn.Parameter(torch.randn(4, 4))
                self.propagation = torch.nn.Identity()

            def modulation(self):
                return torch.exp(1j * self.phase)

        stage = Stage()
        other = Stage()
        model = SimpleNamespace(frame_global=stage, video_global=other)
        original = stage.modulation()
        condition = {'kind': 'phase_shift', 'dy': 1, 'dx': 0}
        with registered_optics(model, condition):
            shifted = stage.modulation()
            self.assertTrue(torch.equal(shifted[0], torch.ones(4, dtype=shifted.dtype)))
            shifted.real.sum().backward()
        self.assertTrue(torch.equal(stage.modulation(), original))
        self.assertGreater(float(stage.phase.grad.abs().sum()), 0.)

    def test_ccd_registration_restores_hooks_after_failure(self):
        stage = SimpleNamespace(propagation=torch.nn.Identity())
        model = SimpleNamespace(frame_global=stage, video_global=stage)
        with self.assertRaises(RuntimeError):
            with registered_optics(model, {'kind': 'ccd_shift', 'dy': 1, 'dx': 0}):
                raise RuntimeError('test cleanup')
        self.assertEqual(len(stage.propagation._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
