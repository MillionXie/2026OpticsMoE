import unittest

import torch

from ..tools.evaluate_detector_noise import perturb_intensity


class DetectorNoiseTests(unittest.TestCase):
    def test_zero_read_noise_preserves_values_and_generator(self):
        intensity = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        generator = torch.Generator().manual_seed(5)
        before = generator.get_state().clone()
        observed = perturb_intensity(intensity, 'gaussian', 0., 2., generator)
        self.assertIs(observed, intensity)
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_read_noise_is_reproducible_and_retains_nonnegative_intensity(self):
        intensity = torch.zeros(20000)
        original = intensity.clone()
        first = perturb_intensity(intensity, 'gaussian', .1, 2., torch.Generator().manual_seed(7))
        second = perturb_intensity(intensity, 'gaussian', .1, 2., torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(intensity, original))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertGreater(float(first.max()), 0.)
        self.assertGreaterEqual(float(first.min()), 0.)

    def test_poisson_mean_and_variance_follow_reference_scaling(self):
        intensity = torch.full((50000,), 3.)
        observed = perturb_intensity(intensity, 'poisson', 100., 2., torch.Generator().manual_seed(11))
        self.assertLess(abs(float(observed.mean()) - 3.), .01)
        self.assertLess(abs(float(observed.var()) - .06), .003)

    def test_invalid_reference_or_level_is_rejected(self):
        intensity = torch.ones(2, 2)
        for kind, level, reference in (('gaussian', -.1, 2.), ('gaussian', .1, 0.),
                                       ('poisson', 0., 2.), ('other', 1., 2.)):
            with self.subTest(kind=kind, level=level, reference=reference), self.assertRaises(ValueError):
                perturb_intensity(intensity, kind, level, reference, torch.Generator())


if __name__ == '__main__':
    unittest.main()
