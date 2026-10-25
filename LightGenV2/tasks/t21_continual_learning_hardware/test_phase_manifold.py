import unittest
import numpy as np
from .analyze_TRAIN_phase_response import phase_manifold


class PhaseManifoldTests(unittest.TestCase):
    def test_recovers_unwrapped_two_level_phase_only_trace(self):
        rng=np.random.default_rng(3);phase=np.array([0,.4,1.4,3.,5.4,8.,9.,9.6,10.])
        matrix=20+rng.normal(size=(1,1000))+np.cos(phase)[:,None]*rng.normal(size=(1,1000))+np.sin(phase)[:,None]*rng.normal(size=(1,1000))
        r=phase_manifold(matrix)
        self.assertGreater(r['rank2_variance_fraction'],.999999)
        self.assertLess(r['ellipse_radial_RMS'],1e-10)
        self.assertTrue(np.allclose(r['positive_orientation_phase_radians'],phase,atol=1e-9))

    def test_nonfinite_or_degenerate_not_a_phase_curve(self):
        with self.assertRaises(ValueError):phase_manifold(np.zeros((9,100)))
        with self.assertRaises(ValueError):phase_manifold(np.full((9,100),np.nan))


if __name__=='__main__':unittest.main()
