import unittest
import numpy as np
import torch
from .probe_TRAIN_phase_response import control_phases,difference


class PhaseProbeTests(unittest.TestCase):
    def test_fixed_conditions_preserve_canonical_and_phase_levels(self):
        source=np.full((1016,1016),.25,np.float32);p=control_phases(source)
        self.assertTrue(np.array_equal(source,p['canonical']));self.assertEqual(set(np.unique(p['checker_pi'])),{0.,.5})
        self.assertEqual(float(p['uniform_pi'].mean()),.5)

    def test_uniform_pi_is_no_intensity_change_in_phase_only_reference(self):
        torch.manual_seed(5);x=torch.rand((16,16),dtype=torch.float64)
        a=torch.fft.fft2(x.to(torch.complex128)).abs().square().numpy()
        b=torch.fft.fft2(x*torch.exp(torch.tensor(1j*torch.pi,dtype=torch.complex128))).abs().square().numpy()
        self.assertLess(difference(a,b)['relative_L2'],1e-12)


if __name__=='__main__':unittest.main()
