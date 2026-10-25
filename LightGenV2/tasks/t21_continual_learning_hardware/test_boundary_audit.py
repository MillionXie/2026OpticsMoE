import unittest
import torch
from .audit_TRAIN_free_space_boundary import padded_intensity
from LightGenV2.tasks.t20_reconfigurable_rgb_sar.reference.optics import AngularSpectrumPropagator


class BoundaryAuditTests(unittest.TestCase):
    def test_same_aperture_and_uniform_phase_invariance(self):
        torch.manual_seed(2);a=torch.rand(1,8,8);p=AngularSpectrumPropagator(532e-9,8e-6,24,.1)
        x=padded_intensity(a,torch.zeros(8,8),p);y=padded_intensity(a,torch.full((8,8),.5),p)
        self.assertEqual(x.shape,(1,8,8));self.assertTrue(torch.allclose(x,y,atol=1e-6,rtol=1e-5))
        with self.assertRaises(ValueError):padded_intensity(a,torch.zeros(8,8),AngularSpectrumPropagator(532e-9,8e-6,9,.1))


if __name__=='__main__':unittest.main()
