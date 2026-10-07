"""Synthetic checks only: no datasets, checkpoints, or devices."""
import unittest
import torch
from LightGenV2.tasks.t04_openmoji_robust_ablation.spatial_alignment import perturb_field


class SpatialAlignmentTests(unittest.TestCase):
    def test_zero_is_exact_identity(self):
        x=torch.randn(2,3,32,32,dtype=torch.complex64)
        self.assertIs(perturb_field(x,0,0,0),x)

    def test_zero_field_and_complex_gradient(self):
        x=torch.zeros(2,3,32,32,dtype=torch.complex64,requires_grad=True)
        y=perturb_field(x,1,.25,.02)
        self.assertEqual(y.shape,x.shape)
        self.assertTrue(torch.equal(y,torch.zeros_like(y)))
        y.abs().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_shared_real_imag_geometry(self):
        r=torch.randn(2,3,32,32)
        y=perturb_field(torch.complex(r,2*r),1,.25,.02)
        self.assertTrue(torch.allclose(y.imag,2*y.real))

    def test_dropout_does_not_rescale(self):
        y=perturb_field(torch.ones(2,3,32,32),0,0,.5)
        self.assertTrue(((y==0)|(y==1)).all())
        self.assertTrue(torch.equal(y[0,0],y[1,2]))


if __name__=='__main__':
    unittest.main()
