import unittest
import torch
from .refine_measured_contrast import augment


class ContrastAugmentationTests(unittest.TestCase):
    def test_background_not_dropped(self):
        x=torch.ones(100,784,dtype=torch.float64)
        self.assertTrue(torch.equal(augment(x,x.mean(0,keepdim=True),torch.zeros(16,784,dtype=x.dtype)),x))

    def test_DC_restored_and_finite(self):
        torch.manual_seed(8);x=1+.01*torch.randn(100,784,dtype=torch.float64);x=x/x.mean(1,keepdim=True)
        noise=.005*torch.randn(16,784,dtype=x.dtype);noise-=noise.mean(1,keepdim=True)
        out=augment(x,x.mean(0,keepdim=True),noise)
        self.assertTrue(torch.isfinite(out).all());self.assertLess(float((out.mean(1)-1).abs().max()),1e-12)
        self.assertLess(float((out-x).std()),.01)


if __name__=='__main__':unittest.main()
