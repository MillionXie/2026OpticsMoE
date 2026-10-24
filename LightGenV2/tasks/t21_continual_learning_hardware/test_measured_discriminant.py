import unittest
import torch
from .refine_measured_discriminant import discriminant_weight
from .refine_measured_ce import contrast_transform


class MeasuredDiscriminantTests(unittest.TestCase):
    def test_finite_rank_deficient_covariance_and_original_fold(self):
        torch.manual_seed(7);y=torch.arange(10).repeat_interleave(4)
        x=1+.01*torch.randn(40,18,dtype=torch.float64);x=x/x.mean(1,keepdim=True)
        t=contrast_transform(x);z=x@t.T;w=discriminant_weight(z,y,.1)
        self.assertTrue(torch.isfinite(w).all());self.assertEqual(w.shape,(10,18))
        self.assertTrue(torch.allclose(z@w.T,x@(w@t).T,atol=1e-9,rtol=1e-9))

    def test_unbalanced_TRAIN_or_invalid_strength_rejected(self):
        z=torch.ones(30,8,dtype=torch.float64);y=torch.zeros(30,dtype=torch.long)
        with self.assertRaises(ValueError):discriminant_weight(z,y,.1)
        with self.assertRaises(ValueError):discriminant_weight(z,y,0)


if __name__=='__main__':unittest.main()
